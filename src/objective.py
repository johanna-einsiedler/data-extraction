import hashlib
import json
import logging
import functools
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from datetime import datetime, timezone
import optuna
import pandas as pd
from dotenv import find_dotenv, load_dotenv

# Load .env BEFORE importing the scorer: evaluate.py freezes JUDGE_BACKEND /
# EQUIV_JUDGE_MODEL at import time. Until 2026-10-01 this ran after the imports,
# so the file's judge setting never reached a worker and every search ran the
# module default (local Phi-3-mini) -- see the judge-identity audit in the
# Phase-2 deviations addendum. .env now declares that same judge explicitly.
load_dotenv(find_dotenv())

from _data_loading import (
    CURATED_MC_ITEMS,
    OUTCOMES,
    QUERIES_JSON_PATH,
    build_system_prompt,
    finalized_item_map,
    load_doc_id_map,
    load_full_document_text,
    load_queries,
    load_true_answers,
    load_true_answers_final,
    save_doc_id_map,
)
from _llm_utils import _clean_llm_text, get_cleaned_text_and_logprobs, get_llm_text
from answer_generation.answer_generator import (
    AnswerGenerator,
    InsufficientCreditsError,
)
from document_parsing.base_parser import ParseResult
from evaluate import evaluate_answer
import batched
import binary_mode
from registry import (
    CHUNKERS,
    EMBEDDERS,
    LLM_META_MAP,
    LLMS,
    LLMS_META,
    OPENAI_API_KEY,
    PARSERS,
    RETRIEVERS,
    TOGETHER_API_KEY,
)
from vectorstore.numpy_store import NumpyVectorStore

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------
# Paper-level concurrency configuration
# ---------------------------------------------------------------------
# PAPER_CONCURRENCY == 1 -> fully sequential, identical to the original behavior.
# PAPER_CONCURRENCY  > 1 -> process papers in a ThreadPoolExecutor so the slow
# per-paper network calls (ans_gen.generate) overlap across papers.
PAPER_CONCURRENCY = int(os.getenv("PAPER_CONCURRENCY", "4"))
# _MODEL_LOCK serializes every torch/model + retrieval op (chunking cache,
# embedding, vector-store load, query embedding, retrieval, and the Phi-3 judge
# invoked inside evaluate_answer). _CACHE_LOCK guards writes to the shared
# `cache` dict and save_cache().
_MODEL_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()

# ---------------------------------------------------------------------
# VLM (image) mode helpers
# ---------------------------------------------------------------------
# When vlm=True the model reads the page images, so the text {context} slot only
# needs a pointer (sending the parsed text too would defeat the purpose and cost
# a lot of extra tokens).
_VLM_CTX = "(The full article is attached below as page images; read them to answer.)"


def _load_page_images(pdf_path):
    """Render a PDF's pages to PIL images for VLM mode. VLM_ZOOM (default 2.0 ≈
    144 DPI) keeps dense tables/numbers legible. Drop to 1.5 for Together/MiniMax,
    whose long papers timed out at 2.0 on >10 MB image payloads (OpenAI/Gemini
    tolerate 2.0). VLM_MAX_PAGES caps pages/paper."""
    import fitz  # PyMuPDF
    from PIL import Image

    zoom = float(os.getenv("VLM_ZOOM", "2.0"))
    cap = os.getenv("VLM_MAX_PAGES")
    cap = int(cap) if cap else None
    mat = fitz.Matrix(zoom, zoom)
    imgs = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            if cap is not None and i >= cap:
                break
            pix = page.get_pixmap(matrix=mat)
            imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return imgs


def _demote_trial(storage_url, study_id, trial_no):
    """Mark a COMPLETE trial FAIL so a rerun replaces it instead of duplicating it
    (sqlite storages only). Its on-disk trial_logs/ data is untouched — only the
    study row changes."""
    if not storage_url.startswith("sqlite"):
        logger.warning(
            "Non-sqlite storage: not auto-demoting #%d; the rerun will add a new "
            "trial alongside the old one.", trial_no
        )
        return
    import sqlite3

    path = storage_url.replace("sqlite:///", "")
    con = sqlite3.connect(path)
    con.execute(
        "UPDATE trials SET state='FAIL', datetime_complete=datetime('now') "
        "WHERE number=? AND study_id=? AND state='COMPLETE'",
        (trial_no, study_id),
    )
    con.commit()
    con.close()
    logger.info("Old trial #%d demoted to FAIL (its trial_logs/ data is kept).", trial_no)


def rerun_specific_trial(study, storage_url, trial_no, keep_old=False,
                         resume_papers=False):
    """Re-run an existing trial's EXACT config as a fresh trial. Current env knobs
    (VLM_ZOOM, REQUEST_TIMEOUT, PAPER_CONCURRENCY, …) apply to the new run — so e.g.
    `VLM_ZOOM=1.5 ... --rerun=332` reruns #332's config at the new zoom.

    resume_papers=True (--resume-papers) seeds the new trial with the OLD trial's
    finished doc_*.json, so a failed/partial run (e.g. gemini quota) picks up from
    the papers it already completed and only computes the missing ones."""
    global _RESUME_FROM_FOLDER
    src = next((t for t in study.trials if t.number == trial_no), None)
    if src is None:
        raise SystemExit(f"[--rerun] trial #{trial_no} not found in the study.")
    # A COMPLETE/RUNNING trial exposes its resolved config in .params; a
    # WAITING/failed-early one keeps it in system_attrs['fixed_params'].
    cfg = dict(src.params) if src.params else dict(
        src.system_attrs.get("fixed_params", {})
    )
    if not cfg:
        raise SystemExit(f"[--rerun] trial #{trial_no} has no recorded config to rerun.")
    logger.info("[--rerun] #%d (%s) config: %s", trial_no, src.state.name, cfg)
    if resume_papers:
        cands = list((PROJECT_ROOT / "trial_logs").glob(f"*_trial_{trial_no:04d}"))
        if cands:
            _RESUME_FROM_FOLDER = str(cands[0])
            logger.info("[--rerun] resuming finished papers from %s", _RESUME_FROM_FOLDER)
        else:
            logger.warning(
                "[--rerun] --resume-papers: no trial_logs folder for #%d; running fresh.",
                trial_no,
            )
    if not keep_old and src.state.name == "COMPLETE":
        _demote_trial(storage_url, study._study_id, trial_no)
    study.enqueue_trial(cfg, skip_if_exists=False)

    def _obj(t):
        acc = objective(t, verbose=True)
        print(
            f"\n>>> Rerun of #{trial_no} -> new trial #{t.number}  acc={acc:.2%}  "
            f"model={t.params.get('llm_model', '?')}"
        )
        return acc

    study.optimize(_obj, n_trials=1, n_jobs=1, catch=(Exception,))
    print("\n" + "=" * 60)
    print(
        f"[--rerun] complete — new trial replaces #{trial_no}"
        + ("" if keep_old else f" (old #{trial_no} -> FAIL).")
    )
    print("=" * 60)


def _build_response_format(
    question_type: str, is_binary: bool = False, api_type: str = "together"
) -> dict | None:
    """Return a response_format dict that constrains model output to the answer only.

    Together supports regex mode; OpenAI supports json_schema with optional pattern
    constraints. Both paths return None for question types where free-form text is
    expected (list, open_ended) so those fall back to prompt-only guidance.

    OpenAI responses will be wrapped in {"answer": "..."} JSON — answer_generator
    automatically unwraps this before returning the text.
    """
    if api_type == "together":
        if is_binary:
            return {"type": "regex", "pattern": "(A|B)"}
        if question_type == "single_choice":
            # SC answers are numeric codes (1, 2, ...) or "Z" (not stated) -- NOT letters
            return {"type": "regex", "pattern": r"(?:\d+|[Zz])"}
        if question_type == "multiple_choice":
            return {"type": "regex", "pattern": "[A-Z](,[A-Z])*"}
        if question_type == "numeric":
            # allow "Z" so the model can abstain (not reported); evaluate maps Z->0
            return {"type": "regex", "pattern": r"(?:-?\d*\.?\d+|[Zz])"}
        return None

    elif api_type == "openai":
        # Map question types to regex patterns for string properties
        if is_binary:
            pattern = "(A|B)"
        elif question_type == "single_choice":
            # SC answers are numeric codes (1, 2, ...) or "Z" (not stated)
            pattern = r"^(?:\d+|[Zz])$"
        elif question_type == "multiple_choice":
            pattern = "^[A-Z](,[A-Z])*$"
        elif question_type == "numeric":
            # allow "Z" so the model can abstain (not reported); evaluate maps Z->0
            pattern = r"^(?:-?\d*\.?\d+|[Zz])$"
        else:
            return None

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string", "pattern": pattern}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        }

    return None


def _select_prompt(
    qid: str, qdata: dict, default_prompt: str, use_default: bool
) -> tuple[str, str | None]:
    """Resolve which prompt variant to use and return (prompt_type, prompt_text)."""

    prompts = qdata.get("prompts", {})
    prompt_type = default_prompt

    if (
        qdata.get("type") in {"multiple_choice", "single_choice"}
        and "binary_prompts" in prompts
        and len(qdata.get("choices", [])) > 2
        and not use_default
    ):
        return "binary_prompts", None

    if prompt_type == "true_few_shot_prompt" and not prompts.get(
        "true_few_shot_prompt"
    ):
        prompt_type = "synthetic_few_shot_prompt"

    prompt_text = prompts.get(prompt_type)
    if not prompt_text:
        raise ValueError(f"No prompt found for query {qid} under '{prompt_type}'")

    return prompt_type, prompt_text


def get_parser(name, **overrides):
    """Factory to lazily instantiate parsers with optional overrides."""
    entry = PARSERS[name]
    cls = entry["cls"]
    kwargs = {**entry["kwargs"], **overrides}
    return cls(**kwargs)


# ---------------------------------------------------------------------
# helper path builders (change base folders if you like)
# ---------------------------------------------------------------------
def parsed_dir(parser_name: str, base: str = None) -> Path:
    if base is None:
        base = str(PROJECT_ROOT / "data" / "parsed")
    return Path(base) / parser_name


# ---------------------------------------------------------------------
# Pre-flight guards for the TPE stage (preregistered cut rules, 2026-09-28).
# All three are outcome-independent: a pruned trial costs seconds and never
# enters the surrogate. Retro-validated on the 56 costed warm-up trials:
# nothing above 0.73 (old basis) would have been refused.
# ---------------------------------------------------------------------
COST_CAP_USD_PER_PAPER = float(os.getenv("COST_CAP_USD_PER_PAPER", "0.50"))
TIME_CAP_S_PER_PAPER = int(os.getenv("TIME_CAP_S_PER_PAPER", "900"))
# Absolute wall-clock ceiling for one trial. The pace and realised-cost guards
# are COMPLETION-triggered: consumer #54 sat at 0/35 papers after 60 min, so
# none of them could ever evaluate. 4 h clears every legitimate trial observed
# (slowest healthy: GLM-5.2 batched 201 min, DeepSeek RAG 172 min) while
# catching the provider-degradation crawls (#53 465 min, #90 21 h).
TRIAL_WALL_BUDGET_S = int(os.getenv("TRIAL_WALL_BUDGET_S", "14400"))


@functools.lru_cache(maxsize=None)
def _model_meta():
    f = PROJECT_ROOT / "src" / "answer_generation" / "filtered_models.json"
    with open(f, "r", encoding="utf-8") as fh:
        return {m["id"]: m for m in json.load(fh)}


@functools.lru_cache(maxsize=None)
def _avg_doc_tokens(parser_name: str) -> float:
    """Mean parsed-document size for this parser, in ~tokens (bytes/4)."""
    sizes = [q.stat().st_size for ext in ("*.md", "*.txt", "*.xml")
             for q in parsed_dir(parser_name).glob(ext)]
    return (sum(sizes) / len(sizes)) / 4.0 if sizes else 16000.0


def model_choice_list() -> list:
    """The llm_model categorical, honouring restriction envs (order-preserving).

    OPEN_MODELS_ONLY=1  -> drop proprietary (api_type openai/gemini).
    MODEL_ALLOWLIST=a,b -> keep exactly these roster ids (applied after the
    open filter). Seeded studies MUST build their trial distributions with
    the same envs, or add_trial rejects them."""
    ids = list(LLMS.keys())
    if os.getenv("OPEN_MODELS_ONLY"):
        meta = _model_meta()
        ids = [i for i in ids
               if (meta.get(i) or {}).get("api_type") not in ("openai", "gemini")]
    allow = os.getenv("MODEL_ALLOWLIST")
    if allow:
        keep = {a.strip() for a in allow.split(",") if a.strip()}
        unknown = keep - set(ids)
        assert not unknown, f"MODEL_ALLOWLIST ids not in roster/filter: {unknown}"
        ids = [i for i in ids if i in keep]
    return ids


def project_cost_per_paper(choices: dict, meta: dict | None = None) -> float:
    """Deterministic $-per-paper projection from the sampled config alone.

    full/batched: one call carrying the parsed doc + all 43 item prompts.
    rag: 43 answer calls of min(k*chunk, doc) tokens; RCSRetriever adds one
    scoring call per item of min(top_m*chunk, doc). Payloads are capped at the
    parser's mean document size because retrieval cannot return more text than
    the paper holds (uncapped, chunk_size=5000 configs would be projected at
    2x their real cost and #122-class trials wrongly cut)."""
    m = (meta if meta is not None else _model_meta()).get(choices.get("llm_model")) or {}
    pin = float((m.get("pricing") or {}).get("input", 0)) / 1e6
    pout = float((m.get("pricing") or {}).get("output", 0)) / 1e6
    doc = _avg_doc_tokens(choices.get("parser", "PyMuPDFParser"))
    if choices.get("mode") == "full":
        return (doc + 43 * 120) * pin + 2500 * pout
    # Completion side stays OPTIMISTIC (80 tok/call): whether a reasoning model
    # actually thinks (gemini realised $0.18 AND $1.25/paper on sibling configs)
    # is unknowable pre-flight; realised spend is guarded in-flight instead, so
    # a borderline-good config is never refused on a pessimistic guess.
    comp = 80.0
    k = int(choices.get("k") or 5)
    # chunk_size is a CHARACTER budget (LengthChunker slices the raw string),
    # so ~/4 for tokens. Retro-validated: token-math over-projected #81/#100/
    # #122 by 2-4x and would have cut a 0.7717 trial.
    chunk_tok = int(choices.get("chunk_size") or 500) / 4.0
    tin = 43 * (min(k * chunk_tok, doc) + 600)
    if choices.get("retriever") == "RCSRetriever":
        tin += 43 * (min(int(choices.get("top_m") or 10) * chunk_tok, doc) + 600)
    return tin * pin + 43 * comp * pout


def preflight_guards(trial, choices: dict) -> None:
    """Raise optuna.TrialPruned for configs that must never spend a cent."""
    m = _model_meta().get(choices.get("llm_model")) or {}
    if m.get("disabled"):
        raise optuna.TrialPruned(
            f"guard: {choices.get('llm_model')} is disabled in the roster")
    # Per-model budget cap (user, 2026-09-28): once the recorded stage spend on
    # gemini reaches GEMINI_BUDGET_USD, further gemini draws are pruned for free.
    # Stage = trials >= GEMINI_BUDGET_FROM_TRIAL (TPE start). Completed spend
    # only, so the cap can overshoot by at most the trials in flight (~$2.5).
    _GB = float(os.getenv("GEMINI_BUDGET_USD", "inf"))
    if choices.get("llm_model") == "gemini-3.1-pro-preview" and _GB != float("inf"):
        _from = int(os.getenv("GEMINI_BUDGET_FROM_TRIAL", "128"))
        spent = sum((t.user_attrs.get("cost_usd") or 0.0)
                    for t in trial.study.trials
                    if t.number >= _from
                    and t.state == optuna.trial.TrialState.COMPLETE
                    and t.params.get("llm_model") == "gemini-3.1-pro-preview")
        trial.set_user_attr("gemini_stage_spend_usd", round(spent, 2))
        if spent >= _GB:
            raise optuna.TrialPruned(
                f"guard: gemini stage spend ${spent:.2f} has reached the "
                f"${_GB:.2f} cap")
    # Qwen3.5-9B: serving degradation at the provider, 2026-09-29/30. The model
    # answers short prompts in ~2 s but every full-document call returns
    # APITimeoutError, so all five post-seed draws (#50, #53-#56) failed without a
    # single completed trial (~13 h wall clock, $0). Failures carry no information
    # to the sampler, so TPE redraws the same region indefinitely. Excluded on the
    # infrastructure criterion alone -- "cannot complete a trial" is observable
    # without reference to any accuracy -- exactly as for the nemotron combination
    # below. Its 8 seed completions stay in the study and keep informing the
    # surrogate. Lift QWEN_TIMEOUT_EXCLUSION=0 once the provider recovers.
    if (choices.get("llm_model") == "Qwen/Qwen3.5-9B"
            and os.getenv("QWEN_TIMEOUT_EXCLUSION", "1") == "1"):
        raise optuna.TrialPruned(
            "guard: Qwen/Qwen3.5-9B times out on full-document calls at the "
            "provider (5/5 post-seed trials failed to complete)")
    if (choices.get("llm_model") == "nvidia/nemotron-3-ultra-550b-a55b"
            and choices.get("mode") == "full"):
        raise optuna.TrialPruned(
            "guard: nemotron x full is infeasible (OpenRouter context-length "
            "filter removes every endpoint; trial #112)")
    if not choices.get("vlm"):
        est = project_cost_per_paper(choices)
        trial.set_user_attr("projected_usd_per_paper", round(est, 4))
        if est > COST_CAP_USD_PER_PAPER:
            raise optuna.TrialPruned(
                f"guard: projected ${est:.2f}/paper exceeds the "
                f"${COST_CAP_USD_PER_PAPER:.2f} cap")


def chunks_dir(parser_name: str, chunker_name: str, base: str = None) -> Path:
    if base is None:
        base = str(PROJECT_ROOT / "data" / "chunks")
    return Path(base) / parser_name / chunker_name


def embeddings_dir(
    parser_name: str,
    chunker_name: str,
    embedder_name: str,
    base: str = None,
) -> Path:
    if base is None:
        base = str(PROJECT_ROOT / "data" / "embeddings")
    return Path(base) / parser_name / chunker_name / embedder_name


def dependency_autoresolve(qid, gold):
    """Prereg2 metrics rule for dependency-child items (3.2.1/3.4.1/3.5.1/1.2.1...):
    a cell whose GOLD is the not-applicable code -99 is resolved by the codebook's
    dependency logic without a model call and counts as correct. Any other gold
    means the item applies to this study, so the model is ALWAYS asked. The model's
    own answer to the parent item is never used to skip a child (the pre-2026-09-28
    behaviour fabricated raw_output="Not applicable"/pred -99 and mis-scored ~24-41
    MC cells per trial; found by analysis/hpo_analysis/unified_rescore.py). Returns
    the eval_result dict for an auto-resolved cell, else None (must evaluate).
    raw_output stays None so offline re-scoring keeps the stored accuracy."""
    if str(gold[0]) == str(-99):
        return {"raw_output": None, "pred_value": -99, "true_value": gold,
                "accuracy": 1}
    return None


def check_dependency(qid, evaluated_results, dependency_rules, doc_id):
    rules = dependency_rules.get(qid, {})
    depends_on = rules.get("depends_on", [])
    not_applicable_if = rules.get("not_applicable_if", [])
    doc_results = [r for r in evaluated_results if r.get("doc_id") == doc_id]

    # Ensure dependencies exist
    for dep_qid in depends_on:
        if dep_qid not in [r["query_id"] for r in doc_results]:
            raise ValueError(
                f"Query {qid} depends on {dep_qid}, which has not been evaluated yet "
                f"for document {doc_id or '[unknown]'}."
            )

    # Check applicability
    for condition in not_applicable_if:
        for dep_qid, blocked_value in condition.items():
            dep_entry = next((r for r in doc_results if r["query_id"] == dep_qid), None)
            if not dep_entry:
                continue  # dependency not evaluated yet

            if dep_entry["type"] == "single_choice":
                dep_value = dep_entry.get("pred_indicator")
            else:
                dep_value = dep_entry.get("pred_value")

            # Normalize dep_value to a list
            if not isinstance(dep_value, list):
                dep_value = [dep_value]

            # Check if all values are blocked; if at least one is not blocked, it's applicable
            all_blocked = True
            for v in dep_value:
                try:
                    v_num = float(v)
                    blocked_num = float(blocked_value)
                    if v_num != blocked_num:
                        all_blocked = False
                        break
                except (ValueError, TypeError):
                    if str(v).strip().lower() != str(blocked_value).strip().lower():
                        all_blocked = False
                        break

            if all_blocked:
                return False  # all values are blocked

    return True


# ---------------------------------------------------------------------
# Ensure functions: run component only if output missing
# ---------------------------------------------------------------------
def ensure_parsed(file_path: Path, parser_name: str, strict: bool = False):
    """Ensure a file is parsed. Return path to cached or newly parsed output.

    strict=True (frozen runs): a cached parse whose meta tag is not the requested
    parser's, or that records a fallback, raises ParserProvenanceError; a parser
    failure raises instead of silently falling back to PyMuPDF. Every NEW parse
    writes enriched meta (parser, version, time, pdf sha256, git commit); existing
    meta files are never rewritten."""
    pdir = parsed_dir(parser_name)
    pdir.mkdir(parents=True, exist_ok=True)

    stem = file_path.stem
    json_path = pdir / f"{stem}.json"
    txt_path = pdir / f"{stem}.txt"
    md_path = pdir / f"{stem}.md"
    xml_path = pdir / f"{stem}.xml"
    meta_path = pdir / f"{stem}.meta.json"

    print(json_path, txt_path)

    # 1️⃣ Use cached version if it exists
    for cached in (json_path, txt_path, md_path, xml_path):
        if cached.exists():
            if strict and meta_path.exists():
                try:
                    _m = json.load(open(meta_path, encoding="utf-8"))
                except Exception:
                    _m = {}
                _tag = PARSER_META_TAG.get(parser_name)
                if _m.get("fallback_from") or (_tag and _m.get("parser") and _m.get("parser") != _tag):
                    raise ParserProvenanceError(
                        f"{file_path.name}: cached parse is '{_m.get('parser')}'"
                        f"{' (fallback from ' + str(_m.get('fallback_from')) + ')' if _m.get('fallback_from') else ''}"
                        f", requested {parser_name}")
            return cached

    # 2️⃣ Run parser. If a parser cannot handle a specific PDF (e.g. MinerU on a
    #    malformed file), fall back to PyMuPDF for that paper so one bad file does
    #    not crash an entire trial. The fallback is written to THIS parser's cache
    #    dir, so it is consistent across trials and not retried.
    parser = get_parser(parser_name)
    _fallback_from = None
    try:
        result = parser.parse(str(file_path))  # make sure your parser uses `out_dir`
    except Exception as e:
        if parser_name == "PyMuPDFParser" or strict:
            raise RuntimeError(f"{parser_name} failed on {file_path.name}: {e!r}") from e
        logger.warning(
            "Parser %s failed on %s (%s); falling back to PyMuPDFParser.",
            parser_name, file_path.name, repr(e)[:140],
        )
        _fallback_from = parser_name
        result = get_parser("PyMuPDFParser").parse(str(file_path))
    _prov = {"parser_registry_name": parser_name, "parser_version": _parser_version(parser_name),
             "parsed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "pdf_sha256": _sha256_file(file_path), "pipeline_git_commit": _pipeline_git_commit()}
    if _fallback_from:
        _prov["fallback_from"] = _fallback_from
    # 3️⃣ Save result depending on type
    if isinstance(result, ParseResult):
        format_hint = result.format or "markdown"
        target_path = {
            "tei_xml": xml_path,
            "markdown": md_path,
        }.get(format_hint, md_path)
        with open(target_path, "w", encoding="utf-8") as fh:
            fh.write(result.content)
        _meta = dict(result.metadata or {})
        _meta.update(_prov)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(_meta, fh, ensure_ascii=False, indent=2)
        return target_path

    if isinstance(result, (dict, list)):
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        return json_path

    elif isinstance(result, str):
        # detect if it looks like XML
        if result.strip().startswith("<"):
            xml_path = pdir / f"{file_path.stem}.xml"
            with open(xml_path, "w", encoding="utf-8") as fh:
                fh.write(result)
            return xml_path
        else:
            with open(md_path, "w", encoding="utf-8") as fh:
                fh.write(result)
            return md_path

    elif isinstance(result, Path):
        return result

    # 4️⃣ fallback: save repr
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(repr(result))
    return md_path


def make_json_safe(obj):
    """Recursively convert any non-serializable objects to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, set, tuple)):
        return list(obj)
    elif hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        return str(obj)
    elif isinstance(obj, bytes):
        return obj.decode(errors="ignore")
    else:
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)


def ensure_chunks(
    parsed_path: Path,
    parser_name: str,
    chunker_name: str,
    chunk_size: int,
    overlap: int,
) -> Path:
    """Ensure parsed file is chunked with specific parameters.
    Cached by docname + chunk size + overlap."""
    cdir = chunks_dir(parser_name, chunker_name)
    cdir.mkdir(parents=True, exist_ok=True)

    # include size + overlap in filename
    out_path = cdir / f"{parsed_path.stem}_s{chunk_size}_o{overlap}.json"

    # Reuse if already exists
    if out_path.exists():
        return out_path

    # Load parsed content
    if parsed_path.suffix == ".json":
        with open(parsed_path, "r", encoding="utf-8") as fh:
            parsed = json.load(fh)
    else:
        parsed = parsed_path.read_text(encoding="utf-8")

    if not isinstance(parsed, str):
        parsed = json.dumps(parsed, ensure_ascii=False)

    # Run chunker (pass params if supported)
    chunker = CHUNKERS[chunker_name](chunk_size=chunk_size, chunk_overlap=overlap)
    chunk_objects = chunker.chunk(parsed)
    chunk_payload = [
        {"text": chunk.text, "metadata": chunk.metadata} for chunk in chunk_objects
    ]

    # Save chunks
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(chunk_payload, fh, ensure_ascii=False, indent=2)
    return out_path


def ensure_embedding(
    chunks_path: Path,
    parser_name: str,
    chunker_name: str,
    embedder_name: str,
    chunk_size: int,
    overlap: int,
) -> Path:
    """Ensure embeddings exist for specific chunks (size + overlap)."""
    edir = embeddings_dir(parser_name, chunker_name, embedder_name)
    edir.mkdir(parents=True, exist_ok=True)

    out_path = edir / f"{chunks_path.stem}"

    if out_path.exists():
        return out_path

    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as fh:
        try:
            chunks_data = json.load(fh)
        except json.JSONDecodeError:
            chunks_data = fh.read().split("\n\n")

    if chunks_data and isinstance(chunks_data[0], dict):
        chunk_texts = [item.get("text", "") for item in chunks_data]
    else:
        chunk_texts = chunks_data

    # Embed
    embeddings = EMBEDDERS[embedder_name].embed(chunk_texts)

    # Save in NumpyVectorStore
    store = NumpyVectorStore(save_path=out_path)
    store.add(embeddings, chunk_texts)
    store.save()

    return out_path


def ensure_query_embeddings(
    embedder_name: str,
    doc_name: str,
    query_id: str,
    query_mode: str = "raw",  # "raw" or "label_def"
    base: str = None,
    queries_base: str = None,
) -> Path:
    """
    Ensure a single query embedding is cached.
    - Embeddings stored under {base}/{embedder_name}/{doc_name}_{query_id}_{query_mode}.npy
    - Query info and prompts are read directly from query_info.json
    """
    if base is None:
        base = str(PROJECT_ROOT / "data" / "query_embeddings")
    if queries_base is None:
        queries_base = str(PROJECT_ROOT / "query_creation")
    edir = Path(base) / embedder_name
    edir.mkdir(parents=True, exist_ok=True)

    out_path = edir / f"{doc_name}_{query_id}_{query_mode}.npy"

    # 1️⃣ If cached, reuse
    if out_path.exists():
        return out_path

    # 2️⃣ Load query_info.json
    qdir = Path(queries_base)
    info_file = qdir / "queries_with_prompts.json"  # single JSON file for all queries
    if not info_file.exists():
        raise FileNotFoundError(f"No queries.json found in {qdir}")

    with open(info_file, "r", encoding="utf-8") as fh:
        all_queries = json.load(fh)

    # Find the query by ID
    qinfo = all_queries.get(query_id)
    if qinfo is None:
        raise ValueError(f"Query ID {query_id} not found in queries.json")

    # 3️⃣ Prepare text to embed
    if query_mode == "label_def":
        parts = [qinfo.get("label", ""), qinfo.get("description", "")]
        if qinfo.get("choices"):
            choices_text = ", ".join(
                f"{k}: {v['value']}" for k, v in qinfo["choices"].items()
            )
            parts.append("Choices: " + choices_text)
        text = " ".join(p for p in parts if p)
    else:
        # "raw" mode: pick base_prompt if available
        text = qinfo.get("prompts", {}).get("base_prompt")
        if not text:
            raise ValueError(f"No 'base_prompt' found for query ID {query_id}")

    # 4️⃣ Embed
    vector = EMBEDDERS[embedder_name].embed([text])[0]
    np.save(out_path, np.asarray(vector))

    return out_path


# ---------------------------------------------------------------------
# Optuna objective (stepwise cached pipeline)
# ---------------------------------------------------------------------
def _cache_dir_default() -> Path:
    """trial_cache/ by default; TRIAL_CACHE_DIR redirects it (frozen runs keep their
    own cache next to their outputs so they never overwrite tuning cells)."""
    return Path(os.getenv("TRIAL_CACHE_DIR") or (PROJECT_ROOT / "trial_cache"))


CACHE_PATH = _cache_dir_default()
CACHE_PATH.mkdir(parents=True, exist_ok=True)

# ---------------- Frozen-config runs (2026-10-02) ----------------
# FROZEN_RUN=1 (or objective(frozen=True)) runs one fixed configuration on an
# explicit paper set with every search guard disabled, stable doc ids, a
# per-run cache and ledger, and per-paper failures recorded instead of raised.
# All defaults reproduce the Optuna path byte-for-byte.
FROZEN_RUN = os.getenv("FROZEN_RUN", "0") == "1"


def _paper_id(path: Path) -> str:
    """'006_Ahnert_2023.pdf' -> '006' (the doc_name convention used everywhere)."""
    m = re.match(r"^(\d+)", path.stem)
    return m.group(1) if m else path.stem


def resolve_run_paths(trial_choices: dict, trial_number: int, *, paper_dir=None,
                      out_dir=None, frozen: bool = False):
    """(paper_folder, trial_folder, cache_dir, trial_hash10). Pure. With the
    defaults this reproduces the historical layout exactly:
    data/tuning, trial_logs/<md5(params)[:10]>_trial_<n:04d>, trial_cache/."""
    import hashlib as _hl
    trial_hash = _hl.md5(json.dumps(trial_choices, sort_keys=True).encode()).hexdigest()[:10]
    paper_folder = Path(paper_dir) if paper_dir else PROJECT_ROOT / "data" / "tuning"
    if frozen and out_dir:
        trial_folder = Path(out_dir)
        cache_dir = Path(out_dir)
    else:
        trial_folder = PROJECT_ROOT / "trial_logs" / f"{trial_hash}_trial_{trial_number:04d}"
        cache_dir = CACHE_PATH
    return paper_folder, trial_folder, cache_dir, trial_hash


# Parser provenance: what each registry parser writes into <stem>.meta.json["parser"].
PARSER_META_TAG = {"MinerUParser": "mineru", "PyMuPDFParser": "pymupdf",
                   "PyMuPDFTesseractParser": "pymupdf_tesseract"}


class ParserProvenanceError(RuntimeError):
    """A cached parse does not come from the requested parser (frozen runs only)."""


def _parser_version(parser_name: str) -> str | None:
    try:
        if parser_name == "MinerUParser":
            import importlib.metadata as _im
            return _im.version("mineru")
        if parser_name in ("PyMuPDFParser", "PyMuPDFTesseractParser"):
            import fitz
            v = f"pymupdf {fitz.version[0]}"
            if parser_name == "PyMuPDFTesseractParser":
                try:
                    import pytesseract
                    v += f"; tesseract {pytesseract.get_tesseract_version()}"
                except Exception:
                    v += "; tesseract ?"
            return v
    except Exception:
        return None
    return None


def _pipeline_git_commit() -> str | None:
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    import hashlib as _hl
    h = _hl.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# When set (by --rerun … --resume-papers), objective() seeds a fresh trial's
# folder with the finished doc_*.json files from this source folder, so the
# per-paper resume (matched on parameters_hash) skips papers already completed by
# an earlier (e.g. failed/partial) run of the same config.
_RESUME_FROM_FOLDER = os.getenv("RESUME_FROM_FOLDER") or None
# Env override exists because --rerun's own lookup globs trial_logs/*_trial_NNNN
# and takes cands[0]; trial numbers are NOT unique across studies (a fresh DB
# restarts at 0), so that glob can pick a stale folder from another study. The
# per-doc parameters_hash check makes that safe but useless — every paper is
# recomputed. Naming the folder explicitly makes a resume actually resume.


def _hash_trial_key(params: dict, doc_idx: int, query_id: str) -> str:
    """
    Create a stable hash key for caching based on params+doc+query.
    """
    key_data = json.dumps(
        {"params": params, "doc_idx": doc_idx, "query_id": query_id},
        sort_keys=True,
    )
    return hashlib.md5(key_data.encode("utf-8")).hexdigest()


def load_cache(cache_dir: Path | None = None):
    cache_file = (cache_dir or CACHE_PATH) / "cache.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, cache_dir: Path | None = None):
    """Atomic write (tmp + rename): a crash mid-dump must never truncate the
    only copy of every paid-for LLM call."""
    cdir = cache_dir or CACHE_PATH
    cdir.mkdir(parents=True, exist_ok=True)
    cache_file = cdir / "cache.json"
    tmp = cdir / "cache.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    os.replace(tmp, cache_file)


# Mid-trial persistence. The end-of-trial save (see "Persist the cache once
# after the merge") is skipped whenever a trial FAILs or the worker is killed,
# which on 2026-09-27 lost ~$20 of gpt-4.1 calls (trial #115: fresh cells were
# only in memory when the 402 abort raised and the queue gate SIGTERMed the
# worker). Flushing after each completed paper, throttled, bounds that loss to
# one throttle window. Guarded by _CACHE_LOCK, same as the per-cell inserts.
_CACHE_LAST_FLUSH = [0.0]
CACHE_FLUSH_EVERY_S = int(os.getenv("CACHE_FLUSH_EVERY_S", "300"))


def flush_cache_throttled(cache: dict, force: bool = False, cache_dir: Path | None = None):
    now = time.time()
    if not force and now - _CACHE_LAST_FLUSH[0] < CACHE_FLUSH_EVERY_S:
        return
    with _CACHE_LOCK:
        if not force and time.time() - _CACHE_LAST_FLUSH[0] < CACHE_FLUSH_EVERY_S:
            return
        save_cache(cache, cache_dir)
        _CACHE_LAST_FLUSH[0] = time.time()


def expand_highest_perf(row_values):
    """
    Expand 4.5 item into subitems (AUC, ACC, BACC, R², r).
    Args:
        row_values (list[str]): List of string lines / entries for item 4.5.
    Returns:
        dict: mapping {"4.5.1": [...], "4.5.2": [...], ...}
    """
    subfields = ["AUC", "ACC", "BACC", "R²", "r"]
    subanswers = {f"4.5.{i}": [] for i in range(1, len(subfields) + 1)}

    # Merge all values into one text blob
    text = " ".join([str(v) for v in row_values if str(v).strip()]).strip()
    text = re.sub(r"\s+", " ", text)

    # Try to extract numeric values for each subfield
    for i, field in enumerate(subfields, start=1):
        pattern = rf"{field}\s*[:=]?\s*([-]?\d*\.?\d+|−?\d*\.?\d+|NA|-99)"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            subanswers[f"4.5.{i}"] = [m.strip() for m in matches]
        else:
            subanswers[f"4.5.{i}"] = ["-99"]  # default if missing

    return subanswers


def objective(
    trial,
    verbose: bool = False,
    query_ids: list[str] | None = None,
    doc_name_filter: str | None = None,
    *,
    paper_dir=None,
    paper_ids: list[str] | None = None,
    out_dir=None,
    frozen: bool | None = None,
):
    import hashlib
    frozen = FROZEN_RUN if frozen is None else bool(frozen)
    paper_dir = paper_dir or os.getenv("PAPER_DIR") or None
    out_dir = out_dir or os.getenv("OUT_DIR") or None
    paper_failures: dict[str, str] = {}
    import json
    from pathlib import Path

    import numpy as np

    trial_choices = {}
    all_results = []
    # Re-ranker (RCS) calls are made by the retriever, outside the per-paper usage
    # accounting; collected here and written as separate trial attrs.
    _rcs_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    llm_choices = []

    # Smoke-test lever: SMOKE_QUERY_IDS restricts to a comma-separated item subset.
    if query_ids is None:
        _sq = os.getenv("SMOKE_QUERY_IDS")
        if _sq:
            query_ids = [x.strip() for x in _sq.split(",") if x.strip()]

    def log(msg: str):
        if verbose:
            logger.info(msg)

    # SAMPLE_ONLY draws configs without running anything, and cache.json is 461 MB:
    # loading it cost ~5 s and ~2.5 GB per draw, which made generating a warm-up set
    # take hours. Skipped for sampling only -- real trials are unaffected.
    _cache_dir = Path(out_dir) if (frozen and out_dir) else CACHE_PATH
    cache = {} if os.getenv("SAMPLE_ONLY") else load_cache(_cache_dir)

    try:
        trial_start_time = time.time()
        mode = trial.suggest_categorical("mode", ["rag", "full"])
        # OPEN_MODELS_ONLY=1 restricts the categorical to open-weight models
        # (api_type not openai/gemini; gpt-oss stays in — its weights are open).
        # Order-preserving filter: seeded trials' distributions must be built
        # with EXACTLY this list, or add_trial rejects them.
        llm_model = trial.suggest_categorical("llm_model", model_choice_list())

        # ---------------- Trial-level hyperparameters ----------------
        # PARSERS_SUBSET (comma-separated) restricts which parsers TPE may sample,
        # so the study can launch with only the fully-warmed parsers and be
        # expanded later (Optuna resumes from the persistent storage).
        _parser_choices = list(PARSERS.keys())
        _ps = os.getenv("PARSERS_SUBSET")
        if _ps:
            _wanted = [p.strip() for p in _ps.split(",") if p.strip()]
            _parser_choices = [p for p in _parser_choices if p in _wanted] or _parser_choices
        parser_name = trial.suggest_categorical("parser", _parser_choices)
        if mode == "rag":
            chunk_size = trial.suggest_categorical(
                "chunk_size", [50, 100, 250, 500, 750, 1000, 2500, 5000]
            )
            chunk_overlap = trial.suggest_categorical("chunk_overlap", [0, 250, 750])
            if chunk_overlap >= chunk_size:
                raise optuna.TrialPruned()
            # `chunker` MUST use ONE fixed choice list for the whole study. Optuna
            # forbids a parameter name having different value-spaces across trials
            # ("CategoricalDistribution does not support dynamic value space"), and
            # the previous parser-conditional list (all chunkers for non-PyMuPDF
            # parsers, restricted for PyMuPDF) meant every non-PyMuPDF-parser RAG
            # trial FAILED at this line once a PyMuPDF trial had recorded the
            # 2-choice distribution — silently killing ~half the RAG search space.
            # Use the study's recorded 2-choice list. StructureChunker is dropped:
            # it only ever applied to the failing parsers and never completed a
            # single trial, so nothing real is lost.
            allowed_chunkers = [
                k for k in CHUNKERS if k not in ["StructureChunker", "TextChunker"]
            ]
            chunker_name = trial.suggest_categorical("chunker", allowed_chunkers)
            embedder_name = trial.suggest_categorical(
                "embedder", list(EMBEDDERS.keys())
            )
            retriever_name = trial.suggest_categorical(
                "retriever", list(RETRIEVERS.keys())
            )
            retriever_cls = RETRIEVERS[retriever_name]
            k = trial.suggest_categorical("k", [2, 3, 5, 7, 10])
            retriever_params = {}

            # TokenBudgetRetriever uses a token budget instead of k
            if retriever_name == "TokenBudgetRetriever":
                retriever_params["token_budget"] = trial.suggest_categorical(
                    "token_budget", [200, 500, 1000, 2000]
                )
            else:
                # TopKRetriever, CrossEncoderRetriever, RCSRetriever all accept k
                retriever_params["k"] = k

            rerank = False
            if getattr(retriever_cls, "supports_rerank", False):
                rerank = trial.suggest_categorical("rerank", [True, False])
                if rerank and getattr(retriever_cls, "supports_top_m", False):
                    top_m = trial.suggest_categorical("top_m", [5, 10, 20])
                    if top_m < k:  # prereg: m >= k
                        raise optuna.TrialPruned()
                    retriever_params["top_m"] = top_m

            # RCSRetriever requires the LLM model name for its re-ranking step
            if retriever_name == "RCSRetriever":
                retriever_params["llm_model"] = llm_model
                # Route the rerank calls through the same generator as the answer
                # calls (provider, wire id, pinned host). No max_tokens, matching
                # the provider-default the Phase-1 RCS trials ran with.
                _rcs_gen = LLMS[llm_model]
                _rcs_lock = threading.Lock()

                def _rcs_generate(_prompt, _g=_rcs_gen):
                    _out = _g.generate(
                        query=_prompt, chunks=[], system_prompt=None,
                        return_logprobs=False, reasoning_enabled=None,
                        ledger_path=_ledger_path, ledger_extra={"query_id": "rcs", "run_id": _run_id},
                    ) or {}
                    _u = _out.get("usage") or {}
                    with _rcs_lock:   # papers run in threads
                        _rcs_usage["calls"] += 1
                        _rcs_usage["prompt_tokens"] += _u.get("prompt_tokens", 0)
                        _rcs_usage["completion_tokens"] += _u.get("completion_tokens", 0)
                    return _out.get("text", "")
                retriever_params["generate_fn"] = _rcs_generate

            # SAMPLE_ONLY only needs the sampled params, not a live retriever
            # (CrossEncoder/RCS would load models) — skip the heavy instantiation.
            retriever = (
                None if os.getenv("SAMPLE_ONLY") else retriever_cls(**retriever_params)
            )
            query_embedding = trial.suggest_categorical(
                "query_embedding", ["raw", "label_def"]
            )
        else:
            chunk_size = None
            chunk_overlap = None
            chunker_name = "FullDocument"
            embedder_name = "N/A"
            retriever_name = "N/A"
            retriever = None
            k = None
            query_embedding = "full_document"

        llm_meta = LLM_META_MAP.get(llm_model, {})
        llm_api_type = llm_meta.get("api_type")
        # Output budget for the single batched call. Falls back to 16384 (4x the
        # old hardcoded 4096) for any roster entry without an explicit cap.
        # Deliberately not the provider maximum: context_length covers input AND
        # output, and rewritten_prompt over a GROBID parse already needs ~94k of
        # input, so reserving 32k would leave only ~3% headroom on the 131k models.
        _batched_max_tokens = int(llm_meta.get("max_output_tokens", 16384))
        langextract_api_key = None
        langextract_model_id = None
        if llm_api_type == "openai":
            langextract_api_key = OPENAI_API_KEY
            langextract_model_id = llm_model
        elif llm_api_type == "together":
            langextract_api_key = TOGETHER_API_KEY
            langextract_model_id = llm_model

        # Choose default prompt at trial level
        # Non-binary prompts

        # Collect all unique non-binary prompt keys across all queries
        all_non_binary_prompts = set()

        for qdata in QUERIES.values():
            # Defensive check in case some query definitions lack a "prompts" key
            if "prompts" not in qdata:
                continue

            # Add all prompt keys except "binary_prompts"
            all_non_binary_prompts.update(
                [k for k in qdata["prompts"].keys() if k != "binary_prompts"]
            )

        # Phase 2: the per-item inference tier was dropped. `mode == "full"` now
        # always means batched, so `question_mode` is no longer a search dimension
        # and is NOT suggested -- copied Phase-1 batched trials omit it too, which
        # keeps the new study free of a param no live trial can vary. Justification
        # is in the triage: individual - RAG = +1.64pp (n=4 paired, sd 1.01pp,
        # one-sided 95% UB +2.82pp) for ~6x RAG's cost and ~31x batched's.
        question_mode = "batched" if mode == "full" else "individual"

        # The prompt lever is now split in two, because the two tiers consume the
        # stored prompts differently and the batched path has an extra level that
        # has no meaning outside it.
        #
        #   mode == "rag"    -> default_prompt_type (the original 6 variants)
        #   question_mode == "batched" -> batched_prompt_type (those 6 PLUS
        #                       "batched_native", the synthesized block that every
        #                       Phase-1 batched trial actually ran)
        #
        # A separate name rather than a 7th level on default_prompt_type: a shared
        # 7-level lever would let rag trials draw a batched-only value, and the
        # only way to reject that is TrialPruned -- but PRUNED counts toward
        # n_startup_trials, so the waste would come straight out of the warm-up
        # budget. Conditional params are the established pattern here (chunk_size,
        # k and the other RAG levers are already rag-only).
        if question_mode == "batched":
            batched_prompt_type = trial.suggest_categorical(
                "batched_prompt_type",
                [
                    "batched_native",
                    "base_prompt",
                    "cot_prompt",
                    "reasoning_prompt",
                    "rewritten_prompt",
                    "synthetic_few_shot_prompt",
                    "true_few_shot_prompt",
                ],
            )
            default_prompt_type = None
            use_default_prompt = True
        else:
            batched_prompt_type = None
            default_prompt_type = trial.suggest_categorical(
                "default_prompt_type",
                [
                    "base_prompt",
                    "cot_prompt",
                    "reasoning_prompt",
                    "rewritten_prompt",
                    "synthetic_few_shot_prompt",
                    "true_few_shot_prompt",
                ],
            )
            use_default_prompt = trial.suggest_categorical(
                "use_default_prompt", [True, False]
            )
        # VLM lever (CONDITIONAL): for multimodal models in full-document mode,
        # feed the PDF page images directly to the model (vlm=True) instead of the
        # parsed text (vlm=False). RAG can't do this (it retrieves text chunks per
        # question), and text-only models can't accept images, so the toggle is
        # only added to the search space for full-doc trials on multimodal models
        # (define-by-run conditional hyperparameter). When vlm=True the parser
        # choice is moot — pages come straight from the source PDF.
        if mode == "full" and llm_meta.get("multimodal", False):
            vlm = trial.suggest_categorical("vlm", [True, False])
        else:
            vlm = False
        # Thinking-mode lever (CONDITIONAL): only offered for reasoning models on
        # Together / OpenRouter, where disabling chain-of-thought yields clean
        # answer-token logprobs for RQ2 calibration, at a possible accuracy cost.
        # (OpenRouter threads the toggle through its own `reasoning` schema in
        # answer_generator.) For instruct or frontier (openai/gemini) models the
        # toggle is a no-op, so it is not added to the search space for those
        # trials (define-by-run conditional hyperparameter).
        # `reasoning_mandatory` models reject the toggle outright -- every
        # OpenRouter provider for Kimi-K2.7-Code returns 400 "Reasoning is
        # mandatory for this endpoint and cannot be disabled", which would fail
        # every trial TPE sampled with disable_reasoning=True. Excluded from the
        # lever rather than left to fail.
        if (
            llm_meta.get("reasoning", False)
            and not llm_meta.get("reasoning_mandatory", False)
            and llm_api_type in ("together", "openrouter")
        ):
            disable_reasoning = trial.suggest_categorical(
                "disable_reasoning", [True, False]
            )
        else:
            disable_reasoning = False
        reasoning_enabled = False if disable_reasoning else None

        # CAVEAT (2026-10-02): this label follows the roster capability flag. It is
        # WRONG for gpt-5.2: the client sends no reasoning_effort and the API default
        # is "none", so gpt-5.2 runs WITHOUT reasoning yet is labelled "on" here
        # (see analysis/tables/hpo/temperature_probe.md). Logic left unchanged so
        # labels stay comparable with the finished studies.
        # Record the EFFECTIVE reasoning mode (not the requested flag): a frontier
        # reasoning model ignores disable_reasoning and always thinks (=on), a
        # togglable model reflects the flag, and a non-reasoning model has no
        # thinking (=off). `disable_reasoning=False` otherwise conflates "on" with
        # "no capability". Stored as a trial attr for a like-for-like comparison.
        if not llm_meta.get("reasoning", False):
            effective_reasoning = "off"
        elif disable_reasoning:
            effective_reasoning = "off"
        else:
            effective_reasoning = "on"
        trial.set_user_attr("effective_reasoning", effective_reasoning)

        trial_choices.update(
            {
                "parser": parser_name,
                "mode": mode,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunker": chunker_name,
                "embedder": embedder_name,
                "retriever": retriever_name,
                "k": k,
                "llm_model": llm_model,
                "default_prompt_type": default_prompt_type,
                "query_embedding": query_embedding,
                "use_default_prompt": use_default_prompt,
                "question_mode": question_mode,
                "batched_prompt_type": batched_prompt_type,
                "vlm": vlm,
                "disable_reasoning": disable_reasoning,
            }
        )
        log("Trial choices: {trial_choices}")

        # SAMPLE_ONLY: stop here and hand back the fully-sampled config (all params
        # are now recorded in trial.params). Used by queue_random_trials.py to draw
        # valid random configs from the REAL search space without running anything.
        if os.getenv("SAMPLE_ONLY"):
            return trial_choices
        if frozen:
            try:
                trial.set_user_attr("projected_usd_per_paper", round(project_cost_per_paper(trial_choices), 4))
            except Exception:
                pass
        else:
            preflight_guards(trial, trial_choices)
        # ---------------- Prepare trial folder ----------------
        folder, trial_folder, _cache_dir, trial_hash = resolve_run_paths(
            trial_choices, trial.number, paper_dir=paper_dir, out_dir=out_dir, frozen=frozen)
        trial_folder.mkdir(parents=True, exist_ok=True)
        # Per-trial provenance ledger (answer_generation/call_ledger.py): passed
        # explicitly to every generate() call; the env var is kept for callers
        # outside this function.
        _ledger_path = Path(trial_folder) / "calls.jsonl"
        _run_id = os.getenv("RUN_ID")
        os.environ["CALL_LEDGER_PATH"] = str(_ledger_path)
        log(f"Writing trial results to: {trial_folder}")

        # --resume-papers: seed this fresh trial with a prior run's finished papers
        # so already-completed docs are skipped (they're only reused if their
        # parameters_hash matches this trial's config — enforced per-paper below).
        if _RESUME_FROM_FOLDER is not None and Path(_RESUME_FROM_FOLDER) != trial_folder:
            import shutil

            _copied = 0
            for _src in Path(_RESUME_FROM_FOLDER).glob("doc_*.json"):
                _dst = trial_folder / _src.name
                if not _dst.exists():
                    shutil.copyfile(_src, _dst)
                    _copied += 1
            log(f"  Resume: copied {_copied} finished paper(s) from {_RESUME_FROM_FOLDER}")

        # ---------------- Parse all documents ----------------
        all_docs = [f for f in folder.glob("*") if f.suffix.lower() == ".pdf"]
        if doc_name_filter is not None:
            all_docs = [
                f
                for f in all_docs
                if re.match(r"^(\d+)", f.stem)
                and re.match(r"^(\d+)", f.stem).group(1) == doc_name_filter
            ]
        # Smoke-test lever: SMOKE_MAX_DOCS caps the number of tuning papers.
        _smoke_max = os.getenv("SMOKE_MAX_DOCS")
        if _smoke_max:
            all_docs = sorted(all_docs, key=lambda f: f.name)[: int(_smoke_max)]
        if paper_ids is not None:
            _want = set(str(x) for x in paper_ids)
            _have = {_paper_id(f): f for f in all_docs}
            for _pid in sorted(_want - set(_have)):
                paper_failures[_pid] = "pdf_missing"
            all_docs = [f for f in all_docs if _paper_id(f) in _want]
        if frozen:
            all_docs = sorted(all_docs, key=lambda f: f.name)
        all_docs_names = [f.name for f in all_docs]

        log(f"  Found {len(all_docs_names)} documents")
        parsed_docs = []
        _parsed_ok = []
        for _f in all_docs:
            try:
                parsed_docs.append(ensure_parsed(_f, parser_name, strict=frozen))
                _parsed_ok.append(_f)
            except Exception as _pe:
                if not frozen:
                    raise
                paper_failures[_paper_id(_f)] = f"parse_failed: {_pe!r}"[:300]
                log(f"  ⚠️ parse failed for {_f.name}: {_pe!r}"[:200])
        all_docs = _parsed_ok
        log("  Parsing complete")

        doc_aggregates = {}
        query_aggregates = {}
        doc_id_map = load_doc_id_map()
        next_id = max(doc_id_map.values(), default=-1) + 1

        # ---------------- Pre-assign doc_ids (must not race across threads) ---
        # Assign every paper its doc_id up front, in the main thread, so no
        # worker mutates doc_id_map / next_id. Each task carries immutable ids.
        paper_tasks = []
        for pdf_path, parsed_path in zip(all_docs, parsed_docs):
            match = re.match(r"^(\d+)", parsed_path.stem)
            doc_name = match.group(1) if match else parsed_path.stem
            if doc_name not in doc_id_map:
                if frozen and doc_name.isdigit():
                    doc_id_map[doc_name] = int(doc_name)   # stable: the paper id itself
                else:
                    doc_id_map[doc_name] = next_id
                    next_id += 1
            doc_id = doc_id_map[doc_name]
            paper_tasks.append((parsed_path, doc_id, doc_name, pdf_path))
        if frozen:
            save_doc_id_map(doc_id_map, str(trial_folder / "doc_id_map.json"))

        # ---------------- Per-paper worker ----------------
        # Each paper is fully independent: dependencies are strictly within-doc
        # (check_dependency filters by doc_id), and doc_results is LOCAL, so this
        # function has no cross-paper shared mutable state except the guarded
        # `cache` dict.
        def _process_paper(parsed_path, doc_id, doc_name, pdf_path):
            # Per-paper token accounting (summed into the trial total on merge;
            # cost is derived later from the roster's per-model $/M prices).
            _usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "calls": 0}
            # Per-paper batched health counters. Kept local and merged by the
            # caller for the same reason as _usage: papers run concurrently
            # (PAPER_CONCURRENCY), so trial-level mutable state would need a lock.
            _batched_stats = {"split": 0, "parse_failed": 0, "truncated": 0, "other_without_y": 0}

            def _acc(_out):
                _u = (_out or {}).get("usage") or {}
                if _u:
                    _usage["prompt_tokens"] += _u.get("prompt_tokens", 0)
                    _usage["completion_tokens"] += _u.get("completion_tokens", 0)
                    _usage["reasoning_tokens"] += _u.get("reasoning_tokens", 0)
                    _usage["calls"] += 1

            # ---------------- Document setup ----------------
            system_prompt = build_system_prompt(
                doc_name,
                fallback_single_prompt=SYSTEM_PROMPTS.get("single_outcome"),
                fallback_multi_prompt=SYSTEM_PROMPTS.get("multiple_outcomes"),
            )
            # Separate system prompt for open-ended items (prereg Prompt 5).
            system_prompt_oe = build_system_prompt(
                doc_name,
                fallback_single_prompt=SYSTEM_PROMPTS.get("single_outcome"),
                fallback_multi_prompt=SYSTEM_PROMPTS.get("multiple_outcomes"),
                oe=True,
            )
            log(f"--- Document {doc_id}: {parsed_path.name} ---")
            doc_file = trial_folder / f"doc_{doc_id:04d}.json"

            # ---------------- Load or initialize document results ----------------
            existing_results = []
            existing_qids = set()
            cached_hash = None

            if doc_file.exists():
                with open(doc_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                cached_hash = existing_data.get("parameters_hash")
                current_hash = hashlib.md5(
                    json.dumps(trial_choices, sort_keys=True).encode()
                ).hexdigest()
                # Resume only when BOTH the config hash and the paper match: a stale
                # doc_NNNN.json belonging to another paper (glob-order shift, or a
                # different paper set) must never be reused (2026-10-02).
                if cached_hash == current_hash and str(existing_data.get("doc_name", doc_name)) == str(doc_name):
                    existing_results = existing_data.get("results", [])
                    existing_qids = {r["query_id"] for r in existing_results}
                    log(f"  Found {len(existing_qids)} cached query results.")
                else:
                    log("  Parameter/paper mismatch → recomputing all queries.")
            else:
                log(f"  No cache found for {doc_name}, evaluating all queries.")

            # Start with cached results (LOCAL to this paper only)
            doc_results = list(existing_results)
            doc_result_map = {r["query_id"]: r for r in existing_results}
            # ---------------- Ensure chunks / embeddings or load full text ----------------
            vector_store = None
            full_doc_text = None
            if mode == "rag":
                with _MODEL_LOCK:
                    chunk_path = ensure_chunks(
                        parsed_path, parser_name, chunker_name, chunk_size, chunk_overlap
                    )
                with _MODEL_LOCK:
                    vector_store_path = ensure_embedding(
                        chunk_path,
                        parser_name,
                        chunker_name,
                        embedder_name,
                        chunk_size,
                        chunk_overlap,
                    )
                vector_store = NumpyVectorStore(
                    save_path=str(vector_store_path) + ".npz"
                )
                with _MODEL_LOCK:
                    vector_store.load()
            else:
                full_doc_text = load_full_document_text(parsed_path)

            # VLM mode: render the source PDF's pages once per paper and reuse the
            # images for every item (the parsed text is not sent to the model).
            page_images = None
            if vlm:
                page_images = _load_page_images(pdf_path)
                log(f"  VLM mode: {len(page_images)} page image(s) for {doc_name}")

            # ---------------- Per-query evaluation ----------------
            # Batched mode: one call over the full document answers every item at
            # once; parse into per-item answers/others, then score each below as in
            # individual mode (no per-item generation).
            batched_answers, batched_others = {}, {}
            _wanted_qids = [q for q in QUERIES if (query_ids is None or q in query_ids)]
            # Skip the (paid) batched call when every requested item is already cached
            # for this paper -- on resume it was a no-op that still cost a call.
            if question_mode == "batched" and not all(q in existing_qids for q in _wanted_qids):
                if full_doc_text is None:
                    full_doc_text = load_full_document_text(parsed_path)
                _bctx = _VLM_CTX if vlm else full_doc_text

                def _batched_call(_ids=None):
                    """One batched generation over `_ids` (all items when None)."""
                    _p = batched.build_batched_prompt(
                        QUERIES, batched_prompt_type, item_ids=_ids
                    )
                    with _MODEL_LOCK:
                        return LLMS[llm_model].generate(
                            query=_p,
                            system_prompt=system_prompt,
                            chunks=[(_bctx, 1.0)],
                            return_logprobs=False,
                            reasoning_enabled=reasoning_enabled,
                            ledger_path=_ledger_path,
                            ledger_extra={"doc_name": doc_name, "query_id": "batched", "run_id": _run_id},
                            # Per-model budget instead of the old hardcoded 4096.
                            # A reply needs only ~550 tokens, but on a reasoning
                            # model the thinking tokens come out of the same
                            # allowance; exhausting it truncates the JSON, and
                            # parse_batched_response turns a truncated reply into
                            # 43 silent abstentions. That collapse hit 7 of 28
                            # Phase-1 batched trials.
                            max_tokens=_batched_max_tokens,
                            images=page_images,
                        )

                _bout = _batched_call()
                _ok, _n = batched.response_is_usable(_bout.get("text", ""), QUERIES)
                _truncated = _bout.get("finish_reason") == "length"
                if not _ok or _truncated:
                    # Fall back to two calls over half the items each. A model that
                    # cannot fit 43 answers in one reply usually manages 22, and
                    # half a paper scored is far better than a silent floor score.
                    log(
                        f"  Batched reply unusable for {doc_name} "
                        f"(parsed {_n}/{len(QUERIES)}, finish_reason="
                        f"{_bout.get('finish_reason')!r}) -> retrying as two calls"
                    )
                    _h1, _h2 = batched.split_item_ids(QUERIES)
                    _o1, _o2 = _batched_call(_h1), _batched_call(_h2)
                    _a1, _x1 = batched.parse_batched_response(
                        _o1.get("text", ""), {k: QUERIES[k] for k in _h1}
                    )
                    _a2, _x2 = batched.parse_batched_response(
                        _o2.get("text", ""), {k: QUERIES[k] for k in _h2}
                    )
                    batched_answers = {**_a1, **_a2}
                    batched_others = {**_x1, **_x2}
                    _acc(_o1)
                    _acc(_o2)
                    _batched_stats["split"] += 1
                    if not _ok:
                        _batched_stats["parse_failed"] += 1
                    if _truncated:
                        _batched_stats["truncated"] += 1
                else:
                    batched_answers, batched_others = batched.parse_batched_response(
                        _bout.get("text", ""), QUERIES
                    )
                    _acc(_bout)
                # A "<id>__other" text whose answer does not contain Y is discarded by
                # the scorer (Y is never inferred from it, matching RAG where the
                # follow-up only runs after Y). Count it so we know how often that
                # happens; the text itself is kept in the per-doc json below.
                _batched_stats["other_without_y"] += sum(
                    1 for _q, _txt in batched_others.items()
                    if _txt and not re.search(
                        r"Y|OTHER", re.sub(r"[^A-Za-z]", "", str(batched_answers.get(_q, ""))).upper()
                    )
                )
            for qid, qdata in QUERIES.items():
                if query_ids is not None and qid not in query_ids:
                    continue
                if qid in existing_qids:
                    log(f"  Skipping query {qid} (cached for this doc).")
                    continue
                log(f"  Evaluating NEW query {qid} for {doc_name}")
                needs_evaluation = True
                if qid in DEPENDENCIES:
                    gold = TRUE_ANSWERS[doc_name][qid]
                    _auto = dependency_autoresolve(qid, gold)
                    if _auto is not None:
                        log(f"  Query {qid}: gold not-applicable (-99) -> auto-correct, no call.")
                        eval_result = _auto
                        result = {
                            "doc_id": doc_id,
                            "query_id": qid,
                            "label": QUERIES[qid]["label"],
                            "type": QUERIES[qid]["type"],
                            "llm_output": "",
                            **eval_result,
                        }
                        needs_evaluation = False

                if needs_evaluation:
                    log("needs evaluation")
                    if mode == "rag":
                        with _MODEL_LOCK:
                            query_emb_path = ensure_query_embeddings(
                                embedder_name, parsed_path.stem, qid, query_embedding
                            )
                        query_embed = np.load(query_emb_path)

                        # Build raw query text for retrievers that need it (e.g. CrossEncoder reranking)
                        if getattr(retriever, "needs_raw_query_text", False):
                            if query_embedding == "label_def":
                                _parts = [
                                    qdata.get("label", ""),
                                    qdata.get("description", ""),
                                ]
                                if qdata.get("choices"):
                                    _choices_text = ", ".join(
                                        f"{k}: {v['value']}"
                                        for k, v in qdata["choices"].items()
                                    )
                                    _parts.append("Choices: " + _choices_text)
                                raw_query_text = " ".join(p for p in _parts if p)
                            else:
                                raw_query_text = qdata.get("prompts", {}).get(
                                    "base_prompt", ""
                                )
                            with _MODEL_LOCK:
                                retrieved_chunks = retriever.retrieve(
                                    query_embed, vector_store, raw_query_text=raw_query_text
                                )
                        else:
                            with _MODEL_LOCK:
                                retrieved_chunks = retriever.retrieve(
                                    query_embed, vector_store
                                )
                    else:
                        if full_doc_text is None:
                            full_doc_text = load_full_document_text(parsed_path)
                        retrieved_chunks = [(_VLM_CTX if vlm else full_doc_text, 1.0)]
                else:
                    # Not-applicable: record result and skip LLM evaluation
                    doc_results.append(result)
                    existing_qids.add(qid)
                    doc_result_map[qid] = result
                    log(
                        f"    ✅ Query {qid} accuracy: {eval_result.get('accuracy', 0):.3f}"
                    )
                    param_hash = hashlib.md5(
                        json.dumps(trial_choices, sort_keys=True).encode()
                    ).hexdigest()
                    doc_data = {
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "parameters_hash": param_hash,
                        "results": doc_results,
                    }
                    with open(doc_file, "w", encoding="utf-8") as f:
                        json.dump(doc_data, f, indent=2, ensure_ascii=False)
                    continue
                # Determine prompt type. Batched resolves its prompt once per
                # paper (build_batched_prompt), not per item, and its
                # default_prompt_type is None -- _select_prompt would raise on that
                # lookup, so it is only consulted on the per-item paths.
                if question_mode == "batched":
                    prompt_type, prompt_text = batched_prompt_type, None
                else:
                    prompt_type, prompt_text = _select_prompt(
                        qid, qdata, default_prompt_type, use_default_prompt
                    )
                # Open-ended items get the concise-description system prompt.
                sp = system_prompt_oe if qdata.get("type") == "open_ended" else system_prompt
                ans_gen = LLMS[llm_model]
                if question_mode == "batched":
                    # Answer already produced by the one-shot batched call above.
                    # Per-item logprobs aren't meaningful for a batched response
                    # (they cover all items at once), so don't store/duplicate them.
                    llm_text = batched_answers.get(qid, "Z")
                    llm_output = {"text": llm_text}
                    llm_logprobs = {}
                elif prompt_type == "binary_prompts" and qdata["type"] in [
                    "multiple_choice",
                    "single_choice",
                ]:
                    # Binary mode, collapsed to ONE call: ask every option as a
                    # yes/no in a single prompt and parse the JSON back into the
                    # selected-letter set (was one call PER option — 22 calls for
                    # item 3.1). Scored identically to any other MC prediction.
                    _bprompt = binary_mode.build_binary_prompt(qid, qdata)
                    llm_output = ans_gen.generate(
                        query=_bprompt,
                        system_prompt=sp,
                        chunks=retrieved_chunks,
                        return_logprobs=True,
                        reasoning_enabled=reasoning_enabled,
                        images=page_images,
                        ledger_path=_ledger_path,
                        ledger_extra={"doc_name": doc_name, "query_id": qid, "run_id": _run_id},
                    )
                    _btext, llm_logprobs = get_cleaned_text_and_logprobs(llm_output)
                    llm_text = binary_mode.parse_binary_response(_btext, qdata)
                    _acc(llm_output)

                else:
                    log(f"prompt agaain {prompt_text}")
                    llm_output = ans_gen.generate(
                        system_prompt=sp,
                        query=prompt_text,
                        chunks=retrieved_chunks,
                        return_logprobs=True,
                        reasoning_enabled=reasoning_enabled,
                        images=page_images,
                        ledger_path=_ledger_path,
                        ledger_extra={"doc_name": doc_name, "query_id": qid, "run_id": _run_id},
                    )
                    llm_text, llm_logprobs = get_cleaned_text_and_logprobs(llm_output)
                    _acc(llm_output)

                def make_handle_other(qid, QUERIES, ans_gen, retrieved_chunks):
                    """Factory function that creates a handle_other callback
                    bound to the current qid, QUERIES, ans_gen, and retrieved_chunks.
                    """

                    def handle_other(llm_text):
                        other_prompt = QUERIES[qid]["prompts"]["follow_up_prompt"]
                        other_response = ans_gen.generate(
                            query=other_prompt,
                            chunks=retrieved_chunks,
                            system_prompt=system_prompt,
                            return_logprobs=True,
                            reasoning_enabled=reasoning_enabled,
                            images=page_images,
                            ledger_path=_ledger_path,
                            ledger_extra={"doc_name": doc_name, "query_id": f"{qid}__other", "run_id": _run_id},
                        )
                        _acc(other_response)
                        return other_response["text"]

                    return handle_other

                # Example usage (where you loop through or handle a query)
                if question_mode == "batched":
                    # 'Other' free-text came inline in the batched JSON (if any).
                    _o = batched_others.get(qid)
                    callback = (lambda _t, _v=_o: _v) if _o else None
                else:
                    callback = make_handle_other(qid, QUERIES, ans_gen, retrieved_chunks)

                with _MODEL_LOCK:
                    eval_result = evaluate_answer(
                        llm_text,
                        question_type=qdata["type"],
                        doc_name=doc_name,
                        query_id=qid,
                        TRUE_ANSWERS=TRUE_ANSWERS,
                        QUERIES=QUERIES,
                        other_callback=callback,
                        curated_mc_items=CURATED_MC_ITEMS,
                    )
                # In full-document mode the whole paper would otherwise be stored
                # in EVERY cell's context (×items ×papers), bloating trial_summary
                # and cache.json to GBs (and corrupting the cache on a killed write).
                # Persist only a marker there; RAG keeps its small retrieved chunks.
                eval_result["context"] = (
                    "[full document]" if mode == "full" else retrieved_chunks
                )
                eval_result["logprobs"] = llm_logprobs

                # ---------------- After evaluating a single query ----------------
                result = {
                    "doc_id": doc_id,
                    "query_id": qid,
                    "label": QUERIES[qid]["label"],
                    "type": QUERIES[qid]["type"],
                    "llm_output": llm_output["text"],
                    **eval_result,
                }

                # ✅ Update caches (shared `cache` dict guarded by _CACHE_LOCK)
                trial_key = _hash_trial_key(trial_choices, doc_id, qid)
                with _CACHE_LOCK:
                    cache[trial_key] = result

                # ✅ Update in-memory lists (LOCAL to this paper)
                doc_results.append(result)
                existing_qids.add(qid)
                doc_result_map[qid] = result
                log(
                    f"    ✅ Query {qid} accuracy: {eval_result.get('accuracy', 0):.3f}"
                )

                # ✅ Save per-document results incrementally
                param_hash = hashlib.md5(
                    json.dumps(trial_choices, sort_keys=True).encode()
                ).hexdigest()
                doc_data = {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "parameters_hash": param_hash,
                    "results": doc_results,
                }
                if question_mode == "batched":
                    # every "<id>__other" the model returned, whether or not the
                    # answer carried Y -- recoverable for an Other-aware rescoring
                    doc_data["batched_other_text"] = dict(batched_others)
                with open(doc_file, "w", encoding="utf-8") as f:
                    json.dump(doc_data, f, indent=2, ensure_ascii=False)
            num_cached = len(existing_results)
            num_new = len(doc_results) - num_cached
            num_total = len(doc_results)
            doc_accuracy = (
                np.mean([r["accuracy"] for r in doc_results]) if doc_results else 0
            )
            log(
                f"📄 Document {doc_id} finished: {num_cached} cached, {num_new} new, total {num_total} queries."
            )
            log(f"    Overall accuracy for document {doc_id}: {doc_accuracy:.3f}")

            # Return LOCAL results only; the caller merges into all_results and
            # builds the aggregates single-threaded after all papers finish.
            return {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_results": doc_results,
                "doc_accuracy": doc_accuracy,
                "token_usage": _usage,
                "batched_stats": _batched_stats,
            }

        # ---------------- Execute papers (sequential or threaded) ----------------
        # PAPER_CONCURRENCY <= 1 preserves the exact original sequential order.
        # PAPER_CONCURRENCY  > 1 runs papers in a thread pool so their network
        # calls (ans_gen.generate) overlap; _MODEL_LOCK / _CACHE_LOCK serialize
        # the model, retrieval, judge, and shared-cache operations.
        # On insufficient funds, tag the trial (so it's easy to find + rerun once
        # topped up — its config is already saved in the study) and abort the whole
        # trial immediately instead of failing every remaining paper.
        def _skip_broke(exc):
            trial.set_user_attr("skip_reason", "insufficient_credits")
            trial.set_user_attr("skip_model", llm_model)
            log(
                f"⏭️  Skipping trial {trial.number}: {llm_model} out of credits. "
                f"Config saved — rerun later with: --rerun={trial.number}"
            )
            raise exc

        paper_outputs = []
        _papers_t0 = time.time()

        def _time_cap_check(done: int):

            if frozen:

                return
            """Racing-style aborts after >=3 completed papers.
            Time: a $0.09/paper trial once ran 21 h — dollar caps cannot catch
            time sinks. Cost: realised $/paper from the papers' own token_usage
            — reasoning volume is unknowable pre-flight (#63 realised $1.25/paper
            where its sibling realised $0.18), so the pre-flight projection stays
            optimistic and the realised guard stops true sinkholes ~paper 3."""
            if done < 3:
                return
            pace = (time.time() - _papers_t0) / done
            if pace > TIME_CAP_S_PER_PAPER:
                trial.set_user_attr("time_cap_s_per_paper", round(pace, 1))
                raise optuna.TrialPruned(
                    f"guard: {pace:.0f}s/paper exceeds the "
                    f"{TIME_CAP_S_PER_PAPER}s cap after {done} papers")
            _pr_ = (_model_meta().get(llm_model) or {}).get("pricing") or {}
            spent = sum(
                (o.get("token_usage", {}).get("prompt_tokens", 0) * float(_pr_.get("input", 0))
                 + o.get("token_usage", {}).get("completion_tokens", 0) * float(_pr_.get("output", 0)))
                for o in paper_outputs) / 1e6
            if spent / done > COST_CAP_USD_PER_PAPER:
                trial.set_user_attr("realized_usd_per_paper", round(spent / done, 4))
                raise optuna.TrialPruned(
                    f"guard: realised ${spent/done:.2f}/paper exceeds the "
                    f"${COST_CAP_USD_PER_PAPER:.2f} cap after {done} papers")

        if PAPER_CONCURRENCY <= 1:
            for parsed_path, doc_id, doc_name, pdf_path in paper_tasks:
                try:
                    paper_outputs.append(
                        _process_paper(parsed_path, doc_id, doc_name, pdf_path)
                    )
                    flush_cache_throttled(cache, cache_dir=_cache_dir)
                    _time_cap_check(len(paper_outputs))
                except InsufficientCreditsError as _ice:
                    _skip_broke(_ice)
                except optuna.TrialPruned:
                    raise
                except Exception as _exc:
                    if not frozen:
                        raise
                    paper_failures[str(doc_name)] = f"{type(_exc).__name__}: {_exc}"[:300]
                    log(f"⚠️ Paper {doc_name} failed and was skipped: {_exc!r}"[:200])
        else:
            # Pre-init the shared AnswerGenerator's HTTP client in the main thread
            # so worker threads don't race on its lazy `_ensure_client()`. The
            # underlying openai/together/httpx clients are safe for concurrent use.
            try:
                LLMS[llm_model]._ensure_client()
            except Exception as _exc:
                log(f"  (client pre-warm skipped: {_exc})")
            # as_completed raises TimeoutError once TRIAL_WALL_BUDGET_S has elapsed
            # since the iteration began, even if nothing has completed; that is the
            # only guard reachable when no paper ever lands.
            # NOT a `with` block: its __exit__ calls shutdown(wait=True), which
            # blocks on futures stuck in the provider retry ladder, so a guard
            # prune could not take effect until the hung calls unwound (hours).
            # shutdown(wait=False, cancel_futures=True) drops queued work and lets
            # the prune propagate immediately; still-running threads finish in the
            # background and are joined at interpreter exit.
            executor = ThreadPoolExecutor(max_workers=PAPER_CONCURRENCY)
            try:
                future_map = {
                    executor.submit(
                        _process_paper, parsed_path, doc_id, doc_name, pdf_path
                    ): (doc_id, doc_name)
                    for parsed_path, doc_id, doc_name, pdf_path in paper_tasks
                }
                # Iterate in COMPLETION order (as_completed), not submission
                # order. With submission order, result() blocks on future #1, so a
                # single paper stuck in the provider retry ladder suspends the
                # in-flight guards indefinitely while other papers finish: consumer
                # trial #53 reached 465 min at 3/35 papers (9300 s/paper, 10x the
                # cap) without _time_cap_check ever being reached. Completion order
                # evaluates the guards whenever ANY paper lands. paper_outputs order
                # is not used downstream (the merge keys by doc_id and sums usage).
                _it = as_completed(future_map, timeout=None if frozen else TRIAL_WALL_BUDGET_S)
                while True:
                    try:
                        future = next(_it)
                    except StopIteration:
                        break
                    except TimeoutError:
                        for _f in future_map:
                            _f.cancel()
                        trial.set_user_attr("wall_budget_exceeded_s", TRIAL_WALL_BUDGET_S)
                        raise optuna.TrialPruned(
                            f"guard: exceeded the {TRIAL_WALL_BUDGET_S}s trial wall-clock "
                            f"budget with {len(paper_outputs)}/{len(paper_tasks)} papers done")
                    fut_doc_id, fut_doc_name = future_map[future]
                    try:
                        paper_outputs.append(future.result())
                        flush_cache_throttled(cache, cache_dir=_cache_dir)
                        try:
                            _time_cap_check(len(paper_outputs))
                        except optuna.TrialPruned:
                            for _f in future_map:
                                _f.cancel()
                            raise
                    except optuna.TrialPruned:
                        # MUST precede the broad handler below: TrialPruned is an
                        # Exception, so without this clause every guard prune was
                        # caught and logged as "paper failed and was skipped" and
                        # the trial carried on (consumer #56: the pace guard fired
                        # at 2261 s/paper and was swallowed).
                        raise
                    except InsufficientCreditsError as _ice:
                        # Out of money: cancel not-yet-started papers and abort now.
                        for _f in future_map:
                            _f.cancel()
                        _skip_broke(_ice)
                    except Exception as exc:
                        paper_failures[str(fut_doc_name)] = f"{type(exc).__name__}: {exc}"[:300]
                        log(
                            f"⚠️ Paper doc_id={fut_doc_id} ({fut_doc_name}) "
                            f"failed and was skipped: {exc}"
                        )
                        continue
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        # ---------------- Merge per-paper results ----------------
        trial_usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "calls": 0}
        trial_batched_stats = {"split": 0, "parse_failed": 0, "truncated": 0, "other_without_y": 0}
        for output in paper_outputs:
            all_results.extend(output["doc_results"])
            doc_aggregates[f"doc_{output['doc_id']}"] = {
                "average_accuracy": output["doc_accuracy"],
                "doc_name": output["doc_name"],
            }
            for _k in trial_usage:
                trial_usage[_k] += output.get("token_usage", {}).get(_k, 0)
            for _k in trial_batched_stats:
                trial_batched_stats[_k] += output.get("batched_stats", {}).get(_k, 0)

        # Fail the whole trial (rather than report a bogus low/0.000 score) when a
        # large share of papers errored out — e.g. an API credit/rate-limit block
        # skips most papers, leaving all_results near-empty and accuracy ~0.000.
        # Raising here records the trial as FAILED (caught by study.optimize's
        # catch=Exception), so the model is reattempted on the next launch instead
        # of polluting the TPE surrogate with an artifact. Threshold via
        # MIN_PAPER_FRAC (default 0.5 = fail if <50% of papers succeeded).
        _min_frac = 0.0 if frozen else float(os.getenv("MIN_PAPER_FRAC", "0.5"))
        _n_ok, _n_total = len(paper_outputs), len(paper_tasks)
        if paper_failures:
            trial.set_user_attr("paper_failures", dict(paper_failures))
        trial.set_user_attr("n_papers_ok", _n_ok)
        trial.set_user_attr("n_papers_requested", _n_total + sum(1 for v in paper_failures.values() if v == "pdf_missing" or v.startswith("parse_failed")))
        if _n_total and _n_ok < _n_total * _min_frac:
            raise RuntimeError(
                f"trial aborted: only {_n_ok}/{_n_total} papers succeeded "
                f"(rest errored — likely API credit/rate-limit); failing for reattempt"
            )

        # ✅ Build query aggregates once over the merged results
        for qid_agg in QUERIES.keys():
            query_scores = [
                r["accuracy"] for r in all_results if r["query_id"] == qid_agg
            ]
            query_aggregates[qid_agg] = {
                "average_accuracy": (
                    np.mean(query_scores) if query_scores else float("nan")
                )
            }

        # ✅ Write the trial summary once, over the merged results
        summary_file = trial_folder / "trial_summary.json"
        per_query_doc_results = []
        for r in all_results:
            doc_info = doc_aggregates.get(f"doc_{r['doc_id']}", {})
            per_query_doc_results.append(
                {
                    "doc_id": r["doc_id"],
                    "query_id": r["query_id"],
                    "doc_name": doc_info.get("doc_name"),
                    **{
                        k: v
                        for k, v in r.items()
                        if k not in ["doc_id", "query_id"]
                    },
                }
            )

        overall_avg = (
            np.mean([r["accuracy"] for r in all_results]) if all_results else 0.0
        )

        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "trial_number": trial.number,
            "parameters": trial_choices,
            "overall_average_accuracy": overall_avg,
            "token_usage": trial_usage,
            "document_aggregates": doc_aggregates,
            "query_aggregates": query_aggregates,
            "per_query_doc_results": per_query_doc_results,
        }

        # Token totals on the trial itself (cost = tokens x roster $/M, computed
        # later). Only trials run after this change carry usage; earlier ones don't.
        trial.set_user_attr("prompt_tokens", trial_usage["prompt_tokens"])
        trial.set_user_attr("completion_tokens", trial_usage["completion_tokens"])
        trial.set_user_attr("llm_calls", trial_usage["calls"])
        try:
            from answer_generation import call_ledger as _cl
            _lp = Path(_ledger_path)
            if _lp.exists():
                _v = _cl.verify(_lp)
                trial.set_user_attr("call_ledger_head", _v["head"])
                trial.set_user_attr("call_ledger_n", _v["n"])
                trial.set_user_attr("call_ledger_ok", bool(_v["ok"]))
        except Exception:
            pass
        if _rcs_usage["calls"]:
            _pr = (llm_meta.get("pricing") or {})
            trial.set_user_attr("rcs_calls", _rcs_usage["calls"])
            trial.set_user_attr("rcs_prompt_tokens", _rcs_usage["prompt_tokens"])
            trial.set_user_attr("rcs_completion_tokens", _rcs_usage["completion_tokens"])
            trial.set_user_attr("rcs_cost_usd", round(
                (_rcs_usage["prompt_tokens"] * float(_pr.get("input", 0))
                 + _rcs_usage["completion_tokens"] * float(_pr.get("output", 0))) / 1e6, 4))
        # Reasoning as OBSERVED from provider usage, not from the toggle we sent:
        # Together's gpt-oss-120b still emitted ~660 reasoning tokens per call
        # with reasoning.enabled=false (probe 2026-09-25), so the toggle-derived
        # `effective_reasoning` attr can say "off" while the model reasoned.
        _rt = trial_usage.get("reasoning_tokens", 0)
        trial.set_user_attr("reasoning_tokens", _rt)
        trial.set_user_attr("effective_reasoning_observed",
                            "on" if _rt > 0 else ("off" if trial_usage["calls"] else "unknown"))
        # Dollar cost from the roster's per-million pricing, so cost analyses need
        # no offline reconstruction. Generation calls only: the judge is local and
        # embedding usage is not metered here.
        _pr = (llm_meta.get("pricing") or {})
        if _pr:
            trial.set_user_attr(
                "cost_usd",
                round(
                    (trial_usage["prompt_tokens"] * float(_pr.get("input", 0))
                     + trial_usage["completion_tokens"] * float(_pr.get("output", 0))) / 1e6,
                    4,
                ),
            )
        # Batched health. Before this, a paper whose JSON failed to parse scored 43
        # silent abstentions and the trial recorded an ordinary-looking number;
        # these attrs make that visible in the study itself.
        if question_mode == "batched":
            trial.set_user_attr("batched_split_used", trial_batched_stats["split"])
            trial.set_user_attr("batched_parse_failed", trial_batched_stats["parse_failed"])
            trial.set_user_attr("batched_truncated", trial_batched_stats["truncated"])
            trial.set_user_attr("batched_other_without_y", trial_batched_stats["other_without_y"])

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        # ✅ Persist the cache once after the merge (avoids per-paper disk churn)
        with _CACHE_LOCK:
            save_cache(cache, _cache_dir)

        if all_results:
            overall_avg = np.mean([r["accuracy"] for r in all_results])
        else:
            overall_avg = 0.0  # fallback if no results at all

        # Log mean accuracy by query_id
        log("\n📊 Mean accuracy by query ID across all documents:")
        for qid_agg in QUERIES.keys():
            query_scores = [
                r["accuracy"] for r in all_results if r["query_id"] == qid_agg
            ]
            mean_acc = np.mean(query_scores) if query_scores else float("nan")
            log(f"    Query {qid_agg}: mean accuracy {mean_acc:.3f}")

        # Log overall per-document accuracy
        log("\n📄 Overall accuracy per document:")
        for doc_id, doc_info in doc_aggregates.items():
            log(
                f"    Document {doc_info['doc_name']} (ID {doc_id}): average accuracy {doc_info['average_accuracy']:.3f}"
            )

        # Log total trial duration
        trial_duration = time.time() - trial_start_time
        log(f"\n⏱ Trial {trial.number} completed in {trial_duration:.2f} seconds")

        return float(overall_avg)
    except Exception as e:
        log(f"Trial failed with choices: {trial_choices}")
        log(f"Error: {e}")
        raise


class PatienceCallback:
    """Preregistered stopping rule (Phase-1 SS6.2.2): stop when ``patience``
    consecutive trials fail to improve the best accuracy.

    STATELESS by design — the streak is derived from the shared study on every
    call, in completion order. The previous per-process counter double-counted
    under two workers (effective patience ~2x the preregistered 15) and reset
    to best=-1.0 on every worker restart, wiping the streak. Only COMPLETE
    trials count: a guard-pruned trial is a refusal to evaluate, not a failed
    evaluation, and must not run down the clock for free."""

    def __init__(self, patience: int = 15, min_trials: int | None = None,
                 from_number: int | None = None):
        self.patience = patience
        # Don't stop before model-based sampling has begun. The floor is the
        # sampler's startup budget, NOT a hard 100: a seeded study (open:
        # TPE_N_STARTUP=88, consumer: 34) models from trial one, so a literal 100
        # silently disabled the rule there -- the consumer search logged
        # "No improvement for 18/15" and kept going because it had only 73
        # completions. Explicit override via PATIENCE_MIN_TRIALS.
        self.min_trials = (
            int(os.getenv("PATIENCE_MIN_TRIALS",
                          os.getenv("TPE_N_STARTUP", "100")))
            if min_trials is None else min_trials
        )
        # streak counts only trials numbered >= from_number (the stage start),
        # while the bar (best) is taken over the whole study incl. seeds. A
        # seeded second study otherwise starts with a phantom streak from seeds
        # that happened to complete after the seeded best.
        self.from_number = (int(os.getenv("PATIENCE_FROM_NUMBER", "0"))
                            if from_number is None else from_number)

    @staticmethod
    def _order(t):
        return (t.datetime_complete or t.datetime_start, t.number)

    def streak(self, study) -> tuple[int, int, float]:
        comp = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.value is not None]
        if not comp:
            return 0, 0, float("nan")
        best_val = max(t.value for t in comp)
        best_first = min((t for t in comp if t.value == best_val), key=self._order)
        streak = sum(1 for t in comp
                     if t.number >= self.from_number
                     and self._order(t) > self._order(best_first))
        return streak, len(comp), best_val

    def __call__(self, study, trial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        streak, n_complete, best_val = self.streak(study)
        if streak == 0 and trial.value == best_val:
            logger.info("  ★ New best accuracy: %.2f%% (trial %d)",
                        best_val * 100, trial.number)
        else:
            logger.info(
                "  No improvement for %d/%d consecutive completed trials "
                "(study best=%.2f%%)", streak, self.patience, best_val * 100)
        if n_complete >= self.min_trials and streak >= self.patience:
            logger.info(
                "Early stopping (prereg SS6.2.2): %d consecutive completed trials "
                "without improvement (%d completed total).", streak, n_complete)
            study.stop()


def run_worker(study):
    study.optimize(objective, n_trials=1)  # each worker runs one trial at a time


def init_globals() -> dict:
    """Populate QUERIES / TRUE_ANSWERS / DEPENDENCIES / SYSTEM_PROMPTS exactly as
    __main__ always did, so that objective() can be called from a runner that
    imports this module (scripts/run_frozen.py). Returns sha256 of each input file."""
    global QUERIES, TRUE_ANSWERS, DEPENDENCIES, SYSTEM_PROMPTS
    QUERIES = load_queries()
    _evaluable = finalized_item_map(QUERIES)
    QUERIES = {qid: q for qid, q in QUERIES.items() if qid in _evaluable}
    TRUE_ANSWERS = load_true_answers_final(QUERIES)
    DEPENDENCIES_FILE = PROJECT_ROOT / "src" / "dependencies.txt"
    with open(DEPENDENCIES_FILE, "r", encoding="utf-8") as f:
        DEPENDENCIES = json.load(f)
    SYSTEM_PROMPTS = {}
    _sp1 = PROJECT_ROOT / "query_creation" / "system_prompt_single_outcome.txt"
    _sp2 = PROJECT_ROOT / "query_creation" / "system_prompt_multiple_outcomes.txt"
    with open(_sp1, "r", encoding="utf-8") as f:
        SYSTEM_PROMPTS["single_outcome"] = f.read()
    with open(_sp2, "r", encoding="utf-8") as f:
        SYSTEM_PROMPTS["multiple_outcomes"] = f.read()
    inputs = {"queries_with_prompts.json": PROJECT_ROOT / "query_creation" / "queries_with_prompts.json",
              "09_data_final.csv": PROJECT_ROOT / "data" / "true_labels" / "09_data_final.csv",
              "dependencies.txt": DEPENDENCIES_FILE, "system_prompt_single_outcome.txt": _sp1,
              "system_prompt_multiple_outcomes.txt": _sp2,
              "filtered_models.json": PROJECT_ROOT / "src" / "answer_generation" / "filtered_models.json"}
    return {k: (_sha256_file(v) if v.exists() else None) for k, v in inputs.items()}


if __name__ == "__main__":
    init_globals()
    _args = sys.argv[1:]

    if "--test" in _args:
        # -------- single fixed-config smoke of the pipeline --------
        fixed_params = {
            "parser": "PyMuPDFParser",
            "mode": os.getenv("TEST_MODE", "rag"),
            "chunk_size": 500,
            "chunk_overlap": 0,
            "chunker": "LengthChunker",
            "embedder": "BGEEmbedder",
            "retriever": "TopKRetriever",
            "k": 5,
            "query_embedding": "raw",
            "llm_model": os.getenv("TEST_MODEL", "Qwen/Qwen3.5-9B"),
            "use_default_prompt": True,
            "prompt_type": "base_prompt",
            "default_prompt_type": "base_prompt",
            "question_mode": os.getenv("TEST_QUESTION_MODE", "individual"),
            "vlm": os.getenv("TEST_VLM", "0") == "1",
            "disable_reasoning": os.getenv("TEST_DISABLE_REASONING", "1") == "1",
        }
        trial = optuna.trial.FixedTrial(fixed_params)
        score = objective(trial, verbose=True)
        print(f"\n[--test] mean accuracy: {score:.2%}")
    else:
        # -------- full TPE study over the tuning set --------
        # Usage: python objective.py [n_trials] [--models=id1,id2,...]
        #        python objective.py --rerun=<N> [--keep-old] [--resume-papers]
        #          --resume-papers: take off from the papers trial N already finished
        n_trials = next((int(a) for a in _args if a.isdigit()), 300)
        # Exclude models flagged disabled (e.g. a provider moved one non-serverless
        # mid-study). They stay in LLMS so the llm_model distribution is unchanged
        # (removing a choice would break Optuna with "dynamic value space"), but we
        # never seed them; if TPE samples one post-startup it just fails fast.
        seed_models = [m for m in LLMS.keys()
                       if not LLM_META_MAP.get(m, {}).get("disabled")]
        for a in _args:
            if a.startswith("--models="):
                seed_models = [m for m in a.split("=", 1)[1].split(",") if m in LLMS]

        optuna.logging.set_verbosity(optuna.logging.INFO)
        # Persistent storage so a crash / Together 402 does not lose the study;
        # re-running resumes (load_if_exists). Override path with OPTUNA_DB.
        storage_url = os.getenv("OPTUNA_DB")
        if not storage_url:
            # Until 2026-10-02 this defaulted to the root optuna_study.db -- the frozen
            # Phase-1 artefact, which still holds 3 WAITING trials that would execute.
            raise SystemExit(
                "OPTUNA_DB is required, e.g. OPTUNA_DB=sqlite:///$PWD/stage2_analyses/phase2_study/phase2.db"
                " -- the root optuna_study.db is the frozen Phase-1 study and is never a default target."
            )
        logger.warning("Optuna storage: %s", storage_url)
        # Two workers share one sqlite file; the default 5 s lock timeout makes
        # concurrent writes raise "database is locked" and fail otherwise-good
        # trials. 120 s matches the admin scripts (requeue_on_recovery.py).
        storage_obj = optuna.storages.RDBStorage(
            storage_url,
            engine_kwargs=({"connect_args": {"timeout": 120}}
                           if storage_url.startswith("sqlite") else {}),
        )
        study = optuna.create_study(
            direction="maximize",
            study_name="full_item_accuracy_tuning",
            storage=storage_obj,
            load_if_exists=True,
            # multivariate+group: model the conditional space jointly instead of
            # per-parameter (RAG-only levers are meaningless outside their branch);
            # constant_liar: two workers sample concurrently without duplicating.
            # Startup (100) completed 2026-09-28 on the unified scoring basis.
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=int(os.getenv("TPE_N_STARTUP", "100")),
                seed=42,
                multivariate=True, group=True, constant_liar=True,
            ),
            # NopPruner: the prereg (§6.2) evaluates EVERY configuration on the full
            # 35-paper tuning set. Optuna's default MedianPruner was killing slower
            # models' trials mid-run (no baseline -> biased model comparison), so
            # disable per-trial pruning. Study-level early stop is the patience cb.
            pruner=optuna.pruners.NopPruner(),
        )

        # ---- rerun a single existing trial by number, then exit ----
        # Enqueues that trial's exact config as a fresh trial (current env knobs
        # apply, e.g. VLM_ZOOM) and runs only it — no model seeding.
        _rerun_no = next(
            (int(a.split("=", 1)[1]) for a in _args if a.startswith("--rerun=")),
            None,
        )
        if _rerun_no is not None:
            rerun_specific_trial(
                study,
                storage_url,
                _rerun_no,
                keep_old=("--keep-old" in _args),
                resume_papers=("--resume-papers" in _args),
            )
            sys.exit(0)

        # Seed one trial per model so every model is evaluated at least once.
        # Fast, deterministic RAG config (k=5, thinking off) keeps seeds cheap;
        # TPE explores full-document mode / thinking-on / other configs afterwards.
        # Re-seed only models that lack an active (complete/running/queued) trial —
        # so a fresh study enqueues all models, and a RESUME re-seeds only models
        # whose seed failed (e.g. a transient network outage), without duplicating.
        # An enqueued (WAITING) trial keeps its params in system_attrs['fixed_params'],
        # NOT in t.params (which is empty until the trial runs) — read both so we
        # don't double-enqueue models that are already queued.
        def _seed_model(t):
            return t.params.get("llm_model") or t.system_attrs.get(
                "fixed_params", {}
            ).get("llm_model")

        active_models = {
            _seed_model(t)
            for t in study.trials
            if t.state.name in ("COMPLETE", "RUNNING", "WAITING")
        }
        seeded = 0
        for model in seed_models:
            if model in active_models:
                continue
            study.enqueue_trial(
                {
                    "mode": "rag",
                    "llm_model": model,
                    "parser": "PyMuPDFParser",
                    "chunk_size": 500,
                    "chunk_overlap": 0,
                    "chunker": "LengthChunker",
                    "embedder": "BGEEmbedder",
                    "retriever": "TopKRetriever",
                    "k": 5,
                    "query_embedding": "raw",
                    "default_prompt_type": "base_prompt",
                    "use_default_prompt": True,
                    "disable_reasoning": True,
                }
            )
            seeded += 1
        logger.info(
            "Enqueued %d seed trial(s); study has %d existing trials",
            seeded, len(study.trials),
        )

        patience = PatienceCallback(patience=15)

        def _obj(t):
            acc = objective(t, verbose=True)
            p = t.params
            print(
                f"\n>>> Trial {t.number}  acc={acc:.2%}  "
                f"model={p.get('llm_model', '?')}  mode={p.get('mode', '?')}  "
                f"parser={p.get('parser', '?')}  prompt={p.get('default_prompt_type', '?')}  "
                f"think_off={p.get('disable_reasoning', 'n/a')}"
            )
            return acc

        # catch=(Exception,): a trial that crashes (transient API 402/timeout,
        # a parser/model hiccup) is recorded as FAILED and the study continues,
        # instead of one bad trial killing the whole multi-hour run. Failed trials
        # are persisted and simply not used by the TPE surrogate.
        #
        # N_JOBS>1 runs that many TRIALS concurrently (Optuna thread pool). Best for
        # the random startup phase (first n_startup_trials=100, which are
        # independent); parallel TPE trials afterwards are slightly less
        # sample-efficient (they sample from a not-yet-updated surrogate). Total
        # concurrent API calls ~= N_JOBS * PAPER_CONCURRENCY, so keep the product
        # within provider rate limits: e.g. startup with N_JOBS=10 PAPER_CONCURRENCY=1
        # (=10-way), TPE with N_JOBS=1 PAPER_CONCURRENCY=4 (=4-way).
        study.optimize(
            _obj,
            n_trials=n_trials,
            n_jobs=int(os.getenv("N_JOBS", "1")),
            callbacks=[patience],
            catch=(Exception,),
        )

        print("\n" + "=" * 60)
        print(f"Best trial #{study.best_trial.number}  acc={study.best_value:.2%}")
        print(f"Params: {study.best_trial.params}")
        print(f"Completed {len(study.trials)} trials")
        print("=" * 60)
