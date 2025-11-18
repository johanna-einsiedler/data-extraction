import hashlib
import json
import os
import re
import sys
import time
import logging
from concurrent.futures import (
    ThreadPoolExecutor,  # <-- Add this at the top of your script
)
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import optuna
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from answer_generation.answer_generator import AnswerGenerator
from document_parsing.base_parser import ParseResult
from evaluate import evaluate_answer
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

_LLM_STRIP_PATTERNS = [
    (r"<(think|reason|thought)>.*?</\1>", dict(flags=re.DOTALL | re.IGNORECASE)),
    (
        r"\((thinking|reasoning|step-by-step reasoning):.*?\)",
        dict(flags=re.DOTALL | re.IGNORECASE),
    ),
    (r"<!--\s*(think|reasoning|steps).*?-->", dict(flags=re.DOTALL | re.IGNORECASE)),
    (
        r"^(Thoughts?:|Reasoning:|Step\s*\d*:|Internal monologue:|Chain-of-thought:)\s*",
        dict(flags=re.IGNORECASE | re.MULTILINE),
    ),
]

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def load_full_document_text(parsed_path: Path) -> str:
    """Return a string representation of the parsed document."""
    if parsed_path.suffix == ".json":
        with open(parsed_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, str):
            return data
        if isinstance(data, list):
            return "\n\n".join(str(item) for item in data)
        return json.dumps(data, ensure_ascii=False, indent=2)

    # XML or plain-text fallback
    return parsed_path.read_text(encoding="utf-8")


def _clean_llm_text(text):
    """Remove internal reasoning markers frequently returned by LLM APIs."""
    if not isinstance(text, str):
        return text
    cleaned = text
    for pattern, kwargs in _LLM_STRIP_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, **kwargs)
    return cleaned.strip()


def get_llm_text(output):
    """
    Normalize assorted LLM responses (string/dict/list) into a cleaned string.
    Handles direct strings, OpenAI-style choices, Together responses, and
    nested structures while stripping out internal reasoning markers.
    """
    if output is None:
        return ""

    if isinstance(output, str):
        return _clean_llm_text(output)

    if isinstance(output, (list, tuple, set)):
        parts = [get_llm_text(item) for item in output]
        return "\n".join([p for p in parts if p])

    if isinstance(output, dict):
        if "text" in output:
            return get_llm_text(output["text"])
        if "message" in output:
            return get_llm_text(output["message"])
        if "content" in output:
            return get_llm_text(output["content"])
        if "choices" in output:
            parts = [get_llm_text(choice) for choice in output.get("choices", [])]
            return "\n".join([p for p in parts if p])

        parts = []
        for key, value in output.items():
            if key in {"logprobs", "usage"}:
                continue
            parts.append(get_llm_text(value))
        return "\n".join([p for p in parts if p])

    if hasattr(output, "content"):
        return get_llm_text(getattr(output, "content"))

    return _clean_llm_text(str(output))


def _select_prompt(qid: str, qdata: dict, default_prompt: str, use_default: bool) -> tuple[str, str | None]:
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

    if prompt_type == "true_few_shot_prompt" and not prompts.get("true_few_shot_prompt"):
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


QUERIES_JSON_PATH = Path(
    "../query_creation/queries_with_prompts.json"
)  # Path to your JSON file
OUTCOMES_PATH = PROJECT_ROOT / "data" / "true_labels" / "outcomes.json"


def load_queries():
    with open(QUERIES_JSON_PATH, "r", encoding="utf-8") as f:
        query_dict = json.load(f)

    queries = {}
    for qid, q in query_dict.items():
        queries[qid] = q  # keep original structure
    return queries


try:
    with open(OUTCOMES_PATH, "r", encoding="utf-8") as f:
        OUTCOMES = json.load(f)
except FileNotFoundError:
    OUTCOMES = {}
    print(
        f"[objective] Warning: outcomes file not found at {OUTCOMES_PATH}. "
        "Proceeding with empty outcomes map."
    )


def load_true_answers(
    file: str = "codes_2025_10_14.xlsx", QUERIES: dict = None
) -> dict:
    """
    Load true coder answers from Excel and remove prefilled template values (from QUERIES[qid]['prefilled'] if available).
    Handles special case 4.5 (metrics and R² version).
    """
    import re
    from pathlib import Path

    import numpy as np

    path = PROJECT_ROOT / "data" / "true_labels" / file
    df = pd.read_excel(path)

    df.columns = [col.strip() for col in df.columns]

    # Fill unnamed headers from second row
    second_row = df.iloc[1]
    df.columns = [
        str(second_row[c]) if str(c).startswith("Unnamed") else c for c in df.columns
    ]

    # Identify valid item codes
    items = df["Item"].dropna().astype(str).str.strip()
    pattern = r"^\d+(\.\d+)*$"
    items = [x for x in items if re.match(pattern, x)]
    to_remove = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "99"]
    items = [x for x in items if x not in to_remove]

    # Identify coding columns (document IDs)
    coding_columns = [c for c in df.columns if re.match(r"^\d+$", str(c).strip())]
    if not coding_columns:
        raise ValueError("No coding columns detected (e.g., '021', '023').")

    true_answers = {}

    for doc_name in coding_columns:
        item_answers = {}

        for item in items:
            # Find all rows for this item
            start_idx_list = df.index[
                df["Item"].astype(str).str.strip() == item
            ].tolist()
            if not start_idx_list:
                continue
            start_idx = start_idx_list[0]
            row_ids = [start_idx]
            for idx in range(start_idx + 1, len(df)):
                cell_value = df.at[idx, "Item"]
                if pd.isna(cell_value) or str(cell_value).strip() == "":
                    row_ids.append(idx)
                else:
                    break

            sub_df = df.loc[row_ids]
            row_values = sub_df[doc_name].astype(str).tolist()

            # --- 🔸 Special case: 4.5 block ---
            if item == "4.5":
                subanswers = {f"4.5.{i}": ["-99"] for i in range(1, 7)}
                for val in row_values:
                    val = str(val).strip()
                    if not val or val.lower() == "nan":
                        continue

                    if val.startswith("AUC"):
                        m = re.search(r"[-−]?\d*\.?\d+", val)
                        if m:
                            subanswers["4.5.1"] = [m.group(0)]
                    elif val.startswith("ACC"):
                        m = re.search(r"[-−]?\d*\.?\d+", val)
                        if m:
                            subanswers["4.5.2"] = [m.group(0)]
                    elif val.startswith("BACC"):
                        m = re.search(r"[-−]?\d*\.?\d+", val)
                        if m:
                            subanswers["4.5.3"] = [m.group(0)]
                    elif val.startswith("R²") or val.startswith("R2"):
                        m = re.search(r"[-−]?\d*\.?\d+", val)
                        if m:
                            subanswers["4.5.4"] = [m.group(0)]
                    elif re.match(r"^r[:\s]", val, re.IGNORECASE):
                        m = re.search(r"[-−]?\d*\.?\d+", val)
                        if m:
                            subanswers["4.5.5"] = [m.group(0)]
                    elif "Specify R" in val or "Character" in val:
                        # Handle textual R² version (4.5.6)
                        m = re.search(r"Specify R.?2.*?:\s*(.*)", val)
                        if m:
                            subanswers["4.5.6"] = [m.group(1).strip()]
                        elif ":" not in val and val not in ["Character", ""]:
                            subanswers["4.5.6"] = [val.strip()]
                        else:
                            subanswers["4.5.6"] = ["not specified"]

                item_answers.update(subanswers)
                continue

            # --- Generic case ---
            answers = [
                a.strip() for a in row_values if a.strip() and a.lower() != "nan"
            ]

            # 🔹 Remove prefilled values if QUERIES is provided
            if QUERIES and item in QUERIES:
                prefilled_vals = QUERIES[item].get("prefilled", [])
                if isinstance(prefilled_vals, str):
                    prefilled_vals = [prefilled_vals]
                prefilled_vals = [str(v).strip().lower() for v in prefilled_vals]
                answers = [
                    a
                    for a in answers
                    if a.strip().lower() not in prefilled_vals
                    and not any(
                        a.strip().lower().startswith(pv) for pv in prefilled_vals
                    )
                ]

            item_answers[item] = answers or ["-99"]

        true_answers[doc_name] = item_answers

    return true_answers


# ---------------------------------------------------------------------
# helper path builders (change base folders if you like)
# ---------------------------------------------------------------------
def parsed_dir(parser_name: str, base: str = "../data/parsed") -> Path:
    return Path(base) / parser_name


def chunks_dir(
    parser_name: str, chunker_name: str, base: str = "../data/chunks"
) -> Path:
    return Path(base) / parser_name / chunker_name


def embeddings_dir(
    parser_name: str,
    chunker_name: str,
    embedder_name: str,
    base: str = "../data/embeddings",
) -> Path:
    return Path(base) / parser_name / chunker_name / embedder_name


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
def ensure_parsed(file_path: Path, parser_name: str):
    """Ensure a file is parsed. Return path to cached or newly parsed output."""
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
            return cached

    # 2️⃣ Run parser
    parser = get_parser(parser_name)
    result = parser.parse(str(file_path))  # make sure your parser uses `out_dir`
    # 3️⃣ Save result depending on type
    if isinstance(result, ParseResult):
        format_hint = result.format or "markdown"
        target_path = {
            "tei_xml": xml_path,
            "markdown": md_path,
        }.get(format_hint, md_path)
        with open(target_path, "w", encoding="utf-8") as fh:
            fh.write(result.content)
        if result.metadata:
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(result.metadata, fh, ensure_ascii=False, indent=2)
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


def get_cleaned_text_and_logprobs(llm_output):
    """
    Returns cleaned text and a dict of token -> logprob.
    Only includes logprobs for tokens that appear in the cleaned text.
    Works whether or not thinking tokens are present.
    Expects llm_output to have:
        - 'text': string
        - 'logprobs': LogprobsPart object with 'tokens' and 'token_logprobs'
    """
    text = llm_output["text"]
    logprobs_obj = llm_output["logprobs"]

    tokens = getattr(logprobs_obj, "tokens", []) if logprobs_obj else []
    token_logprobs = getattr(logprobs_obj, "token_logprobs", []) if logprobs_obj else []

    # Clean text
    cleaned_text = _clean_llm_text(text)

    # Build logprobs for only tokens in cleaned text
    logprobs_dict = {}
    if tokens and token_logprobs:
        idx = 0  # position in cleaned_text
        cleaned_len = len(cleaned_text)
        for tok, lp in zip(tokens, token_logprobs):
            tok_len = len(tok)
            if idx >= cleaned_len:
                break

            # Check if token matches the next part of cleaned_text
            if cleaned_text[idx : idx + tok_len] == tok:
                logprobs_dict[tok] = lp
                idx += tok_len
            # If token is whitespace or punctuation in cleaned text, still include it
            elif tok.strip() != "" and tok in cleaned_text[idx:]:
                # Find next occurrence of token in cleaned_text
                next_idx = cleaned_text.find(tok, idx)
                if next_idx != -1:
                    logprobs_dict[tok] = lp
                    idx = next_idx + tok_len

    return cleaned_text, logprobs_dict


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
    base: str = "../data/query_embeddings",
    queries_base: str = "../query_creation",
) -> Path:
    """
    Ensure a single query embedding is cached.
    - Embeddings stored under {base}/{embedder_name}/{doc_name}_{query_id}_{query_mode}.npy
    - Query info and prompts are read directly from query_info.json
    """
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
CACHE_PATH = Path("../trial_cache")
CACHE_PATH.mkdir(exist_ok=True)


def _hash_trial_key(params: dict, doc_idx: int, query_id: str) -> str:
    """
    Create a stable hash key for caching based on params+doc+query.
    """
    key_data = json.dumps(
        {"params": params, "doc_idx": doc_idx, "query_id": query_id},
        sort_keys=True,
    )
    return hashlib.md5(key_data.encode("utf-8")).hexdigest()


def load_cache():
    cache_file = CACHE_PATH / "cache.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    cache_file = CACHE_PATH / "cache.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def load_doc_id_map(path: str = "../data/doc_id_map.json") -> dict:
    """Load or create a persistent mapping from document name to ID."""
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_doc_id_map(doc_id_map: dict, path: str = "../data/doc_id_map.json"):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc_id_map, f, indent=2, ensure_ascii=False)


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


def objective(trial, verbose: bool = False):
    import hashlib
    import json
    from pathlib import Path

    import numpy as np

    trial_choices = {}
    all_results = []
    llm_choices = []

    def log(msg: str):
        if verbose:
            logger.info(msg)

    cache = load_cache()

    try:
        trial_start_time = time.time()
        mode = trial.suggest_categorical("mode", ["rag", "full"])

        # ---------------- Trial-level hyperparameters ----------------
        parser_name = trial.suggest_categorical("parser", list(PARSERS.keys()))
        if mode == "rag":
            chunk_size = trial.suggest_categorical(
                "chunk_size", [100, 250, 500, 750, 1000, 2500, 5000]
            )
            chunk_overlap = trial.suggest_categorical("chunk_overlap", [0, 250, 750])
            if chunk_overlap >= chunk_size:
                raise optuna.TrialPruned()
            if parser_name in ["PyMuPDFParser", "PyMuPDFTesseractParser"]:
                allowed_chunkers = [
                    k for k in CHUNKERS if k not in ["StructureChunker", "TextChunker"]
                ]
            else:
                allowed_chunkers = list(CHUNKERS.keys())
            chunker_name = trial.suggest_categorical("chunker", allowed_chunkers)
            embedder_name = trial.suggest_categorical(
                "embedder", list(EMBEDDERS.keys())
            )
            retriever_name = trial.suggest_categorical(
                "retriever", list(RETRIEVERS.keys())
            )
            retriever_cls = RETRIEVERS[retriever_name]
            k = trial.suggest_categorical("k", [1, 3, 5, 10, 20])
            retriever_params = {"k": k}
            if hasattr(retriever_cls, "token_budget"):
                retriever_params["token_budget"] = trial.suggest_categorical(
                    "token_budget", [200, 500, 1000, 2000]
                )
            if getattr(retriever_cls, "supports_rerank", False):
                retriever_params["rerank"] = trial.suggest_categorical(
                    "rerank", [True, False]
                )
                if retriever_params["rerank"] and getattr(
                    retriever_cls, "supports_top_m", False
                ):
                    retriever_params["top_m"] = trial.suggest_categorical(
                        "top_m", [5, 10, 20, 50]
                    )
            retriever = retriever_cls(**retriever_params)
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

        llm_model = trial.suggest_categorical("llm_model", list(LLMS.keys()))
        llm_meta = LLM_META_MAP.get(llm_model, {})
        llm_api_type = llm_meta.get("api_type")
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

        default_prompt_type = trial.suggest_categorical(
            "default_prompt_type",
            [
                "base_prompt",
                "reasoning_prompt",
                "rewritten_prompt",
                "synthetic_few_shot_prompt",
                "true_few_shot_prompt",
            ],
        )
        use_default_prompt = trial.suggest_categorical(
            "use_default_prompt", [True, False]
        )

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
            }
        )
        log("Trial choices: {trial_choices}")
        # ---------------- Prepare trial folder ----------------
        trial_hash = hashlib.md5(
            json.dumps(trial_choices, sort_keys=True).encode()
        ).hexdigest()[:10]
        trial_folder = Path("../trial_logs") / f"{trial_hash}_trial_{trial.number:04d}"
        trial_folder.mkdir(parents=True, exist_ok=True)
        log(f"Writing trial results to: {trial_folder}")

        # ---------------- Parse all documents ----------------
        folder = Path("../data/raw/")
        all_docs = [f for f in folder.glob("*") if f.suffix.lower() == ".pdf"]
        all_docs_names = [
            f.name for f in folder.glob("*") if f.suffix.lower() == ".pdf"
        ]

        log(f"  Found {len(all_docs_names)} documents")
        parsed_docs = [ensure_parsed(f, parser_name) for f in all_docs]
        log("  Parsing complete")

        doc_aggregates = {}
        query_aggregates = {}
        doc_id_map = load_doc_id_map()
        next_id = max(doc_id_map.values(), default=-1) + 1
        for parsed_path in parsed_docs:
            # ---------------- Document setup ----------------
            match = re.match(r"^(\d+)", parsed_path.stem)
            doc_name = match.group(1) if match else parsed_path.stem
            outcome = OUTCOMES.get(doc_name, "")
            if outcome == "":
                system_prompt = SYSTEM_PROMPTS["single_outcome"]
                print("outcome none")
            else:
                system_prompt = SYSTEM_PROMPTS["multiple_outcomes"].format(
                    OUTCOME=outcome
                )
                print("multiple outcomes")
            print("outcome: ", outcome)
            print(system_prompt)
            if doc_name not in doc_id_map:
                doc_id_map[doc_name] = next_id
                next_id += 1
            doc_id = doc_id_map[doc_name]
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
                if cached_hash == current_hash:
                    existing_results = existing_data.get("results", [])
                    existing_qids = {r["query_id"] for r in existing_results}
                    log(f"  Found {len(existing_qids)} cached query results.")
                else:
                    log("  Parameter mismatch → recomputing all queries.")
            else:
                log(f"  No cache found for {doc_name}, evaluating all queries.")

            # Start with cached results
            doc_results = list(existing_results)
            all_results = all_results + doc_results
            doc_result_map = {r["query_id"]: r for r in existing_results}
            # ---------------- Ensure chunks / embeddings or load full text ----------------
            vector_store = None
            full_doc_text = None
            if mode == "rag":
                chunk_path = ensure_chunks(
                    parsed_path, parser_name, chunker_name, chunk_size, chunk_overlap
                )
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
                vector_store.load()
            else:
                full_doc_text = load_full_document_text(parsed_path)

            # ---------------- Per-query evaluation ----------------
            for qid, qdata in QUERIES.items():
                if qid in existing_qids:
                    log(f"  Skipping query {qid} (cached for this doc).")
                    continue
                log(f"  Evaluating NEW query {qid} for {doc_name}")
                needs_evaluation = True
                if qid in DEPENDENCIES:
                    applicable = check_dependency(
                        qid, all_results, DEPENDENCIES, doc_id=doc_id
                    )
                    if not applicable:
                        log(f"Skipping query {qid} because it is not applicable.")
                        if TRUE_ANSWERS[doc_name][qid][0] != str(-99):
                            log(
                                f"Warning: Unexpected value for item {qid}; expected -99 / not applicable but true value is {TRUE_ANSWERS[doc_name][qid]}"
                            )

                        eval_result = {
                            "raw_output": "Not applicable",
                            "pred_value": -99,
                            "true_value": TRUE_ANSWERS[doc_name][qid],
                            "accuracy": 1,
                        }
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
                        query_emb_path = ensure_query_embeddings(
                            embedder_name, parsed_path.stem, qid, query_embedding
                        )
                        query_embed = np.load(query_emb_path)
                        retrieved_chunks = retriever.retrieve(query_embed, vector_store)
                    else:
                        if full_doc_text is None:
                            full_doc_text = load_full_document_text(parsed_path)
                        retrieved_chunks = [(full_doc_text, 1.0)]
                # Determine prompt type
                prompt_type, prompt_text = _select_prompt(
                    qid, qdata, default_prompt_type, use_default_prompt
                )
                ans_gen = LLMS[llm_model]
                if prompt_type == "binary_prompts" and qdata["type"] in [
                    "multiple_choice",
                    "single_choice",
                ]:
                    binary_outputs = {}
                    combined_logprobs = []
                    llm_output = {}
                    llm_output["text"] = ""
                    # Store the mapping from binary prompts to the multiple choice letters
                    binary_to_choice = {}

                    for i, subprompt in enumerate(qdata["prompts"]["binary_prompts"]):
                        llm_output_binary = ans_gen.generate(
                            query=subprompt,
                            system_prompt=system_prompt,
                            chunks=retrieved_chunks,
                            return_logprobs=True,
                        )
                        text, logprobs = get_cleaned_text_and_logprobs(
                            llm_output_binary
                        )
                        binary_outputs[f"binary_{i}"] = text.strip()
                        combined_logprobs.append(logprobs)
                        llm_output["text"] = (
                            llm_output["text"] + "\n" + llm_output_binary["text"]
                        )
                        # Map 'A' -> the letter for this choice, 'B' -> empty
                        if text.strip().upper() == "A":
                            letter_for_choice = list(qdata["choices"].keys())[
                                i
                            ]  # assumes ordering matches
                            binary_to_choice[f"binary_{i}"] = letter_for_choice
                        else:
                            binary_to_choice[f"binary_{i}"] = ""

                    # Concatenate all non-empty mapped letters
                    final_answer = ",".join(
                        letter for letter in binary_to_choice.values() if letter
                    )
                    if final_answer == "":
                        final_answer = "Z"

                    llm_text = final_answer
                    llm_logprobs = combined_logprobs

                else:
                    log(f"prompt agaain {prompt_text}")
                    llm_output = ans_gen.generate(
                        system_prompt=system_prompt,
                        query=prompt_text,
                        chunks=retrieved_chunks,
                        return_logprobs=True,
                    )
                    llm_text, llm_logprobs = get_cleaned_text_and_logprobs(llm_output)

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
                        )
                        return other_response["text"]

                    return handle_other

                # Example usage (where you loop through or handle a query)
                callback = make_handle_other(qid, QUERIES, ans_gen, retrieved_chunks)

                eval_result = evaluate_answer(
                    llm_text,
                    question_type=qdata["type"],
                    doc_name=doc_name,
                    query_id=qid,
                    TRUE_ANSWERS=TRUE_ANSWERS,
                    QUERIES=QUERIES,
                    other_callback=callback,
                )
                eval_result["context"] = retrieved_chunks
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

                # ✅ Update caches
                trial_key = _hash_trial_key(trial_choices, doc_id, qid)
                cache[trial_key] = result

                # ✅ Update in-memory lists
                doc_results.append(result)
                all_results.append(result)
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
                with open(doc_file, "w", encoding="utf-8") as f:
                    json.dump(doc_data, f, indent=2, ensure_ascii=False)
            num_cached = len(existing_results)
            num_new = len(doc_results) - num_cached
            num_total = len(doc_results)
            doc_accuracy = (
                np.mean([r["accuracy"] for r in doc_results]) if doc_results else 0
            )
            # Report intermediate value to the pruner
            trial.report(
                doc_accuracy, step=doc_id
            )  # step can be any integer, e.g., doc_id

            # Check if the trial should be pruned
            if trial.should_prune():
                log(
                    f"⚠️ Pruning trial {trial.number} at document {doc_id} (doc_accuracy={doc_accuracy:.3f})"
                )
                raise optuna.TrialPruned()
            log(
                f"📄 Document {doc_id} finished: {num_cached} cached, {num_new} new, total {num_total} queries."
            )
            log(f"    Overall accuracy for document {doc_id}: {doc_accuracy:.3f}")

            # ✅ Update doc aggregates unconditionally
            doc_aggregates[f"doc_{doc_id}"] = {
                "average_accuracy": doc_accuracy,
                "doc_name": doc_name,
            }

            # ✅ Update query aggregates
            for qid_agg in QUERIES.keys():
                query_scores = [
                    r["accuracy"] for r in all_results if r["query_id"] == qid_agg
                ]
                query_aggregates[qid_agg] = {"average_accuracy": np.mean(query_scores)}

            # ✅ Save partial trial summary after each query
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

            overall_avg = np.mean([r["accuracy"] for r in all_results])

            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "trial_number": trial.number,
                "parameters": trial_choices,
                "overall_average_accuracy": overall_avg,
                "document_aggregates": doc_aggregates,
                "query_aggregates": query_aggregates,
                "per_query_doc_results": per_query_doc_results,
            }

            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)

            # ✅ Persist the cache periodically (avoids loss if run stops)
            save_cache(cache)

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


def run_worker(study):
    study.optimize(objective, n_trials=1)  # each worker runs one trial at a time


if __name__ == "__main__":
    folder = Path("../data/raw/")
    all_docs = list(folder.glob("*"))
    QUERIES = load_queries()

    TRUE_ANSWERS = load_true_answers()

    DEPENDENCIES_FILE = PROJECT_ROOT / "src" / "dependencies.txt"
    with open(DEPENDENCIES_FILE, "r", encoding="utf-8") as f:
        DEPENDENCIES = json.load(f)

    QUERIES = dict(list(QUERIES.items()))

    SYSTEM_PROMPTS = {}
    with open(
        PROJECT_ROOT / "query_creation" / "system_prompt_single_outcome.txt",
        "r",
        encoding="utf-8",
    ) as f:
        SYSTEM_PROMPTS["single_outcome"] = f.read()
    with open(
        PROJECT_ROOT / "query_creation" / "system_prompt_multiple_outcomes.txt",
        "r",
        encoding="utf-8",
    ) as f:
        SYSTEM_PROMPTS["multiple_outcomes"] = f.read()
    ## FIXED TRIAL
    fixed_params = {
        "parser": "PyMuPDFParser",
        "mode": "rag",
        "chunk_size": 500,
        "chunk_overlap": 0,
        "chunker": "LengthChunker",
        "embedder": "BGEEmbedder",
        "retriever": "TopKRetriever",
        "k": 5,
        "query_embedding": "raw",
        "llm_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        # "gpt-4.1-2025-04-14",
        "use_default_prompt": True,
        "prompt_type": "base_prompt",
        "default_prompt_type": "base_prompt",
    }

    # Wrap trial as FixedTrial
    trial = optuna.trial.FixedTrial(fixed_params)
    score = objective(trial, verbose=True)

    # ACUTAL ONE WITH PRUNING
    # sampler = optuna.samplers.TPESampler(
    #     n_startup_trials=10,  # first 10 trials are random
    #     n_ei_candidates=24,  # number of candidates for expected improvement
    #     multivariate=True,  # allow correlated sampling
    # )
    # # --- Create study with pruner ---
    # pruner = optuna.pruners.MedianPruner(
    #     n_startup_trials=5,  # don’t prune before 5 trials complete
    #     n_warmup_steps=2,  # don’t prune before 2 steps of report()
    # )

    # study = optuna.create_study(
    #     direction="maximize",
    #     sampler=sampler,
    #     pruner=pruner,
    #     storage="postgresql+psycopg2://einsie0004:optuna@localhost:5432/trials_v1",
    #     load_if_exists=True,
    # )

    # def stop_if_no_improvement(study, trial, threshold=0.01, n_last=10):
    #     completed = [
    #         t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    #     ]
    #     if len(completed) < n_last:
    #         return
    #     recent = completed[-n_last:]
    #     if max(recent) - min(recent) < threshold:
    #         print(
    #             f"Stopping study early: improvement {max(recent) - min(recent):.5f} < {threshold}"
    #         )
    #         study.stop()

    # # Launch 40 workers in threads (fine for I/O bound tasks)
    # with ThreadPoolExecutor(max_workers=40) as executor:
    #     futures = [executor.submit(run_worker, study) for _ in range(10)]
    #     for f in futures:
    #         f.result()
# study.optimize(objective, n_trials=100, callbacks=[stop_if_no_improvement])
# --- Run trials ---

######################## OLD OBJECTIVE

# def objective(trial, verbose: bool = False):
#     import hashlib
#     import json
#     from pathlib import Path

#     import numpy as np

#     trial_choices = {}
#     all_results = []
#     llm_choices = []

#     # Helper logging function
#     def log(msg: str):
#         if verbose:
#             print(msg, file=sys.stderr, flush=True)

#     # Load cache globally
#     cache = load_cache()

#     try:
#         # ---------------- Parameter suggestions ----------------
#         parser_name = trial.suggest_categorical("parser", list(PARSERS.keys()))
#         mode = "rag"
#         trial_choices["parser"] = parser_name
#         trial_choices["mode"] = mode

#         # Compute hash from parameter choices so far (will be updated as more are added)
#         trial_hash = hashlib.md5(
#             json.dumps(trial_choices, sort_keys=True).encode()
#         ).hexdigest()[:10]
#         trial_folder = Path("../trial_logs") / f"{trial_hash}_trial_{trial.number:04d}"
#         trial_folder.mkdir(parents=True, exist_ok=True)
#         log(f"Writing trial results to: {trial_folder}")

#         log(f"[Trial {trial.number}] Starting with parser={parser_name}, mode={mode}")

#         folder = Path("../data/raw/")
#         all_docs = [f for f in folder.glob("*") if f.suffix.lower() == ".pdf"]
#         log(f"  Found {len(all_docs)} documents")

#         parsed_docs = [ensure_parsed(f, parser_name) for f in all_docs]
#         log("  Parsing complete")

#         doc_aggregates = {}
#         query_aggregates = {}

#         doc_id_map = load_doc_id_map()
#         next_id = max(doc_id_map.values(), default=-1) + 1

#         # ---------------- Per-document loop ----------------
#         for parsed_path in parsed_docs:
#             doc_name = parsed_path.stem
#             if doc_name not in doc_id_map:
#                 doc_id_map[doc_name] = next_id
#                 next_id += 1
#             doc_id = doc_id_map[doc_name]
#             doc_results = []

#             log(f"--- Document {doc_id}: {parsed_path.name} ---")

#             # Compute the full parameter hash *after* all trial parameters are decided later
#             # so we’ll temporarily just set up doc file here
#             doc_file = trial_folder / f"doc_{doc_id:04d}.json"

#             # Load cached file if it exists and matches this parameter hash
#             if doc_file.exists():
#                 with open(doc_file, "r", encoding="utf-8") as f:
#                     existing_data = json.load(f)
#                 cached_hash = existing_data.get("parameters_hash")
#                 current_hash = hashlib.md5(
#                     json.dumps(trial_choices, sort_keys=True).encode()
#                 ).hexdigest()

#                 if cached_hash == current_hash:
#                     log(f"  Using existing document results from {doc_file.name}")
#                     for r in existing_data["results"]:
#                         all_results.append(r)
#                     doc_aggregates[f"doc_{doc_id}"] = {
#                         "average_accuracy": np.mean(
#                             [r["accuracy"] for r in existing_data["results"]]
#                         ),
#                         "doc_name": doc_name,
#                         "cached": True,
#                     }
#                     continue
#                 else:
#                     log(
#                         f"  Parameter mismatch (old={cached_hash[:6]}, new={current_hash[:6]}), recalculating."
#                     )

#             # ---------------- Per-query loop ----------------
#             for qid, qdata in QUERIES.items():
#                 log(f"  > Query {qid}")
#                 start_stage = time.time()

#                 # ---------------- Retrieval stage ----------------
#                 if mode == "rag":
#                     chunk_size = trial.suggest_categorical(
#                         "chunk_size", [100, 250, 500, 750, 1000, 2500, 5000]
#                     )
#                     overlap = trial.suggest_categorical(
#                         "chunk_overlap",
#                         [o for o in [0, 250, 750] if o < chunk_size] or [0],
#                     )
#                     if parser_name in ["PyMuPDFParser", "PyMuPDFTesseractParser"]:
#                         allowed_chunkers = [
#                             k
#                             for k in CHUNKERS
#                             if k not in ["StructureChunker", "TextChunker"]
#                         ]
#                     else:
#                         allowed_chunkers = list(CHUNKERS.keys())

#                     chunker_name = trial.suggest_categorical(
#                         "chunker", allowed_chunkers
#                     )
#                     embedder_name = trial.suggest_categorical(
#                         "embedder", list(EMBEDDERS.keys())
#                     )

#                     trial_choices.update(
#                         {
#                             "chunk_size": chunk_size,
#                             "chunk_overlap": overlap,
#                             "chunker": chunker_name,
#                             "embedder": embedder_name,
#                         }
#                     )

#                     chunk_path = ensure_chunks(
#                         parsed_path, parser_name, chunker_name, chunk_size, overlap
#                     )
#                     vector_store_path = ensure_embedding(
#                         chunk_path,
#                         parser_name,
#                         chunker_name,
#                         embedder_name,
#                         chunk_size,
#                         overlap,
#                     )
#                     vector_store = NumpyVectorStore(
#                         save_path=str(vector_store_path) + ".npz"
#                     )
#                     vector_store.load()

#                     retriever_name = trial.suggest_categorical(
#                         "retriever", list(RETRIEVERS.keys())
#                     )
#                     retriever_cls = RETRIEVERS[retriever_name]
#                     k = trial.suggest_categorical("k", [1, 3, 5, 10, 20])
#                     params = {"k": k}
#                     if hasattr(retriever_cls, "token_budget"):
#                         params["token_budget"] = trial.suggest_categorical(
#                             "token_budget", [200, 500, 1000, 2000]
#                         )
#                     if getattr(retriever_cls, "supports_rerank", False):
#                         params["rerank"] = trial.suggest_categorical(
#                             "rerank", [True, False]
#                         )
#                         if params["rerank"] and getattr(
#                             retriever_cls, "supports_top_m", False
#                         ):
#                             params["top_m"] = trial.suggest_categorical(
#                                 "top_m", [5, 10, 20, 50]
#                             )

#                     retriever = retriever_cls(**params)
#                     query_embedding_mode = trial.suggest_categorical(
#                         "query_embedding", ["raw", "label_def"]
#                     )
#                     query_emb_path = ensure_query_embeddings(
#                         embedder_name, parsed_path.stem, qid, query_embedding_mode
#                     )
#                     query_embed = np.load(query_emb_path)
#                     retrieved_chunks = retriever.retrieve(query_embed, vector_store)

#                 log(f"    Retrieval done in {time.time() - start_stage:.2f}s")

#                 # ---------------- Caching check ----------------
#                 trial_key = _hash_trial_key(trial_choices, doc_id, qid)
#                 if trial_key in cache:
#                     result = cache[trial_key]
#                     log("    Using cached result")
#                 else:
#                     # ---------------- LLM stage ----------------
#                     llm_model = trial.suggest_categorical(
#                         "llm_model", list(LLMS.keys())
#                     )
#                     non_binary_prompts = [
#                         key
#                         for key in qdata["prompts"].keys()
#                         if key != "binary_prompts"
#                     ]
#                     prompt_type_general = trial.suggest_categorical(
#                         "prompt_type_general", non_binary_prompts
#                     )

#                     # Step 2: choose prompt type for the current question
#                     if qdata["type"] in ["multiple_choice", "single_choice"]:
#                         # options: either the base prompt or binary
#                         prompt_type = trial.suggest_categorical(
#                             "prompt_type_choice",
#                             [prompt_type_general, "binary_prompts"],
#                         )
#                     else:  # open-ended, list, numeric
#                         prompt_type = prompt_type_general

#                     prompt_text = qdata["prompts"].get(prompt_type)
#                     if not prompt_text:
#                         raise ValueError(
#                             f"No prompt found for query {qid} under '{prompt_type}'"
#                         )

#                     trial_choices.update(
#                         {"llm_model": llm_model, "prompt_type": prompt_type}
#                     )
#                     ans_gen = LLMS[llm_model]

#                     if prompt_type == "binary_prompts":
#                         if (
#                             qdata["type"] == "single_choice"
#                             or qdata["type"] == "multiple_choice"
#                         ):
#                             # Run all binary subprompts and collect outputs
#                             binary_outputs = {}
#                             for i, subprompt in enumerate(
#                                 qdata["prompts"]["binary_prompts"]
#                             ):
#                                 llm_output_binary = ans_gen.generate(
#                                     query=subprompt,
#                                     chunks=retrieved_chunks,
#                                     return_logprobs=True,
#                                 )
#                                 # Store each subresult under a key like "binary_0"
#                                 binary_outputs[f"binary_{i}"] = get_llm_text(
#                                     llm_output_binary
#                                 )

#                             # Combine according to your get_llm_text logic
#                             combined_texts = get_llm_text(binary_outputs)

#                             # Concatenate them into one string for evaluation
#                             if isinstance(combined_texts, list):
#                                 llm_output = ",".join(combined_texts)
#                             else:
#                                 llm_output = str(combined_texts)

#                         else:
#                             prompt_text = qdata["prompts"]["base_prompt"]
#                             llm_output = ans_gen.generate(
#                                 query=prompt_text,
#                                 chunks=retrieved_chunks,
#                                 return_logprobs=True,
#                             )

#                     else:
#                         # Non-binary case (unchanged)
#                         llm_output = ans_gen.generate(
#                             query=prompt_text,
#                             chunks=retrieved_chunks,
#                             return_logprobs=True,
#                         )

#                     def handle_other(llm_text):
#                         """Called when 'Other' (Z) is selected in multiple choice. Uses qdata to generate a follow-up LLM query."""

#                         other_prompt = QUERIES[qid]["follow_up_prompt"]

#                         other_response = ans_gen.generate(
#                             query=other_prompt,
#                             chunks=retrieved_chunks,  # same as original
#                             return_logprobs=True,
#                            # system_prompt=qdata["system_prompt"],
#                         )
#                         return other_response["text"]

#                     eval_result = evaluate_answer(
#                         get_llm_text(llm_output["text"]),
#                         question_type="multiple_choice",
#                         doc_name=doc_name,
#                         query_id=qid,
#                         #answer_file="human_codes_test.xlsx",
#                         TRUE_ANSWERS,
#                         QUERIES,
#                         #choices_file="query_choice_mapping.json",
#                         other_callback=handle_other,
#                     )

#                     result = {"doc_id": doc_id, "query_id": qid, **eval_result}
#                     cache[trial_key] = result

#                 llm_choices.append(result)
#                 doc_results.append(result)
#                 all_results.append(result)

#             # ---------------- Save per-document file ----------------
#             param_hash = hashlib.md5(
#                 json.dumps(trial_choices, sort_keys=True).encode()
#             ).hexdigest()
#             doc_data = {
#                 "doc_id": doc_id,
#                 "doc_name": doc_name,
#                 "parameters_hash": param_hash,
#                 "results": doc_results,
#             }
#             with open(doc_file, "w", encoding="utf-8") as f:
#                 json.dump(doc_data, f, indent=2, ensure_ascii=False)
#             log(f"  Saved {len(doc_results)} results to {doc_file.name}")

#             doc_avg = np.mean([r["accuracy"] for r in doc_results])
#             doc_aggregates[f"doc_{doc_id}"] = {
#                 "average_accuracy": doc_avg,
#                 "doc_name": doc_name,
#             }

#         # ---------------- Aggregate all results ----------------
#         for qid in QUERIES.keys():
#             query_scores = [r["accuracy"] for r in all_results if r["query_id"] == qid]
#             query_aggregates[qid] = {"average_accuracy": np.mean(query_scores)}

#         overall_avg = np.mean([r["accuracy"] for r in all_results])
#         log(f"[Trial {trial.number}] Finished. Overall acc={overall_avg:.3f}")

#         # ---------------- Trial summary ----------------
#         summary_file = trial_folder / "trial_summary.json"
#         with open(summary_file, "w", encoding="utf-8") as f:
#             json.dump(
#                 {
#                     "timestamp": datetime.now().isoformat(),
#                     "trial_number": trial.number,
#                     "parameters": trial_choices,
#                     "overall_average_accuracy": overall_avg,
#                     "document_aggregates": doc_aggregates,
#                     "query_aggregates": query_aggregates,
#                 },
#                 f,
#                 indent=2,
#                 ensure_ascii=False,
#             )
#         log(f"Wrote trial summary: {summary_file.name}")

#         save_cache(cache)
#         return float(overall_avg)

#     except Exception as e:
#         log(f"Trial failed with choices: {trial_choices}")
#         log(f"Error: {e}")
#         raise
