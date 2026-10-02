"""Reconstruct prompt/completion tokens and $ cost for trials that never recorded them.

Only 18 of 96 COMPLETE trials carry `prompt_tokens` user attrs, and no $ figure is
computed anywhere in the codebase. Both are recoverable offline, with zero API spend.

Route per mode (the stored `context` is the literal placeholder "[full document]"
for full-document trials, so it cannot be used there):

  full + individual  per cell : system + item prompt + parsed document text
  full + batched     per doc  : system + ALL item prompts + parsed document text
  rag                per cell : system + item prompt + retrieved context
  vlm=True                    : NOT recoverable (pages go as images) -> NaN

Binary decomposition (`use_default_prompt=False`) turns one call into one call per
answer option for the 29 eligible items — 217 sub-prompts in total — so it is a
large cost multiplier and is modelled explicitly.

Completion tokens are a LOWER BOUND: reasoning traces were never persisted, so for
`effective_reasoning='on'` the output cost is an interval, not a point.

Validate:  pipeline_env/bin/python -m analysis.hpo_analysis.tokens --validate
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import (CACHE, CELLS_PQ, OUTPUT_BLACKOUT, PARSED, ROOT, TABLES,
                    TRIALS_PQ)

sys.path.insert(0, str(ROOT / "src"))
_NULL = io.StringIO()
MODELS_JSON = ROOT / "src" / "answer_generation" / "filtered_models.json"
QUERIES_JSON = ROOT / "query_creation" / "queries_with_prompts.json"


# ------------------------------------------------------------------ tokenizer
# tiktoken 0.4.0 in pipeline_env ships only up to cl100k_base. Every model in the
# roster uses its own tokenizer anyway, so this is an explicit approximation; the
# per-mode calibration factor from validate() absorbs the systematic part of it.
_ENCODINGS = ("o200k_base", "cl100k_base", "gpt2")


@lru_cache(maxsize=1)
def _enc():
    import tiktoken
    for name in _ENCODINGS:
        try:
            return tiktoken.get_encoding(name)
        except Exception:
            continue
    raise RuntimeError("no usable tiktoken encoding")


@lru_cache(maxsize=1)
def encoding_name() -> str:
    import tiktoken
    for name in _ENCODINGS:
        try:
            tiktoken.get_encoding(name)
            return name
        except Exception:
            continue
    return "?"


def count(text: str | None) -> int:
    if not text:
        return 0
    return len(_enc().encode(str(text), disallowed_special=()))


# ------------------------------------------------------------------ pricing
@lru_cache(maxsize=1)
def pricing() -> dict[str, dict]:
    return {m["id"]: m for m in json.loads(MODELS_JSON.read_text())}


# ------------------------------------------------------------------ documents
# objective.ensure_parsed resolves a cached parse in THIS order and returns the
# first hit. Every parser dir holds both a .json and a .md, so the pipeline always
# read the .json — and load_full_document_text re-serialises that with json.dumps
# (indent=2), which is far larger than the raw markdown. Picking the .md instead
# understates full-document prompt tokens substantially, and parser-dependently.
_PARSE_SUFFIX_ORDER = (".json", ".txt", ".md", ".xml")


@lru_cache(maxsize=1)
def _doc_files() -> dict:
    """(parser, doc_name) -> the parsed file objective.py would actually have used."""
    out = {}
    for pdir in sorted(PARSED.iterdir()):
        if not pdir.is_dir():
            continue
        by_doc = {}
        for f in pdir.iterdir():
            if f.name.endswith(".meta.json") or f.suffix not in _PARSE_SUFFIX_ORDER:
                continue
            m = re.match(r"^(\d+)", f.stem)
            if m:
                by_doc.setdefault(m.group(1), {})[f.suffix] = f
        for doc, variants in by_doc.items():
            for suf in _PARSE_SUFFIX_ORDER:
                if suf in variants:
                    out[(pdir.name, doc)] = variants[suf]
                    break
    return out

@lru_cache(maxsize=None)
def doc_tokens(parser: str, doc_name: str) -> int:
    from _data_loading import load_full_document_text
    f = _doc_files().get((parser, str(doc_name)))
    if f is None:
        return 0
    try:
        return count(load_full_document_text(f))
    except Exception:
        return 0


@lru_cache(maxsize=1)
def char_per_token() -> float:
    """Calibrated chars-per-token over the parsed corpus, for RAG contexts whose
    text is not retained in cells.parquet (only its length is)."""
    from _data_loading import load_full_document_text
    chars = toks = 0
    for (parser, _), f in list(_doc_files().items()):
        if parser != "PyMuPDFParser":
            continue
        try:
            t = load_full_document_text(f)
        except Exception:
            continue
        chars += len(t)
        toks += count(t)
    return chars / toks if toks else 4.0


# ------------------------------------------------------------------ prompts
@lru_cache(maxsize=1)
def _queries() -> dict:
    return json.loads(QUERIES_JSON.read_text())


@lru_cache(maxsize=1)
def _evaluable_queries() -> dict:
    """The 43 evaluable items, exactly as objective.py filters them."""
    import _data_loading as D
    with contextlib.redirect_stdout(_NULL):
        Q = D.load_queries()
        keep = D.finalized_item_map(Q)
    return {k: v for k, v in Q.items() if k in keep}


@lru_cache(maxsize=1)
def batched_prompt_tokens() -> int:
    """The single combined prompt used by question_mode='batched'.

    src/batched.py renders a COMPACT per-item block, not the verbose stored
    prompt; summing the stored item prompts overestimates this by ~30%.
    """
    import batched
    with contextlib.redirect_stdout(_NULL):
        txt = batched.build_batched_prompt(_evaluable_queries())
    return count(txt.replace("{context}", ""))


@lru_cache(maxsize=None)
def binary_prompt_tokens(qid: str) -> int:
    """Binary decomposition is ONE call per cell with a synthesized prompt
    (src/objective.py:1181 — 'collapsed to ONE call ... was one call PER option'),
    so it is a prompt-shape change, not a call multiplier."""
    import binary_mode
    q = _evaluable_queries().get(qid) or _queries().get(qid)
    if not q:
        return 0
    with contextlib.redirect_stdout(_NULL):
        txt = binary_mode.build_binary_prompt(qid, q)
    return count(txt.replace("{context}", ""))


@lru_cache(maxsize=None)
def system_tokens(doc_name: str, oe: bool) -> int:
    from _data_loading import build_system_prompt
    with contextlib.redirect_stdout(_NULL):
        try:
            return count(build_system_prompt(str(doc_name), oe=oe))
        except Exception:
            return 0


@lru_cache(maxsize=None)
def item_prompt_tokens(qid: str, default_prompt: str, use_default: bool) -> tuple[int, int]:
    """(tokens_per_call, n_calls) for one item under this trial's prompt policy.

    Mirrors objective._select_prompt, including the binary-decomposition switch and
    the silent true_few_shot -> synthetic_few_shot fallback.
    """
    q = _queries().get(qid)
    if not q:
        return 0, 1
    prompts = q.get("prompts", {})
    if (q.get("type") in {"multiple_choice", "single_choice"}
            and prompts.get("binary_prompts")
            and len(q.get("choices", [])) > 2
            and not use_default):
        return binary_prompt_tokens(qid), 1
    ptype = default_prompt
    if ptype == "true_few_shot_prompt" and not prompts.get("true_few_shot_prompt"):
        ptype = "synthetic_few_shot_prompt"
    return count(prompts.get(ptype)), 1


# ------------------------------------------------------------------ estimation
def estimate_trial(trial: pd.Series, cells: pd.DataFrame) -> dict:
    mode = trial.get("mode")
    qmode = trial.get("question_mode")
    parser = trial.get("parser")
    dpt = trial.get("default_prompt_type") or "base_prompt"
    udp = trial.get("use_default_prompt")
    udp = True if udp is None or (isinstance(udp, float) and np.isnan(udp)) else bool(udp)
    vlm = trial.get("vlm")

    if vlm is True or str(vlm).lower() == "true":
        return {"prompt_tokens_est": np.nan, "n_calls_est": np.nan,
                "method": "vlm_images", "recoverable": False}

    docs = cells.doc_name.astype(str).unique()
    if mode == "full":
        if not parser:
            return {"prompt_tokens_est": np.nan, "n_calls_est": np.nan,
                    "method": "no_parser", "recoverable": False}
        dtok = {d: doc_tokens(parser, d) for d in docs}
        if not any(dtok.values()):
            return {"prompt_tokens_est": np.nan, "n_calls_est": np.nan,
                    "method": "parsed_missing", "recoverable": False}
        if qmode == "batched":
            total = calls = 0
            per_doc_items = batched_prompt_tokens()
            for d in docs:
                total += system_tokens(d, False) + per_doc_items + dtok[d]
                calls += 1
            return {"prompt_tokens_est": total, "n_calls_est": calls,
                    "method": "full_batched", "recoverable": True}
        total = calls = 0
        for c in cells.itertuples():
            d = str(c.doc_name)
            ptok, n = item_prompt_tokens(str(c.query_id), dpt, udp)
            total += n * (system_tokens(d, c.type == "open_ended") + ptok + dtok[d])
            calls += n
        return {"prompt_tokens_est": total, "n_calls_est": calls,
                "method": "full_individual", "recoverable": True}

    if mode == "rag":
        cpt = char_per_token()
        total = calls = 0
        for c in cells.itertuples():
            d = str(c.doc_name)
            ptok, n = item_prompt_tokens(str(c.query_id), dpt, udp)
            ctx = (c.ctx_chars / cpt) if not c.ctx_placeholder else doc_tokens(parser, d)
            total += n * (system_tokens(d, c.type == "open_ended") + ptok + ctx)
            calls += n
        return {"prompt_tokens_est": int(total), "n_calls_est": calls,
                "method": "rag_context", "recoverable": True}

    return {"prompt_tokens_est": np.nan, "n_calls_est": np.nan,
            "method": f"unknown_mode:{mode}", "recoverable": False}


def completion_floor(cells: pd.DataFrame) -> int:
    """Lower bound on output tokens: the visible answers only. Reasoning traces
    were never stored, so this understates cost for reasoning-on trials."""
    return int(cells.raw_output_chars.sum() / char_per_token())


def build(trials: pd.DataFrame | None = None,
          cells: pd.DataFrame | None = None) -> pd.DataFrame:
    trials = pd.read_parquet(TRIALS_PQ) if trials is None else trials
    cells = pd.read_parquet(CELLS_PQ) if cells is None else cells
    tmap = trials.set_index("number")
    rows = []
    for number, g in cells.groupby("trial_number"):
        if number not in tmap.index:
            continue
        t = tmap.loc[number]
        est = estimate_trial(t, g)
        model = t.get("llm_model")
        pr = pricing().get(model, {}).get("pricing", {})
        cin, cout = pr.get("input", np.nan), pr.get("output", np.nan)
        out_lo = completion_floor(g)
        rec = {
            "trial_number": int(number),
            "llm_model": model,
            "mode": t.get("mode"),
            "question_mode": t.get("question_mode"),
            "parser": t.get("parser"),
            "effective_reasoning": t.get("effective_reasoning"),
            "n_docs": g.doc_name.nunique(),
            "n_cells": len(g),
            "accuracy": g.accuracy_fixed.mean(),
            **est,
            "completion_tokens_lo": out_lo,
            "prompt_tokens_true": t.get("prompt_tokens"),
            "completion_tokens_true": t.get("completion_tokens"),
            "llm_calls_true": t.get("llm_calls"),
            "usage_source": "optuna_attr" if pd.notna(t.get("prompt_tokens")) else None,
        }
        pin = rec["prompt_tokens_est"]
        rec["cost_in_usd"] = pin * cin / 1e6 if pd.notna(pin) else np.nan
        rec["cost_out_lo_usd"] = out_lo * cout / 1e6 if pd.notna(cout) else np.nan
        rec["cost_lo_usd"] = rec["cost_in_usd"] + rec["cost_out_lo_usd"]
        rec["usd_per_paper"] = rec["cost_lo_usd"] / rec["n_docs"] if rec["n_docs"] else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)

    # The Batch-API files carry real per-call usage for the trials whose ingest
    # discarded it — this triples the ground truth available for validation, and
    # in particular lifts full/individual off n=2.
    rec_usage = CACHE / "recovered_usage.parquet"
    if rec_usage.exists():
        ru = pd.read_parquet(rec_usage).rename(columns={
            "prompt_tokens": "rec_prompt", "completion_tokens": "rec_completion",
            "calls": "rec_calls"})
        df = df.merge(ru, on="trial_number", how="left")
        fill = df.prompt_tokens_true.isna() & df.rec_prompt.notna()
        df.loc[fill, "prompt_tokens_true"] = df.loc[fill, "rec_prompt"]
        df.loc[fill, "completion_tokens_true"] = df.loc[fill, "rec_completion"]
        df.loc[fill, "llm_calls_true"] = df.loc[fill, "rec_calls"]
        df.loc[fill, "usage_source"] = "batch_recovered"
        df = df.drop(columns=["rec_prompt", "rec_completion", "rec_calls"])

    df.to_parquet(CACHE / "tokens.parquet", index=False)
    return df


# ------------------------------------------------------------------ validation
def validate(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df.prompt_tokens_true.notna() & df.prompt_tokens_est.notna()].copy()
    g["ape"] = (g.prompt_tokens_est - g.prompt_tokens_true).abs() / g.prompt_tokens_true
    g["ratio"] = g.prompt_tokens_est / g.prompt_tokens_true
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    df = build()
    cal = calibration(df)
    df = apply_calibration(df, cal)
    df.to_parquet(CACHE / "tokens.parquet", index=False)
    import json as _json
    _json.dump(cal, open(TABLES / "T2_calibration.json", "w"), indent=2)
    df.to_csv(TABLES / "T2_tokens_cost.csv", index=False)
    ok = df[df.recoverable]
    print(f"tokens.parquet: {len(df)} trials  ({len(ok)} recoverable, "
          f"{len(df) - len(ok)} not)")
    print(f"  tokenizer: {encoding_name()}  |  chars-per-token: {char_per_token():.3f}")
    print("\n  not recoverable by reason:")
    for m, n in df[~df.recoverable].method.value_counts().items():
        print(f"     {m:20s} {n}")

    v = validate(df)
    print(f"\n--- validation against {len(v)} trials with true token attrs ---")
    print(f"{'method':18s} {'n':>3} {'MAPE':>8} {'med ratio':>10}")
    for meth, gg in v.groupby("method"):
        print(f"{meth:18s} {len(gg):>3} {gg.ape.mean()*100:>7.1f}% {gg.ratio.median():>10.3f}")
    print(f"{'ALL':18s} {len(v):>3} {v.ape.mean()*100:>7.1f}% {v.ratio.median():>10.3f}")
    if a.validate:
        worst = v.sort_values("ape", ascending=False).head(6)
        print("\n  worst trials:")
        for _, r in worst.iterrows():
            print(f"     #{int(r.trial_number):>3} {r.method:16s} "
                  f"est={r.prompt_tokens_est:>12,.0f} true={r.prompt_tokens_true:>12,.0f} "
                  f"ape={r.ape*100:5.1f}%")
    print(f"\nwrote {CACHE/'tokens.parquet'}\n      {TABLES/'T2_tokens_cost.csv'}")




# ------------------------------------------------------------------ calibration
def calibration(df: pd.DataFrame) -> dict[str, dict]:
    """Per-mode multiplicative correction from the trials with true token attrs.

    full_batched validates at ~3% and rag at ~9%; full_individual has only n=2 and
    systematically UNDER-estimates, so the batched-vs-individual cost saving
    computed from raw estimates is a conservative lower bound.
    """
    v = validate(df)
    out = {}
    for meth, g in v.groupby("method"):
        out[str(meth)] = {
            "n": int(len(g)),
            "factor": float(1.0 / g.ratio.median()),
            "mape": float(g.ape.mean()),
            "trust": "high" if g.ape.mean() < 0.10 and len(g) >= 4 else "low",
        }
    return out


def apply_calibration(df: pd.DataFrame, cal: dict[str, dict]) -> pd.DataFrame:
    df = df.copy()
    f = df.method.map(lambda m: cal.get(str(m), {}).get("factor", 1.0))
    df["prompt_tokens_cal"] = df.prompt_tokens_est * f
    df["calibration_factor"] = f
    df["calibration_trust"] = df.method.map(
        lambda m: cal.get(str(m), {}).get("trust", "none"))
    pr = df.llm_model.map(lambda m: pricing().get(m, {}).get("pricing", {}).get("input", np.nan))
    df["cost_in_cal_usd"] = df.prompt_tokens_cal * pr / 1e6
    df["cost_cal_usd"] = df.cost_in_cal_usd + df.cost_out_lo_usd
    df["usd_per_paper_cal"] = df.cost_cal_usd / df.n_docs
    return df


if __name__ == "__main__":
    main()
