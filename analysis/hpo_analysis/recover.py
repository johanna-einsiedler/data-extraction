"""Recover the model outputs the Batch-API ingest discarded.

Trials 382-398 were ingested by ingest_batch_results.py / ingest_individual_results.py,
which score each answer and then write only {query_id, type, doc_name, accuracy} —
dropping the answer, its logprobs and the token usage. All three survive in the raw
batch output files, so those trials are recoverable rather than lost.

  individual  custom_id "icfg0|006|1.8"  -> one answer per (config, paper, item)
  batched     custom_id "cfg0|006"       -> one JSON blob per (config, paper),
                                            parsed back through batched.parse_batched_response

Per-item logprobs exist only for the individual path; a batched response is a single
sequence, so its logprobs are recorded at paper level and left null per item.

Build:  pipeline_env/bin/python -m analysis.hpo_analysis.recover
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import CACHE, ROOT, TRIALS_PQ

sys.path.insert(0, str(ROOT / "src"))
_NULL = io.StringIO()

RECOVERED_PQ = CACHE / "recovered_cells.parquet"
RECOVERED_USAGE_PQ = CACHE / "recovered_usage.parquet"

# The raw Batch-API payloads were moved to archive/batch_api/ during the 2026-09-04
# cleanup. They are the only source for the 17 output-less trials, so resolve from
# either location and keep the recovery reproducible.
BATCH_DIRS = [ROOT.parent / "deposit" / "batch_api", ROOT / "archive" / "batch_api", ROOT]  # moved to the deposit folder 2026-11-07


def _find(name: str) -> Path | None:
    for d in BATCH_DIRS:
        if (d / name).exists():
            return d / name
    return None


MANIFESTS = {
    "individual": _find("batch_individual_manifest.json"),
    "batched": _find("batch_warmup_manifest.json"),
}
MATCH_KEYS = ["mode", "llm_model", "parser", "default_prompt_type",
              "use_default_prompt", "question_mode"]


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(_NULL):
        return fn(*a, **kw)


def load_manifests() -> dict[str, dict]:
    out = {}
    for kind, p in MANIFESTS.items():
        if p is not None and p.exists():
            for cid, cfg in json.loads(p.read_text()).items():
                out[cid] = {**cfg, "_kind": kind}
    return out


def _norm(v):
    if isinstance(v, bool):
        return str(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "None"
    return str(v)


def map_configs_to_trials(cfgs: dict[str, dict]) -> dict[str, int]:
    """Match each batch config to its Optuna trial by the sampled parameters."""
    tr = pd.read_parquet(TRIALS_PQ)
    tr = tr[tr.state == "COMPLETE"]
    index: dict[tuple, list[int]] = {}
    for t in tr.itertuples():
        key = tuple(_norm(getattr(t, k, None)) for k in MATCH_KEYS)
        index.setdefault(key, []).append(int(t.number))

    out, unmatched = {}, []
    for cid, cfg in cfgs.items():
        key = tuple(_norm(cfg.get(k)) for k in MATCH_KEYS)
        hits = index.get(key, [])
        if len(hits) == 1:
            out[cid] = hits[0]
        elif len(hits) > 1:
            # prefer a trial that is currently output-less (382-398)
            bl = [h for h in hits if 382 <= h <= 398]
            out[cid] = bl[0] if bl else hits[0]
        else:
            unmatched.append(cid)
    if unmatched:
        print(f"  [warn] {len(unmatched)} configs matched no trial: {unmatched}")
    return out


def _lp(choice) -> tuple[int, float]:
    lp = (choice or {}).get("logprobs") or {}
    content = lp.get("content") or []
    vals = [c.get("logprob") for c in content
            if isinstance(c.get("logprob"), (int, float))]
    return len(vals), float(np.sum(vals)) if vals else (0, np.nan)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    import batched as batched_mod
    import binary_mode
    import _data_loading as D
    from _llm_utils import _clean_llm_text

    Q = _quiet(D.load_queries)
    Q = {k: v for k, v in Q.items() if k in _quiet(D.finalized_item_map, Q)}

    # Items where the binary switch fires (MC/SC, >2 choices, binary_prompts present)
    binary_items = {qid for qid, q in Q.items()
                    if q.get("type") in {"multiple_choice", "single_choice"}
                    and q.get("prompts", {}).get("binary_prompts")
                    and len(q.get("choices", [])) > 2}

    cfgs = load_manifests()
    cmap = map_configs_to_trials(cfgs)
    print(f"  manifests: {len(cfgs)} configs -> {len(cmap)} matched to trials")

    rows, usage = [], []
    out_files = sorted(
        f for d in BATCH_DIRS for f in glob.glob(str(d / "batch*_out.jsonl"))
    )
    for f in out_files:
        for line in open(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            cid_full = d.get("custom_id", "")
            parts = cid_full.split("|")
            cfg = parts[0]
            trial = cmap.get(cfg)
            if trial is None or len(parts) < 2:
                continue
            body = (d.get("response") or {}).get("body") or {}
            ch = (body.get("choices") or [{}])[0]
            text = (ch.get("message") or {}).get("content")
            if text is None:
                continue
            u = body.get("usage") or {}
            # one usage record per API CALL: batched = one per paper, individual
            # = one per (paper, item). Deduplicating on the paper alone would
            # divide an individual trial's true usage by ~43.
            usage.append({"trial_number": trial, "doc_name": parts[1],
                          "call_key": parts[2] if len(parts) >= 3 else "__batched__",
                          "prompt_tokens": u.get("prompt_tokens", 0),
                          "completion_tokens": u.get("completion_tokens", 0),
                          "calls": 1})
            n_lp, lp_sum = _lp(ch)

            if len(parts) >= 3:                       # individual: one item
                qid = parts[2]
                # Reproduce the ingest's text pipeline exactly: strip reasoning
                # markers, and where binary decomposition fired the payload is a
                # {letter: bool} JSON that must be parsed back to a letter set
                # before it can be scored as a normal MC answer.
                clean = _clean_llm_text(str(text))
                use_default = bool(cfgs.get(cfg, {}).get("use_default_prompt", True))
                if qid in binary_items and not use_default and qid in Q:
                    try:
                        clean = _quiet(binary_mode.parse_binary_response, clean, Q[qid])
                    except Exception:
                        pass
                rows.append({"trial_number": trial, "doc_name": parts[1],
                             "query_id": qid, "raw_output": str(clean),
                             "n_logprob_tokens": n_lp, "logprob_sum": lp_sum,
                             "source": "batch_individual"})
            else:                                     # batched: one blob per paper
                try:
                    # returns (answers, others); missing items default to "Z"
                    parsed, _others = _quiet(
                        batched_mod.parse_batched_response, str(text), Q)
                except Exception:
                    continue
                for qid, ans in (parsed or {}).items():
                    if qid not in Q:
                        continue
                    rows.append({"trial_number": trial, "doc_name": parts[1],
                                 "query_id": str(qid), "raw_output": str(ans),
                                 # a batched reply is ONE sequence: per-item
                                 # logprobs are not meaningful and stay null
                                 "n_logprob_tokens": 0, "logprob_sum": np.nan,
                                 "source": "batch_batched"})

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["trial_number", "doc_name", "query_id"], keep="last")
    us = (pd.DataFrame(usage)
          .drop_duplicates(subset=["trial_number", "doc_name", "call_key"], keep="last")
          .groupby("trial_number")[["prompt_tokens", "completion_tokens", "calls"]]
          .sum().reset_index())
    if len(df):
        df["answer_prob"] = np.exp(np.clip(df.logprob_sum, -700, 0))
        df["answer_prob_norm"] = np.exp(
            np.clip(df.logprob_sum / df.n_logprob_tokens.replace(0, np.nan), -700, 0))
        df.to_parquet(RECOVERED_PQ, index=False)
        us.to_parquet(RECOVERED_USAGE_PQ, index=False)
    return df, us


def main() -> None:
    df, us = build()
    print(f"\nrecovered {len(df):,} cells across {df.trial_number.nunique()} trials")
    g = df.groupby("trial_number").agg(cells=("query_id", "size"),
                                       docs=("doc_name", "nunique"),
                                       lp=("n_logprob_tokens", lambda s: (s > 0).sum()),
                                       src=("source", "first"))
    print(f"\n{'trial':>6}{'cells':>7}{'docs':>6}{'w/ logprobs':>13}  source")
    for n, r in g.sort_index().iterrows():
        flag = "  <- full grid" if r.cells == 1505 and r.docs == 35 else ""
        print(f"{n:>6}{int(r.cells):>7}{int(r.docs):>6}{int(r.lp):>13}  {r.src}{flag}")
    full = g[(g.cells == 1505) & (g.docs == 35)]
    print(f"\nfull 35-paper grids recovered: {len(full)}")
    print(f"token usage recovered for {len(us)} trials")
    print(f"\nwrote {RECOVERED_PQ}\n      {RECOVERED_USAGE_PQ}")


if __name__ == "__main__":
    main()
