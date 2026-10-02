"""Derived design variables — the treatments the raw params don't express.

Three matter, and getting them wrong changes the answer:

  prompt_used      resolved per (trial, ITEM), not per trial: `use_default_prompt`
                   is a binary-decomposition switch that fires only for MC/SC items
                   with >2 choices and a binary_prompts entry (25 of 43 items), and
                   `true_few_shot_prompt` silently falls back to synthetic for the
                   7 items that have no true few-shot example.
  binary_eligible  that 25-item set — fixed, pre-determined, item-level. It is what
                   makes the binary switch identified WITHIN a trial (Spec B).
  parser_effective NA when vlm=True: VLM trials send page images and the sampled
                   parser is inert, so leaving it in contaminates parser effects.
"""
from __future__ import annotations

import contextlib
import io
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from .paths import CELLS_PQ, OUTPUT_BLACKOUT, ROOT, TRIALS_PQ

sys.path.insert(0, str(ROOT / "src"))
_NULL = io.StringIO()

TRIAL_LEVERS = ["llm_model", "mode", "question_mode", "parser_effective",
                "default_prompt_type", "use_default_prompt", "vlm",
                "effective_reasoning"]
RAG_LEVERS = ["chunker", "chunk_size", "chunk_overlap", "embedder", "retriever",
              "k", "query_embedding", "rerank", "top_m", "token_budget"]


@lru_cache(maxsize=1)
def _queries() -> dict:
    import _data_loading as D
    with contextlib.redirect_stdout(_NULL):
        Q = D.load_queries()
        keep = D.finalized_item_map(Q)
    return {k: v for k, v in Q.items() if k in keep}


@lru_cache(maxsize=1)
def binary_eligible_items() -> frozenset:
    """Items the binary switch can fire on — mirrors objective._select_prompt."""
    out = set()
    for qid, q in _queries().items():
        if (q.get("type") in {"multiple_choice", "single_choice"}
                and q.get("prompts", {}).get("binary_prompts")
                and len(q.get("choices", [])) > 2):
            out.add(qid)
    return frozenset(out)


@lru_cache(maxsize=1)
def tfs_missing_items() -> frozenset:
    """Items where true_few_shot silently degrades to synthetic_few_shot."""
    return frozenset(
        qid for qid, q in _queries().items()
        if not q.get("prompts", {}).get("true_few_shot_prompt")
    )


@lru_cache(maxsize=None)
def prompt_used(qid: str, default_prompt: str, use_default: bool) -> str:
    if qid in binary_eligible_items() and not use_default:
        return "binary_prompts"
    if default_prompt == "true_few_shot_prompt" and qid in tfs_missing_items():
        return "synthetic_few_shot_prompt"
    return default_prompt


def _as_bool(x, default=True):
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes")
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    return bool(x)


def analysis_trials(cells: pd.DataFrame, trials: pd.DataFrame) -> set:
    """The 70 trials that are actually comparable: COMPLETE, all 35 papers x 43
    items, and with the model's outputs still on disk.

    Two separate exclusions, both fatal to a like-for-like comparison:
      * 9 trials finished only 19-34 papers, and paper difficulty dominates the
        trial mean (switching to a common paper set moves means by up to 0.038);
      * 17 trials (382-398) were ingested from the OpenAI Batch API, whose ingest
        path scores each answer and then discards it, so they carry no outputs,
        no logprobs and no context and can never be re-scored.
    """
    comp = set(trials.loc[trials.state == "COMPLETE", "number"])
    g = (cells[cells.trial_number.isin(comp)]
         .groupby("trial_number")
         .agg(n_cells=("accuracy", "size"),
              n_docs=("doc_name", "nunique"),
              out=("has_output", "mean")))
    ok = g[(g.n_docs == 35) & (g.n_cells == 1505) & (g.out > 0.99)]
    return set(ok.index)


def prepare(drop_blackout: bool = True,
            complete_only: bool = True,
            require_full_grid: bool = True,
            outcome: str = "accuracy_fixed") -> pd.DataFrame:
    """Cell frame + trial design, ready for estimation.

    Defaults to the 70-trial comparable set; pass require_full_grid=False only for
    inventory/diagnostic views that need to show what was excluded.
    """
    cells = pd.read_parquet(CELLS_PQ)
    trials = pd.read_parquet(TRIALS_PQ)

    if require_full_grid:
        cells = cells[cells.trial_number.isin(analysis_trials(cells, trials))]
    else:
        if complete_only:
            keep = set(trials.loc[trials.state == "COMPLETE", "number"])
            cells = cells[cells.trial_number.isin(keep)]
        if drop_blackout:
            cells = cells[~cells.trial_number.isin(OUTPUT_BLACKOUT)]

    cols = ["number", "llm_model", "mode", "question_mode", "parser",
            "default_prompt_type", "use_default_prompt", "vlm",
            "effective_reasoning", "disable_reasoning"] + RAG_LEVERS
    cols = [c for c in cols if c in trials.columns]
    df = cells.merge(trials[cols].rename(columns={"number": "trial_number"}),
                     on="trial_number", how="left")

    df["y"] = df[outcome].astype(float)
    df["doc_name"] = df.doc_name.astype(str)
    df["query_id"] = df.query_id.astype(str)

    df["is_vlm"] = df.vlm.map(lambda v: _as_bool(v, False))
    # parser is inert under VLM: the model sees page images, not parsed text
    df["parser_effective"] = np.where(df.is_vlm, "NA_vlm", df.parser.astype(str))

    df["udp"] = df.use_default_prompt.map(lambda v: _as_bool(v, True))
    df["binary_eligible"] = df.query_id.isin(binary_eligible_items())
    df["tfs_missing"] = df.query_id.isin(tfs_missing_items())
    df["binary_fired"] = df.binary_eligible & ~df.udp
    df["prompt_used"] = [
        prompt_used(q, str(d) if isinstance(d, str) else "base_prompt", u)
        for q, d, u in zip(df.query_id, df.default_prompt_type, df.udp)
    ]
    df["use_default_prompt"] = df.udp
    df["n_docs_trial"] = df.groupby("trial_number").doc_name.transform("nunique")
    return df
