"""Tidy frames for the production study. Raw artifacts are read-only.

    load_trials()   one row per Optuna trial (all 509, every state)
    match_folders() trial -> trial_logs folder, by (number, model, n_docs, value)
    build_cells()   one row per (trial, paper, item)  ->  cells.parquet

Standing rule: a trial's value is recomputed from its own cells; the DB value is
carried only as provenance. That makes the analysis immune to the in-place
mutation that desynced trial 389.

Build:  pipeline_env/bin/python -m analysis.hpo_analysis.loader
"""
from __future__ import annotations

import gc
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import jsonio
from .paths import (CELLS_PQ, DB_RO, OUTPUT_BLACKOUT, RESCORE_PQ, STUDY,
                    TRIAL_LOGS, TRIALS_PQ)

# Enqueue scripts that drew configs uniformly at random (outcome-independent).
_RANDOM_ENQUEUE = ("queue_random_trials", "generate_warmup_configs", "warmup")


# ------------------------------------------------------------------ trials
def load_trials() -> pd.DataFrame:
    """All trials, read-only, straight from sqlite.

    Deliberately not via optuna.load_study: FAILed trials keep their
    `fixed_params` system attr that way, which is how the informative failures
    are distinguished from the 274 information-free ones.
    """
    con = sqlite3.connect(DB_RO, uri=True)
    sid = con.execute(
        "SELECT study_id FROM studies WHERE study_name=?", (STUDY,)
    ).fetchone()[0]

    base = pd.read_sql_query(
        """SELECT trial_id, number, state, datetime_start, datetime_complete
           FROM trials WHERE study_id=? ORDER BY number""",
        con, params=(sid,),
    )
    vals = pd.read_sql_query(
        "SELECT trial_id, value FROM trial_values WHERE objective=0", con
    )
    prm = pd.read_sql_query(
        "SELECT trial_id, param_name, param_value, distribution_json FROM trial_params",
        con,
    )
    ua = pd.read_sql_query(
        "SELECT trial_id, key, value_json FROM trial_user_attributes", con
    )
    sa = pd.read_sql_query(
        "SELECT trial_id, key, value_json FROM trial_system_attributes", con
    )
    con.close()

    df = base.merge(vals, on="trial_id", how="left").rename(columns={"value": "db_value"})

    # decode categorical params: param_value is an index into distribution choices
    prm = prm[prm.trial_id.isin(df.trial_id)]
    choices: dict[str, list] = {}
    for name, dj in prm[["param_name", "distribution_json"]].drop_duplicates().values:
        try:
            choices.setdefault(name, json.loads(dj)["attributes"]["choices"])
        except Exception:
            pass

    def _decode(row):
        ch = choices.get(row.param_name)
        if ch is None:
            return row.param_value
        try:
            return ch[int(row.param_value)]
        except Exception:
            return row.param_value

    if len(prm):
        prm = prm.assign(val=prm.apply(_decode, axis=1))
        wide = prm.pivot_table(
            index="trial_id", columns="param_name", values="val", aggfunc="first"
        )
        df = df.merge(wide, on="trial_id", how="left")

    # user attrs of interest
    for key, col in [("effective_reasoning", "effective_reasoning"),
                     ("prompt_tokens", "prompt_tokens"),
                     ("completion_tokens", "completion_tokens"),
                     ("llm_calls", "llm_calls"),
                     ("skip_reason", "skip_reason"),
                     ("source", "source")]:
        sub = ua[ua.key == key][["trial_id", "value_json"]]
        if len(sub):
            sub = sub.assign(**{col: sub.value_json.map(_loads)})[["trial_id", col]]
            df = df.merge(sub, on="trial_id", how="left")
        else:
            df[col] = np.nan

    # fixed_params marks an *enqueued* trial and preserves the config of FAILures
    fp = sa[sa.key == "fixed_params"][["trial_id", "value_json"]]
    fp = fp.assign(fixed_params=fp.value_json.map(_loads))[["trial_id", "fixed_params"]]
    df = df.merge(fp, on="trial_id", how="left")

    df["has_params"] = df.trial_id.isin(prm.trial_id) if len(prm) else False
    df["enqueued"] = df.fixed_params.notna()
    # a trial carries usable design information if it has params OR a queued config
    df["informative"] = df.has_params | df.enqueued
    df["assignment"] = np.where(df.enqueued, "enqueued", "sampler")
    df["output_blackout"] = df.number.isin(OUTPUT_BLACKOUT)

    for c in ("datetime_start", "datetime_complete"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["duration_s"] = (df.datetime_complete - df.datetime_start).dt.total_seconds()
    return df.drop(columns=["trial_id"]).sort_values("number").reset_index(drop=True)


def _loads(s):
    try:
        return json.loads(s)
    except Exception:
        return s


# ------------------------------------------------------------------ folders
def match_folders(trials: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """Map each trial to its trial_logs folder.

    Generalises load_baselines from the retired seed-phase script
    (archive/superseded_2026-09/seed_phase_analysis.py). Trial *numbers*
    are reused across the archived v1 study, so a candidate must also agree on
    llm_model and on the recorded value; n_docs is scored, not required, so that
    partial trials still match.
    """
    rows = []
    for t in trials.itertuples():
        best = None
        for folder in sorted(TRIAL_LOGS.glob(f"*_trial_{t.number:04d}")):
            summ = folder / "trial_summary.json"
            if not summ.exists():
                rows.append({"number": t.number, "folder": folder.name,
                             "n_cells": 0, "n_docs": 0, "disk_value": np.nan,
                             "value_match": False, "model_match": False})
                best = best or rows[-1]
                continue
            try:
                ts = jsonio.load(summ)
            except Exception:
                continue
            cells = ts.get("per_query_doc_results", [])
            model = (ts.get("parameters") or {}).get("llm_model")
            disk = ts.get("overall_average_accuracy", np.nan)
            model_ok = (getattr(t, "llm_model", None) is None) or (model == t.llm_model)
            val_ok = (t.db_value is not None
                      and disk is not None
                      and not (isinstance(disk, float) and math.isnan(disk))
                      and abs(float(disk) - float(t.db_value)) <= tol)
            cand = {
                "number": t.number, "folder": folder.name, "n_cells": len(cells),
                "n_docs": len(ts.get("document_aggregates", {}) or {}),
                "disk_value": float(disk) if disk is not None else np.nan,
                "value_match": bool(val_ok), "model_match": bool(model_ok),
            }
            score = (model_ok, val_ok, len(cells))
            if best is None or score > (best["model_match"], best["value_match"], best["n_cells"]):
                best = cand
        if best is not None:
            rows.append(best)
    out = pd.DataFrame(rows).drop_duplicates(subset="number", keep="first")
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ cells
_SCALARS = ("doc_id", "query_id", "label", "type", "accuracy",
            "accuracy_secondary", "accuracy_exact", "accuracy_jaccard",
            "accuracy_per_level", "judge_equivalent")


def _answer_prob(lp) -> tuple[int, float, float, float]:
    """(n_tokens, sum_logprob, seq_prob, length-normalised prob)."""
    if not isinstance(lp, dict) or not lp:
        return 0, np.nan, np.nan, np.nan
    vals = [v for v in lp.values() if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return 0, np.nan, np.nan, np.nan
    s = float(np.sum(vals))
    n = len(vals)
    return n, s, float(np.exp(np.clip(s, -700, 0))), float(np.exp(np.clip(s / n, -700, 0)))


def build_cells(trials: pd.DataFrame | None = None,
                folders: pd.DataFrame | None = None,
                force: bool = False) -> pd.DataFrame:
    if CELLS_PQ.exists() and not force:
        return pd.read_parquet(CELLS_PQ)

    trials = load_trials() if trials is None else trials
    folders = match_folders(trials) if folders is None else folders
    fmap = dict(zip(folders.number, folders.folder))

    rows = []
    for i, (number, fname) in enumerate(sorted(fmap.items())):
        summ = TRIAL_LOGS / fname / "trial_summary.json"
        if not summ.exists():
            continue
        ts = jsonio.load(summ)
        cells = ts.get("per_query_doc_results", [])
        id2name = {}
        if any("doc_name" not in c for c in cells):
            from .rescore import doc_id_to_name
            id2name = doc_id_to_name(TRIAL_LOGS / fname)
        for c in cells:
            doc = c.get("doc_name") or id2name.get(c.get("doc_id"))
            n_lp, lp_sum, prob, prob_n = _answer_prob(c.get("logprobs"))
            ctx = c.get("context")
            raw = c.get("raw_output")
            llm = c.get("llm_output")
            rows.append({
                "trial_number": number,
                "doc_name": str(doc) if doc is not None else None,
                **{k: c.get(k) for k in _SCALARS},
                "has_output": (raw is not None) or (llm is not None),
                "raw_output_chars": len(raw) if isinstance(raw, str) else 0,
                "llm_output_chars": len(llm) if isinstance(llm, str) else 0,
                "ctx_chars": (len(ctx) if isinstance(ctx, str)
                              else sum(len(str(x)) for x in ctx) if isinstance(ctx, list)
                              else 0),
                "ctx_n_chunks": len(ctx) if isinstance(ctx, list) else (0 if ctx is None else 1),
                "ctx_placeholder": ctx == "[full document]",
                "n_logprob_tokens": n_lp,
                "logprob_sum": lp_sum,
                "answer_prob": prob,
                "answer_prob_norm": prob_n,
            })
        del ts, cells
        if i % 20 == 0:
            gc.collect()

    df = pd.DataFrame(rows)
    df = _apply_recovered(df)
    df = _apply_rescore(df)

    df["trial_number"] = df.trial_number.astype("int32")
    for col in ("doc_name", "query_id", "label", "type"):
        df[col] = df[col].astype("category")
    df.to_parquet(CELLS_PQ, index=False)
    return df


def _apply_recovered(df: pd.DataFrame) -> pd.DataFrame:
    """Fill outputs/logprobs recovered from the raw Batch-API files.

    Only ever FILLS cells that have no output — a trial that genuinely ran keeps
    its own record, so an overlapping recovery can never overwrite real data.
    """
    from .recover import RECOVERED_PQ
    if not RECOVERED_PQ.exists():
        df["recovered"] = False
        return df
    r = pd.read_parquet(RECOVERED_PQ)
    r["doc_name"] = r.doc_name.astype(str)
    r["query_id"] = r.query_id.astype(str)
    r = r.rename(columns={"n_logprob_tokens": "r_nlp", "logprob_sum": "r_lps",
                          "answer_prob": "r_ap", "answer_prob_norm": "r_apn"})
    df = df.merge(r[["trial_number", "doc_name", "query_id", "raw_output",
                     "r_nlp", "r_lps", "r_ap", "r_apn"]],
                  on=["trial_number", "doc_name", "query_id"], how="left")
    fill = (~df.has_output) & df.raw_output.notna()
    df["recovered"] = fill
    df.loc[fill, "raw_output_chars"] = df.loc[fill, "raw_output"].str.len()
    df.loc[fill, "has_output"] = True
    for src, dst in (("r_nlp", "n_logprob_tokens"), ("r_lps", "logprob_sum"),
                     ("r_ap", "answer_prob"), ("r_apn", "answer_prob_norm")):
        df.loc[fill, dst] = df.loc[fill, src]
    return df.drop(columns=["raw_output", "r_nlp", "r_lps", "r_ap", "r_apn"])


def _apply_rescore(df: pd.DataFrame) -> pd.DataFrame:
    """Join the concat-fix sidecar; never mutate the source artifacts."""
    if not RESCORE_PQ.exists():
        df["accuracy_fixed"] = df["accuracy"]
        df["rescored"] = False
        return df
    sc = pd.read_parquet(RESCORE_PQ)
    sc = sc.assign(doc_name=sc.doc_name.astype(str), query_id=sc.query_id.astype(str))
    df = df.merge(
        sc[["trial_number", "doc_name", "query_id", "accuracy_fixed", "rescored"]],
        on=["trial_number", "doc_name", "query_id"], how="left",
    )
    # non-MC cells are untouched by the fix
    df["accuracy_fixed"] = df.accuracy_fixed.fillna(df.accuracy)
    df["rescored"] = df.rescored.astype("boolean").fillna(False).astype(bool)
    return df


def main() -> None:
    trials = load_trials()
    trials.to_parquet(TRIALS_PQ, index=False)
    print(f"trials.parquet : {len(trials)} trials  "
          f"({(trials.state == 'COMPLETE').sum()} COMPLETE, "
          f"{int(trials.informative.sum())} informative)")

    folders = match_folders(trials)
    print(f"folders matched: {len(folders)}  "
          f"(value_match {int(folders.value_match.sum())}, "
          f"full 1505-cell grids {int((folders.n_cells == 1505).sum())})")

    cells = build_cells(trials, folders, force=True)
    usable = cells[~cells.trial_number.isin(OUTPUT_BLACKOUT)]
    print(f"cells.parquet  : {len(cells):,} rows, {cells.trial_number.nunique()} trials, "
          f"{CELLS_PQ.stat().st_size/1e6:.1f} MB")
    print(f"  excl. blackout: {len(usable):,} rows, {usable.trial_number.nunique()} trials")
    print(f"  with logprobs : {int((cells.n_logprob_tokens > 0).sum()):,} cells")
    print(f"  rescored (MC) : {int(cells.rescored.sum()):,} cells, "
          f"{int((cells.accuracy_fixed != cells.accuracy).sum()):,} changed")
    print(f"\nwrote {TRIALS_PQ}\n      {CELLS_PQ}")


if __name__ == "__main__":
    main()
