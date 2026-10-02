#!/usr/bin/env python
"""Integrity gate for the production Optuna study — run BEFORE any analysis.

Answers four questions, writes CSVs to analysis/tables/hpo/, modifies nothing:

  1. db_vs_disk        Does each trial's recorded Optuna value match the mean of
                       its own stored cells? (in-place mutations by
                       rerun_oe_judge.py / rerun_47_followup.py desync these)
  2. output_coverage   Which trials still have the model's raw_output stored?
                       Trials 382-398 stored NO outputs at all, so any re-score
                       from raw_output silently rewrites them as wrong.
  3. rescore_delta     What does the map_values_to_letters concat fix change,
                       restricted to trials where re-scoring is actually valid?
  4. ranking_stability Does the argmax move? Does the top-10 set survive?

Usage:  pipeline_env/bin/python analysis/check_integrity.py
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DB = ROOT / "optuna_study.db"
STUDY = "full_item_accuracy_tuning"
TRIAL_LOGS = ROOT / "trial_logs"
TABLES = ROOT / "analysis" / "tables" / "hpo"
CACHE = ROOT / "analysis" / "cache"
TABLES.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

# mc_rule.rater_iu prints debug lines to stdout on every call; silence it.
_NULL = io.StringIO()


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(_NULL):
        return fn(*a, **kw)


# ---------------------------------------------------------------- study access
def load_complete_trials() -> pd.DataFrame:
    """number, db_value, state, llm_model — read-only, no optuna dependency."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    sid = con.execute(
        "SELECT study_id FROM studies WHERE study_name=?", (STUDY,)
    ).fetchone()[0]
    vals = dict(
        con.execute(
            """SELECT t.number, v.value FROM trials t
               JOIN trial_values v ON v.trial_id = t.trial_id
               WHERE t.study_id=? AND v.objective=0""",
            (sid,),
        )
    )
    states = dict(
        con.execute("SELECT number, state FROM trials WHERE study_id=?", (sid,))
    )
    models: dict[int, str] = {}
    dist: dict[str, list] = {}
    for num, name, val, dj in con.execute(
        """SELECT t.number, p.param_name, p.param_value, p.distribution_json
           FROM trials t JOIN trial_params p ON p.trial_id=t.trial_id
           WHERE t.study_id=? AND p.param_name='llm_model'""",
        (sid,),
    ):
        dist.setdefault(name, json.loads(dj)["attributes"]["choices"])
        try:
            models[num] = dist[name][int(val)]
        except Exception:
            models[num] = str(val)
    con.close()
    rows = [
        {"number": n, "state": s, "db_value": vals.get(n), "llm_model": models.get(n)}
        for n, s in states.items()
    ]
    return pd.DataFrame(rows).sort_values("number").reset_index(drop=True)


def find_folder(number: int) -> Path | None:
    hits = sorted(TRIAL_LOGS.glob(f"*_trial_{number:04d}"))
    for f in hits:
        if (f / "trial_summary.json").exists():
            return f
    return hits[0] if hits else None


# ---------------------------------------------------------------- (1) + (2)
def db_vs_disk_and_coverage(trials: pd.DataFrame) -> pd.DataFrame:
    """One row per COMPLETE trial: disk mean, cell counts, output coverage."""
    import orjson

    out = []
    for _, t in trials[trials.state == "COMPLETE"].iterrows():
        n = int(t.number)
        folder = find_folder(n)
        rec = {
            "number": n,
            "llm_model": t.llm_model,
            "db_value": t.db_value,
            "folder": folder.name if folder else None,
            "n_cells": 0,
            "n_mc": 0,
            "mc_output_present": 0,
            "disk_value": np.nan,
            "delta_db_disk": np.nan,
            "rescoreable": False,
        }
        if folder and (folder / "trial_summary.json").exists():
            ts = orjson.loads((folder / "trial_summary.json").read_bytes())
            cells = ts.get("per_query_doc_results", [])
            mc = [c for c in cells if c.get("type") == "multiple_choice"]
            present = sum(
                1
                for c in mc
                if c.get("raw_output") is not None or c.get("llm_output") is not None
            )
            accs = [c.get("accuracy") for c in cells if c.get("accuracy") is not None]
            rec.update(
                n_cells=len(cells),
                n_mc=len(mc),
                mc_output_present=present,
                disk_value=float(np.mean(accs)) if accs else np.nan,
            )
            rec["delta_db_disk"] = (
                rec["disk_value"] - t.db_value
                if t.db_value is not None and not np.isnan(rec["disk_value"])
                else np.nan
            )
            # A cell is re-scoreable iff its own output survives; a trial is
            # re-scoreable iff ANY do. Trials with a total output blackout
            # (382-398) cannot be re-scored at all and are excluded.
            rec["rescoreable"] = len(mc) > 0 and present > 0
            rec["mc_output_missing"] = len(mc) - present
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- (3)
def rescore_delta(cov: pd.DataFrame):
    """Re-score MC cells through the fixed evaluator, ONLY where valid."""
    import orjson

    import _data_loading as D
    import mc_rule
    from evaluate import evaluate_answer

    Q = _quiet(D.load_queries)
    Q = {q: v for q, v in Q.items() if q in _quiet(D.finalized_item_map, Q)}
    TRUE = _quiet(D.load_true_answers_final, Q)
    _quiet(mc_rule._load_raw)

    _orig, _memo = mc_rule.rater_iu, {}

    def cached(doc, item, label, choices):
        k = (str(doc), str(item))
        if k not in _memo:
            _memo[k] = _orig(doc, item, label, choices)
        return _memo[k]

    mc_rule.rater_iu = cached

    per_trial, per_item = [], defaultdict(lambda: [0, 0])
    for _, r in cov[cov.rescoreable].iterrows():
        folder = TRIAL_LOGS / r.folder
        ts = orjson.loads((folder / "trial_summary.json").read_bytes())
        accs, n_up, n_dn, n_skip = [], 0, 0, 0
        with contextlib.redirect_stdout(_NULL):
            for c in ts.get("per_query_doc_results", []):
                a = c.get("accuracy")
                out = c.get("raw_output")
                if out is None:
                    out = c.get("llm_output")
                # never re-score a cell whose output was not stored: doing so
                # silently rewrites a real answer as wrong. Keep the stored score.
                if c.get("type") == "multiple_choice" and out is not None:
                    res = evaluate_answer(
                        out,
                        question_type="multiple_choice",
                        doc_name=str(c["doc_name"]),
                        query_id=str(c["query_id"]),
                        TRUE_ANSWERS=TRUE,
                        QUERIES=Q,
                        other_callback=None,
                        curated_mc_items=mc_rule.CURATED_MC,
                    )
                    if res["accuracy"] != a:
                        idx = 0 if res["accuracy"] > (a or 0) else 1
                        per_item[c["query_id"]][idx] += 1
                        n_up += idx == 0
                        n_dn += idx == 1
                    a = res["accuracy"]
                elif c.get("type") == "multiple_choice":
                    n_skip += 1
                accs.append(a)
        new = float(np.mean(accs)) if accs else np.nan
        per_trial.append(
            {
                "number": int(r.number),
                "llm_model": r.llm_model,
                "old": r.disk_value,
                "new": new,
                "delta": new - r.disk_value,
                "cells_up": n_up,
                "cells_down": n_dn,
                "cells_skipped_no_output": n_skip,
            }
        )
    mc_rule.rater_iu = _orig
    items = pd.DataFrame(
        [{"item": k, "up": v[0], "down": v[1]} for k, v in sorted(per_item.items())]
    )
    return pd.DataFrame(per_trial).sort_values("delta"), items


# ---------------------------------------------------------------- (4)
def ranking_stability(delta: pd.DataFrame, k: int = 10) -> dict:
    old = delta.sort_values("old", ascending=False)
    new = delta.sort_values("new", ascending=False)
    from scipy.stats import spearmanr

    rho = spearmanr(delta.old, delta.new).statistic
    o, n = list(old.number[:k]), list(new.number[:k])
    return {
        "argmax_old": int(old.number.iloc[0]),
        "argmax_old_value": float(old.old.iloc[0]),
        "argmax_new": int(new.number.iloc[0]),
        "argmax_new_value": float(new.new.iloc[0]),
        "argmax_moves": int(old.number.iloc[0]) != int(new.number.iloc[0]),
        "spearman_rho": float(rho),
        f"top{k}_overlap": len(set(o) & set(n)),
        f"top{k}_old": o,
        f"top{k}_new": n,
    }


# ---------------------------------------------------------------- main
def main() -> None:
    trials = load_complete_trials()
    n_complete = int((trials.state == "COMPLETE").sum())
    print(f"Study `{STUDY}`: {len(trials)} trials, {n_complete} COMPLETE\n")

    cov = db_vs_disk_and_coverage(trials)
    cov.to_csv(TABLES / "T0_1_db_vs_disk.csv", index=False)

    matched = cov[cov.n_cells > 0]
    mism = matched[matched.delta_db_disk.abs() > 1e-6]
    print("--- T0.1  DB value vs mean of stored cells ---")
    print(f"  COMPLETE trials with a summary on disk : {len(matched)}")
    print(f"  value mismatches (|Δ| > 1e-6)          : {len(mism)}")
    for _, r in mism.iterrows():
        print(f"     #{r.number:>3}  db={r.db_value:.7f}  disk={r.disk_value:.7f}  Δ={r.delta_db_disk:+.7f}")

    blackout = sorted(matched[~matched.rescoreable].number.astype(int))
    partial = sorted(matched[matched.rescoreable & (matched.mc_output_missing > 0)]
                     .number.astype(int))
    print("\n--- T0.2  model-output coverage ---")
    full_ok = int((matched.rescoreable & (matched.mc_output_missing == 0)).sum())
    print(f"  trials with every MC output stored : {full_ok}")
    print(f"  trials with a few cells missing    : {len(partial)}  {partial}")
    print(f"  trials with a TOTAL output blackout: {len(blackout)}  -> not rescoreable")
    if blackout:
        lo, hi = min(blackout), max(blackout)
        contig = blackout == list(range(lo, hi + 1))
        print(f"     numbers {lo}-{hi}{' (CONTIGUOUS block)' if contig else ''}")
        print("     re-scoring these from raw_output would rewrite every MC cell as wrong.")

    print("\n--- T0.3  concat-fix effect (rescoreable trials only) ---")
    delta, items = rescore_delta(cov)
    delta.to_csv(TABLES / "T0_3_rescore_delta.csv", index=False)
    items.to_csv(TABLES / "T0_3_item_flips.csv", index=False)
    print(f"  trials re-scored : {len(delta)}")
    print(f"  mean Δ           : {delta.delta.mean():+.4f}   "
          f"(min {delta.delta.min():+.4f}, max {delta.delta.max():+.4f})")
    print(f"  cells corrected 0->1 : {int(items.up.sum())}")
    print(f"  cells corrected 1->0 : {int(items.down.sum())}")

    stab = ranking_stability(delta)
    json.dump(stab, open(TABLES / "T0_4_ranking_stability.json", "w"), indent=2)
    print("\n--- T0.4  ranking stability ---")
    print(f"  argmax: #{stab['argmax_old']} ({stab['argmax_old_value']:.4f})"
          f"  ->  #{stab['argmax_new']} ({stab['argmax_new_value']:.4f})"
          f"   [{'MOVES' if stab['argmax_moves'] else 'stable'}]")
    print(f"  Spearman rho old vs new : {stab['spearman_rho']:.4f}")
    print(f"  top-10 set overlap      : {stab['top10_overlap']}/10")

    verdict = stab["argmax_moves"] or stab["top10_overlap"] < 8
    print("\n=== GATE VERDICT ===")
    print(f"  Apply the concat fix before the call: {'YES' if verdict else 'NO'}")
    print(f"  It MUST skip the {len(blackout)} blackout trials "
          f"({blackout[0]}-{blackout[-1]}) — they stored no outputs.")
    print(f"  Cells skipped for lack of an output: "
          f"{int(delta.cells_skipped_no_output.sum())}")
    print(f"  Consistent-comparison sample = {len(delta)} trials "
          f"(of {n_complete} COMPLETE).")
    print(f"\nWrote: {TABLES}/T0_*.csv|json")


if __name__ == "__main__":
    main()
