"""Unified re-scoring of the 61 Phase-1 trials copied into phase2.db — sidecar first,
DB write only with --write-db.

The copied trials carry Phase-1 values (exact-set MC, exact numeric, NLI open-ended,
pre-semicolon single-choice). Every natively-run Phase-2 trial is scored by the
current scheme (prereg2 metrics section: MC agreement band I<=P<=U for the 17
non-curated items, curated exact; numeric rounded to lesser reported precision;
list normalized set; open-ended Phi-3 judge v2-supersetrelaxed). This module pushes
each copied trial's stored raw outputs through the CURRENT evaluator so every value
in the study means the same thing.

Guards (same policy as rescore.py):
  * a cell is re-scored only if its output survives on disk; otherwise its stored
    accuracy is kept (never rewrite a real answer as wrong);
  * source folders are matched by PARAMS EQUALITY against the folder summary's own
    "parameters" dict — trial numbers collide across studies and the copy transform
    (question_mode dropped, batched_prompt_type introduced) changes the folder hash,
    so neither number nor hash alone is safe.

Validation: --self-test re-scores natively-run Phase-2 trials, whose stored values
were produced by this same evaluator; the recomputed mean must match the DB value.

    pipeline_env/bin/python -m analysis.hpo_analysis.unified_rescore --self-test
    pipeline_env/bin/python -m analysis.hpo_analysis.unified_rescore
    pipeline_env/bin/python -m analysis.hpo_analysis.unified_rescore --write-db
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from . import jsonio
from .paths import CACHE, ROOT, TRIAL_LOGS
from .rescore import _evaluator, doc_id_to_name

P1_DB = ROOT / "optuna_study.db"
P2_DB = ROOT / "stage2_analyses" / "phase2_study" / "phase2.db"
OUT_PQ = CACHE / "unified_rescore_cells.parquet"
OUT_CSV = CACHE / "unified_rescore_trials.csv"
SPEC_ID = "prereg2-metrics-2026-09-28"
_NULL = io.StringIO()


def _study(db: Path):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.load_study(study_name="full_item_accuracy_tuning",
                             storage=f"sqlite:///{db}")


def match_source_folder(number: int, params: dict) -> Path | None:
    """The Phase-1 folder for source trial `number`: every candidate with that
    number whose summary 'parameters' agree with the source params on every
    shared key. Exactly one match required."""
    hits = []
    for cand in sorted(TRIAL_LOGS.glob(f"*_trial_{number:04d}")):
        summ = cand / "trial_summary.json"
        if not summ.exists():
            continue
        try:
            sp = jsonio.load(summ).get("parameters") or {}
        except Exception:
            continue
        shared = [k for k in params if k in sp]
        # summaries omit some conditional params (e.g. rerank/top_m on #120),
        # so require equality on the echoed keys only — but on enough of them
        # that a number collision cannot slip through.
        if len(shared) >= 4 and all(sp[k] == params[k] for k in shared):
            hits.append(cand)
    return hits[0] if len(hits) == 1 else None


def rescore_folder(folder: Path, evaluate_answer, Q, TRUE, CUR) -> list[dict]:
    ts = jsonio.load(folder / "trial_summary.json")
    cells = ts.get("per_query_doc_results", [])
    id2name = (doc_id_to_name(folder)
               if any("doc_name" not in c for c in cells) else {})
    rows = []
    with contextlib.redirect_stdout(_NULL):
        for c in cells:
            qid = str(c.get("query_id"))
            if qid not in Q:
                continue
            doc = c.get("doc_name") or id2name.get(c.get("doc_id"))
            if doc is None:
                rows.append({"doc_name": "?", "query_id": qid,
                             "accuracy_stored": c.get("accuracy"),
                             "accuracy_unified": c.get("accuracy"),
                             "rescored": False})
                continue
            doc = str(doc)
            out = c.get("raw_output")
            if out is None:
                out = c.get("llm_output")
            stored = c.get("accuracy")
            if out is None:
                acc, rescored = stored, False
            else:
                r = evaluate_answer(
                    out,
                    question_type=Q[qid]["type"],
                    doc_name=doc,
                    query_id=qid,
                    TRUE_ANSWERS=TRUE,
                    QUERIES=Q,
                    other_callback=None,
                    curated_mc_items=CUR,
                )
                acc, rescored = r["accuracy"], True
            rows.append({"doc_name": doc, "query_id": qid,
                         "accuracy_stored": stored, "accuracy_unified": acc,
                         "rescored": rescored})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-db", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="unify every COMPLETE trial (copied AND native), not just copied")
    ap.add_argument("--self-test", action="store_true",
                    help="re-score natively-run Phase-2 trials instead; expect ~0 delta")
    args = ap.parse_args()

    evaluate_answer, Q, TRUE, CUR = _evaluator()
    p2 = _study(P2_DB)

    if args.self_test:
        # native = has params, no phase1_trial attr, COMPLETE, full folder
        targets = []
        for t in p2.trials:
            if t.state.name != "COMPLETE" or "phase1_trial" in t.user_attrs:
                continue
            if "partial_completion" in t.user_attrs:
                continue
            f = match_source_folder(t.number, dict(t.params))
            if f is not None:
                targets.append((t, f))
            if len(targets) >= 3:
                break
        for t, folder in targets:
            rows = rescore_folder(folder, evaluate_answer, Q, TRUE, CUR)
            m = pd.DataFrame(rows).accuracy_unified.mean()
            print(f"self-test #{t.number}: stored={t.value:.6f} rescored={m:.6f} "
                  f"delta={m - t.value:+.6f}")
        return

    p1_vals = {t.number: (dict(t.params), t.value) for t in _study(P1_DB).trials
               if t.value is not None}
    if args.all:
        targets = [t for t in p2.trials
                   if t.state.name == "COMPLETE"
                   and "partial_completion" not in t.user_attrs]
    else:
        targets = [t for t in p2.trials if "phase1_trial" in t.user_attrs]
    print(f"trials to unify: {len(targets)}")

    all_rows, trial_rows, unmatched = [], [], []
    for t in sorted(targets, key=lambda x: x.number):
        if "phase1_trial" in t.user_attrs:
            src = int(t.user_attrs["phase1_trial"])
            if src not in p1_vals:
                unmatched.append((t.number, src, "no source trial")); continue
            sparams, _ = p1_vals[src]
            folder = match_source_folder(src, sparams)
        else:
            src = t.number
            folder = match_source_folder(t.number, dict(t.params))
        if folder is None:
            unmatched.append((t.number, src, "no unique folder")); continue
        rows = rescore_folder(folder, evaluate_answer, Q, TRUE, CUR)
        df = pd.DataFrame(rows)
        new_val = float(df.accuracy_unified.mean())
        for r in rows:
            r.update({"p2_number": t.number, "p1_number": src})
        all_rows += rows
        trial_rows.append({
            "p2_number": t.number, "p1_number": src, "folder": folder.name,
            "old_value": t.value, "new_value": new_val,
            "delta": new_val - t.value, "n_cells": len(df),
            "n_rescored": int(df.rescored.sum()),
            "n_changed": int((df.accuracy_unified != df.accuracy_stored).sum()),
        })
        print(f"  p2#{t.number:3d} <- p1#{src:3d} {folder.name[:26]:26} "
              f"{t.value:.4f} -> {new_val:.4f} ({new_val - t.value:+.4f}) "
              f"[{int(df.rescored.sum())}/{len(df)} cells rescored]")

    cells = pd.DataFrame(all_rows)
    trials = pd.DataFrame(trial_rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(OUT_PQ, index=False)
    trials.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_PQ} and {OUT_CSV}")
    if unmatched:
        print("UNMATCHED (kept at old value):")
        for row in unmatched:
            print("  ", row)
    print(f"mean delta {trials.delta.mean():+.4f}, range "
          f"[{trials.delta.min():+.4f}, {trials.delta.max():+.4f}]")

    if args.all:
        # ---- trial #123 (partial: 31/35 papers, value imputed) on the unified basis
        import glob as _glob, os as _os, re as _re
        t123 = next((x for x in p2.trials if "partial_completion" in x.user_attrs), None)
        if t123 is not None:
            best = {}
            for d in _glob.glob(str(TRIAL_LOGS / "0b4c900e2c_trial_*")):
                for fdoc in _glob.glob(d + "/doc_*.json"):
                    b = _os.path.basename(fdoc)
                    if b not in best or _os.path.getmtime(fdoc) > _os.path.getmtime(best[b]):
                        best[b] = fdoc
            accs = []
            with contextlib.redirect_stdout(_NULL):
                for fdoc in best.values():
                    dj = jsonio.load(fdoc)
                    doc = str(dj.get("doc_name"))
                    for c in (dj.get("results") or dj.get("doc_results") or []):
                        qid = str(c.get("query_id"))
                        if qid not in Q:
                            continue
                        out = c.get("raw_output") or c.get("llm_output")
                        if out is None:
                            accs.append(c.get("accuracy")); continue
                        r = evaluate_answer(out, question_type=Q[qid]["type"],
                                            doc_name=doc, query_id=qid,
                                            TRUE_ANSWERS=TRUE, QUERIES=Q,
                                            other_callback=None, curated_mc_items=CUR)
                        accs.append(r["accuracy"])
            mean31 = sum(accs) / len(accs)
            # empirical 35-vs-31 gap on the SAME unified basis, from this run's cells
            miss = {"371", "385", "190", "460"}
            g = cells.groupby(["p2_number", "doc_name"], observed=True).accuracy_unified.mean().reset_index()
            gap_list = []
            for n, grp in g.groupby("p2_number"):
                if grp.doc_name.nunique() == 35:
                    m35 = grp.accuracy_unified.mean()
                    m31 = grp[~grp.doc_name.astype(str).isin(miss)].accuracy_unified.mean()
                    gap_list.append(m35 - m31)
            gap = sum(gap_list) / len(gap_list)
            new123 = mean31 + gap
            print(f"#123 unified: raw31={mean31:.4f} gap={gap:+.4f} -> {new123:.4f} "
                  f"(old {t123.value:.4f}); n_cells={len(accs)}")
            trial_rows.append({"p2_number": t123.number, "p1_number": -1,
                               "folder": "union(0b4c900e2c)", "old_value": t123.value,
                               "new_value": float(new123),
                               "delta": float(new123) - t123.value,
                               "n_cells": len(accs), "n_rescored": len(accs),
                               "n_changed": -1})
            trials = pd.DataFrame(trial_rows)
            trials.to_csv(OUT_CSV, index=False)

    if args.write_db and len(trials):
        bak = P2_DB.with_suffix(".db.bak_before_unified_rescore")
        if not bak.exists():
            shutil.copy2(P2_DB, bak)
            print(f"backup -> {bak.name}")
        con = sqlite3.connect(P2_DB, timeout=120)
        cur = con.cursor()
        for r in trials.itertuples():
            tid, = cur.execute("SELECT trial_id FROM trials WHERE number=?",
                               (int(r.p2_number),)).fetchone()
            cur.execute("UPDATE trial_values SET value=? WHERE trial_id=? AND objective=0",
                        (float(r.new_value), tid))
            for k, v in (("value_before_unified_rescore", r.old_value),
                         ("unified_rescore", SPEC_ID),
                         ("unified_rescore_cells", int(r.n_rescored))):
                cur.execute("DELETE FROM trial_user_attributes WHERE trial_id=? AND key=?",
                            (tid, k))
                cur.execute("INSERT INTO trial_user_attributes (trial_id, key, value_json)"
                            " VALUES (?,?,?)", (tid, k, json.dumps(v)))
        con.commit(); con.close()
        print(f"phase2.db updated: {len(trials)} trial values now follow {SPEC_ID}")


if __name__ == "__main__":
    main()
