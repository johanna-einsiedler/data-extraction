"""Stamp completeness / provenance attributes on every COMPLETE trial of the three
Phase-2 study DBs. Never touches trial values; never touches the Phase-1 DB.

usage: pipeline_env/bin/python scripts/stamp_trial_attrs.py
         [--db all|phase2|open_tpe|consumer_tpe] [--write] [--expected scripts/study_expected.json]

Attributes: n_papers, n_cells, n_doc_files, missing_docs, partial_completion
(True iff n_papers < 35; an existing non-empty value is never overwritten),
source_folder, config_hash10, config_md5, judge (gpt-oss-120b for copies of the
Aug-4/5 Phase-1 batch until write_aug_batch_values re-scores them, else Phi-3),
judge_basis, attrs_stamped. Idempotent: prints a diff and writes only the keys
that differ. --write takes a backup of each DB first (backup_db) and verifies
PRAGMA integrity_check plus a re-read afterwards.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _study_record import (AUG_BATCH, DB_PATHS, JUDGE_GPTOSS, JUDGE_PHI3, PHASE2_DBS, ROOT, TODAY,  # noqa: E402
                           assert_quiescent, backup_db, db_md5, folder_stats, integrity_ok,
                           load_studies, matched_folder, resolve_source, tuning_set,
                           union_folder_stats, write_user_attrs)

JUDGE_BASIS = "verdict-cache attribution 2026-10-01 (analysis/hpo_analysis/judge_attribution.py)"


def desired_attrs(name, t, studies, exp, tuning):
    ua = t.user_attrs
    agg = exp["dbs"][name].get("aggregated", {}).get(str(t.number))
    if agg:
        fs = union_folder_stats(agg, tuning)
    else:
        origin, num, params = resolve_source(name, t, studies)
        f = matched_folder(num, params)
        if f is None:
            raise RuntimeError(f"{name}#{t.number}: no unique folder for source {origin}#{num}")
        fs = folder_stats(f, tuning)
    origin, num, _ = resolve_source(name, t, studies)
    is_aug = origin == "phase1" and num in AUG_BATCH
    judge = JUDGE_GPTOSS if (is_aug and "rescore_note" not in ua) else JUDGE_PHI3
    d = {
        "n_papers": fs["n_papers"], "n_cells": fs["n_cells"], "n_doc_files": fs["n_doc_files"],
        "missing_docs": fs["missing_docs"], "source_folder": fs["folder"],
        "judge": judge, "judge_basis": JUDGE_BASIS, "attrs_stamped": f"scripts/stamp_trial_attrs.py {TODAY}",
    }
    if fs["config_md5"]:
        d["config_hash10"] = fs["config_md5"][:10]; d["config_md5"] = fs["config_md5"]
    partial = fs["n_papers"] < exp["tuning_papers"]
    if partial and not ua.get("partial_completion"):
        d["partial_completion"] = True
    elif not partial and "partial_completion" in ua and ua["partial_completion"] is True:
        d["partial_completion"] = False
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="all"); ap.add_argument("--write", action="store_true")
    ap.add_argument("--expected", default=str(ROOT / "scripts/study_expected.json"))
    a = ap.parse_args()
    exp = json.load(open(a.expected))
    dbs = list(PHASE2_DBS) if a.db == "all" else [a.db]
    studies = load_studies(("phase1",) + tuple(dbs)); tuning = tuning_set()
    p1_md5 = db_md5(DB_PATHS["phase1"])
    plans = {}
    for name in dbs:
        changes = {}
        for t in studies[name].trials:
            if t.state.name != "COMPLETE" or t.value is None:
                continue
            want = desired_attrs(name, t, studies, exp, tuning)
            diff = {k: v for k, v in want.items() if t.user_attrs.get(k) != v}
            if diff:
                changes[t.number] = diff
        plans[name] = changes
        n_keys = sum(len(v) for v in changes.values())
        print(f"{name}: {len(changes)} trials, {n_keys} attribute writes pending")
        partial = sorted((n, d["missing_docs"]) for n, d in changes.items() if d.get("partial_completion"))
        if partial:
            print("  partial_completion to set:", partial)
    if not a.write:
        print("\ndry run; add --write to apply"); return
    assert_quiescent(studies)
    for name, changes in plans.items():
        if not changes:
            continue
        path = DB_PATHS[name]; backup_db(path, "stamp_attrs")
        con = sqlite3.connect(path, timeout=120)
        try:
            for number, attrs in changes.items():
                write_user_attrs(con, number, attrs)
            con.commit()
        finally:
            con.close()
        assert integrity_ok(path), f"{name}: integrity_check failed after write"
        st = load_studies((name,))[name]
        for number, attrs in changes.items():
            got = next(t for t in st.trials if t.number == number).user_attrs
            for k, v in attrs.items():
                assert got.get(k) == v, f"{name}#{number}.{k}: wrote {v!r} read {got.get(k)!r}"
        print(f"{name}: wrote {sum(len(v) for v in changes.values())} attrs on {len(changes)} trials; verified")
    assert db_md5(DB_PATHS["phase1"]) == p1_md5, "Phase-1 DB changed!"


if __name__ == "__main__":
    main()
