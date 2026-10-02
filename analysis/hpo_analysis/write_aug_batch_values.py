"""Write the unified-basis, Phi-3-judged values of the Aug-4/5 Phase-1 batch
(#382-#398) into their SEED COPIES in the three Phase-2 study DBs, and publish a
sidecar table for the Phase-1 originals. The Phase-1 DB is never a target.

usage: pipeline_env/bin/python -m analysis.hpo_analysis.write_aug_batch_values [--write] [--sidecar-only]

Inputs: analysis/cache/mode_by_type/aug_phi3/resc/<p1>.parquet (17 files; per-cell
accuracy_unified under prereg2-metrics-2026-09-28 with judge Phi-3), the Phase-1
study (read-only), the three Phase-2 studies, and scripts/study_expected.json
(the 16 copy targets). Dry run by default; --write backs up each DB BEFORE
opening it for writing, updates trial_values + provenance attrs in one
transaction, verifies integrity and re-reads. Idempotent.
"""
from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from _study_record import (AUG_BATCH, DB_PATHS, JUDGE_GPTOSS, JUDGE_PHI3, PHASE2_DBS, TODAY,  # noqa: E402
                           assert_quiescent, backup_db, db_md5, folder_stats, integrity_ok,
                           load_studies, matched_folder, trial_by_number, tuning_set,
                           update_value, write_user_attrs)
from analysis.hpo_analysis.unified_rescore import SPEC_ID  # noqa: E402

EXPECTED = ROOT / "scripts/study_expected.json"


def load_parquets(exp):
    d = ROOT / exp["aug_batch"]["parquet_dir"]
    new = {}
    for f in sorted(glob.glob(str(d / "*.parquet"))):
        df = pd.read_parquet(f); n = int(df.p1_number.iloc[0])
        assert n in AUG_BATCH and len(df) in (1505, 1500), (f, n, len(df))
        new[n] = (float(df.accuracy_unified.mean()), len(df), str(Path(f).relative_to(ROOT)))
    assert sorted(new) == exp["aug_batch"]["p1_trials"], sorted(new)
    return new


def write_sidecar(exp, new, studies):
    p1 = studies["phase1"]; tuning = tuning_set(); rows = []
    copies = {}
    for db, m in exp["aug_batch"]["copies"].items():
        for num, p1n in m.items():
            copies.setdefault(int(p1n), []).append(f"{db}#{num}")
    for n in sorted(new):
        t = trial_by_number(p1, n); f = matched_folder(n, dict(t.params))
        fm = folder_stats(f, tuning)["mean"] if f else None
        val, cells, pq = new[n]
        rows.append({"p1_trial": n, "n_cells": cells, "p1_db_value": t.value, "folder_mean": fm,
                     "phi3_unified": val, "delta_vs_db": val - t.value, "judge_before": JUDGE_GPTOSS,
                     "judge": JUDGE_PHI3, "spec": SPEC_ID, "copies": ";".join(copies.get(n, [])) or "—",
                     "parquet": pq})
    out = ROOT / exp["aug_batch"]["sidecar"]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"sidecar: {out.relative_to(ROOT)} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true"); ap.add_argument("--sidecar-only", action="store_true")
    a = ap.parse_args()
    exp = json.load(open(EXPECTED))
    studies = load_studies(("phase1",) + PHASE2_DBS)
    p1_md5 = db_md5(DB_PATHS["phase1"])
    new = load_parquets(exp)
    if a.sidecar_only:
        write_sidecar(exp, new, studies); return
    plan = []
    for db in PHASE2_DBS:
        targets = {t.number: int(t.user_attrs["phase1_trial"]) for t in studies[db].trials
                   if t.state.name == "COMPLETE" and "phase1_trial" in t.user_attrs
                   and int(t.user_attrs["phase1_trial"]) in range(382, 389)}
        expct = {int(k): v for k, v in exp["aug_batch"]["copies"][db].items()}
        assert targets == expct, f"{db}: targets {targets} != expected {expct}"
        for num, p1n in sorted(targets.items()):
            t = trial_by_number(studies[db], num); val, cells, pq = new[p1n]
            done = "rescore_note" in t.user_attrs and abs(t.value - val) < 1e-9
            plan.append((db, num, p1n, t.value, val, int(t.user_attrs.get("unified_rescore_cells", 0)), cells, pq, done))
    print(f"{'WRITE' if a.write else 'dry run'}: {sum(not p[-1] for p in plan)} pending of {len(plan)} targets")
    for db, num, p1n, old, val, oc, cells, pq, done in plan:
        print(f"  {db:12} #{num:<4} <- p1#{p1n}  {old:.6f} -> {val:.6f}  cells {oc}->{cells}  {'(already written)' if done else ''}")
    if not a.write:
        return
    assert_quiescent(studies)
    for db in PHASE2_DBS:
        rows = [p for p in plan if p[0] == db and not p[-1]]
        if not rows:
            continue
        path = DB_PATHS[db]; backup_db(path, "aug_phi3")
        con = sqlite3.connect(path, timeout=120)
        try:
            for _, num, p1n, old, val, oc, cells, pq, _d in rows:
                t = trial_by_number(studies[db], num)
                vb = t.user_attrs.get("value_before_unified_rescore", old)
                p1v = trial_by_number(studies["phase1"], p1n).value
                assert abs(float(vb) - p1v) < 1e-9, f"{db}#{num}: value_before {vb} != P1 {p1v}"
                update_value(con, num, val)
                write_user_attrs(con, num, {
                    "value_before_unified_rescore": float(vb), "unified_rescore": SPEC_ID,
                    "unified_rescore_cells": cells, "judge": JUDGE_PHI3, "judge_before": JUDGE_GPTOSS,
                    "rescore_date": TODAY, "rescore_cells_parquet": pq,
                    "rescore_note": (f"Aug-4/5 batch (Phase-1 #{p1n}): value recomputed on {TODAY} from recovered "
                                     f"batch-API outputs under {SPEC_ID} with judge {JUDGE_PHI3}; prior value "
                                     f"{old:.6f} ({JUDGE_GPTOSS}-judged, pre-semicolon-fix); Phase-1 study record "
                                     f"unchanged — see analysis/tables/hpo/aug_batch_phi3_rescore.csv")})
            con.commit()
        finally:
            con.close()
        assert integrity_ok(path), f"{db}: integrity_check failed"
        st = load_studies((db,))[db]
        for _, num, p1n, old, val, *_r in rows:
            t = trial_by_number(st, num)
            assert abs(t.value - val) < 1e-9 and t.user_attrs.get("judge") == JUDGE_PHI3, f"{db}#{num} re-read mismatch"
        print(f"{db}: {len(rows)} values written and verified")
    write_sidecar(exp, new, studies)
    assert db_md5(DB_PATHS["phase1"]) == p1_md5, "Phase-1 DB changed!"


if __name__ == "__main__":
    main()
