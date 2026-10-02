"""Shared helpers for the study-record tools (check_study_coherence.py,
stamp_trial_attrs.py, analysis/hpo_analysis/write_aug_batch_values.py).

All reads go through optuna.load_study on the sqlite files; the callers hash
every DB before and after so that a read-only run can prove it wrote nothing.
Writes (only from the two writer scripts, only with --write) use raw sqlite3 on
trial_values / trial_user_attributes, because Optuna >= 4 refuses attribute and
value updates on finished trials. The Phase-1 study (optuna_study.db) is never a
write target.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.hpo_analysis import jsonio  # noqa: E402
from analysis.hpo_analysis import unified_rescore as ur  # noqa: E402
from analysis.hpo_analysis.paths import TRIAL_LOGS  # noqa: E402
from analysis.hpo_analysis.rescore import doc_id_to_name  # noqa: E402

STUDY_NAME = "full_item_accuracy_tuning"
P2_DIR = ROOT / "stage2_analyses" / "phase2_study"
DB_PATHS: dict[str, Path] = {
    "phase1": ur.P1_DB,
    "phase2": ur.P2_DB,
    "open_tpe": P2_DIR / "open_tpe.db",
    "consumer_tpe": P2_DIR / "consumer_tpe.db",
}
PHASE2_DBS = ("phase2", "open_tpe", "consumer_tpe")
TODAY = _dt.date.today().isoformat()
JUDGE_PHI3 = "microsoft/Phi-3-mini-4k-instruct"
JUDGE_GPTOSS = "openai/gpt-oss-120b"
AUG_BATCH = tuple(range(382, 399))


def tuning_set() -> set[str]:
    """Numeric stems of the 35 tuning PDFs (the paper identifiers used as doc_name)."""
    out = set()
    for f in (ROOT / "data" / "tuning").glob("*.pdf"):
        m = re.match(r"^(\d+)", f.stem)
        if m:
            out.add(m.group(1))
    return out


def db_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_studies(names=("phase1",) + PHASE2_DBS) -> dict:
    return {n: ur._study(DB_PATHS[n]) for n in names}


def trial_by_number(study, number: int):
    for t in study.trials:
        if t.number == number:
            return t
    return None


def resolve_source(name: str, t, studies: dict) -> tuple[str, int, dict]:
    """Follow seed links to the trial whose outputs produced this value.
    Returns (origin_db, number, params) with origin_db in {phase1, phase2, <name>}."""
    ua = t.user_attrs
    if "phase1_trial" in ua:
        n = int(ua["phase1_trial"])
        src = trial_by_number(studies["phase1"], n)
        return "phase1", n, dict(src.params)
    if "seeded_from_trial" in ua and name != "phase2":
        n = int(ua["seeded_from_trial"])
        src = trial_by_number(studies["phase2"], n)
        return resolve_source("phase2", src, studies)
    return name, t.number, dict(t.params)


def matched_folder(number: int, params: dict) -> Path | None:
    return ur.match_source_folder(number, params)


def config_hash(params_from_summary: dict) -> str:
    return hashlib.md5(json.dumps(params_from_summary, sort_keys=True).encode()).hexdigest()


def folder_stats(folder: Path, tuning: set[str] | None = None) -> dict:
    ts = jsonio.load(folder / "trial_summary.json")
    cells = ts.get("per_query_doc_results", [])
    id2 = doc_id_to_name(folder) if any("doc_name" not in c for c in cells) else {}
    docs = set()
    accs = []
    for c in cells:
        d = c.get("doc_name")
        if d is None:
            d = id2.get(c.get("doc_id"))
        if d is not None:
            docs.add(str(d))
        if c.get("accuracy") is not None:
            accs.append(float(c["accuracy"]))
    tuning = tuning if tuning is not None else tuning_set()
    params = ts.get("parameters") or {}
    return {
        "folder": folder.name,
        "n_papers": len(docs),
        "n_cells": len(cells),
        "n_doc_files": len(list(folder.glob("doc_*.json"))),
        "docs": sorted(docs),
        "missing_docs": sorted(tuning - docs),
        "mean": (sum(accs) / len(accs)) if accs else None,
        "config_md5": config_hash(params) if params else None,
    }


def union_folder_stats(hash10: str, tuning: set[str] | None = None) -> dict:
    """Newest doc_*.json per document across every folder of one config hash
    (the #123 partial-completion aggregate). Stored accuracies only."""
    best: dict[str, Path] = {}
    for d in TRIAL_LOGS.glob(f"{hash10}_trial_*"):
        for f in d.glob("doc_*.json"):
            if f.name not in best or f.stat().st_mtime > best[f.name].stat().st_mtime:
                best[f.name] = f
    docs, accs, n_cells = set(), [], 0
    for f in best.values():
        dj = jsonio.load(f)
        docs.add(str(dj.get("doc_name")))
        for c in (dj.get("results") or dj.get("doc_results") or []):
            n_cells += 1
            if c.get("accuracy") is not None:
                accs.append(float(c["accuracy"]))
    tuning = tuning if tuning is not None else tuning_set()
    return {"folder": f"union({hash10})", "n_papers": len(docs), "n_cells": n_cells,
            "n_doc_files": len(best), "docs": sorted(docs),
            "missing_docs": sorted(tuning - docs),
            "mean": (sum(accs) / len(accs)) if accs else None, "config_md5": None}


def value_chain(t) -> list[float]:
    ua = t.user_attrs
    chain = [ua.get("value_before_semicolon_rescore"), ua.get("value_before_unified_rescore"), t.value]
    return [float(v) for v in chain if v is not None]


def assert_quiescent(studies: dict) -> None:
    for n in PHASE2_DBS:
        if n in studies:
            bad = [t.number for t in studies[n].trials if t.state.name in ("RUNNING", "WAITING")]
            if bad:
                raise SystemExit(f"{n}: trials in RUNNING/WAITING state: {bad}")
    r = subprocess.run(["pgrep", "-f", "src/objective.py"], capture_output=True, text=True)
    if r.stdout.strip():
        raise SystemExit(f"a study worker is running (pids {r.stdout.split()}); refusing to write")


def backup_db(path: Path, tag: str) -> Path:
    base = path.with_name(f"{path.name}.bak_before_{tag}_{TODAY}")
    cand, k = base, 1
    while cand.exists():
        k += 1
        cand = base.with_name(f"{base.name}_{k}")
    shutil.copy2(path, cand)
    print(f"  backup {path.name} -> {cand.name}  md5={db_md5(cand)}")
    return cand


def _trial_id(cur: sqlite3.Cursor, number: int) -> int:
    row = cur.execute("SELECT trial_id FROM trials WHERE number=?", (number,)).fetchone()
    if row is None:
        raise KeyError(f"trial number {number} not in DB")
    return int(row[0])


def write_user_attrs(con: sqlite3.Connection, number: int, attrs: dict[str, Any]) -> None:
    cur = con.cursor()
    tid = _trial_id(cur, number)
    for k, v in attrs.items():
        cur.execute("DELETE FROM trial_user_attributes WHERE trial_id=? AND key=?", (tid, k))
        cur.execute("INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (?,?,?)",
                    (tid, k, json.dumps(v)))


def update_value(con: sqlite3.Connection, number: int, value: float) -> None:
    cur = con.cursor()
    tid = _trial_id(cur, number)
    n = cur.execute("UPDATE trial_values SET value=? WHERE trial_id=? AND objective=0",
                    (float(value), tid)).rowcount
    if n != 1:
        raise RuntimeError(f"trial {number}: expected one value row, updated {n}")


def integrity_ok(path: Path) -> bool:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        con.close()
