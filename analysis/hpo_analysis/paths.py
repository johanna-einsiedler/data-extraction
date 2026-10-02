"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "optuna_study.db"
DB_RO = f"file:{DB}?mode=ro"
STUDY = "full_item_accuracy_tuning"

TRIAL_LOGS = ROOT / "trial_logs"
PARSED = ROOT / "data" / "parsed"
TRUE_LABELS = ROOT / "data" / "true_labels"
SPLITS = ROOT / "data" / "dataset_splits"

ANALYSIS = ROOT / "analysis"
CACHE = ANALYSIS / "cache"
TABLES = ANALYSIS / "tables" / "hpo"
FIGURES = ANALYSIS / "figures" / "hpo"
for _d in (CACHE, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

RESCORE_PQ = CACHE / "rescored_cells.parquet"
TRIALS_PQ = CACHE / "trials.parquet"
CELLS_PQ = CACHE / "cells.parquet"

# Trials 382-398 stored NO model outputs (raw_output and llm_output both None on
# every cell). Any re-score from raw_output rewrites them as wrong, so they are
# excluded from the consistent-comparison sample. Established by check_integrity.py.
OUTPUT_BLACKOUT = tuple(range(382, 399))

TYPE_ORDER = ["single_choice", "multiple_choice", "numeric", "list", "open_ended"]
TYPE_LABEL = {
    "single_choice": "Single-choice",
    "multiple_choice": "Multiple-choice",
    "numeric": "Numeric",
    "list": "List",
    "open_ended": "Open-ended",
}
CURATED_MC_ITEMS = frozenset({"3.1", "4.1", "4.2", "4.3"})


def short_model(m: str | None) -> str:
    return m.split("/")[-1] if m else "?"
