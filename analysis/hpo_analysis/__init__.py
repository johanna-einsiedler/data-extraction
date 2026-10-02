"""Analysis layer for the production Optuna study `full_item_accuracy_tuning`.

Design rule: the raw artifacts (optuna_study.db, trial_logs/*) are READ-ONLY.
The MC concat-fix is applied as a *sidecar* (analysis/cache/rescored_cells.parquet)
rather than by rewriting trial_summary.json in place — in-place mutation is what
desynced trial 389 from the DB, and it destroys the ability to re-derive anything.
"""
from .paths import ROOT, DB_RO, TRIAL_LOGS, CACHE, TABLES, FIGURES  # noqa: F401
