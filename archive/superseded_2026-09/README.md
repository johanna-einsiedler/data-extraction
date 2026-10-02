# Superseded code, retired 2026-09-03

Removed from the working tree when the production analysis layer
(`analysis/hpo_analysis/`) replaced the ad-hoc seed-phase and one-off scripts.
Kept here rather than deleted because most were never git-tracked, so deletion
would be permanent. `analysis/create_figures.py` WAS tracked and was deleted
outright — recover it with `git show HEAD:analysis/create_figures.py`.

| file | why retired |
|---|---|
| `rescore_concat_fix.py` | **Unsafe.** Re-scores MC cells from stored `raw_output`; with `--write` it would have scored trials 382-398 (which stored no outputs) as all-wrong and rewritten them. Replaced by `hpo_analysis/rescore.py`, which applies the same fix as a read-only sidecar and guards per cell. |
| `analysis/rerun_oe_judge.py`, `analysis/rerun_47_followup.py` | Mutate `trial_summary.json` in place without updating Optuna — the bug class that desynced trial 389 (disk .6850498 vs db .6837209). The sidecar pattern replaces them. |
| `analysis/apply_47_C_to_Y.py` | One-off codebook migration (item 4.7 option `C` -> `Y`), already applied; re-running would corrupt `query_creation/*.json`. |
| `analysis/rescore_seed.py`, `analysis/score_secondary.py` | Seed-phase re-scorers, superseded by the sidecar + `stats.py` secondary outcomes. |
| `analysis/seed_phase_analysis.py` | The 16-trial seed analysis. Its trial->folder matching rule was generalised into `hpo_analysis/loader.match_folders`; outputs survive as `analysis/seed_*.csv` and `seed_phase_report.md`. |
| `analysis/create_figures.py` | Dead schema: defaults point at `trial_logs_sample_size_acc` / `trial_cache_sample_size_acc`, neither of which exists, and its JSON keys share nothing with the current `trial_summary.json`. Reusable parts were ported into `hpo_analysis/figures.py`. |
| `src/objective_sample_size.py`, `src/objective_sample_size_accuracy.py` | Objectives for the retired sample-size study; their data directories no longer exist. |
| `scripts/regenerate_trial_summaries.py` | Replays the sample-size study from `trial_logs_sample_size_acc` / `trial_cache_sample_size_acc`; both directories are gone, and it imported `objective_sample_size_accuracy` above. |
