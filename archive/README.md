# archive/

Material that is **not** part of the current analysis and **not** needed to run the pipeline.
Nothing here is imported by `src/`, `analysis/`, `scripts/` or `query_creation/` — verified before
each item was moved.

Reorganised 2026-09-04. Two exceptions to "not needed" are called out below.

| folder | what | why it is here |
|---|---|---|
| `batch_api/` | OpenAI Batch-API orchestration (`build_*`, `submit_*`, `ingest_*`, `prototype_batch_trial.py`), the 43 request/response `.jsonl` payloads (~2.6 GB) and their manifests | One-off execution route for 17 trials. **⚠ Still required to reproduce the output recovery** — `analysis/hpo_analysis/recover.py` resolves `batch*_out.jsonl` and the manifests from here. Do not delete. |
| `trial_orchestration/` | `queue_random_trials.py`, `generate_warmup_configs.py`, `enqueue_*.py`, `warmup_configs.json` | How trials were enqueued. Kept as **provenance for the claim that configuration assignment was random and outcome-independent**, not because anything runs them. |
| `seed_phase_outputs/` | `seed_*.csv`, `rescore_*.csv`, `summaries*.csv`, audit CSVs, `judge_candidate_eval.py`, `seed_phase_report_files/` | Outputs of the superseded 16-trial seed phase. The narrative survives in `seed_phase_report.md` (kept — the Phase-2 prereg cites its effect sizes). |
| `superseded_2026-09/` | The retired analysis layer — see its own README | Replaced by `analysis/hpo_analysis/`. Contains `rescore_concat_fix.py`, which is **unsafe to run**. |
| `superseded_2026-10/` | Phase-2 run control + logs, study backups, `triage/`, the three-tier memos, orphan figures/caches — see its own README | Retired 2026-10-02 after all searches finished. |
| `seed_phase/`, `pre_threading/`, `dev_logs/` | Archived Optuna studies, the pre-threading objective, run logs | Older studies must **not** be resumed: they ran at default temperature under pre-revision scoring and would corrupt the current study's surrogate. |
| `deploy/` | Remote-execution bundle (`make_bundle.sh`, `run_study.sh`, 65 MB tarball) | Infrastructure for running the study on a wired box; not part of any analysis. |
| `db_backups/` | `optuna_study.db.bak_*` snapshots | Point-in-time copies. The live DB is `optuna_study.db` in the repo root. |
| `one_off_scripts/` | `check_model_availability.py`, `run_holistic_test.py`, `warmup_parse.py`, synthetic-example generators | Development utilities, unreferenced. |
| `dead_code/` | root-level `utils_langextract/` (duplicate of `src/utils_langextract/`, plus the vendored Together provider) | LangExtract has **no live code path** — confirmed by grep for real imports. The Phase-2 prereg records it as the one Phase-1 prompt variant never evaluated. `src/utils_langextract/` was left in place so that statement stays literally true. |

## What stayed, and why

The working tree now holds only the current pipeline and its analysis:

- `src/`, `query_creation/`, `data/`, `tests/`, `pipeline_cli.py` — the pipeline.
- `analysis/hpo_analysis/`, `analysis/check_integrity.py`, `analysis/build_deck.py` — the current analysis.
- `analysis/cache/`, `analysis/tables/`, `analysis/figures/`, `analysis/deck/` — its outputs.
- `analysis/translation_list.csv` and `analysis/build_translation_list.py` — the gold→option alias
  map is **read at scoring time by `src/evaluate.py`**, so both the file and its generator stay.
- `optuna_study.db`, `trial_logs/`, `trial_cache/` — the source of truth for every reported number.
- `scoring_scheme.md`, `seed_phase_report.md`, `README.md` — cited by the Phase-2 preregistration.
- `scripts/` — `compute_irr.py`, `materialize_splits.py`, `build_filtered_models.py`,
  `test_logprobs.py`, `export_cells_csv.py`, `export_true_labels.R`.
