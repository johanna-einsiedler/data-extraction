# LLM data-extraction pipeline for a systematic review of machine learning in psychology

Code, configuration and study record for the two-phase preregistered evaluation of an LLM-based
extraction pipeline (43 codebook items, 500 coded studies). This repository is the citable companion of the
Phase-2 preregistration: everything here is either an input to the pipeline runs, an output of the
hyperparameter searches, or an analysis whose numbers appear in the preregistration.

## What is here

| Path | Content |
|---|---|
| `src/` | The pipeline: parsers, chunkers, embedders, retrievers, answer generation (`answer_generation/`), the Optuna objective (`objective.py`), scoring (`evaluate.py`, `mc_rule.py`), gold loading (`_data_loading.py`), the batched prompt builder (`batched.py`), the per-call provenance ledger. The model roster is `src/answer_generation/filtered_models.json`. |
| `query_creation/` | The codebook (`coding_scheme.xlsx`), every item with its prompt variants (`queries_with_prompts.json`), the prompt templates and system prompts, and the scripts that build them. |
| `configs/` | `frozen_pipelines.json` (the three frozen configurations with their hashes) and the paper splits (`splits/v3_2026-10-30`: 35 tuning / 63 calibration / 150 test / 60 RQ2 subset; `v1_2026-07-03` is the Phase-1 split). |
| `optuna_study.db`, `stage2_analyses/phase2_study/` | The complete search record: the Phase-1 study and the three Phase-2 studies (unrestricted, open-weight, consumer reference) with every trial, parameter, value and attribute, plus the seeding scripts. |
| `scripts/` | Evaluation-stage tooling: `run_frozen.py` (fixed-configuration runs on a paper set), `make_splits.py` / `materialize_splits.py`, `access_audit_receipt.py`, `check_study_coherence.py` / `stamp_trial_attrs.py` / `_study_record.py` (study-record audit), `freeze_config_from_db.py`, `export_predictions_for_irr.py` and `compute_irr.py` (inter-rater analyses), `build_filtered_models.py` (roster), `export_true_labels.R` (gold export), `verify_call_ledger.py`. |
| `analysis/hpo_analysis/`, `analysis/tables/hpo/`, `analysis/cache/` | The analyses whose numbers are cited in the preregistration: extraction-mode comparison by item type, judge attribution and unified re-scoring, the August-batch re-score, temperature/determinism probe, prompt-wording confound, answer failure modes, cost reconstruction, human-agreement baseline; their tables and the re-scored cell caches. |
| `tests/` | Unit and integration tests (`pipeline_env/bin/python -m pytest tests -q`). |
| `scoring_scheme.md` | The scoring rules per item type, as implemented. |

## What is not here, and why

- **Papers.** The PDFs of the 500 studies are copyrighted; the id lists in `configs/splits/` identify them.
- **Gold standard.** The coded dataset (`data/true_labels/`) belongs to the review team and is available from them; the loader expects `09_data_final.csv` and the raw two-coder file there.
- **Trial outputs.** The per-paper outputs of every completed trial (`trial_logs/`, about 2.8 GB) and the answer/judge caches (`trial_cache/`) are deposited separately; the search databases reference them by folder name and configuration hash.
- **Secrets.** API keys live in an untracked `.env`.

## Running

Use the project environment (`pipeline_env/bin/python`, Python 3.11). A search needs `OPTUNA_DB` set explicitly;
a frozen run is `scripts/run_frozen.py --config {primary|open|consumer} --paper-set {calibration|test} --run-id <id>`
and writes a manifest, the per-paper outputs and a hash-chained call ledger under `runs/`. The equivalence judge for
open-ended items is `microsoft/Phi-3-mini-4k-instruct`, run locally.
