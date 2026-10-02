# Superseded material, retired 2026-10-02

Moved out of the working tree after all three Phase-2 Optuna searches finished
(phase2.db 2026-09-28, open_tpe.db 2026-09-29, consumer_tpe.db 2026-09-30).
The next stage runs frozen configurations on the calibration/test papers, so the
run-control scripts, their logs, the study backups and the three-tier-era memos
have no caller. Nothing here is imported by `src/`, `analysis/`, `scripts/`,
`tests/` or `query_creation/` (grep-verified before each move).

Layout mirrors the origin path, so **restore = `mv archive/superseded_2026-10/<path> <path>`**
(for the five rows marked *tracked*, use `git mv` back instead). `archive/` is git-ignored;
only this README is versioned.

| archived under `superseded_2026-10/` | origin | why superseded |
|---|---|---|
| `stage2_analyses/phase2_study/run_control/*.py, *.sh` (8) | `stage2_analyses/phase2_study/` | One-off supervision of the live searches (queue gating, memory watch, resume-after-cut, prompt-check enqueue, final-pair handoff, credit-recovery requeue). All searches are complete; nothing enqueues trials any more. |
| `stage2_analyses/phase2_study/logs/` (36 files, 72 MB) | `stage2_analyses/phase2_study/logs/` | Worker stdout of the finished runs. The DBs are the record; `launch.sh` recreates `logs/` on demand. |
| `stage2_analyses/phase2_study/db_backups/` (6 files) | `stage2_analyses/phase2_study/` | `phase2.db.bak_before_*`: point-in-time copies taken by `unified_rescore.py --write-db` and the semicolon/partial-complete fixes (all applied). `smoke.db`, `sanity.db(.bak)`: 2026-09-23/24 throwaway studies. |
| `stage2_analyses/triage/` (whole dir, 12 MB, mostly logs) | `stage2_analyses/triage/` | Three-tier triage (8 models x RAG/batched/individual). Its result is the evidence for dropping the individual tier; `triage.db` + `triage_cells.parquet` kept intact here. Cited by `stage2_analyses/phase2_study/README.md` (pointer updated) and a comment in `src/objective.py` (~line 1027). |
| `stage2_analyses/{README,cost_and_config_sensitivity,excluding_individual,rag_vs_fulldoc_test,tier_confounding_and_rag_headroom}.md`, `cost_by_tier_summary.csv`, `trial_summary_by_route.csv`, `plot_cost_tiers.py`, `figures/S1_cost_by_tier.png` | `stage2_analyses/` | 2026-09-04/05 three-tier-era memos and the five-model-probe proposal; superseded by the Phase-2 design (`phase2_study/README.md`) and the preregistered mode comparison (`analysis/hpo_analysis/mode_by_type.py`). |
| `analysis/figures_legacy/*.png` (17) | `analysis/figures/` | Outputs of the deleted `analysis/create_figures.py` / retired seed-phase scripts; nothing references them. Live figures are `analysis/figures/hpo/`; `rescore_ranking.png` stayed (embedded by `seed_phase_report.md`). |
| `analysis/cache_orphans/{cost_headline.json, stats_87.log, optuna_study.snapshot_20260902.db}` | `analysis/cache/` | Unreferenced by any script; the snapshot predates the unified rescore. |
| `analysis/aug_phi3_run_control/{chain.sh, chain387.sh, chain.log}` | `analysis/cache/mode_by_type/aug_phi3/` | Shell chains that ran `rescore_aug_batch_phi3.py` for trials 382-396 and 387 (absolute paths); `chain.log` records stored vs unified accuracy per trial. The parquet outputs stayed in `aug_phi3/resc/`. |
| `src/objective_threaded.py` | `src/` | Threading prototype, merged into `src/objective.py` (2026-07). |
| `src/answer_generation/get_model_info.py` *(tracked)*, `removed_models_nonserverless.json` | `src/answer_generation/` | Roster-building helpers from before `scripts/build_filtered_models.py`; the 17-id roster is frozen. |
| `query_creation/{queries_with_prompts,query_info}.json.bak` | `query_creation/` | Pre-2026-07-16 copies of the live JSON; the live files are tracked. |
| `tests/temp_page.png`, `tests/temp_page 22.26.39.png` *(tracked)* | `tests/` | Stray identical 8.9 KB page renders (Oct 2025), referenced by nothing. |
| `tests/test_rcs_rerank_results_*.json` (2) *(tracked)* | `tests/` | `tests/retrieval/test_retrieval.py` writes these into the cwd; the canonical copies live in `tests/retrieval/`. Pattern now git-ignored. |

Deleted outright (not archived): `query_creation/~$coding_scheme.xlsx`, `data/true_labels/~$codes_2025_10_14.xlsx`
(Excel lock stubs), `.pytest_cache/` at root, `tests/`, `src/`.
