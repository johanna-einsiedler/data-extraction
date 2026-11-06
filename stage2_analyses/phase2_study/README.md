# Phase-2 study

Drops the per-item (individual) inference tier, makes the prompt lever actually work
in batched, and re-seeds a fresh Optuna study from the Phase-1 trials that are still
valid under the new design.

`phase2.db` is the new study. `../../optuna_study.db` is read-only input and is
never written — `copy_trials.py` verifies this by mtime.

## Why

**Individual is dropped.** The triage (`../../archive/superseded_2026-10/stage2_analyses/triage/`) puts individual − RAG at
**+1.64 pp** (n=4 paired models, sd 1.01 pp, one-sided 95% upper bound +2.82 pp,
inside a 4 pp margin) for roughly 6× RAG's cost and 31× batched's.

**The prompt lever was never live in batched.** `build_batched_prompt()` synthesized
the prompt from each item's metadata and never read `default_prompt_type`. All 26
copied batched trials ran one byte-identical prompt while recording six different
values for the lever. They are re-recorded under `batched_prompt_type="batched_native"`
— what they actually ran — with the original inert value kept in
`user_attrs["phase1_inert_default_prompt_type"]` so the substitution is auditable.

## Search space changes

| lever | Phase 1 | Phase 2 |
|---|---|---|
| `question_mode` | `individual` / `batched` | **gone** — `mode=="full"` means batched |
| `default_prompt_type` | always, 6 levels | **rag only**, 6 levels |
| `use_default_prompt` | always | **rag only** |
| `batched_prompt_type` | — | **batched only**, `batched_native` + the 6 variants |

A separate param name rather than a 7th level on `default_prompt_type`: a shared
lever would let rag trials draw a batched-only value, rejectable only via
`TrialPruned` — and PRUNED counts toward `n_startup_trials`, so the waste would come
out of the warm-up budget.

**The 17 roster ids are frozen.** Changing the `llm_model` choice list would raise
`CategoricalDistribution does not support dynamic value space` against every copied
trial. Models that moved provider keep their id and set `api_model`; models with no
reachable endpoint (cogito) stay in the roster as `disabled`.

## Usage

```bash
python stage2_analyses/phase2_study/copy_trials.py --dry     # report only
python stage2_analyses/phase2_study/copy_trials.py           # idempotent
python stage2_analyses/phase2_study/warmup_configs.py --n 39 # writes JSON
python stage2_analyses/phase2_study/warmup_configs.py --n 39 --enqueue
OPTUNA_DB=sqlite:///$PWD/stage2_analyses/phase2_study/phase2.db \
  PAPER_CONCURRENCY=4 N_JOBS=1 \
  pipeline_env/bin/python src/objective.py 39 --models=none
```

`--models=none` suppresses the 16-model seed loop, which would otherwise add RAG
seed trials that duplicate the copied ones.

## State (2026-10-02)

All three searches are finished; nothing enqueues trials any more.

| DB | COMPLETE | FAIL | PRUNED | last completion |
|---|---|---|---|---|
| `phase2.db` | 119 | 31 | 0 | 2026-09-28 |
| `open_tpe.db` (open-weight seed of phase2, `seed_open_study.py`) | 104 | 8 | 3 | 2026-09-29 |
| `consumer_tpe.db` | 73 | 9 | 9 | 2026-09-30 |

The 61 copied Phase-1 trials and the 39 `warmup_configs.json` warm-ups are inside `phase2.db`.
Run-control scripts (`queue_gate.py`, `requeue_on_recovery.py`, ...), worker `logs/`, the
`phase2.db.bak_before_*` backups and the `smoke`/`sanity` DBs are in
`../../archive/superseded_2026-10/stage2_analyses/phase2_study/`. `launch.sh` recreates `logs/` if a study
is ever resumed. The three DBs are git-tracked: a `git status` change on any of them means something wrote to a finished study.

## Caveat to carry into the analysis

DeepSeek-V4-Pro, Kimi-K2.6, Kimi-K2.7-Code and nemotron moved from Together to
OpenRouter with a pinned provider. Copied Phase-1 trials for those ids were served by
a different stack (different quantisation, possibly different sampling), so copied and
new trials for the same `llm_model` are **not freely exchangeable**. Check before
pooling them.

**2026-11-07 trial_logs prune.** trial_logs/ now holds exactly the folders of the COMPLETE trials of the four studies (208 folders); sanity/smoke-study and failed-trial folders were deleted. One loss: the four union folders of phase2 #123 (gpt-4.1 + RCS, partial 31/35 papers) were removed by mistake; 486 of its cells (12 papers) are kept in `analysis/cache/trial123_recovered_cells.json`, and its scoring record lives in the trial's attributes (`check_study_coherence.py` reports this as a WARN and verifies #123 against those attributes).
