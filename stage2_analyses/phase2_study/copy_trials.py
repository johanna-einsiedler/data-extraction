#!/usr/bin/env python
"""Seed the Phase-2 study with the Phase-1 RAG and batched trials.

Phase 2 drops the per-item (individual) tier and makes the prompt lever live in
batched, so the search space is not the Phase-1 one. Three things have to be
reconciled for a copied trial to mean what it says:

1. `question_mode` is gone. Every full-mode trial in Phase 2 is batched, so the
   param is not suggested and is dropped from the copied params.

2. `default_prompt_type` was INERT in batched. build_batched_prompt() synthesized
   the prompt from item metadata and never read the lever, so all 26 copied batched
   trials ran one byte-identical prompt while recording six different values for it.
   Copying that as-is would teach TPE a prompt->accuracy relationship that does not
   exist. Instead each is recorded under `batched_prompt_type="batched_native"`,
   which is exactly the prompt they ran, and the inert `default_prompt_type` /
   `use_default_prompt` are dropped.

3. The `llm_model` categorical must stay byte-identical to Phase 1, or Optuna
   raises "CategoricalDistribution does not support dynamic value space". This is
   why re-routed models keep their roster key and change only `api_model`, and why
   cogito stays in the roster as `disabled` rather than being deleted.

Read-only with respect to optuna_study.db. Idempotent: re-running skips trials
already copied (matched on the source trial number in user_attrs).

    python stage2_analyses/phase2_study/copy_trials.py --dry
    python stage2_analyses/phase2_study/copy_trials.py
"""
import argparse
import json
import pathlib
import sys

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState, create_trial

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

SRC_DB = f"sqlite:///{ROOT / 'optuna_study.db'}"
DST_DB = f"sqlite:///{ROOT / 'stage2_analyses' / 'phase2_study' / 'phase2.db'}"
STUDY = "full_item_accuracy_tuning"

PROMPT_VARIANTS = [
    "base_prompt",
    "cot_prompt",
    "reasoning_prompt",
    "rewritten_prompt",
    "synthetic_few_shot_prompt",
    "true_few_shot_prompt",
]
# Must match the suggest_categorical calls in src/objective.py exactly.
DISTS = {
    "mode": CategoricalDistribution(["rag", "full"]),
    "parser": CategoricalDistribution(
        ["GROBIDParser", "PyMuPDFParser", "PyMuPDFTesseractParser",
         "GeminiParser", "QwenParser", "MinerUParser"]),
    "chunk_size": CategoricalDistribution([50, 100, 250, 500, 750, 1000, 2500, 5000]),
    "chunk_overlap": CategoricalDistribution([0, 250, 750]),
    "chunker": CategoricalDistribution(["LengthChunker", "TextStructureChunker"]),
    "embedder": CategoricalDistribution(["BGEEmbedder", "E5Embedder", "OpenAIEmbedder"]),
    "retriever": CategoricalDistribution(
        ["TopKRetriever", "CrossEncoderRetriever", "TokenBudgetRetriever", "RCSRetriever"]),
    "k": CategoricalDistribution([2, 3, 5, 7, 10]),
    "token_budget": CategoricalDistribution([200, 500, 1000, 2000]),
    "rerank": CategoricalDistribution([True, False]),
    "top_m": CategoricalDistribution([5, 10, 20]),
    "query_embedding": CategoricalDistribution(["raw", "label_def"]),
    "default_prompt_type": CategoricalDistribution(PROMPT_VARIANTS),
    "use_default_prompt": CategoricalDistribution([True, False]),
    "batched_prompt_type": CategoricalDistribution(["batched_native"] + PROMPT_VARIANTS),
    "vlm": CategoricalDistribution([True, False]),
    "disable_reasoning": CategoricalDistribution([True, False]),
}
# Never copied: question_mode (dropped in Phase 2), and for batched trials the two
# inert prompt levers.
DROP_ALWAYS = {"question_mode"}
DROP_FOR_BATCHED = {"default_prompt_type", "use_default_prompt"}


def roster_ids():
    with open(ROOT / "src" / "answer_generation" / "filtered_models.json") as fh:
        return [m["id"] for m in json.load(fh)]


def analysis_grade():
    """Trial numbers passing the same bar as the 87-trial analysis set: COMPLETE,
    35 papers, 1505 cells, >99% of cells carrying a model output. Reuses
    hpo_analysis.design.analysis_trials so the Phase-2 seed and the Phase-1
    write-ups cannot drift apart on what counts as a usable trial."""
    import pandas as pd
    from hpo_analysis import design

    cache = ROOT / "analysis" / "cache"
    cells = pd.read_parquet(cache / "cells.parquet")
    trials = pd.read_parquet(cache / "trials.parquet")
    return design.analysis_trials(cells, trials)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    src = optuna.load_study(study_name=STUDY, storage=SRC_DB)
    keep = analysis_grade()
    ids = roster_ids()
    dists = dict(DISTS, llm_model=CategoricalDistribution(ids))

    picked, skipped = [], {"not_analysis_grade": 0, "individual": 0, "no_params": 0}
    for t in src.trials:
        if t.state != TrialState.COMPLETE or t.value is None:
            continue
        if t.number not in keep:
            skipped["not_analysis_grade"] += 1
            continue
        p = dict(t.params)
        if not p:
            skipped["no_params"] += 1
            continue
        mode, qm = p.get("mode"), p.get("question_mode")
        if mode == "full" and qm != "batched":
            skipped["individual"] += 1      # incl. the 3 with no question_mode
            continue
        is_batched = mode == "full"
        drop = DROP_ALWAYS | (DROP_FOR_BATCHED if is_batched else set())
        params = {k: v for k, v in p.items() if k not in drop and k in dists}
        if is_batched:
            params["batched_prompt_type"] = "batched_native"
        assert params["llm_model"] in ids, params["llm_model"]
        picked.append((t, params, is_batched))

    print(f"source : {SRC_DB}")
    print(f"target : {DST_DB}")
    print(f"selected {len(picked)} trials "
          f"({sum(1 for _,_,b in picked if not b)} rag / "
          f"{sum(1 for _,_,b in picked if b)} batched)")
    print(f"skipped: {skipped}")
    if args.dry:
        for t, params, b in picked[:4]:
            print(f"  e.g. #{t.number} -> {params}")
        return

    dst = optuna.create_study(
        study_name=STUDY, storage=DST_DB, direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=100, seed=42),
        pruner=optuna.pruners.NopPruner(),
    )
    already = {
        t.user_attrs.get("phase1_trial")
        for t in dst.trials
        if t.user_attrs.get("phase1_trial") is not None
    }
    added = 0
    for t, params, is_batched in picked:
        if t.number in already:
            continue
        attrs = dict(t.user_attrs)
        attrs["phase1_trial"] = t.number
        attrs["phase1_question_mode"] = "batched" if is_batched else "rag"
        # The prompt lever was inert for these; keep the original recorded value as
        # provenance so the substitution is auditable, not silent.
        if is_batched:
            attrs["phase1_inert_default_prompt_type"] = t.params.get("default_prompt_type")
        dst.add_trial(create_trial(
            params=params,
            distributions={k: dists[k] for k in params},
            value=t.value, state=TrialState.COMPLETE, user_attrs=attrs))
        added += 1
    n_done = sum(1 for t in dst.trials if t.state == TrialState.COMPLETE)
    print(f"added {added}; target study now holds {n_done} COMPLETE trials "
          f"({max(0, 100 - n_done)} more needed to finish TPE startup)")


if __name__ == "__main__":
    main()
