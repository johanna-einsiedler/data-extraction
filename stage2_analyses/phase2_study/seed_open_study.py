#!/usr/bin/env python
"""Seed the open-weight-only TPE study from the finished unrestricted study.

Every COMPLETE trial of phase2.db whose model is open-weight (api_type not
openai/gemini; gpt-oss counts as open) is copied into open_tpe.db with its
unified value and provenance attrs. The llm_model categorical is rebuilt with
objective.model_choice_list() under OPEN_MODELS_ONLY=1 — the exact list the
workers will suggest from, or add_trial would reject the distribution.

Proprietary trials (gemini / gpt-4.1 / gpt-5.2) cannot be represented in the
restricted space and are left behind; their information about parser/prompt
levers is the price of the clean categorical (design note from 2026-09-28).

    pipeline_env/bin/python stage2_analyses/phase2_study/seed_open_study.py [--dry]
"""
import json, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ["SAMPLE_ONLY"] = "1"
os.environ["OPEN_MODELS_ONLY"] = "1"
sys.path.insert(0, str(ROOT / "src"))

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState, create_trial
from objective import model_choice_list, _model_meta

SRC = f"sqlite:///{ROOT/'stage2_analyses'/'phase2_study'/'phase2.db'}"
DST = f"sqlite:///{ROOT/'stage2_analyses'/'phase2_study'/os.getenv('SEED_DST_DB','open_tpe.db')}"
STUDY = "full_item_accuracy_tuning"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    dry = "--dry" in sys.argv
    open_ids = model_choice_list()
    meta = _model_meta()
    assert all((meta.get(i) or {}).get("api_type") not in ("openai", "gemini")
               for i in open_ids)
    print(f"restricted llm_model categorical: {len(open_ids)} ids")

    src = optuna.load_study(study_name=STUDY, storage=SRC)
    seeds, skipped = [], []
    for t in sorted(src.trials, key=lambda x: x.number):
        if t.state != TrialState.COMPLETE or t.value is None:
            continue
        m = t.params.get("llm_model")
        if m not in open_ids:
            skipped.append((t.number, m)); continue
        dists = dict(t.distributions)
        dists["llm_model"] = CategoricalDistribution(open_ids)
        ua = dict(t.user_attrs); ua["seeded_from_trial"] = t.number
        seeds.append(create_trial(state=TrialState.COMPLETE, value=t.value,
                                  params=dict(t.params), distributions=dists,
                                  user_attrs=ua))
    best = max(t.value for t in seeds)
    print(f"seeds: {len(seeds)}  skipped (proprietary): {len(skipped)}  "
          f"seeded best: {best:.4f}")
    if dry:
        return
    storage = optuna.storages.RDBStorage(DST, engine_kwargs={"connect_args": {"timeout": 120}})
    dst = optuna.create_study(study_name=STUDY, storage=storage,
                              direction="maximize", load_if_exists=True)
    if len(dst.trials):
        print(f"open_tpe.db already holds {len(dst.trials)} trials — aborting, not double-seeding")
        sys.exit(1)
    for tr in seeds:
        dst.add_trial(tr)
    dst_name = os.getenv("SEED_DST_DB", "open_tpe.db")
    print(f"seeded {len(seeds)} trials into {dst_name} "
          f"(env for workers: TPE_N_STARTUP={len(seeds)} PATIENCE_FROM_NUMBER={len(seeds)})")


if __name__ == "__main__":
    main()
