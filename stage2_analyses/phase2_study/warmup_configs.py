#!/usr/bin/env python
"""Generate the warm-up configs that top the Phase-2 study up to TPE's startup budget.

TPESampler(n_startup_trials=100) samples at random until 100 COMPLETE-or-PRUNED
trials exist; copy_trials.py supplies 61 of those from Phase 1, so ~39 remain.

Left to plain random sampling those 39 would spread `batched_prompt_type` evenly
over all seven levels, while "batched_native" already has 26 copied trials. TPE
would then enter its modelling phase with the one level that is *not* under test
holding ~70% of the batched evidence. So the warm-up excludes batched_native
entirely and balances the six real variants instead.

Configs are drawn through objective() itself under SAMPLE_ONLY=1, so every one is a
genuine point of the define-by-run space (conditional levers, pruning guards and
model gating all applied) rather than a hand-built dict that might not be reachable.

    python stage2_analyses/phase2_study/warmup_configs.py --n 39 --dry
    python stage2_analyses/phase2_study/warmup_configs.py --n 39 --enqueue
"""
import argparse
import collections
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("SAMPLE_ONLY", "1")
sys.path.insert(0, str(ROOT / "src"))

import optuna  # noqa: E402

DST_DB = f"sqlite:///{ROOT / 'stage2_analyses' / 'phase2_study' / 'phase2.db'}"
STUDY = "full_item_accuracy_tuning"
OUT = ROOT / "stage2_analyses" / "phase2_study" / "warmup_configs.json"
VARIANTS = [
    "base_prompt", "cot_prompt", "reasoning_prompt",
    "rewritten_prompt", "synthetic_few_shot_prompt", "true_few_shot_prompt",
]


def existing_configs(study):
    """Every config already present in the study, in any state, as a frozen key."""
    out = set()
    for t in study.trials:
        p = dict(t.params) or dict(t.system_attrs.get("fixed_params", {}))
        if p:
            out.add(json.dumps(p, sort_keys=True, default=str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=39, help="configs to generate")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batched-share", type=float, default=0.6,
                    help="fraction of warm-up configs that should be batched")
    ap.add_argument("--enqueue", action="store_true",
                    help="enqueue into the Phase-2 study (default: write JSON only)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import objective as O  # heavy import; after SAMPLE_ONLY is set

    # objective() reads module globals that are only assigned in objective.py's
    # __main__ block, so an imported module has to seed them or every call raises
    # NameError on QUERIES. SAMPLE_ONLY returns before TRUE_ANSWERS / DEPENDENCIES
    # / SYSTEM_PROMPTS are touched, so QUERIES alone is enough (same two lines as
    # archive/trial_orchestration/generate_warmup_configs.py).
    _q = O.load_queries()
    O.QUERIES = {k: v for k, v in _q.items() if k in O.finalized_item_map(_q)}

    # Models with no reachable endpoint stay in the roster (the Optuna categorical
    # must not change) but must never be enqueued.
    disabled = {
        m["id"]
        for m in json.load(open(ROOT / "src" / "answer_generation" / "filtered_models.json"))
        if m.get("disabled")
    }

    dst = optuna.load_study(study_name=STUDY, storage=DST_DB)
    seen = existing_configs(dst)
    n_done = sum(1 for t in dst.trials
                 if t.state.name in ("COMPLETE", "PRUNED"))
    print(f"study holds {n_done} COMPLETE/PRUNED; TPE startup needs 100")

    want_batched = int(round(args.n * args.batched_share))
    want_rag = args.n - want_batched
    # even coverage of the six real variants
    quota = collections.Counter()
    for i in range(want_batched):
        quota[VARIANTS[i % len(VARIANTS)]] += 1

    sampler = optuna.samplers.RandomSampler(seed=args.seed)
    probe = optuna.create_study(sampler=sampler)
    picked, rag_n, tries = [], 0, 0
    while (sum(quota.values()) > 0 or rag_n < want_rag) and tries < args.n * 400:
        tries += 1
        t = probe.ask()
        try:
            cfg = O.objective(t, verbose=False)
        except Exception:
            probe.tell(t, state=optuna.trial.TrialState.FAIL)
            continue
        probe.tell(t, 0.5)
        params = dict(t.params)
        if params.get("llm_model") in disabled:
            continue
        key = json.dumps(params, sort_keys=True, default=str)
        if key in seen:
            continue
        if cfg["mode"] == "full":
            bpt = cfg.get("batched_prompt_type")
            if bpt == "batched_native" or quota[bpt] <= 0:
                continue            # already covered by the 26 copied trials
            quota[bpt] -= 1
        else:
            if rag_n >= want_rag:
                continue
            rag_n += 1
        seen.add(key)
        picked.append(params)

    nb = sum(1 for p in picked if p.get("mode") == "full")
    print(f"generated {len(picked)} configs ({nb} batched / {len(picked)-nb} rag) "
          f"in {tries} draws")
    got = collections.Counter(p.get("batched_prompt_type") for p in picked
                              if p.get("mode") == "full")
    print("batched_prompt_type coverage:", dict(got))
    if sum(quota.values()):
        print(f"WARNING: unmet quota {dict(+quota)} -- raise --n or the draw budget")

    if args.dry:
        for p in picked[:3]:
            print("  e.g.", p)
        return
    OUT.write_text(json.dumps(picked, indent=2))
    print(f"wrote {OUT}")
    if args.enqueue:
        for p in picked:
            dst.enqueue_trial(p, skip_if_exists=True)
        print(f"enqueued {len(picked)} trials into {DST_DB}")


if __name__ == "__main__":
    main()
