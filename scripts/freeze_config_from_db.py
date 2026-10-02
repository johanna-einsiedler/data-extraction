"""Build / refresh configs/frozen_pipelines.json: the frozen configurations as
(a) the Optuna params a FixedTrial must supply and (b) the resolved 16-field
trial_choices dict whose canonical JSON md5 is the trial-folder prefix.

usage: pipeline_env/bin/python scripts/freeze_config_from_db.py            # rebuild all entries
       pipeline_env/bin/python scripts/freeze_config_from_db.py --db stage2_analyses/phase2_study/phase2.db --trial 131 --name primary

trial_choices is derived by running objective() under SAMPLE_ONLY=1 on a
FixedTrial(optuna_params): that exercises the same conditional-parameter logic
the search used (vlm only for multimodal models, disable_reasoning only for
togglable reasoning models) and raises if a required param is missing."""
from __future__ import annotations

import argparse, datetime, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["SAMPLE_ONLY"] = "1"
sys.path.insert(0, str(ROOT / "src"))
import optuna  # noqa: E402
optuna.logging.set_verbosity(optuna.logging.WARNING)
import objective  # noqa: E402
objective.init_globals()   # objective() reads QUERIES even under SAMPLE_ONLY

REG = ROOT / "configs" / "frozen_pipelines.json"
STUDY = "full_item_accuracy_tuning"
DEFAULT = [
    ("primary", "stage2_analyses/phase2_study/phase2.db", 131, "P_primary", "mineru"),
    ("open", "stage2_analyses/phase2_study/phase2.db", 109, "P_open", "mineru"),
    ("consumer", "stage2_analyses/phase2_study/consumer_tpe.db", 62, "consumer reference", "pymupdf_tesseract"),
]
STUB = {"mode": "full", "llm_model": "openai/gpt-oss-20b", "parser": "PyMuPDFParser",
        "batched_prompt_type": "base_prompt", "disable_reasoning": True}


def trial_choices_for(params: dict) -> tuple[dict, dict]:
    """Run the sampling path on a FixedTrial; return (trial_choices, params actually
    suggested). A FixedTrial key that objective() never asks for is silently
    ignored, so the registry records the suggested set, not the input."""
    ft = optuna.trial.FixedTrial(params)
    tc = objective.objective(ft)
    assert isinstance(tc, dict), "objective() under SAMPLE_ONLY must return trial_choices"
    return tc, dict(ft.params)


def md5_of(tc: dict) -> str:
    return hashlib.md5(json.dumps(tc, sort_keys=True).encode()).hexdigest()


def entry_from_db(db: str, number: int, label: str, expected_parser_meta: str) -> dict:
    st = optuna.load_study(study_name=STUDY, storage=f"sqlite:///{ROOT / db}")
    t = next(x for x in st.trials if x.number == number)
    params = dict(t.params)
    tc, params = trial_choices_for(params)
    md5 = md5_of(tc)
    folder = sorted((ROOT / "trial_logs").glob(f"{md5[:10]}_trial_{number:04d}"))
    if folder:
        fp = json.load(open(folder[0] / "trial_summary.json"))["parameters"]
        assert md5_of(fp) == md5, f"{db}#{number}: derived trial_choices hash {md5[:10]} != folder {md5_of(fp)[:10]}"
    return {
        "label": label,
        "source": {"db": db, "study_name": STUDY, "trial_number": number, "state": t.state.name,
                   "tuning_value": t.value, "tuning_cost_usd": t.user_attrs.get("cost_usd"),
                   "n_papers": t.user_attrs.get("n_papers"), "judge": t.user_attrs.get("judge")},
        "optuna_params": params, "trial_choices": tc,
        "trial_hash10": md5[:10], "parameters_hash_md5": md5,
        "expected_parser_meta": expected_parser_meta,
        "folder": folder[0].name if folder else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db"); ap.add_argument("--trial", type=int); ap.add_argument("--name")
    ap.add_argument("--label"); ap.add_argument("--parser-meta")
    a = ap.parse_args()
    reg = json.load(open(REG)) if REG.exists() else {"schema_version": 1, "pipelines": {}}
    if a.db:
        reg["pipelines"][a.name] = entry_from_db(a.db, a.trial, a.label or a.name, a.parser_meta or "")
    else:
        for name, db, num, label, meta in DEFAULT:
            reg["pipelines"][name] = entry_from_db(db, num, label, meta)
        tc, stub_params = trial_choices_for(STUB); md5 = md5_of(tc)
        reg["pipelines"]["test_stub"] = {"label": "test-only stub (never a preregistered run)", "test_only": True,
                                         "source": None, "optuna_params": stub_params, "trial_choices": tc,
                                         "trial_hash10": md5[:10], "parameters_hash_md5": md5,
                                         "expected_parser_meta": "pymupdf", "folder": None}
    reg["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    reg["hash_rule"] = "md5(json.dumps(trial_choices, sort_keys=True)); [:10] is the trial_logs folder prefix"
    json.dump(reg, open(REG, "w"), indent=1)
    for k, v in reg["pipelines"].items():
        print(f"{k:10} {v['trial_hash10']}  {v['optuna_params'].get('llm_model')} / {v['optuna_params'].get('parser')}  value={(v.get('source') or {}).get('tuning_value')}")
    print("wrote", REG.relative_to(ROOT))


if __name__ == "__main__":
    main()
