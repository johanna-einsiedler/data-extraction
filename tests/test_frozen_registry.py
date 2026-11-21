"""configs/frozen_pipelines.json must be internally consistent with the hashing rule,
the roster, and the conditional-parameter rules a FixedTrial will be asked for."""
import hashlib, json, os, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
REG = json.load(open(ROOT / "configs" / "frozen_pipelines.json"))
ROSTER = {m["id"]: m for m in json.load(open(ROOT / "src/answer_generation/filtered_models.json"))}


def test_hashes_recompute():
    for name, e in REG["pipelines"].items():
        md5 = hashlib.md5(json.dumps(e["trial_choices"], sort_keys=True).encode()).hexdigest()
        assert md5 == e["parameters_hash_md5"] and md5[:10] == e["trial_hash10"], name


def test_known_frozen_hashes():
    want = {"primary": "5f762d817b2d1acc07b13dfcd4cc684d", "open": "d22703dbce" + REG["pipelines"]["open"]["parameters_hash_md5"][10:],
            "consumer": "4db0f5c971b00dd042eb80793f70fb3d"}
    for k, v in want.items():
        assert REG["pipelines"][k]["parameters_hash_md5"] == v, k


def test_params_consistent_with_choices_and_roster():
    for name, e in REG["pipelines"].items():
        p, tc = e["optuna_params"], e["trial_choices"]
        for k, v in p.items():
            assert tc.get(k) == v, (name, k, tc.get(k), v)
        m = ROSTER[p["llm_model"]]
        assert not m.get("disabled"), name
        if "vlm" in p:
            assert m.get("multimodal"), f"{name}: vlm suggested for a non-multimodal model"
        if "disable_reasoning" in p:
            assert m.get("reasoning") and m["api_type"] in ("together", "openrouter"), name
        if e.get("test_only"):
            assert e["source"] is None
        else:
            assert e["source"]["state"] == "COMPLETE"
            if e.get("provider_change"):
                # re-frozen on another provider after the search: the search champion keeps its
                # trial folder; the entry's folder is the tuning-check run of the new configuration
                sc = e["search_champion"]
                assert sc["folder"].startswith(sc["trial_hash10"]) and (ROOT / e["folder"]).exists(), name
            else:
                assert e["folder"].startswith(e["trial_hash10"])
