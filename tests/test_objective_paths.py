"""resolve_run_paths() must reproduce the historical layout byte-for-byte with the
defaults (so the Optuna path is unchanged) and route frozen runs to their out_dir."""
import hashlib, json, os, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("SAMPLE_ONLY", "1")
sys.path.insert(0, str(ROOT / "src"))
import objective  # noqa: E402

FROZEN = {  # trial_choices as written to trial_summary.json["parameters"] -> known hash10
    "5f762d817b": ROOT / "trial_logs" / "5f762d817b_trial_0131" / "trial_summary.json",
    "f2609ec809": ROOT / "trial_logs" / "f2609ec809_trial_0109" / "trial_summary.json",
    "4db0f5c971": ROOT / "trial_logs" / "4db0f5c971_trial_0062" / "trial_summary.json",
}


def _params(p):
    return json.load(open(p))["parameters"]


def test_defaults_reproduce_historical_layout():
    for h10, summ in FROZEN.items():
        if not summ.exists():
            continue
        tc = _params(summ)
        for n in (0, 1, 131):
            paper, trial_folder, cache_dir, h = objective.resolve_run_paths(tc, n)
            assert h == h10 == hashlib.md5(json.dumps(tc, sort_keys=True).encode()).hexdigest()[:10]
            assert paper == ROOT / "data" / "tuning"
            assert trial_folder == ROOT / "trial_logs" / f"{h10}_trial_{n:04d}"
            assert cache_dir == objective.CACHE_PATH


def test_frozen_routes_to_out_dir(tmp_path):
    tc = {"mode": "full", "llm_model": "x", "parser": "PyMuPDFParser"}
    paper, trial_folder, cache_dir, h = objective.resolve_run_paths(
        tc, 0, paper_dir=tmp_path / "papers", out_dir=tmp_path / "out", frozen=True)
    assert paper == tmp_path / "papers" and trial_folder == tmp_path / "out" and cache_dir == tmp_path / "out"
    # frozen without out_dir still falls back to the historical folder
    _, tf2, cd2, _ = objective.resolve_run_paths(tc, 0, frozen=True)
    assert tf2 == ROOT / "trial_logs" / f"{h}_trial_0000" and cd2 == objective.CACHE_PATH


def test_paper_id():
    assert objective._paper_id(pathlib.Path("006_Ahnert_2023.pdf")) == "006"
    assert objective._paper_id(pathlib.Path("weird-name.pdf")) == "weird-name"
