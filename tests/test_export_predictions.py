"""export_predictions_for_irr.py: representation contract for scripts/compute_irr.py."""
import csv, json, pathlib, subprocess, sys
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/true_labels/01_study_data_long_raw.csv").exists(), reason="needs gold files")


def test_export_from_handwritten_run(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    doc = {"doc_id": 6, "doc_name": "006", "parameters_hash": "x" * 32, "results": [
        {"query_id": "1.8", "pred_indicator": "1", "accuracy": 1, "raw_output": "1"},
        {"query_id": "3.2", "pred_indicator": ["A", "Y"], "accuracy": 0, "raw_output": "A, Y"},
        {"query_id": "2.2", "pred_indicator": ["Z"], "accuracy": 1, "raw_output": "Z"},
        {"query_id": "1.2.1", "pred_value": 2084.0, "accuracy": 1, "raw_output": "2084"},
        {"query_id": "4.5.1", "pred_value": 0.83, "accuracy": 1, "raw_output": "0.83"},
        {"query_id": "1.3", "pred_value": "adolescents", "accuracy": 1, "raw_output": "adolescents"}]}
    json.dump(doc, open(run / "doc_0006.json", "w")); json.dump({"006": 6}, open(run / "doc_id_map.json", "w"))
    r = subprocess.run([sys.executable, "scripts/export_predictions_for_irr.py", "--run-dir", str(run)], cwd=ROOT,
                       capture_output=True, text=True); assert r.returncode == 0, r.stderr[-1500:]
    rows = {(a, b): c for a, b, c in list(csv.reader(open(run / "predictions_irr.csv")))[1:]}
    q = json.load(open(ROOT / "query_creation/queries_with_prompts.json"))
    assert rows[("006", q["1.8"]["label"])] == "1"
    mc = rows[("006", q["3.2"]["label"])]
    assert mc.endswith(", Other") and len(mc.split(", ")) == 2 and mc.split(", ")[0] not in ("A", "")
    assert rows[("006", q["2.2"]["label"])] == ""
    assert rows[("006", q["1.2.1"]["label"])] == "2084"
    assert rows[("006", "highestperf_auc")] == "0.83"
    assert ("006", q["1.3"]["label"]) not in rows, "open-ended items are not exported to the IRR file"
    cells = list(csv.DictReader(open(run / "cells.csv"))); assert len(cells) == 6 and {c["type"] for c in cells} >= {"single_choice", "multiple_choice", "numeric", "open_ended"}
    rep = json.load(open(run / "export_report.json")); assert rep["n_predictions"] == 5


def test_compute_irr_accepts_the_file(tmp_path):
    """The predictions CSV must be loadable by compute_irr.py's model configuration."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib; irr = importlib.import_module("compute_irr")
    assert hasattr(irr, "parse_mc") and hasattr(irr, "variable_types")
