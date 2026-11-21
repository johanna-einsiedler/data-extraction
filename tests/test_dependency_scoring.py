"""Model-driven dependency rule (prereg2 "Conventions shared by all item types").

Applicability of a child item follows the PIPELINE'S OWN parent answer. When the
parent answer makes the child not applicable, the child is set to -99 without a
model call and counts as correct iff the gold marks the child not applicable
(-99 for the numeric children 1.2.1/4.2.1; 0 or -99 for the multiple-choice
children, where the coders used the not-reported code). Otherwise the child is
scored like any other cell. The gold never enters the decision."""
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("SAMPLE_ONLY", "1")
sys.path.insert(0, str(ROOT / "src"))

import objective  # noqa: E402
from objective import dependency_resolve  # noqa: E402


@pytest.fixture(autouse=True)
def _deps(monkeypatch):
    import json
    monkeypatch.setattr(objective, "DEPENDENCIES", json.load(open(ROOT / "src" / "dependencies.txt")), raising=False)
    monkeypatch.setattr(objective, "QUERIES", {q: {} for q in
                        ["1.8", "1.2.1", "3.2", "3.2.1", "3.4", "3.4.1", "3.5", "3.5.1", "4.1", "4.2", "4.2.1"]}, raising=False)


def sc(qid, code):      # a scored single-choice parent cell
    return {"query_id": qid, "type": "single_choice", "pred_indicator": str(code), "pred_value": "x"}


def mc(qid, *values):   # a scored multiple-choice parent cell (stored option values)
    return {"query_id": qid, "type": "multiple_choice", "pred_value": list(values)}


def test_continuous_outcome_makes_cases_not_applicable():
    res = {"1.8": sc("1.8", 4)}
    r = dependency_resolve("1.2.1", "numeric", res, ["-99"])
    assert r["accuracy"] == 1 and r["pred_value"] == -99 and r["raw_output"] is None
    assert r["resolved_by"] == "model_parent_not_applicable"
    assert dependency_resolve("1.2.1", "numeric", res, ["3397"])["accuracy"] == 0   # gold applicable -> wrong
    assert dependency_resolve("1.2.1", "numeric", res, ["0"])["accuracy"] == 0      # numeric gold 0 = not reported


def test_applicable_parent_asks_the_model():
    assert dependency_resolve("1.2.1", "numeric", {"1.8": sc("1.8", 1)}, ["-99"]) is None
    assert dependency_resolve("1.2.1", "numeric", {"1.8": sc("1.8", "Z")}, ["-99"]) is None


def test_mc_child_gold_codes():
    res = {"3.4": sc("3.4", 0)}
    assert dependency_resolve("3.4.1", "multiple_choice", res, ["0"])["accuracy"] == 1
    assert dependency_resolve("3.4.1", "multiple_choice", res, ["-99"])["accuracy"] == 1
    assert dependency_resolve("3.4.1", "multiple_choice", res, ["grouped / nested cv"])["accuracy"] == 0


def test_mc_parent_values_are_normalised_against_the_rule_text():
    # stored pred_values are lower-cased with punctuation stripped
    assert dependency_resolve("4.2.1", "numeric", {"4.2": mc("4.2", "internal no split")}, ["-99"])["accuracy"] == 1
    assert dependency_resolve("4.2.1", "numeric", {"4.2": mc("4.2", "internal single split")}, ["-99"]) is None
    # one selected option that is NOT blocked keeps the child applicable
    assert dependency_resolve("4.2.1", "numeric", {"4.2": mc("4.2", "no validation", "internal single split")}, ["-99"]) is None


def test_chain_3_5_to_3_5_1():
    res = {"1.8": sc("1.8", 4)}
    child = dependency_resolve("3.5", "multiple_choice", res, ["0"])
    assert child["accuracy"] == 1
    res["3.5"] = {"query_id": "3.5", "type": "multiple_choice", **child}
    assert dependency_resolve("3.5.1", "multiple_choice", res, ["0"])["accuracy"] == 1
    assert dependency_resolve("3.5.1", "multiple_choice", res, ["smogn"])["accuracy"] == 0


def test_gold_never_enters_the_decision_and_may_be_absent():
    res = {"1.8": sc("1.8", 4)}
    r = dependency_resolve("1.2.1", "numeric", res, None, model_answer="12")
    assert r is not None and r["accuracy"] is None and r["model_answer_unused"] == "12"
    assert dependency_resolve("1.2.1", "numeric", {"1.8": sc("1.8", 1)}, None) is None


def test_missing_parent_means_applicable():
    assert dependency_resolve("1.2.1", "numeric", {}, ["-99"]) is None


def test_non_child_is_untouched():
    assert dependency_resolve("1.1", "numeric", {"1.8": sc("1.8", 4)}, ["-99"]) is None
