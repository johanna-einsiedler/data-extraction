"""Registered numeric rule (prereg2): round both, half away from zero, to the
precision of the less precise value when that value has >= 2 significant digits;
a single-significant-digit value must match exactly; integers exact; symmetric."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import mc_rule  # noqa: E402


@pytest.mark.parametrize("a,b,exp", [
    ("0.849", "0.85", 1), ("0.85", "0.849", 1),       # symmetric
    ("1", "0.85", 0), ("0.85", "1", 0),               # single significant digit -> exact
    ("0.8", "0.849", 0), ("0.849", "0.8", 0),
    ("0.07", "0.065", 0),                             # 0.07 has one significant digit
    ("0.75", "0.749", 1), ("0.7", "0.749", 0),
    ("49.57", "50", 1), ("12.5", "13", 1), ("12.5", "12", 0),
    ("16", "16", 1), ("16", "17", 0), ("3397", "3397.0", 1),
    ("90%", "90", 1), ("1,200", "1200", 1), ("n = 135 participants", "135", 1),
    ("0.751", "0.749", 0),
    ("Z", "0.85", 0), ("", "0.85", 0), ("abc", "5", 0),
])
def test_numeric_match(a, b, exp):
    assert mc_rule.numeric_match(a, b) == exp


def test_parse_numeric_significant_digits():
    assert mc_rule.parse_numeric("0.07") == (0.07, 2, 1)
    assert mc_rule.parse_numeric("50") == (50.0, 0, 2)
    assert mc_rule.parse_numeric("0") == (0.0, 0, 1)
    assert mc_rule.parse_numeric("Z") is None


def test_child_gold_not_applicable_codes():
    assert mc_rule.child_gold_not_applicable("numeric", ["-99"])
    assert not mc_rule.child_gold_not_applicable("numeric", ["0"])
    assert mc_rule.child_gold_not_applicable("multiple_choice", ["0"])
    assert mc_rule.child_gold_not_applicable("multiple_choice", ["-99"])
    assert not mc_rule.child_gold_not_applicable("multiple_choice", ["grid search"])
