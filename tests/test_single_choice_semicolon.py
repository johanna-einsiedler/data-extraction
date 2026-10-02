"""Regression: a single-choice code followed by '; explanation' must score on the code."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from evaluate import clean_answer


def _code(s):
    return clean_answer(str(s).split(";")[0])


def test_code_then_semicolon_explanation():
    assert _code("1; outcome is binary depression presence.") == "1"
    assert _code("Z; not reported.") == "Z"
    assert _code("2 ; ordinal ranks") == "2"


def test_plain_code_unchanged():
    assert _code("3") == "3"
    assert _code("  B  ") == "B"
    assert _code("Z") == "Z"


def test_prose_first_is_still_not_rescued():
    # Non-compliant shape: reasoning before the code. Not a semicolon problem.
    assert _code("Answer: 1; because ...") != "1"
