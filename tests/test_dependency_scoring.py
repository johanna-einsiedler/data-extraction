"""Regression tests for the gold-driven dependency rule (prereg2 metrics).

The pre-2026-09-28 live scorer skipped dependency-child items based on the
MODEL's parent answer, fabricated raw_output="Not applicable"/pred -99, and
scored the cell against gold — mis-scoring ~24-41 MC cells per trial (found by
the unified rescore). The rule now: gold == -99 -> auto-correct without a model
call; any real gold -> the model is always asked.

The parquet-driven test replays every cell the unified rescore CHANGED on the
freshest trial (#124) and asserts the new resolver would have asked the model on
each one — i.e. the entire class of mis-scored cells is unreachable now.
"""
import json
import os
import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("SAMPLE_ONLY", "1")
sys.path.insert(0, str(ROOT / "src"))

from objective import dependency_autoresolve  # noqa: E402

PQ = ROOT / "analysis" / "cache" / "unified_rescore_cells.parquet"


def test_gold_not_applicable_autoresolves_correct():
    for gold in (["-99"], [-99], [-99, "x"]):
        r = dependency_autoresolve("3.4.1", gold)
        assert r is not None
        assert r["accuracy"] == 1
        assert r["pred_value"] == -99
        # raw_output must stay None so offline re-scoring keeps the stored value
        assert r["raw_output"] is None


@pytest.mark.parametrize("gold", [["0"], ["1"], ["A", "B"], [0], ["Z"]])
def test_real_gold_always_asks_the_model(gold):
    assert dependency_autoresolve("3.4.1", gold) is None


def test_every_previously_misscored_cell_would_now_be_asked():
    if not PQ.exists():
        pytest.skip("unified rescore parquet not built")
    import _data_loading as D
    cells = pd.read_parquet(PQ)
    changed = cells[(cells.p2_number == 124)
                    & (cells.accuracy_unified != cells.accuracy_stored)]
    assert len(changed) > 0, "expected the known #124 divergent cells"
    Q = D.load_queries()
    Q = {q: v for q, v in Q.items() if q in D.finalized_item_map(Q)}
    TRUE = D.load_true_answers_final(Q)
    for row in changed.itertuples():
        gold = TRUE[str(row.doc_name)][str(row.query_id)]
        assert dependency_autoresolve(str(row.query_id), gold) is None, (
            f"cell ({row.doc_name},{row.query_id}) gold={gold} would still be "
            f"auto-skipped — the old bug class is reachable again")
