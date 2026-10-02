"""MC concat-fix as a sidecar — never mutates trial_logs or the Optuna DB.

`map_values_to_letters` had a string-concatenation bug that under-credited
multiple-choice cells. The fix is applied here by re-scoring each MC cell from
its stored `raw_output` through the current evaluator, and writing the result to
analysis/cache/rescored_cells.parquet. The loader joins that on.

Two guards, both established empirically by analysis/check_integrity.py:
  * a cell is re-scored ONLY if its own output survives on disk — scoring a
    missing output rewrites a real answer as wrong;
  * trials 382-398 stored no outputs at all and are skipped entirely.

Build:  pipeline_env/bin/python -m analysis.hpo_analysis.rescore
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import jsonio

from .paths import CACHE, OUTPUT_BLACKOUT, RESCORE_PQ, ROOT, TRIAL_LOGS
from .recover import RECOVERED_PQ

sys.path.insert(0, str(ROOT / "src"))
_NULL = io.StringIO()


def _quiet(fn, *a, **kw):
    """src.mc_rule prints debug lines to stdout on every rater_iu call."""
    with contextlib.redirect_stdout(_NULL):
        return fn(*a, **kw)


def doc_id_to_name(folder: Path) -> dict[int, str]:
    """Recover doc_id -> doc_name for summaries written in the older format.

    Trial 330's per_query_doc_results carry no `doc_name` (774 cells) and its
    document_aggregates are keyed by name with bare float values, so the mapping
    has to come from the per-document files instead.
    """
    out: dict[int, str] = {}
    for f in sorted(folder.glob("doc_*.json")):
        try:
            d = jsonio.load(f)
        except Exception:
            continue
        if d.get("doc_id") is not None and d.get("doc_name") is not None:
            out[int(d["doc_id"])] = str(d["doc_name"])
    return out


def _evaluator():
    """Load the scoring stack once, with a memoised rater_iu."""
    import _data_loading as D
    import mc_rule
    from evaluate import evaluate_answer

    Q = _quiet(D.load_queries)
    Q = {q: v for q, v in Q.items() if q in _quiet(D.finalized_item_map, Q)}
    TRUE = _quiet(D.load_true_answers_final, Q)
    _quiet(mc_rule._load_raw)

    _orig, _memo = mc_rule.rater_iu, {}

    def cached(doc, item, label, choices):
        k = (str(doc), str(item))
        if k not in _memo:
            _memo[k] = _orig(doc, item, label, choices)
        return _memo[k]

    mc_rule.rater_iu = cached
    return evaluate_answer, Q, TRUE, mc_rule.CURATED_MC


def build(force: bool = False) -> pd.DataFrame:
    if RESCORE_PQ.exists() and not force:
        return pd.read_parquet(RESCORE_PQ)

    evaluate_answer, Q, TRUE, CUR = _evaluator()

    # Outputs recovered from the raw Batch-API files (analysis/hpo_analysis/recover.py).
    # The ingest path discarded them, so for trials 382-398 this is the only source.
    rec: dict[tuple, str] = {}
    if RECOVERED_PQ.exists():
        rdf = pd.read_parquet(RECOVERED_PQ)
        rec = {(int(t), str(d), str(q)): o for t, d, q, o in
               zip(rdf.trial_number, rdf.doc_name, rdf.query_id, rdf.raw_output)}

    rows = []
    folders = sorted(TRIAL_LOGS.glob("*_trial_*"))
    for folder in folders:
        summ = folder / "trial_summary.json"
        if not summ.exists():
            continue
        try:
            number = int(folder.name.rsplit("_trial_", 1)[1])
        except (IndexError, ValueError):
            continue
        blackout = number in OUTPUT_BLACKOUT
        ts = jsonio.load(summ)
        cells = ts.get("per_query_doc_results", [])
        id2name = (doc_id_to_name(folder)
                   if any("doc_name" not in c for c in cells) else {})
        with contextlib.redirect_stdout(_NULL):
            for c in cells:
                if c.get("type") != "multiple_choice":
                    continue
                doc = c.get("doc_name")
                if doc is None:
                    doc = id2name.get(c.get("doc_id"))
                if doc is None:
                    continue  # unmappable cell: cannot look up rater sets
                doc = str(doc)
                stored = c.get("accuracy")
                out = c.get("raw_output")
                if out is None:
                    out = c.get("llm_output")
                if out is None:
                    out = rec.get((number, doc, str(c["query_id"])))
                if out is None:
                    fixed, rescored = stored, False
                else:
                    r = evaluate_answer(
                        out,
                        question_type="multiple_choice",
                        doc_name=doc,
                        query_id=str(c["query_id"]),
                        TRUE_ANSWERS=TRUE,
                        QUERIES=Q,
                        other_callback=None,
                        curated_mc_items=CUR,
                    )
                    fixed, rescored = r["accuracy"], True
                rows.append(
                    {
                        "trial_number": number,
                        "doc_name": doc,
                        "query_id": str(c["query_id"]),
                        "accuracy_stored": stored,
                        "accuracy_fixed": fixed,
                        "rescored": rescored,
                        "blackout": blackout,
                        "from_recovered": (c.get("raw_output") is None
                                          and c.get("llm_output") is None
                                          and out is not None),
                    }
                )

    df = pd.DataFrame(rows)
    df["changed"] = df.accuracy_fixed.ne(df.accuracy_stored)
    for col in ("doc_name", "query_id"):
        df[col] = df[col].astype("category")
    df["trial_number"] = df.trial_number.astype("int32")
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESCORE_PQ, index=False)
    return df


def main() -> None:
    df = build(force=True)
    n_tr = df.trial_number.nunique()
    resc = df[df.rescored]
    print(f"rescored_cells.parquet: {len(df):,} MC cells over {n_tr} trials")
    print(f"  re-scored           : {len(resc):,} cells "
          f"({resc.trial_number.nunique()} trials)")
    print(f"  skipped (blackout)  : {int(df.blackout.sum()):,} cells "
          f"({df[df.blackout].trial_number.nunique()} trials)")
    print(f"  skipped (no output) : {int((~df.blackout & ~df.rescored).sum()):,} cells")
    up = int((resc.changed & (resc.accuracy_fixed > resc.accuracy_stored)).sum())
    dn = int((resc.changed & (resc.accuracy_fixed < resc.accuracy_stored)).sum())
    print(f"  corrected 0->1      : {up:,}")
    print(f"  corrected 1->0      : {dn:,}")
    print(f"\nwrote {RESCORE_PQ}")


if __name__ == "__main__":
    main()
