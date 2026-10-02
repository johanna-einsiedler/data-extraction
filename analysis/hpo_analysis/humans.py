"""Inter-coder (dis)agreement per paper x item, under the SCORER's normalization.

The whole point: a naive string comparison of rater cells gives 91% disagreement on
open-ended items, which is an artefact of free text, not a measurement. Agreement
must be judged the same way the pipeline's answers are judged, or the
disagreement-bucket analysis compares apples to oranges.

Per type, mirroring src/evaluate.py and src/mc_rule.py:
  multiple_choice  rater_iu() letter sets; agree iff I == U
  single_choice    map_values_to_letters on each rater, compare sets
  numeric          first numeric token, compared at the lesser reported precision
  list             normalized set match
  open_ended       LLM judge equivalence  (requires --judge-oe; else 'unknown')

Buckets: agree | partial (MC only: I subset-of U, I non-empty) | disjoint | unknown

Build:  pipeline_env/bin/python -m analysis.hpo_analysis.humans [--judge-oe]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from .paths import CACHE, CELLS_PQ, ROOT, TABLES, TRUE_LABELS

sys.path.insert(0, str(ROOT / "src"))
_NULL = io.StringIO()

# The five 4.5.x sub-items share the label 'highestperf' and cannot be resolved by
# label alone. _data_loading already owns this mapping — reuse it rather than
# duplicating, so a change there cannot silently desync the agreement analysis.
def _subitem_cols() -> dict:
    from _data_loading import HIGHESTPERF_COLS
    return dict(HIGHESTPERF_COLS)


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(_NULL):
        return fn(*a, **kw)


@lru_cache(maxsize=1)
def _stack():
    import _data_loading as D
    import mc_rule
    Q = _quiet(D.load_queries)
    Q = {k: v for k, v in Q.items() if k in _quiet(D.finalized_item_map, Q)}
    _quiet(mc_rule._load_raw)
    return Q, mc_rule


@lru_cache(maxsize=1)
def raw_long() -> pd.DataFrame:
    df = pd.read_csv(TRUE_LABELS / "01_study_data_long_raw.csv", low_memory=False)
    df["study_ID"] = df.study_ID.astype(str).str.strip()
    return df


def rater_column(qid: str, label: str) -> str | None:
    sub = _subitem_cols()
    if qid in sub:
        return sub[qid]
    _, mc_rule = _stack()
    return mc_rule._RAW_COLS.get(re.sub(r"\s+", " ", str(label)).strip())


@lru_cache(maxsize=1)
def _by_study() -> dict[str, pd.DataFrame]:
    return {k: v for k, v in raw_long().groupby("study_ID")}


def rater_cells(doc: str, col: str) -> list:
    sub = _by_study().get(str(doc))
    if sub is None or col not in sub.columns:
        return []
    return list(sub[col])


# ------------------------------------------------------------------ comparators
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
# The highestperf_* columns store the metric name inline: "AUC: 0.83", "AUC: -99"
# (not applicable) or a bare "AUC:" (not reported). Strip the prefix before any
# NA test or numeric parse, or every 4.5.x pair looks unparseable.
_METRIC_PREFIX = re.compile(r"^\s*(auc|acc|bacc|r2|r\u00b2|r)\s*:\s*", re.I)


def strip_metric_prefix(x) -> str:
    return _METRIC_PREFIX.sub("", str(x)).strip()


def _num_parts(x):
    s = strip_metric_prefix(x).replace(",", "").replace("%", "")
    m = _NUM.search(s)
    if not m:
        return None, 0
    tok = m.group(0)
    v = float(tok)
    if v in (-99.0, -97.0, -96.0):      # not-applicable / not-reported sentinels
        return None, 0
    return v, (len(tok.split(".")[1]) if "." in tok else 0)


def numeric_agree(a, b) -> bool | None:
    va, da = _num_parts(a)
    vb, db = _num_parts(b)
    if va is None or vb is None:
        return None
    nd = min(da, db)
    return round(va, nd) == round(vb, nd)


def _norm_list(x) -> frozenset:
    from text_utils import normalize
    s = str(x)
    parts = re.split(r"[;,\n]", s)
    out = set()
    for p in parts:
        p = re.sub(r"\(.*?\)", "", p)
        p = re.sub(r"\b(v\.?\s*)?\d+(\.\d+)*\b", "", p)
        p = re.sub(r"\b(package|library)\b", "", p, flags=re.I)
        p = normalize(p).strip()
        if p:
            out.add(p)
    return frozenset(out)


def scalar_str(x) -> str:
    """pandas reads the single-choice rater columns as floats, so a coded answer
    arrives as 1.0 and never matches the choice key "1" — map_values_to_letters
    then falls through to Y (Other) for EVERY rater, making every pair look like
    agreement. Coerce integral floats back to their code string first."""
    if isinstance(x, float) and not np.isnan(x) and float(x).is_integer():
        return str(int(x))
    return str(x).strip()


def sc_letters(val, qid, choices):
    """Reuse the scorer's own rater normalizer so agreement and accuracy agree."""
    _, mc_rule = _stack()
    try:
        return _quiet(mc_rule._rater_letters, scalar_str(val), qid, choices)
    except Exception:
        return frozenset()


# ------------------------------------------------------------------ build
def build(judge_oe: bool = False) -> pd.DataFrame:
    Q, mc_rule = _stack()
    cells = pd.read_parquet(CELLS_PQ)
    docs = sorted(cells.doc_name.astype(str).unique())

    rows = []
    for qid, q in Q.items():
        typ = q.get("type")
        label = q.get("label", "")
        choices = q.get("choices", {})
        col = rater_column(qid, label)
        for doc in docs:
            raw = [c for c in rater_cells(doc, col)] if col else []
            vals = [v for v in raw
                    if not mc_rule.is_na(v)
                    and not mc_rule.is_na(strip_metric_prefix(v))]
            n_raters = len(raw)
            rec = {"doc_name": doc, "query_id": qid, "type": typ, "label": label,
                   "rater_column": col, "n_raters": n_raters,
                   "n_nonempty": len(vals), "bucket": "unknown", "agree": np.nan,
                   "I_size": np.nan, "U_size": np.nan}
            if col is None or n_raters < 2:
                rows.append(rec); continue

            if typ == "multiple_choice":
                I, U = _quiet(mc_rule.rater_iu, doc, qid, label, choices)
                rec["I_size"], rec["U_size"] = len(I), len(U)
                if not U:
                    rec["bucket"] = "agree"; rec["agree"] = True
                elif I == U:
                    rec["bucket"] = "agree"; rec["agree"] = True
                elif I:
                    rec["bucket"] = "partial"; rec["agree"] = False
                else:
                    rec["bucket"] = "disjoint"; rec["agree"] = False

            elif typ == "single_choice":
                # evaluate.py scores single-choice by comparing the numeric CODE
                # (pred_number == true_value_key), and the rater columns store that
                # same code — so compare codes, not letters. Mapping a code through
                # map_values_to_letters matches no option text and returns Y for
                # every rater, which would report 0% disagreement.
                codes = {scalar_str(v).lower() for v in raw}
                ok = len(codes) == 1
                rec["agree"] = ok
                rec["bucket"] = "agree" if ok else "disjoint"

            elif typ == "numeric":
                if len(vals) < 2:
                    # both abstained -> agreement; one abstained -> disagreement
                    ok = len(vals) == 0
                    rec["agree"] = ok
                    rec["bucket"] = "agree" if ok else "disjoint"
                else:
                    r = numeric_agree(vals[0], vals[1])
                    rec["agree"] = r
                    rec["bucket"] = ("unknown" if r is None
                                     else "agree" if r else "disjoint")

            elif typ == "list":
                sets = [_norm_list(v) for v in raw]
                ok = len(set(sets)) == 1
                rec["agree"] = ok
                rec["bucket"] = "agree" if ok else "disjoint"

            elif typ == "open_ended":
                if len(vals) < 2:
                    ok = len(vals) == 0
                    rec["agree"] = ok
                    rec["bucket"] = "agree" if ok else "disjoint"
                elif judge_oe:
                    from evaluate import judge_equivalent
                    try:
                        eq, _, _ = _quiet(judge_equivalent, str(vals[0]), str(vals[1]))
                        rec["agree"] = bool(eq)
                        rec["bucket"] = "agree" if eq else "disjoint"
                    except Exception:
                        rec["bucket"] = "unknown"
                # else: leave 'unknown' — raw string equality on free text is
                # meaningless and would report ~91% disagreement.
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_parquet(CACHE / "human_agreement.parquet", index=False)
    return df


def difficulty_ratings() -> pd.DataFrame:
    """Per-paper human/LLM difficulty ratings (1-5). These are per study x rater,
    NOT per item — so RQ1.7 can only be answered at the study level as designed."""
    raw = raw_long()
    g = raw.groupby("study_ID").agg(
        diff_human_mean=("diff_human", "mean"),
        diff_human_n=("diff_human", "count"),
        diff_human_range=("diff_human", lambda s: s.max() - s.min()),
        diff_llm_mean=("diff_llm", "mean"),
        diff_llm_n=("diff_llm", "count"),
    ).reset_index().rename(columns={"study_ID": "doc_name"})
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-oe", action="store_true",
                    help="use the LLM judge for open-ended agreement (makes API calls)")
    a = ap.parse_args()

    df = build(judge_oe=a.judge_oe)
    df.to_csv(TABLES / "T4_human_agreement.csv", index=False)
    dd = difficulty_ratings()
    dd.to_csv(TABLES / "T4_difficulty_ratings.csv", index=False)

    ev = df[df.n_raters >= 2]
    print(f"paper x item pairs with >=2 raters: {len(ev):,} "
          f"({df.doc_name.nunique()} papers x {df.query_id.nunique()} items)")
    known = ev[ev.bucket != "unknown"]
    print(f"  resolvable: {len(known):,}   unknown: {len(ev) - len(known):,}")
    print(f"  OVERALL DISAGREEMENT: {1 - known.agree.mean():.3%}")

    print("\n--- by question type ---")
    print(f"{'type':>16} {'pairs':>6} {'unknown':>8} {'disagree':>9}")
    for t, g in ev.groupby("type"):
        k = g[g.bucket != "unknown"]
        d = f"{1 - k.agree.mean():.1%}" if len(k) else "n/a"
        print(f"{str(t):>16} {len(g):>6} {int((g.bucket=='unknown').sum()):>8} {d:>9}")

    print("\n--- MC buckets ---")
    mc = ev[ev.type == "multiple_choice"]
    for b, n in mc.bucket.value_counts().items():
        print(f"   {b:>9}: {n:>5} ({n/len(mc):.1%})")

    print("\n--- hardest items for humans (top 8) ---")
    it = (known.groupby(["query_id", "label"]).agree.agg(["mean", "size"])
          .sort_values("mean").head(8))
    for (qid, lab), r in it.iterrows():
        print(f"   {qid:>7} {lab:<16} disagreement {1-r['mean']:.1%}  (n={int(r['size'])})")

    print(f"\ndifficulty ratings: {dd.diff_human_n.gt(0).sum()} papers with >=1 rating, "
          f"{dd.diff_human_n.ge(2).sum()} with 2")
    print(f"wrote {TABLES/'T4_human_agreement.csv'}\n      {TABLES/'T4_difficulty_ratings.csv'}")


if __name__ == "__main__":
    main()
