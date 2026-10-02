"""Post-hoc inter-rater reliability (Krippendorff's alpha + percent agreement).

Computes reliability for three "rater" configurations, per the study design:

  (a) raters       - the original human coders of each double-coded paper
                     (inter-coder reliability).
  (b) raters_gold  - the original coders PLUS the adjudicated consensus gold
                     (09_data_final) as an additional "third rater".
  (c) model        - the original coders PLUS the optimized pipeline's
                     predictions as an additional rater (RQ1.1). Requires a
                     predictions CSV; skipped gracefully if not supplied.

Alpha is computed per codebook variable with a distance appropriate to its type:
  * single_choice / numeric  -> nominal distance
  * multiple_choice          -> Jaccard set-distance (partial credit), with all
                                free "Other" entries collapsed to a single
                                "Other" level (as in the human IRR table).

Inputs (data/true_labels/, produced by scripts/export_true_labels.R):
  01_study_data_long_raw.csv   one row per (study, rater)
  09_data_final.csv            adjudicated consensus gold (one row per study)

Usage:
  python scripts/compute_irr.py                      # configs (a) and (b)
  python scripts/compute_irr.py --predictions preds.csv   # also (c)
  python scripts/compute_irr.py --out results/irr.csv

The predictions CSV for (c) must have columns: study_ID, variable, value
(value in the same representation as the gold: SC=code, MC=comma-joined labels).
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRUE_LABELS = PROJECT_ROOT / "data" / "true_labels"
LONG_CSV = TRUE_LABELS / "01_study_data_long_raw.csv"
FINAL_CSV = TRUE_LABELS / "09_data_final.csv"
QUERY_INFO = PROJECT_ROOT / "query_creation" / "query_info.json"

# Missing / structural codes that should not count as a substantive category.
_MISSING = {"", "nan", "-96", "-97", "-99", "0"}


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def nominal_distance(a, b) -> float:
    return 0.0 if a == b else 1.0


def jaccard_distance(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return 1.0 - (len(a & b) / union) if union else 0.0


# ---------------------------------------------------------------------------
# Krippendorff's alpha (general form, any distance; handles variable #coders)
# ---------------------------------------------------------------------------
def krippendorff_alpha(units: dict, distance) -> float | None:
    """units: {unit_id: [value, value, ...]} with >=2 values per usable unit.

    Uses the pooled-coincidence formulation (Krippendorff 2011):
      Do = sum_u [ 1/(m_u-1) * sum_{i!=j} delta(v_i,v_j) ] / sum_u m_u
      De = sum_{a,b over all pooled values} delta(a,b) / (N*(N-1))
      alpha = 1 - Do/De
    Returns None if fewer than 2 units or De == 0 (no observable disagreement).
    """
    usable = {u: vs for u, vs in units.items() if len(vs) >= 2}
    if len(usable) < 2:
        return None

    pooled = [v for vs in usable.values() for v in vs]
    N = len(pooled)
    if N < 2:
        return None

    # Observed disagreement
    do_num = 0.0
    for vs in usable.values():
        m = len(vs)
        pair_sum = sum(
            distance(vs[i], vs[j]) for i in range(m) for j in range(m) if i != j
        )
        do_num += pair_sum / (m - 1)
    Do = do_num / N

    # Expected disagreement (all ordered pairs of pooled values)
    de_num = sum(
        distance(pooled[i], pooled[j])
        for i in range(N)
        for j in range(N)
        if i != j
    )
    De = de_num / (N * (N - 1))

    if De == 0:
        return 1.0 if Do == 0 else None
    return 1.0 - Do / De


def percent_agreement(units: dict, equal) -> float | None:
    """Mean pairwise agreement across all coder pairs within each unit."""
    agree = total = 0
    for vs in units.values():
        if len(vs) < 2:
            continue
        for a, b in combinations(vs, 2):
            total += 1
            agree += int(equal(a, b))
    return agree / total if total else None


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------
def parse_mc(raw) -> frozenset:
    """Comma-joined MC labels -> frozenset; free 'Other' entries collapse to
    a single 'Other' level; missing codes -> empty set."""
    s = "" if raw is None else str(raw).strip()
    if s in _MISSING:
        return frozenset()
    out = set()
    for tok in s.split(","):
        t = tok.strip()
        if not t:
            continue
        out.add("Other" if t.lower().startswith("other") else t)
    return frozenset(out)


def norm_sc(raw) -> str:
    s = "" if raw is None else str(raw).strip()
    return "MISSING" if s.lower() in _MISSING else s


# ---------------------------------------------------------------------------
# Variable type map (from query_info.json label -> type)
# ---------------------------------------------------------------------------
def variable_types() -> dict:
    info = json.loads(QUERY_INFO.read_text())
    types = {}
    for e in info:
        label = str(e.get("label", "")).strip()
        if label:
            types[label] = e["type"]
    return types


# ---------------------------------------------------------------------------
# Build per-variable coder values for a configuration
# ---------------------------------------------------------------------------
def build_units(long_df, gold_df, variable, vtype, add_gold, model_df=None):
    """Return {study_id: [coder_value, ...]} for one variable."""
    if variable not in long_df.columns:
        return {}
    parse = parse_mc if vtype == "multiple_choice" else norm_sc

    units: dict[str, list] = {}
    for sid, grp in long_df.groupby("study_ID"):
        vals = [parse(v) for v in grp[variable].tolist()]
        units[sid] = vals

    if add_gold and gold_df is not None and variable in gold_df.columns:
        gmap = dict(zip(gold_df["study_ID"], gold_df[variable]))
        for sid in units:
            if sid in gmap:
                units[sid].append(parse(gmap[sid]))

    if model_df is not None:
        msub = model_df[model_df["variable"] == variable]
        mmap = dict(zip(msub["study_ID"], msub["value"]))
        for sid in units:
            if sid in mmap:
                units[sid].append(parse(mmap[sid]))

    # keep only double-coded-style units (>=2 coders)
    return {u: vs for u, vs in units.items() if len(vs) >= 2}


def run_config(name, long_df, gold_df, vtypes, add_gold, model_df=None):
    rows = []
    for variable, vtype in vtypes.items():
        if vtype not in ("single_choice", "multiple_choice", "numeric"):
            continue
        units = build_units(long_df, gold_df, variable, vtype, add_gold, model_df)
        if len(units) < 2:
            continue
        if vtype == "multiple_choice":
            dist, eq = jaccard_distance, lambda a, b: a == b
        else:
            dist, eq = nominal_distance, lambda a, b: a == b
        rows.append(
            {
                "config": name,
                "variable": variable,
                "type": vtype,
                "n_units": len(units),
                "alpha": krippendorff_alpha(units, dist),
                "pct_agreement": percent_agreement(units, eq),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest():
    # nominal: 1 agreeing unit + 1 disagreeing unit -> alpha 0
    u = {"1": ["a", "a"], "2": ["a", "b"]}
    assert abs(krippendorff_alpha(u, nominal_distance) - 0.0) < 1e-9
    # perfect agreement -> alpha 1
    u2 = {"1": ["a", "a"], "2": ["b", "b"]}
    assert abs(krippendorff_alpha(u2, nominal_distance) - 1.0) < 1e-9
    # jaccard: identical sets -> alpha 1
    u3 = {"1": [frozenset({"x"}), frozenset({"x"})],
          "2": [frozenset({"y"}), frozenset({"y"})]}
    assert abs(krippendorff_alpha(u3, jaccard_distance) - 1.0) < 1e-9
    print("[selftest] Krippendorff alpha OK (nominal 0.0/1.0, jaccard 1.0)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=str, default=None,
                    help="CSV with columns study_ID,variable,value for config (c)")
    ap.add_argument("--out", type=str, default=str(TRUE_LABELS / "irr_results.csv"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    _selftest()
    if args.selftest:
        return

    long_df = pd.read_csv(LONG_CSV, dtype=str, keep_default_na=False)
    gold_df = pd.read_csv(FINAL_CSV, dtype=str, keep_default_na=False)
    vtypes = variable_types()

    all_rows = []
    all_rows += run_config("raters", long_df, gold_df, vtypes, add_gold=False)
    all_rows += run_config("raters_gold", long_df, gold_df, vtypes, add_gold=True)

    if args.predictions:
        model_df = pd.read_csv(args.predictions, dtype=str, keep_default_na=False)
        all_rows += run_config("model", long_df, gold_df, vtypes,
                               add_gold=False, model_df=model_df)
        all_rows += run_config("model_gold", long_df, gold_df, vtypes,
                               add_gold=True, model_df=model_df)

    res = pd.DataFrame(all_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(res)} rows)")
    # summary by config
    summ = res.groupby(["config", "type"]).agg(
        n_vars=("variable", "count"),
        mean_alpha=("alpha", "mean"),
        mean_pct=("pct_agreement", "mean"),
    )
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
