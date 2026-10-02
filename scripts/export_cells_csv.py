"""Export a per-item comparison CSV from trial_logs, RE-SCORED with the current
evaluate.py (so stale stored scores/representations don't leak in).

One row per codebook item. Columns are grouped by paper; for each paper you get
the ground truth plus one column per trial (labeled by its model/mode):

  item | label | type | gold[006] | <model>·<mode>[006] ... | gold[079] | ...

Each cell is the interpreted answer, re-derived from the model's stored raw
output via evaluate.evaluate_answer:
  - multiple_choice -> selected option LETTERS (aligned with gold letters)
  - everything else -> the interpreted pred_value
Gold is the current (unescaped, letter-mapped) true answer.

  python scripts/export_cells_csv.py                 # -> cells_comparison.csv
  python scripts/export_cells_csv.py --acc           # cells hold accuracy (0/1..1)
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRIAL_LOGS = ROOT / "trial_logs"
sys.path.insert(0, str(ROOT / "src"))

import evaluate as E  # noqa: E402  (needs src on path; loads NLI model at import)
from _data_loading import CURATED_MC_ITEMS, load_true_answers_final  # noqa: E402

QUERIES = json.load(open(ROOT / "query_creation" / "queries_with_prompts.json"))
TRUE = load_true_answers_final(QUERIES)

SKIP_SENTINELS = {"not applicable", "not applicable.", ""}


def sval(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        return "; ".join(str(x) for x in v)
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        return str(int(v)) if v == int(v) else str(v)
    return str(v)


def rescore(doc, qid, qtype, raw):
    """Re-evaluate a cell from its raw model output with the current evaluate.
    Returns (pred_display, gold_display, accuracy) or None if it can't."""
    try:
        ev = E.evaluate_answer(
            str(raw), qtype, doc, qid, TRUE, QUERIES,
            curated_mc_items=CURATED_MC_ITEMS,
        )
    except Exception:
        return None
    if qtype == "multiple_choice":
        pred = ",".join(ev.get("pred_indicator", []))
        gold = ",".join(ev.get("true_indicator", []))
    else:
        pred = sval(ev.get("pred_value"))
        gold = sval(ev.get("true_value"))
    return pred, gold, ev.get("accuracy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", action="store_true", help="cells hold accuracy instead of the answer")
    ap.add_argument("--out", default=str(ROOT / "cells_comparison.csv"))
    args = ap.parse_args()

    trials = []
    gold = {}       # (doc, qid) -> gold display
    labels = {}     # qid -> (label, type)
    order = []      # qid order (first trial's first doc)
    used = {}       # column label -> count

    def short(m):
        return m.split("/")[-1]

    for summ in sorted(glob.glob(str(TRIAL_LOGS / "*" / "trial_summary.json"))):
        d = json.load(open(summ))
        params = d.get("parameters", {})
        col = f"{short(params.get('llm_model', '?'))}·{params.get('mode', '?')}"
        used[col] = used.get(col, 0) + 1
        if used[col] > 1:  # disambiguate duplicate model/mode runs
            col = f"{col}#{used[col]}"
        tdir = Path(summ).parent
        cells = {}
        for docf in sorted(glob.glob(str(tdir / "doc_*.json"))):
            dd = json.load(open(docf))
            doc = str(dd.get("doc_name"))
            for r in dd.get("results", []):
                qid = r["query_id"]
                qtype = r.get("type", "")
                raw = r.get("raw_output")
                if raw is None:
                    raw = r.get("llm_output", "")
                if isinstance(raw, list):
                    raw = ", ".join(map(str, raw))

                res = None
                if str(raw).strip().lower() not in SKIP_SENTINELS:
                    res = rescore(doc, qid, qtype, raw)
                if res is None:  # dependency-skip or eval failure -> stored values
                    pred, g, acc = sval(r.get("pred_value")), sval(r.get("true_value")), r.get("accuracy")
                else:
                    pred, g, acc = res

                cells[(doc, qid)] = acc if args.acc else pred
                gold[(doc, qid)] = g
                if qid not in labels:
                    labels[qid] = (r.get("label", ""), qtype)
                    order.append(qid)
        trials.append({"col": col, "cells": cells})

    papers = sorted({d for (d, _q) in gold})

    rows = []
    for qid in order:
        row = {"item": qid, "label": labels[qid][0], "type": labels[qid][1]}
        for p in papers:
            row[f"gold[{p}]"] = gold.get((p, qid), "")
            for t in trials:
                v = t["cells"].get((p, qid), "")
                if args.acc and isinstance(v, float):
                    v = round(v, 3)
                row[f"{t['col']}[{p}]"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({df.shape[0]} items x {df.shape[1]} cols)  [re-scored with current evaluate.py]")
    print(f"papers: {papers}")
    print("trials: " + ", ".join(t["col"] for t in trials))


if __name__ == "__main__":
    main()
