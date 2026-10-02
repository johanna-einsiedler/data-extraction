"""Export a frozen run's stored outputs for the human-agreement analyses.

usage: pipeline_env/bin/python scripts/export_predictions_for_irr.py --run-dir runs/<id>/<config> [--out-dir DIR]

Writes, into the run dir by default:
  predictions_irr.csv  study_ID, variable, value   -- the representation scripts/compute_irr.py
                       expects: single_choice = option CODE; multiple_choice = comma-joined
                       labels in the RATERS' vocabulary (most frequent raw rater token per
                       option letter, Y -> "Other", {Z} -> ""); numeric as stored
                       (4.5.x -> highestperf_<metric>); list / open_ended are not exported
                       here (compute_irr ignores those types).
  cells.csv            doc, item, label, type, raw_output, pred, gold, accuracy (+ secondary fields)
  export_report.json   counts, letter->token map provenance, unmapped letters

Scoring reruns from stored outputs are allowed by the preregistered single-touch
policy; this script never calls a model. It reads stored per-item results only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SAMPLE_ONLY", "1")
sys.path.insert(0, str(ROOT / "src"))
from evaluate import map_values_to_letters  # noqa: E402
from _data_loading import HIGHESTPERF_COLS  # noqa: E402

LONG = ROOT / "data" / "true_labels" / "01_study_data_long_raw.csv"


def load_queries() -> dict:
    return json.load(open(ROOT / "query_creation" / "queries_with_prompts.json"))


def rater_vocabulary(queries: dict) -> tuple[dict, dict]:
    """item -> {letter: most frequent raw rater token}. The raters' file is WIDE
    (one column per item label, multi-select values joined by ', '), so exported MC
    values compare raw-string-equal to what compute_irr reads from it."""
    df = pd.read_csv(LONG, dtype=str).fillna("")
    vocab, report = {}, {}
    for qid, q in queries.items():
        if q.get("type") != "multiple_choice" or q.get("label") not in df.columns:
            continue
        choices = q.get("choices") or {}
        counts: dict[str, Counter] = defaultdict(Counter)
        for raw in df[q["label"]]:
            for tok in [t.strip() for t in str(raw).split(", ") if t.strip()]:
                try:
                    letters = map_values_to_letters([tok], choices, item_id=qid)
                except Exception:
                    letters = []
                if len(letters) == 1 and letters[0] != "Y":
                    counts[letters[0]][tok] += 1
        vocab[qid] = {L: c.most_common(1)[0][0] for L, c in counts.items()}
        report[qid] = {L: dict(c.most_common(3)) for L, c in counts.items()}
    return vocab, report


def choice_value(choices: dict, letter: str) -> str:
    v = choices.get(letter)
    return (v.get("value") if isinstance(v, dict) else v) or letter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--out-dir")
    a = ap.parse_args()
    run = Path(a.run_dir); out = Path(a.out_dir) if a.out_dir else run
    out.mkdir(parents=True, exist_ok=True)
    queries = load_queries()
    vocab, vocab_report = rater_vocabulary(queries)
    cells, preds, unmapped, counts = [], [], defaultdict(set), Counter()
    for f in sorted(run.glob("doc_[0-9]*.json")):
        d = json.load(open(f)); doc = str(d.get("doc_name"))
        for r in d.get("results", []):
            qid = str(r.get("query_id")); q = queries.get(qid) or {}
            typ = q.get("type"); label = q.get("label")
            cells.append({"doc": doc, "item": qid, "label": label, "type": typ,
                          "raw_output": r.get("raw_output", r.get("llm_output")), "pred": r.get("pred_value", r.get("pred_indicator")),
                          "gold": r.get("true_value", r.get("true_indicator")), "accuracy": r.get("accuracy"),
                          "accuracy_exact": r.get("accuracy_exact"), "accuracy_jaccard": r.get("accuracy_jaccard"),
                          "judge_equivalent": r.get("judge_equivalent"), "judge_confidence": r.get("judge_confidence")})
            counts[typ] += 1
            if typ == "single_choice":
                v = r.get("pred_indicator", r.get("pred_value"))
                preds.append((doc, label, "" if v in (None, "Z") else str(v)))
            elif typ == "multiple_choice":
                letters = r.get("pred_indicator") or r.get("pred_value") or []
                if isinstance(letters, str):
                    letters = [x.strip() for x in letters.split(",") if x.strip()]
                toks = []
                for L in letters:
                    if L == "Z":
                        continue
                    if L == "Y":
                        toks.append("Other"); continue
                    tok = (vocab.get(qid) or {}).get(L)
                    if tok is None:
                        tok = choice_value(q.get("choices") or {}, L); unmapped[qid].add(L)
                    toks.append(tok)
                preds.append((doc, label, ", ".join(toks)))
            elif typ == "numeric":
                v = r.get("pred_value")
                var = HIGHESTPERF_COLS.get(qid, label) if qid.startswith("4.5") else label
                if v not in (None, "", "Z"):
                    try:
                        fv = float(v); v = str(int(fv)) if fv.is_integer() else str(fv)
                    except Exception:
                        v = str(v)
                else:
                    v = ""
                preds.append((doc, var, v))
    with open(out / "predictions_irr.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["study_ID", "variable", "value"]); w.writerows(preds)
    pd.DataFrame(cells).to_csv(out / "cells.csv", index=False)
    json.dump({"n_docs": len(list(run.glob("doc_[0-9]*.json"))), "cells_by_type": dict(counts), "n_predictions": len(preds),
               "letter_vocabulary": vocab_report, "unmapped_letters": {k: sorted(v) for k, v in unmapped.items()}},
              open(out / "export_report.json", "w"), indent=1)
    print(f"wrote {out/'predictions_irr.csv'} ({len(preds)} rows), cells.csv ({len(cells)} rows), export_report.json")


if __name__ == "__main__":
    main()
