"""Classify every stored cell of every COMPLETE trial by HOW it failed (if it did):
missing output, refusal text, abstention (gold has / has no answer), format-invalid
(by item type), well-formed wrong, correct. Also sums the batched parse-health
counters. Writes analysis/tables/hpo/answer_failure_modes_{cells,by_group}.csv.
usage: pipeline_env/bin/python analysis/hpo_analysis/answer_failure_modes.py"""
import json, re, sys
from collections import Counter
from pathlib import Path
import pandas as pd, optuna
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from _study_record import load_studies, matched_folder, jsonio  # noqa: E402
optuna.logging.set_verbosity(optuna.logging.WARNING)
Q = json.load(open(ROOT / "query_creation/queries_with_prompts.json"))
REFUSAL = re.compile(r"\b(i cannot|i can't|i can not|i am unable|i'm unable|unable to (determine|provide|answer|find)|as an ai|i'm sorry|i am sorry|cannot (determine|provide|be determined)|not able to|i do not have|i don't have access)\b", re.I)
ABSTAIN = {"z", "n/a", "na", "none", "not stated", "not reported", "not mentioned", "unknown", "not specified", "-", "null", "nan"}
NA_GOLD = {"", "z", "nan", "none", "[]", "['0']", "0", "0.0", "['z']", "[0]", "-99", "-99.0", "['-99']"}


def classify(c):
    typ = c.get("type") or (Q.get(str(c.get("query_id"))) or {}).get("type")
    raw = c.get("raw_output"); acc = c.get("accuracy")
    try:
        acc = float(acc)
    except Exception:
        acc = None
    if raw is None or str(raw).strip() == "":
        # no model call was made (dependency child auto-resolved) vs output genuinely absent
        return typ, ("no_call_autoresolved" if acc == 1 else "missing_output"), acc
    s = str(raw).strip()
    if s.lower() == "not applicable":
        # written by the pipeline for a skipped dependency-child item, not by the model
        return typ, ("dependency_placeholder_correct" if acc == 1 else "dependency_placeholder_wrong"), acc
    if REFUSAL.search(s) and typ != "open_ended":
        return typ, "refusal_text", acc
    low = s.lower().strip(" .\"'[]")
    if low in ABSTAIN:
        gold_na = str(c.get("true_value")).strip().lower() in NA_GOLD or str(c.get("true_indicator")).strip().lower() in ("['z']", "z", "none", "nan", "[]")
        return typ, ("abstain_correct" if acc == 1 else "abstain_gold_has_answer" if not gold_na else "abstain_scored_wrong"), acc
    choices = set(((Q.get(str(c.get("query_id"))) or {}).get("choices") or {}).keys())
    if typ == "single_choice":
        first = re.split(r"[;,:\s\-—]", s, maxsplit=1)[0].strip(" .\"'").upper()
        if choices and first not in choices and first not in {x.upper() for x in choices}:
            return typ, "format_invalid", acc
        if len(s) > 12:
            return typ, ("correct_with_explanation" if acc == 1 else "wrong_with_explanation"), acc
    elif typ == "multiple_choice":
        pi = str(c.get("pred_indicator"))
        if pi in ("None", "[]", "", "nan"):
            return typ, "format_invalid", acc
    elif typ == "numeric":
        if not re.search(r"\d", s):
            return typ, "format_invalid", acc
        if len(s) > 25:
            return typ, ("correct_with_explanation" if acc == 1 else "wrong_with_explanation"), acc
    return typ, ("correct" if acc == 1 else "wrong_wellformed"), acc


def main():
    S = load_studies(("phase1", "phase2", "open_tpe", "consumer_tpe"))
    rows, health, examples = [], [], Counter()
    for db in ("phase1", "phase2", "open_tpe", "consumer_tpe"):
        for t in S[db].trials:
            if t.state.name != "COMPLETE" or t.value is None: continue
            if db != "phase1" and ("phase1_trial" in t.user_attrs or "seeded_from_trial" in t.user_attrs): continue
            p = t.params; f = matched_folder(t.number, dict(p))
            if f is None: continue
            if p.get("mode") == "rag": mode = "rag"
            elif db == "phase1": mode = "batched_native(P1)" if p.get("question_mode") == "batched" else "individual(P1)"
            else: mode = "batched(P2)"
            if db == "phase1" and p.get("mode") == "rag": mode = "rag"
            model = str(p.get("llm_model")).split("/")[-1]
            ua = t.user_attrs
            health.append(dict(db=db, n=t.number, model=model, mode=mode, parse_failed=ua.get("batched_parse_failed"),
                               truncated=ua.get("batched_truncated"), split_used=ua.get("batched_split_used"),
                               other_without_y=ua.get("batched_other_without_y")))
            if db == "phase1" and t.number in range(382, 399):
                continue  # output blackout batch: nothing stored, not a model failure
            cells = jsonio.load(f / "trial_summary.json").get("per_query_doc_results", [])
            cnt = Counter(); per_doc = {}
            for c in cells:
                typ, cat, acc = classify(c); cnt[(typ, cat)] += 1
                pd_ = per_doc.setdefault(str(c.get("doc_name", c.get("doc_id"))), [0, 0])
                pd_[0] += 1; pd_[1] += cat in ("abstain_gold_has_answer", "abstain_correct", "abstain_scored_wrong", "missing_output", "format_invalid")
                if cat in ("refusal_text", "format_invalid") and len(examples) < 4000:
                    examples[(model, typ, cat, str(c.get("raw_output"))[:90])] += 1
            collapsed = sum(1 for tot, bad in per_doc.values() if tot >= 20 and bad / tot >= 0.9)
            health[-1].update(papers=len(per_doc), collapsed_papers=collapsed)
            for (typ, cat), k in cnt.items():
                rows.append(dict(db=db, n=t.number, model=model, mode=mode, type=typ, cat=cat, k=k))
    d = pd.DataFrame(rows); h = pd.DataFrame(health)
    out = ROOT / "analysis/tables/hpo"
    d.to_csv(out / "answer_failure_modes_cells.csv", index=False); h.to_csv(out / "answer_failure_modes_health.csv", index=False)
    pd.DataFrame([dict(model=k[0], type=k[1], cat=k[2], raw=k[3], n=v) for k, v in examples.most_common(400)]).to_csv(out / "answer_failure_modes_examples.csv", index=False)
    return d, h, examples


if __name__ == "__main__":
    d, h, ex = main()
    pd.set_option("display.width", 230); pd.set_option("display.max_rows", 200)
    order = ["correct", "correct_with_explanation", "abstain_correct", "no_call_autoresolved", "dependency_placeholder_correct",
             "wrong_wellformed", "wrong_with_explanation", "abstain_gold_has_answer", "abstain_scored_wrong",
             "dependency_placeholder_wrong", "format_invalid", "refusal_text", "missing_output"]
    def share(g):
        t = g.groupby("cat").k.sum(); t = (t / t.sum() * 100).reindex(order).fillna(0).round(2); t["cells"] = int(g.k.sum()); return t
    print("=== share of cells by outcome, per mode (%) ==="); print(d.groupby("mode").apply(share).to_string())
    print("\n=== non-open-ended failure modes per item type (all modes, %) ==="); print(d.groupby("type").apply(share)[["abstain_gold_has_answer", "format_invalid", "refusal_text", "missing_output", "wrong_wellformed", "cells"]].to_string())
    b = d[d["mode"] == "batched(P2)"]
    print("\n=== Phase-2 batched, by model (%) ==="); print(b.groupby("model").apply(share)[["correct", "wrong_wellformed", "abstain_gold_has_answer", "format_invalid", "refusal_text", "missing_output", "cells"]].sort_values("abstain_gold_has_answer", ascending=False).to_string())
    print("\n=== batched parse health (Phase-2 natives): papers per counter ==="); hb = h[h["mode"] == "batched(P2)"]
    print(hb.groupby("model")[["papers", "parse_failed", "truncated", "split_used", "collapsed_papers", "other_without_y"]].sum().assign(trials=hb.groupby("model").size()).to_string())
    print("collapsed papers by mode:", h.groupby("mode")[["papers", "collapsed_papers"]].sum().to_dict("index"))
    print("\n=== frozen pipelines ===")
    for db, n in (("phase2", 131), ("phase2", 109), ("consumer_tpe", 62)):
        g = d[(d.db == db) & (d.n == n)]; print(f"{db}#{n}:", share(g).to_dict())
    print("\n=== most frequent refusal / format-invalid raw outputs ===")
    for (model, typ, cat, raw), v in ex.most_common(14): print(f"  {v:5d}  {cat:15} {typ:15} {model[:22]:22} {raw!r}")
