"""Points lost to answer FORMATTING: every wrong non-open-ended cell whose raw text
changes under a lenient clean-up (code fences, 'Answer:' prefixes, duplicated lines,
trailing punctuation) is re-scored with the deterministic scorer; a flip to correct
is a point the strict parse lost. Excludes the Aug-4/5 output-blackout batch.
usage: pipeline_env/bin/python analysis/hpo_analysis/format_loss.py"""
import contextlib, io, json, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from _study_record import jsonio, load_studies, matched_folder  # noqa: E402
from analysis.hpo_analysis import unified_rescore as ur  # noqa: E402


def clean(s):
    s = str(s).strip()
    s = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", s).strip()
    s = re.sub(r"^(final answer|answer|output|result)\s*[:\-]\s*", "", s, flags=re.I).strip()
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    return (lines[0] if lines else s).strip(" .\"'`*")


def main():
    ev, Q, TRUE, CUR = ur._evaluator()
    S = load_studies(("phase1", "phase2", "open_tpe", "consumer_tpe"))
    tot, wrong, flips = Counter(), Counter(), Counter()
    for db in S:
        for t in S[db].trials:
            if t.state.name != "COMPLETE" or t.value is None: continue
            if db != "phase1" and ("phase1_trial" in t.user_attrs or "seeded_from_trial" in t.user_attrs): continue
            if db == "phase1" and t.number in range(382, 399): continue
            p = t.params; f = matched_folder(t.number, dict(p))
            if f is None: continue
            mode = "rag" if p.get("mode") == "rag" else (("batched_native(P1)" if p.get("question_mode") == "batched" else "individual(P1)") if db == "phase1" else "batched(P2)")
            for c in jsonio.load(f / "trial_summary.json").get("per_query_doc_results", []):
                qid = str(c.get("query_id")); typ = (Q.get(qid) or {}).get("type")
                if typ in (None, "open_ended"): continue
                tot[mode] += 1
                try: acc = float(c.get("accuracy"))
                except Exception: acc = None
                raw = c.get("raw_output")
                if raw is None or acc != 0: continue
                wrong[mode] += 1
                cl = clean(raw)
                if not cl or cl == str(raw).strip(): continue
                with contextlib.redirect_stdout(io.StringIO()):
                    try: r = ev(cl, question_type=typ, doc_name=str(c.get("doc_name")), query_id=qid, TRUE_ANSWERS=TRUE, QUERIES=Q, other_callback=None, curated_mc_items=CUR)
                    except Exception: continue
                flips[mode] += int(r.get("accuracy") == 1)
    out = {m: {"cells": tot[m], "wrong": wrong[m], "recoverable": flips[m], "pp_of_accuracy": round(flips[m] / tot[m] * 100, 2)} for m in tot}
    json.dump(out, open(ROOT / "analysis/tables/hpo/format_loss.json", "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
