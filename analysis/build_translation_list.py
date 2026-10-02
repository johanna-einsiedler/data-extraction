#!/usr/bin/env python
"""Build the gold/rater value -> codebook-option-letter map for every MC item, and
flag values that DON'T map to a real option (they currently collapse to phantom
'Y'/Other). The flagged list is the "translation list" to review: a value that is
really a codebook option but phrased differently (e.g. 'iml' vs 'interpretable ML')
should be added as an alias; a value with no matching option is a genuine 'Other'.

Reads the two rater sources (per-rater raw + finalized gold) so we see every
phrasing that actually occurs. Reuses evaluate.map_values_to_letters for the
match logic. Writes analysis/translation_list.csv. Read-only over the data.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import evaluate as E  # noqa: E402
from _data_loading import _unescape_unicode  # noqa: E402

RAW = ROOT / "data" / "true_labels" / "01_study_data_long_raw.csv"
FINAL = ROOT / "data" / "true_labels" / "09_data_final.csv"
QWP = ROOT / "query_creation" / "queries_with_prompts.json"

# User-reviewed aliases: gold value IS a codebook option, phrased differently ->
# force it to that option's letter instead of falling through to Other.
# {item: {raw_gold_value: option_letter}}
REVIEWED_ALIASES_RAW = {
    "2.3": {
        "Calculations of indices": "C",                                  # -> Calculation of indices
        "Extracting features from other high-dimensional data": "G",     # -> Extracting features from text/image/audio
    },
    "3.3": {
        "Optimism correction": "G",                                      # -> Optimism-corrected bootstrap
        "Single k fold": "B",                                            # -> Single k fold CV
    },
    "5.2": {
        "Caution in interpretation of IML techniques": "H",              # -> ...interpretable ML techniques
    },
}
REVIEWED_ALIASES = {
    (it, E.normalize_text(k)): v
    for it, d in REVIEWED_ALIASES_RAW.items() for k, v in d.items()
}

# Sentinels handled by _map_gold_code (not real content): -99 not-applicable,
# -96/-97 not-reported, 0 not-reported, blanks.
NA_TOKENS = {"-99", "-96", "-97", "0", "z", "not reported", "not applicable",
             "", "nan", "none"}


def other_key(choices: dict):
    """The option KEY that acts as 'Other' — its value contains the word 'other'
    (covers both 'Y: Other' and 4.7's 'C: For other reported results')."""
    for k, v in choices.items():
        if re.search(r"\bother\b", str(v), re.I):
            return k
    return None


def best_option_match(value: str, choices: dict, other_k):
    """Return (best_letter, best_value, similarity) of the closest NON-Other option
    to `value` — a high score suggests the value is really that option, phrased
    differently (an alias candidate) rather than a genuine Other."""
    from difflib import SequenceMatcher

    def toks(s):
        return set(re.sub(r"[^\w\s]", " ", E.normalize_text(s)).split())

    v_norm, v_toks = E.normalize_text(value), toks(value)
    best = (None, None, 0.0)
    for k, val in choices.items():
        if k == other_k:
            continue
        o_norm = E.normalize_text(val)
        seq = SequenceMatcher(None, v_norm, o_norm).ratio()
        jac = (len(v_toks & toks(val)) / len(v_toks | toks(val))) if (v_toks | toks(val)) else 0
        score = max(seq, jac)
        if score > best[2]:
            best = (k, val, score)
    return best


def mc_items():
    qi = json.load(open(QWP))
    items = qi.values() if isinstance(qi, dict) else qi
    out = {}
    for it in items:
        if it.get("type") != "multiple_choice":
            continue
        iid = it.get("id") or it.get("query_id")
        choices = {str(k).upper(): (v.get("value") if isinstance(v, dict) else v)
                   for k, v in (it.get("choices", {}) or {}).items()}
        out[iid] = {"label": it.get("label"), "choices": choices}
    return out


def split_values(cell):
    """Split a raw MC cell ('A, B, Other: foo') into option tokens."""
    s = str(cell).strip()
    if s.lower() in NA_TOKENS:
        return []
    parts = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        # 'Other: free text' -> the option is 'Other'
        if p.lower().startswith("other"):
            parts.append("Other")
        else:
            parts.append(p)
    return parts


def main():
    items = mc_items()
    raw = pd.read_csv(RAW, dtype=str)
    final = pd.read_csv(FINAL, dtype=str)

    rows = []          # translation-list rows
    review = defaultdict(list)   # item -> [(raw_value, status)] needing review
    for iid, meta in items.items():
        col = meta["label"]
        choices_norm = {k: {"value": v} for k, v in meta["choices"].items()}
        ok = other_key(meta["choices"])   # the item's Other-equivalent key, or None
        seen = set()
        for df in (raw, final):
            if col not in df.columns:
                continue
            for cell in df[col].dropna():
                for tok in split_values(cell):
                    seen.add(tok)
        for tok in sorted(seen):
            unesc = _unescape_unicode(tok)             # decode <U+XXXX> like the pipeline
            if unesc.strip().lower() in NA_TOKENS or unesc.strip().lower().startswith("other"):
                continue
            letter = E.map_values_to_letters([E.normalize_text(unesc)], choices_norm)
            letter = letter[0] if letter else "Y"
            alias = REVIEWED_ALIASES.get((iid, E.normalize_text(unesc)))
            if alias:
                letter = alias                       # reviewed override wins
            clean = letter in meta["choices"] and letter not in ("Y",)
            # 'Y' default means "did not match a real option"
            # Closest NON-Other codebook option for EVERY value, so the reviewer can
            # judge gold-value vs option text side by side.
            bl, bv, sim = best_option_match(unesc, meta["choices"], ok)
            if clean:
                status = "OK"
            elif sim >= 0.60:
                status = "LIKELY-ALIAS"            # probably option `bl`, phrased differently
                review[iid].append((unesc, "alias", bl, bv, sim))
            elif ok is not None:
                status = "genuine-Other"           # correctly -> Other; no action
            else:
                status = "FORCED-0 (no Other option)"
                review[iid].append((unesc, "forced0", None, None, sim))
            cur = letter if clean else (ok or "(none)")
            rows.append({
                "item": iid, "variable": col,
                "gold_value": unesc,
                "currently_maps_to": cur,
                "current_option_text": meta["choices"].get(cur, ""),
                "closest_option": bl or "",
                "closest_option_text": bv or "",
                "similarity": round(sim, 2),
                "status": status,
            })

    # Canonical mapping file the evaluation reads: exactly the 5 columns requested.
    # `maps_to` = the current mapping decision (letter); `option_text` = its text.
    canon = pd.DataFrame(rows)[
        ["item", "variable", "gold_value", "currently_maps_to", "current_option_text"]
    ].rename(columns={"currently_maps_to": "maps_to", "current_option_text": "option_text"})
    canon = canon.drop_duplicates().sort_values(["item", "gold_value"])
    out = ROOT / "analysis" / "translation_list.csv"
    canon.to_csv(out, index=False)
    # richer review copy (with closest_option + similarity) alongside, for auditing
    pd.DataFrame(rows).to_csv(ROOT / "analysis" / "translation_list_annotated.csv", index=False)

    print(f"MC items: {len(items)} | distinct value tokens examined: {len(rows)}")
    counts = defaultdict(int)
    for r in rows:
        counts[r["status"].split(":")[0].split(" ")[0]] += 1
    print("status counts:", dict(counts))

    print("\n=== LIKELY ALIASES — value closely matches a listed option (CONFIRM) ===")
    print("(these currently fall through to 'Other' but probably ARE the option shown)")
    for iid in items:
        aliases = [r for r in review.get(iid, []) if r[1] == "alias"]
        if not aliases:
            continue
        print(f"\n  {iid} ({items[iid]['label']})")
        for val, _, bl, bv, sim in sorted(aliases, key=lambda x: -x[4]):
            print(f"     {sim:.2f}  {val!r}  ->  '{bl}': {bv!r}")

    forced = [(iid, r) for iid in items for r in review.get(iid, []) if r[1] == "forced0"]
    if forced:
        print("\n=== FORCED-0 (item has NO Other option; value can't map) ===")
        for iid, r in forced:
            print(f"  {iid} ({items[iid]['label']}): {r[0]!r}")

    print(f"\nFull map written: {out}")
    n_alias = sum(1 for iid in items for r in review.get(iid, []) if r[1] == "alias")
    n_other = sum(1 for r in rows if r["status"] == "genuine-Other")
    print(f"Summary: {counts.get('OK',0)} map cleanly | {n_alias} LIKELY-ALIAS to confirm | "
          f"{n_other} genuine-Other (no action) | {len(forced)} forced-0")


if __name__ == "__main__":
    main()
