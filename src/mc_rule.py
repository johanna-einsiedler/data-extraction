"""Shared revised-scoring rules, used by BOTH the live scorer (evaluate.py) and the
offline re-scorer (archive/superseded_2026-09/rescore_seed.py) so there is one
verified source of truth.

  * MC non-curated : I ⊆ P ⊆ U  (raters' intersection ⊆ prediction ⊆ union), letters
                     only; gold-NA (both raters abstained) -> model must abstain.
  * MC curated {3.1,4.1,4.2,4.3} : strict exact set-match, letters only.
  * dependency-gated numerics (1.2.1,4.2.1) & list (0.7.3) : gold-NA + model-abstain -> 1.

`I`/`U` come from the per-rater raw data, mapped to letters via evaluate.map_values_to_letters
(which applies the reviewed gold map). evaluate's functions are imported lazily to
avoid an import cycle.
"""
import re
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_RAW = None
_RAW_COLS = None
_BY_STUDY = None            # {study_id: sub-DataFrame} — built once, avoids per-cell scans

CURATED_MC = {"3.1", "4.1", "4.2", "4.3"}
DEP_NUMERIC = {"1.2.1", "4.2.1"}
# Dependency-gated MULTIPLE-CHOICE items (per src/dependencies.txt). When the
# parent condition makes the item not-applicable the finalized gold is NA
# (e.g. ['0']); a model abstention should then count correct — the raw-rater
# agreement band in score_mc doesn't see the resolved NA gold, so these need the
# same na_pair reconciliation as DEP_NUMERIC.
DEP_MC = {"3.2.1", "3.4.1", "3.5", "3.5.1"}
NA_SENT = {"-99", "-96", "-97", "0", "", "nan", "none", "z",
           "not reported", "not applicable"}


def _load_raw():
    global _RAW, _RAW_COLS, _BY_STUDY
    if _RAW is None:
        _RAW = pd.read_csv(_ROOT / "data" / "true_labels" / "01_study_data_long_raw.csv",
                           dtype=str)
        _RAW_COLS = {re.sub(r"\s+", " ", c).strip(): c for c in _RAW.columns}
        _BY_STUDY = {sid: sub for sid, sub in _RAW.groupby(_RAW.study_ID.astype(str))}
    return _RAW, _RAW_COLS


def _norm_scalar(x):
    if isinstance(x, (list, tuple)):
        x = ",".join(str(t) for t in x)
    return str(x).strip().strip("[]'\" ").lower()


def is_na(x):
    return _norm_scalar(x) in NA_SENT


def _rater_letters(cell, item_id, choices):
    from evaluate import map_values_to_letters, normalize_text
    s = str(cell).strip()
    if s.lower() in NA_SENT:
        return frozenset()
    toks = ["Other" if t.strip().lower().startswith("other") else t.strip()
            for t in s.split(",") if t.strip()]
    lets = map_values_to_letters([normalize_text(t) for t in toks], choices, item_id=item_id)
    return frozenset(l for l in lets if l != "Z")


def rater_iu(doc_name, item_id, item_label, choices):
    """(I, U) letter sets from the paper's raters; empty if the item is absent."""
    _load_raw()
    col = _RAW_COLS.get(re.sub(r"\s+", " ", str(item_label)).strip())
    sub = _BY_STUDY.get(str(doc_name))
    if col is None or sub is None:
        return frozenset(), frozenset()
    rows = sub[col].tolist()
    sets = [_rater_letters(c, item_id, choices) for c in rows]
    if not sets:
        return frozenset(), frozenset()
    return frozenset.intersection(*sets), frozenset.union(*sets)


def model_abstains(pred_letters, pred_value):
    P = set(pred_letters)
    return (not P) or (P == {"Z"}) or is_na(pred_value)


def score_mc(item_id, pred_letters, true_letters, pred_value,
             doc_name, item_label, choices):
    """Revised letters-only MC primary score (0/1)."""
    P = set(pred_letters)
    if item_id in CURATED_MC:
        gold = set(true_letters)
        if gold != {"Z"}:
            gold.discard("Z")
        Pc = set(P)
        if Pc != {"Z"}:
            Pc.discard("Z")
        if not gold or gold == {"Z"}:
            return int(not Pc or Pc == {"Z"})
        return int(Pc == gold)
    I, U = rater_iu(doc_name, item_id, item_label, choices)
    if not U:                              # both raters abstained -> not applicable
        return int(model_abstains(P, pred_value))
    Pc = frozenset(x for x in P if x != "Z")
    return int(I <= Pc <= U)


def na_pair(pred_value, true_value):
    """True when a dependency-gated numeric / list cell is gold-NA + model-abstain."""
    return is_na(pred_value) and is_na(true_value)


# --------------------------------------------------------------------------
# Dependency children (prereg2 "Conventions shared by all item types"): the coded
# dataset marks a child "not applicable" with -99 for the numeric children and
# with 0 for the multiple-choice children (shared with "not reported").
# --------------------------------------------------------------------------
def child_gold_not_applicable(qtype, gold):
    """Does the gold mark a dependency child not applicable?"""
    g = _norm_scalar(gold[0] if isinstance(gold, (list, tuple)) and gold else gold)
    if g in ("-99", "-99.0", "not applicable"):
        return True
    return qtype == "multiple_choice" and g in ("0", "")


# --------------------------------------------------------------------------
# Numeric items (prereg2 "Numeric items: precision-tolerant matching"): round
# both values, half away from zero, to the precision of the less precise of the
# two, provided that value carries at least two significant digits; a value with
# a single significant digit must match exactly; integers match exactly. The
# rule is symmetric in prediction and gold.
# --------------------------------------------------------------------------
_NUM_TOKEN = re.compile(r"-?\d*\.?\d+")


def parse_numeric(s):
    """(value, decimals, significant_digits) from the first numeric token after
    stripping %, units and thousands separators; None for 'Z' / unparseable."""
    s = str(s).strip()
    if s == "" or s.upper() == "Z" or s.lower() in ("none", "nan"):
        return None
    m = _NUM_TOKEN.search(s.replace(",", ""))
    if not m:
        return None
    tok = m.group()
    dec = len(tok.split(".")[1]) if "." in tok else 0
    digits = tok.lstrip("-").replace(".", "").lstrip("0")
    # Trailing zeros of an integer count as significant ("50" -> 2); a bare zero
    # has one significant digit.
    sig = len(digits) if digits else 1
    return float(tok), dec, sig


def _round_half_away(x, ndp):
    from decimal import Decimal, ROUND_HALF_UP
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-ndp), rounding=ROUND_HALF_UP)


def numeric_match(pred, gold):
    """0/1 under the registered precision-tolerant rule. Both arguments are raw
    strings (or numbers); abstentions / -99 are handled by the caller."""
    pp, gg = parse_numeric(pred), parse_numeric(gold)
    if pp is None or gg is None:
        return 0
    (pv, pdec, psig), (gv, gdec, gsig) = pp, gg
    if pdec == gdec:                       # same precision (incl. two integers): exact
        return int(pv == gv)
    less_dec, less_sig = (pdec, psig) if pdec < gdec else (gdec, gsig)
    if less_sig >= 2:
        return int(_round_half_away(pv, less_dec) == _round_half_away(gv, less_dec))
    return int(pv == gv)
