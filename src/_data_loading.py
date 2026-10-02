"""Data loading utilities: queries, true answers, outcomes, and document text."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUERIES_JSON_PATH = PROJECT_ROOT / "query_creation" / "queries_with_prompts.json"
OUTCOMES_PATH = PROJECT_ROOT / "data" / "true_labels" / "outcomes.json"

try:
    with open(OUTCOMES_PATH, "r", encoding="utf-8") as _f:
        OUTCOMES = json.load(_f)
except FileNotFoundError:
    OUTCOMES = {}
    print(
        f"[_data_loading] Warning: outcomes file not found at {OUTCOMES_PATH}. "
        "Proceeding with empty outcomes map."
    )


# ---------------------------------------------------------------------------
# Study-level outcome + sample from CSV
# ---------------------------------------------------------------------------
STUDY_INFO_CSV_PATH = (
    PROJECT_ROOT / "data" / "true_labels" / "study_outcomes_samples.csv"
)


def _parse_study_info_field(info_str: str) -> dict:
    """Parse a semicolon-separated info string into outcome / sample.

    Returns {"outcome": str | None, "sample": str | None}.
    """
    info = info_str.strip().rstrip(";").strip()
    if not info:
        return {"outcome": None, "sample": None}

    outcome = None
    sample = None

    parts = [p.strip() for p in info.split(";") if p.strip()]

    for part in parts:
        pl = part.lower().strip()
        if pl.startswith("outcome"):
            val = re.sub(r"^outcome\s*[/:]?\s*", "", part, flags=re.IGNORECASE).strip()
            if val and outcome is None:
                outcome = val
        elif pl.startswith("sample"):
            val = re.sub(r"^sample\s*[/:]?\s*", "", part, flags=re.IGNORECASE).strip()
            if val and sample is None:
                sample = val
        elif "outcome:" in pl:
            m = re.search(r"outcome:\s*(.+)", part, re.IGNORECASE)
            if m and outcome is None:
                outcome = m.group(1).strip()
        elif "sample:" in pl:
            m = re.search(r"sample:\s*(.+)", part, re.IGNORECASE)
            if m and sample is None:
                sample = m.group(1).strip()

    return {"outcome": outcome, "sample": sample}


def load_study_info(
    csv_path: Path = STUDY_INFO_CSV_PATH,
) -> dict[str, dict]:
    """Load per-study outcome and sample specifications from CSV.

    Returns
    -------
    dict
        study_id → {"outcome": str | None, "sample": str | None}
    """
    result: dict[str, dict] = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(
            f"[_data_loading] Warning: study info CSV not found at {csv_path}. "
            "Proceeding with empty study info."
        )
        return result

    for line in lines[1:]:  # skip header
        parts = line.strip().split(",", 1)
        if len(parts) < 2:
            continue
        sid = parts[0].strip().strip('"')
        info_raw = parts[1].strip().strip('"').replace('""', '"')
        parsed = _parse_study_info_field(info_raw)
        if parsed["outcome"] or parsed["sample"]:
            result[sid] = parsed

    return result


# Module-level load (lazy-safe: dict is empty if file missing)
STUDY_INFO = load_study_info()

# System prompt template
_SYSTEM_PROMPT_TEMPLATE_PATH = (
    PROJECT_ROOT / "query_creation" / "system_prompt_template.txt"
)
try:
    with open(_SYSTEM_PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as _f:
        _SYSTEM_PROMPT_TEMPLATE = _f.read()
except FileNotFoundError:
    _SYSTEM_PROMPT_TEMPLATE = None

# Separate system prompt for open-ended items (prereg Prompt 5): concise
# factual descriptions rather than the coding-scheme framing.
_SYSTEM_PROMPT_TEMPLATE_OE_PATH = (
    PROJECT_ROOT / "query_creation" / "system_prompt_template_oe.txt"
)
try:
    with open(_SYSTEM_PROMPT_TEMPLATE_OE_PATH, "r", encoding="utf-8") as _f:
        _SYSTEM_PROMPT_TEMPLATE_OE = _f.read()
except FileNotFoundError:
    _SYSTEM_PROMPT_TEMPLATE_OE = None


def build_system_prompt(
    study_id: str,
    study_info: dict | None = None,
    *,
    fallback_single_prompt: str | None = None,
    fallback_multi_prompt: str | None = None,
    oe: bool = False,
) -> str:
    """Build the appropriate system prompt for a study.

    Uses the new unified template when available, with a {FOCUS_SECTION}
    placeholder that is filled depending on which specifications exist.

    Falls back to the legacy single/multiple-outcome prompts when the
    template is missing.

    Parameters
    ----------
    study_id : str
        Numeric study identifier (e.g. "006").
    study_info : dict, optional
        Override for STUDY_INFO lookup.  Keys: "outcome", "sample".
    fallback_single_prompt : str, optional
        Legacy single-outcome prompt text (used when template is missing).
    fallback_multi_prompt : str, optional
        Legacy multiple-outcomes prompt text with ``{OUTCOME}`` placeholder.
    """
    if study_info is None:
        study_info = STUDY_INFO.get(study_id, {})

    outcome = study_info.get("outcome")
    sample = study_info.get("sample")

    # --- Build using unified template (OE variant for open-ended items) ---
    template = (
        _SYSTEM_PROMPT_TEMPLATE_OE
        if (oe and _SYSTEM_PROMPT_TEMPLATE_OE is not None)
        else _SYSTEM_PROMPT_TEMPLATE
    )
    if template is not None:
        if outcome and sample:
            focus = (
                "All decisions and coding should be made with respect to the "
                "following analytic unit:\n\n"
                f"  👉 Chosen outcome: {outcome}\n"
                f"  👉 Chosen sample:  {sample}"
            )
        elif outcome:
            focus = (
                "All decisions and coding should be made with respect to the "
                "following analytic unit / outcome of interest:\n\n"
                f"  👉 Chosen outcome: {outcome}"
            )
        elif sample:
            focus = (
                "All decisions and coding should be made with respect to the "
                "following specific sample:\n\n"
                f"  👉 Chosen sample: {sample}"
            )
        else:
            focus = (
                "In this case, the study reports only one analytic unit or "
                "outcome, so no specific outcome or sample needs to be selected.\n"
                "All decisions and coding should therefore be made with respect "
                "to this single outcome."
            )
        return template.replace("{FOCUS_SECTION}", focus)

    # --- Legacy fallback (no template file) ---
    # outcome from the old outcomes.json via OUTCOMES dict
    legacy_outcome = outcome or OUTCOMES.get(study_id, "")
    if legacy_outcome:
        if fallback_multi_prompt:
            return fallback_multi_prompt.format(OUTCOME=legacy_outcome)
    if fallback_single_prompt:
        return fallback_single_prompt
    return ""


def load_full_document_text(parsed_path: Path) -> str:
    """Return a string representation of the parsed document."""
    if parsed_path.suffix == ".json":
        with open(parsed_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, str):
            return data
        if isinstance(data, list):
            return "\n\n".join(str(item) for item in data)
        return json.dumps(data, ensure_ascii=False, indent=2)

    # XML or plain-text fallback
    return parsed_path.read_text(encoding="utf-8")


def load_queries():
    with open(QUERIES_JSON_PATH, "r", encoding="utf-8") as f:
        query_dict = json.load(f)

    queries = {}
    for qid, q in query_dict.items():
        queries[qid] = q
    return queries


def load_true_answers(
    file: str = "codes_2025_10_14.xlsx", QUERIES: dict = None
) -> dict:
    """
    Load true coder answers from Excel and remove prefilled template values.
    Handles special case 4.5 (metrics and R² version).
    """
    path = PROJECT_ROOT / "data" / "true_labels" / file
    df = pd.read_excel(path)

    df.columns = [col.strip() for col in df.columns]

    # Fill unnamed headers from second row
    second_row = df.iloc[1]
    df.columns = [
        str(second_row[c]) if str(c).startswith("Unnamed") else c for c in df.columns
    ]

    # Identify valid item codes
    items = df["Item"].dropna().astype(str).str.strip()
    pattern = r"^\d+(\.\d+)*$"
    items = [x for x in items if re.match(pattern, x)]
    to_remove = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "99"]
    items = [x for x in items if x not in to_remove]

    # Identify coding columns (document IDs)
    coding_columns = [c for c in df.columns if re.match(r"^\d+$", str(c).strip())]
    if not coding_columns:
        raise ValueError("No coding columns detected (e.g., '021', '023').")

    true_answers = {}

    for doc_name in coding_columns:
        item_answers = {}

        for item in items:
            start_idx_list = df.index[
                df["Item"].astype(str).str.strip() == item
            ].tolist()
            if not start_idx_list:
                continue
            start_idx = start_idx_list[0]
            row_ids = [start_idx]
            for idx in range(start_idx + 1, len(df)):
                cell_value = df.at[idx, "Item"]
                if pd.isna(cell_value) or str(cell_value).strip() == "":
                    row_ids.append(idx)
                else:
                    break

            sub_df = df.loc[row_ids]
            row_values = sub_df[doc_name].astype(str).tolist()

            # --- Special case: 4.5 block ---
            if item == "4.5":
                subanswers = {f"4.5.{i}": ["-99"] for i in range(1, 7)}
                for val in row_values:
                    val = str(val).strip()
                    if not val or val.lower() == "nan":
                        continue

                    if val.startswith("AUC"):
                        m = re.search(r"[-−]?\d*\.?\d+", val)
                        if m:
                            subanswers["4.5.1"] = [m.group(0)]
                    elif val.startswith("ACC"):
                        m = re.search(r"[-−]?\d*\.?\d+", val)
                        if m:
                            subanswers["4.5.2"] = [m.group(0)]
                    elif val.startswith("BACC"):
                        m = re.search(r"[-−]?\d*\.?\d+", val)
                        if m:
                            subanswers["4.5.3"] = [m.group(0)]
                    elif val.startswith("R²") or val.startswith("R2"):
                        m = re.search(r"[-−]?\d*\.?\d+", val)
                        if m:
                            subanswers["4.5.4"] = [m.group(0)]
                    elif re.match(r"^r[:\s]", val, re.IGNORECASE):
                        m = re.search(r"[-−]?\d*\.?\d+", val)
                        if m:
                            subanswers["4.5.5"] = [m.group(0)]
                    elif "Specify R" in val or "Character" in val:
                        m = re.search(r"Specify R.?2.*?:\s*(.*)", val)
                        if m:
                            subanswers["4.5.6"] = [m.group(1).strip()]
                        elif ":" not in val and val not in ["Character", ""]:
                            subanswers["4.5.6"] = [val.strip()]
                        else:
                            subanswers["4.5.6"] = ["not specified"]

                item_answers.update(subanswers)
                continue

            # --- Generic case ---
            answers = [
                a.strip() for a in row_values if a.strip() and a.lower() != "nan"
            ]

            if QUERIES and item in QUERIES:
                prefilled_vals = QUERIES[item].get("prefilled", [])
                if isinstance(prefilled_vals, str):
                    prefilled_vals = [prefilled_vals]
                prefilled_vals = [str(v).strip().lower() for v in prefilled_vals]
                answers = [
                    a
                    for a in answers
                    if a.strip().lower() not in prefilled_vals
                    and not any(
                        a.strip().lower().startswith(pv) for pv in prefilled_vals
                    )
                ]

            item_answers[item] = answers or ["-99"]

        true_answers[doc_name] = item_answers

    return true_answers


# ---------------------------------------------------------------------------
# Finalized consensus gold set (09_data_final)
# ---------------------------------------------------------------------------
# Produced from data/true_labels/09_data_final.rds via scripts/export_true_labels.R.
FINAL_LABELS_CSV = PROJECT_ROOT / "data" / "true_labels" / "09_data_final.csv"

# Codebook items whose consolidated gold was manually adjudicated (like the
# single-choice items) and are therefore scored with strict exact set-match.
# All OTHER multiple-choice items use the rule-based rater-union as gold and are
# scored with a Jaccard set-overlap share (see evaluate.py). Item ids:
#   3.1 algo, 4.1 compare, 4.2 valid, 4.3 perfmetric
CURATED_MC_ITEMS = {"3.1", "4.1", "4.2", "4.3"}

# Codebook items intentionally excluded from the finalized gold (too many rater
# conflicts / too subjective to adjudicate objectively): rationale, splitprep,
# perftrain, interpret, general. Not evaluated (1.10 has a column but the data
# owner did not content-process it, so it is dropped like the others).
DROPPED_ITEMS = {"1.10", "2.6", "4.4", "5.3", "5.4"}

# 4.5.x numeric sub-items map onto the split highestperf_* columns. 4.5.6
# (open-ended "Specify R2 version") and 4.1.1 (bestmodel) have no gold column.
HIGHESTPERF_COLS = {
    "4.5.1": "highestperf_auc",
    "4.5.2": "highestperf_acc",
    "4.5.3": "highestperf_bacc",
    "4.5.4": "highestperf_r2",
    "4.5.5": "highestperf_r",
}

# Missing-data codes in 09_data_final, mapped onto the pipeline's answer space.
# The pipeline convention (see the old Excel gold and objective.py) is:
#   "0"   = not reported  (what the model emits as 0 / "Z / not stated")
#   "-99" = not applicable (dependency-gated; objective.py auto-scores it correct)
_NOT_REPORTED = "0"
_NOT_APPLICABLE = "-99"


def _unescape_unicode(s: str) -> str:
    """Decode R-export literal Unicode escapes like ``R<U+00B2>`` -> ``R²`` and
    ``na<U+00EF>ve`` -> ``naïve`` so gold text can match the option strings."""
    return re.sub(
        r"<U\+([0-9A-Fa-f]{4,6})>",
        lambda m: chr(int(m.group(1), 16)),
        s,
    )


def _map_gold_code(raw) -> str:
    """Normalize a raw 09_data_final cell to the pipeline's answer space."""
    s = "" if raw is None else str(raw).strip()
    if s == "" or s.lower() == "nan":
        return _NOT_APPLICABLE
    if s in ("-96", "-97"):  # not reported / unclear -> pipeline "not reported"
        return _NOT_REPORTED
    return _unescape_unicode(s)  # keeps "-99" and real values; decodes <U+XXXX>


def finalized_item_map(QUERIES: dict, csv_path: Path = FINAL_LABELS_CSV) -> dict:
    """Return {query_id: gold_column} for items that have a column in the
    finalized gold. Items without one (dropped subjective items, 4.1.1, 4.5.6)
    are omitted, so callers can filter QUERIES to what is actually evaluable."""
    cols = {c.strip() for c in pd.read_csv(csv_path, nrows=0).columns}
    item_to_col: dict[str, str] = {}
    for qid, q in QUERIES.items():
        if qid in DROPPED_ITEMS:
            continue
        col = HIGHESTPERF_COLS.get(qid, str(q.get("label", "")).strip())
        if col and col in cols:
            item_to_col[qid] = col
    return item_to_col


def load_true_answers_final(
    QUERIES: dict,
    csv_path: Path = FINAL_LABELS_CSV,
) -> dict:
    """Load the finalized consensus gold set into the nested shape the
    evaluation path expects: ``{study_id: {item: [gold_string]}}``.

    Drop-in replacement for :func:`load_true_answers`. Column per item is found
    by the query ``label`` (4.5.x -> highestperf_* columns). Missing codes are
    normalized via :func:`_map_gold_code`. MC gold stays comma-joined in a
    single-element list; ``get_true_answer_from_dict`` splits it downstream.
    Studies are keyed by zero-padded ``study_ID`` (matching the PDF-derived
    doc names) and, as a fallback, by the un-padded integer form.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    item_to_col = finalized_item_map(QUERIES, csv_path)

    true_answers: dict[str, dict] = {}
    for _, row in df.iterrows():
        sid = str(row["study_ID"]).strip()
        item_answers = {
            qid: [_map_gold_code(row[col])] for qid, col in item_to_col.items()
        }
        true_answers[sid] = item_answers
        sid_int = sid.lstrip("0") or "0"
        if sid_int != sid:
            true_answers.setdefault(sid_int, item_answers)

    return true_answers


def load_doc_id_map(path: str = "../data/doc_id_map.json") -> dict:
    """Load or create a persistent mapping from document name to ID."""
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_doc_id_map(doc_id_map: dict, path: str = "../data/doc_id_map.json"):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc_id_map, f, indent=2, ensure_ascii=False)
