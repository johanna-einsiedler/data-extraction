"""Utilities for extracting structured query metadata from the coding scheme Excel file."""

from __future__ import annotations

import copy
import json
import re
import string
from typing import List

import pandas as pd

ITEM_PATTERN = re.compile(r"^\d+(\.\d+)*$")
EXCLUDED_ITEMS = {"0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "99"}


def _sanitize_item(value: str | float | int) -> str:
    """Return a trimmed string for an item code, normalising NaN-like values to an empty string."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def collect_item_info(path: str) -> List[dict]:
    """Read the coding scheme Excel file and return structured metadata for each query."""
    df = pd.read_excel(path)
    df.columns = [col.strip() for col in df.columns]
    df["Item"] = df["Item"].map(_sanitize_item)

    # Identify relevant item codes while preserving their original order.
    seen = set()
    items: List[str] = []
    for raw_item in df["Item"]:
        if not raw_item or raw_item in seen:
            continue
        if ITEM_PATTERN.match(raw_item) and raw_item not in EXCLUDED_ITEMS:
            items.append(raw_item)
            seen.add(raw_item)

    query_info: List[dict] = []

    for item in items:
        row = df.loc[df["Item"] == item]
        query_id = item
        query_label = row["Variable label"].values[0]
        description = row["Description"].values[0]
        instructions = row["Instructions for coders"].values[0]

        # Determine the span of rows that belong to this item (handles multi-line options).
        start_idx_list = df.index[df["Item"] == item].tolist()
        if not start_idx_list:
            continue
        row_ids = [start_idx_list[0]]
        for idx in range(row_ids[0] + 1, len(df)):
            cell_value = df.at[idx, "Item"]
            if not cell_value:  # still part of the current item block
                row_ids.append(idx)
            else:
                break

        choices = df.loc[row_ids, "Values"].values.tolist()
        descriptions = df.loc[row_ids, "Examples and Notes"].values
        descriptions = ["" if pd.isna(d) else d for d in descriptions]

        # Filter out empty or NaN-like entries while keeping alignment with descriptions.
        filtered_choices = []
        filtered_descriptions = []
        for choice, desc in zip(choices, descriptions):
            if choice is not None and str(choice).strip().lower() != "nan":
                filtered_choices.append(choice)
                filtered_descriptions.append(desc)

        choices = filtered_choices
        descriptions = filtered_descriptions

        prefilled_values = df.loc[row_ids, "empty_col"]
        prefilled_values = [
            str(v).strip()
            for v in prefilled_values
            if pd.notna(v) and str(v).strip() != ""
        ]

        query_type = "open_ended"
        description_detailed = ""
        mapping = {}

        if len(choices) == 1:
            description_detailed = df.loc[row_ids[0], "Examples and Notes"]

        elif len(choices) > 1:
            description_detailed = ""

            if pd.isna(df.loc[row_ids[0], "Values"]):
                description_detailed = df.loc[row_ids[0], "Examples and Notes"]

            # Checkbox-style multi-select questions scatter choices across rows.
            if any("[" in str(x).replace(" ", "") for x in choices):
                cleaned_choices = []
                cleaned_descriptions = []
                for val, desc in zip(choices, descriptions):
                    val_str = str(val).strip()
                    if re.match(r"^\d+\.\s*", val_str):
                        continue
                    val_str = re.sub(r"\[\s*\]", "", val_str)
                    val_str = val_str.replace(":", "").replace("_", "").strip()
                    if val_str:
                        cleaned_choices.append(val_str)
                        cleaned_descriptions.append(desc)
                choices = cleaned_choices
                descriptions = cleaned_descriptions

                other_present = any("Other" in x for x in choices)
                not_reported_present = any(
                    x.startswith("0") and "Not reported" in x for x in choices
                )
                filtered_choices = []
                filtered_descriptions = []
                for val, desc in zip(choices, descriptions):
                    if "-99" not in val and "Not applicable" not in val:
                        filtered_choices.append(val)
                        filtered_descriptions.append(desc)
                choices = filtered_choices
                descriptions = filtered_descriptions

                choices_no_special = [
                    v for v in choices if not (v.startswith("0") or "Other" in v)
                ]
                descriptions_no_special = [
                    d for v, d in zip(choices, descriptions) if v in choices_no_special
                ]
                letters = list(string.ascii_uppercase)
                mapping = {
                    letters[i]: {"value": val, "description": desc}
                    for i, (val, desc) in enumerate(
                        zip(choices_no_special, descriptions_no_special)
                    )
                }

                if other_present:
                    idx = next(i for i, x in enumerate(choices) if "Other" in x)
                    mapping["Y"] = {
                        "value": "Other",
                        "description": descriptions[idx]
                        if idx < len(descriptions)
                        else "",
                    }

                if not_reported_present:
                    idx = next(i for i, x in enumerate(choices) if x.startswith("0"))
                    mapping["Z"] = {
                        "value": "Not reported",
                        "description": descriptions[idx]
                        if idx < len(descriptions)
                        else "",
                    }

                query_type = "multiple_choice"

            else:
                if any("Numeric" in str(x).replace(" ", "") for x in choices):
                    query_type = "numeric"
                    notes = df.loc[row_ids, "Examples and Notes"].dropna()
                    notes_list = [str(x).strip() for x in notes if str(x).strip()]
                    description_detailed = " ".join(notes_list)
                    instructions_series = df.loc[
                        row_ids, "Instructions for coders"
                    ].dropna()
                    instructions_list = [
                        str(x).strip() for x in instructions_series if str(x).strip()
                    ]
                    instructions = " ".join(instructions_list)
                    if "Numeric" in choices:
                        choices.remove("Numeric")
                    descriptions = [""] * len(choices)

                elif any("1=" in str(x).replace(" ", "") for x in choices):
                    query_type = "single_choice"
                    description_detailed = ""

                elif query_id == "0.7.3":
                    query_type = "list"
                    choices.remove("Specific packages mentioned: __________")

                mapping = {}
                for val, desc in zip(choices, descriptions):
                    if "=" in str(val):
                        key, value = [x.strip() for x in str(val).split("=", 1)]
                        mapping[key] = {"value": value, "description": desc}

            if query_id == "4.1.1":
                query_type = "open_ended"

        else:
            query_type = "open_ended"
            description_detailed = ""
            mapping = {}

        # Build few-shot examples from the coding spreadsheet when available.
        example_cols = [c for c in df.columns if c.startswith("correct_answer_")]
        examples = []
        for col in example_cols:
            suffix = col.split("_")[-1]
            answer = row[col].values[0] if col in row else None
            paragraph_col = f"paragraph_{suffix}"
            paragraph = (
                row[paragraph_col].values[0]
                if paragraph_col in row and not pd.isna(row[paragraph_col].values[0])
                else None
            )
            if paragraph and str(paragraph).strip():
                examples.append(
                    {"context": str(paragraph).strip(), "answer": str(answer).strip()}
                )

        query_dict = {
            "query_id": query_id,
            "label": query_label,
            "description": description if not pd.isna(description) else "",
            "description_detailed": description_detailed
            if not pd.isna(description_detailed)
            else "",
            "instructions": instructions if not pd.isna(instructions) else "",
            "choices": mapping,
            "type": query_type,
            "prefilled": prefilled_values,
            "examples": examples,
        }

        # Expand question 4.5 into per-metric sub-items driven by the prefilled values.
        if query_id == "4.5":
            for i, value in enumerate(prefilled_values, start=1):
                sub = copy.deepcopy(query_dict)
                sub["query_id"] = f"{query_dict['query_id']}.{i}"
                sub["description"] = value.strip(":")
                sub["prefilled"] = []
                sub["examples"] = examples
                sub["type"] = "open_ended" if "specify" in value.lower() else "numeric"
                query_info.append(sub)
        else:
            query_info.append(query_dict)

    return query_info


if __name__ == "__main__":
    path = "coding_scheme.xlsx"
    item_info = collect_item_info(path)
    with open("query_info.json", "w", encoding="utf-8") as f:
        json.dump(item_info, f, ensure_ascii=False, indent=4)
