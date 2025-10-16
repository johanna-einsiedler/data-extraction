import copy
import json
import re
import string

import numpy as np
import pandas as pd


def collect_item_info(path: str):
    """
    Extract structured metadata about coding items from an Excel codebook.

    The Excel sheet is expected to have columns like:
    ['Item', 'Variable label', 'Description', 'Values',
     'Examples and Notes', 'Instructions for coders', 'empty_col']

    The function groups information for each "Item" ID (e.g., '0.7', '0.7.1', etc.)
    and builds a structured JSON-like record containing:
        - query_id: the item number (e.g. "0.7.1")
        - label: short variable name
        - description: text description of the variable
        - instructions: instructions for coders
        - type: question type (single_choice, multiple_choice, numeric, etc.)
        - choices: possible response options and their descriptions
        - prefilled: any pre-existing values found in the "empty_col" column

    Args:
        path (str): Path to the Excel file.

    Returns:
        list[dict]: A list of dictionaries, one per item.
    """

    # ------------------------
    # 1. Load and clean data
    # ------------------------
    df = pd.read_excel(path)

    # Strip whitespace from column headers to avoid lookup errors
    df.columns = [col.strip() for col in df.columns]

    # ------------------------
    # 2. Identify item codes
    # ------------------------
    # Extract all unique entries in the "Item" column
    items = df["Item"].unique()

    # Remove NaN and invalid entries
    items = [x for x in items if x is not None and str(x).strip() != "nan"]

    # Keep only entries that look like hierarchical item codes (e.g. "0.7.2", "1.10")
    # Allow multiple digits between dots and remove leading/trailing spaces
    pattern = re.compile(r"^\d+(\.\d+)*$")
    items = [
        x.strip() for x in items if isinstance(x, str) and pattern.match(x.strip())
    ]

    # Exclude generic meta-rows that are not part of the coding scheme
    to_remove = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "99"]
    items = [x for x in items if x not in to_remove]
    # Initialize container for structured query information
    query_info = []

    # ------------------------
    # 3. Loop through each item
    # ------------------------
    for item in items:
        # Subset the row corresponding to this item
        df["Item"] = df["Item"].astype(str).str.strip()
        row = df.loc[df["Item"] == item]
        # Extract basic metadata fields
        query_id = item
        query_label = row["Variable label"].values[0]
        description = row["Description"].values[0]
        instructions = row["Instructions for coders"].values[0]

        # ------------------------
        # 4. Identify all rows belonging to this item
        # ------------------------
        # Each item can span multiple rows if choices or sub-values follow
        # Find the starting index of the item
        start_idx_list = df.index[
            df["Item"].astype(str).str.strip() == item.strip()
        ].tolist()

        if start_idx_list:  # Only proceed if the item exists
            start_idx = start_idx_list[0]
            row_ids = [start_idx]
            # Continue adding subsequent rows until the next "Item" appears
            for idx in range(start_idx + 1, len(df)):
                # Check if the Item cell is empty or NaN after stripping
                cell_value = df.at[idx, "Item"]
                if (
                    cell_value is pd.NA
                    or pd.isna(cell_value)
                    or cell_value == ""
                    or cell_value == "nan"
                ):
                    row_ids.append(idx)
                else:
                    break
        # --------------    ----------
        # 5. Extract all relevant cell values
        # ------------------------
        # Possible answer values (e.g., "1 = No", "2 = Yes")
        choices = df.loc[row_ids, "Values"].values.tolist()
        # Descriptive examples or notes for each value
        descriptions = df.loc[row_ids, "Examples and Notes"].values
        descriptions = ["" if pd.isna(d) else d for d in descriptions]
        filtered_choices = []
        filtered_descriptions = []
        for choice, desc in zip(choices, descriptions):
            if choice is not None and str(choice).strip().lower() != "nan":
                filtered_choices.append(choice)
                filtered_descriptions.append(desc)

        choices = filtered_choices
        descriptions = filtered_descriptions
        # 🆕 NEW: Collect any prefilled data (manual or auto-coded) from "empty_col"
        prefilled_values = df.loc[row_ids, "empty_col"]
        prefilled_values = [
            str(v).strip()
            for v in prefilled_values
            if pd.notna(v) and str(v).strip() != ""
        ]

        # ------------------------
        # 6. Determine question type and structure choices
        # ------------------------
        if len(choices) == 1:
            # Likely an open-ended question
            query_type = "open_ended"
            description_detailed = df.loc[start_idx, "Examples and Notes"]
            mapping = {}

        elif len(choices) > 1:
            description_detailed = ""

            # Case 1: Multiple-choice with checkboxes (identified by "[ ]" markers)
            if pd.isna(df.loc[start_idx, "Values"]):
                description_detailed = df.loc[start_idx, "Examples and Notes"]
            if any("[" in str(x).replace(" ", "") for x in choices):
                # --- Normalize and clean choices and keep descriptions aligned ---
                cleaned_choices = []
                cleaned_descriptions = []

                for val, desc in zip(choices, descriptions):
                    val_str = str(val).strip()

                    # Skip entries that are just numbered headers
                    if re.match(r"^\d+\.\s*", val_str):
                        continue

                    # Clean the choice further (remove [ ], colons, underscores)
                    val_str = re.sub(r"\[\s*\]", "", val_str)
                    val_str = val_str.replace(":", "").replace("_", "").strip()

                    # Only keep non-empty choices
                    if val_str:
                        cleaned_choices.append(val_str)
                        cleaned_descriptions.append(desc)

                choices = cleaned_choices
                descriptions = cleaned_descriptions
                # --- Detect special categories ---
                other_present = any("Other" in x for x in choices)
                not_reported_present = any(
                    x.startswith("0") and "Not reported" in x for x in choices
                )
                not_applicable_present = any(
                    "-99" in x or "Not applicable" in x for x in choices
                )

                # --- Remove "Not applicable" entirely and keep descriptions aligned ---
                filtered_choices = []
                filtered_descriptions = []
                for val, desc in zip(choices, descriptions):
                    if "-99" not in val and "Not applicable" not in val:
                        filtered_choices.append(val)
                        filtered_descriptions.append(desc)

                choices = filtered_choices
                descriptions = filtered_descriptions
                # --- Prepare filtered regular options ---
                choices_no_special = [
                    v for v in choices if not (v.startswith("0") or "Other" in v)
                ]
                descriptions_no_special = [
                    d for v, d in zip(choices, descriptions) if v in choices_no_special
                ]
                # --- Assign A, B, C... to regular options ---
                letters = list(string.ascii_uppercase)
                mapping = {
                    letters[i]: {"value": val, "description": desc}
                    for i, (val, desc) in enumerate(
                        zip(choices_no_special, descriptions_no_special)
                    )
                }

                # --- Assign Y = Other, Z = Not reported ---
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

                # --- Determine type ---
                query_type = "multiple_choice"

            # Case 2: Numeric or single-choice question
            else:
                if any("Numeric" in x.replace(" ", "") for x in choices):
                    query_type = "numeric"
                    notes = df.loc[row_ids, "Examples and Notes"].dropna()
                    notes_list = [str(x).strip() for x in notes if str(x).strip()]
                    # Concatenate into one string, separated by spaces (or choose another separator)
                    description_detailed = " ".join(notes_list)
                    instructions = df.loc[row_ids, "Instructions for coders"].dropna()
                    instructions = [
                        str(x).strip() for x in instructions if str(x).strip()
                    ]
                    # Concatenate into one string, separated by spaces (or choose another separator)
                    instructions = " ".join(instructions)
                    if "Numeric" in choices:
                        choices.remove("Numeric")
                    descriptions = [""] * len(choices)

                elif any("1=" in x.replace(" ", "") for x in choices):
                    query_type = "single_choice"
                    description_detailed = ""

                # Case 3: Special case — list of items (e.g., 0.7.3 packages)
                elif query_id == "0.7.3":
                    query_type = "list"
                    choices.remove("Specific packages mentioned: __________")

                # Convert choices like "1 = Yes" to structured key/value mappings
                mapping = {}
                for val, desc in zip(choices, descriptions):
                    if "=" in val:
                        key, value = [x.strip() for x in val.split("=", 1)]
                        mapping[key] = {"value": value, "description": desc}

            # Special Case Question 4.1.1
            if query_id == "4.1.1":
                query_type = "open_ended"

        # ------------------------
        # 7. Collect few-shot examples
        # ------------------------
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

        # ------------------------
        # 8. Assemble final record
        # ------------------------
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
            "examples": examples,  # ✅ NEW FIELD
        }

        # Special case for 4.5 expansion
        if query_id == "4.5":
            for i, value in enumerate(prefilled_values, start=1):
                sub = copy.deepcopy(query_dict)
                sub["query_id"] = f"{query_dict['query_id']}.{i}"
                sub["description"] = value.strip(":")
                sub["prefilled"] = []
                sub["examples"] = examples  # ✅ propagate examples to sub-entries
                sub["type"] = "open_ended" if "specify" in value.lower() else "numeric"
                query_info.append(sub)
        else:
            query_info.append(query_dict)

    return query_info

    # ------------------------
    # 9. Return all item info
    # ------------------------
    return query_info


if __name__ == "__main__":
    path = "coding_scheme.xlsx"
    item_info = collect_item_info(path)
    with open("query_info.json", "w", encoding="utf-8") as f:
        json.dump(item_info, f, ensure_ascii=False, indent=4)
