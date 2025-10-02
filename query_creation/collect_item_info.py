import json
import re
import string

import pandas as pd


def collect_item_info(path: str):
    # Load Excel file
    df = pd.read_excel(path)
    df.columns = [col.strip() for col in df.columns]

    # create item list
    items = df["Item"].unique()
    items = [x for x in items if x is not None and str(x) != "nan"]
    items = [x for x in items if not isinstance(x, int)]
    items = [x for x in items if all(c.isdigit() or c == "." for c in x)]
    to_remove = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "99"]
    items = [x for x in items if x not in to_remove]

    query_info = []
    for item in items:
        row = df.loc[df["Item"] == item]
        query_id = item
        query_label = row["Variable label"].values[0]
        description = row["Description"].values[0]
        instructions = row["Instructions for coders"].values[0]

        # Find the index of the row where item == '0.7.1'
        start_idx = df.index[df["Item"] == item].tolist()

        if start_idx:  # make sure it exists
            start_idx = start_idx[0]
            row_ids = [start_idx]

            # Iterate over subsequent rows
            for idx in range(start_idx + 1, len(df)):
                if pd.isna(df.at[idx, "Item"]):
                    row_ids.append(idx)
                else:
                    break

        choices = df.loc[row_ids, "Values"].values.tolist()

        descriptions = df.loc[row_ids, "Examples and Notes"].values
        descriptions = ["" if pd.isna(d) else d for d in descriptions]

        choices = [x for x in choices if x is not None and str(x) != "nan"]

        if len(choices) == 1:
            query_type = "open_ended"
            description_detailed = df.loc[start_idx, "Examples and Notes"]
            mapping = {}
        elif len(choices) > 1:
            if any("[" in x.replace(" ", "") for x in choices):
                choices = [x for x in choices if not x.strip()[0].isdigit()]
                choices = [
                    re.sub(r"\[\s*\]", "", x)  # remove [] or [ ]
                    .replace(":", "")  # remove colons
                    .replace("_", "")  # remove underscores
                    .strip()  # remove leading/trailing spaces
                    for x in choices
                ]
                other_present = "Other" in choices
                not_reported_present = "0 = Not reported" in choices
                # Exclude "Other" for letter assignment
                choices_no_other = [
                    (v, d) for v, d in zip(choices, descriptions) if v != "Other"
                ]
                choices_final = [
                    (v, d) for v, d in choices_no_other if v != "0 = Not reported"
                ]
                # Assign letters A, B, C, ... to the rest
                letters = list(string.ascii_uppercase)
                mapping = {
                    letters[i]: {"value": val, "description": desc}
                    for i, (val, desc) in enumerate(choices_final)
                }
                # Assign "Other" to Y if present
                if other_present:
                    other_desc = descriptions[choices.index("Other")]
                    mapping["Y"] = {"value": "Other", "description": other_desc}
                if not_reported_present:
                    not_reported_desc = descriptions[choices.index("Other")]
                    mapping["Z"] = {
                        "value": "Not reported",
                        "description": not_reported_desc,
                    }

                query_type = "multiple_choice"
                description_detailed = ""

            else:
                if any("Numeric" in x.replace(" ", "") for x in choices):
                    query_type = "numeric"
                    description_detailed = df.loc[start_idx + 1, "Examples and Notes"]
                    if "Numeric" in choices:
                        choices.remove("Numeric")
                    descriptions = [""] * len(choices)
                # Check if single choice
                elif any("1=" in x.replace(" ", "") for x in choices):
                    query_type = "single_choice"
                    description_detailed = ""
                elif query_id == "0.7.3":
                    query_type = "list"
                    choices.remove("Specific packages mentioned: __________")

                # Convert to dictionary with descriptions
                mapping = {}
                for val, desc in zip(choices, descriptions):
                    if "=" in val:
                        key, value = [x.strip() for x in val.split("=", 1)]
                        mapping[key] = {"value": value, "description": desc}

        # if len(choices)==1:
        #     if choices[0] == ''
        if pd.isna(description_detailed):
            description_detailed = ""
        if pd.isna(instructions):
            instructions = ""

        query_dict = {
            "query_id": query_id,
            "label": query_label,
            "description": description,
            "description_detailed": description_detailed,
            "instructions": instructions,
            "choices": mapping,
            "type": query_type,
        }
        query_info.append(query_dict)
    return query_info


if __name__ == "__main__":
    path = "coding_scheme.xlsx"
    item_info = collect_item_info(path)
    with open("query_info.json", "w", encoding="utf-8") as f:
        json.dump(item_info, f, ensure_ascii=False, indent=4)
