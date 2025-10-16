import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ----------------------------
# Utilities
# ----------------------------
def clean_answer(ans: str) -> str:
    if not ans:
        return ""

    ans = ans.strip()

    # Remove any HTML/XML-like tags such as <...> or </>
    ans = re.sub(r"<[^>]*>", "", ans)

    # Remove leading and trailing non-word characters
    ans = re.sub(r"^[^\w]*", "", ans)
    ans = re.sub(r"[^\w]*$", "", ans)

    # Extract first token (word or number)
    tokens = ans.split()
    if not tokens:
        return ""

    first_token = tokens[0]
    return first_token.upper()


def normalize(text: str) -> str:
    return re.sub(r"\W+", "", text.strip().lower())


# def load_choices(query_id: str, base_path: str = "../queries") -> tuple[dict, list]:
#     """
#     Load 'choices' and 'prefilled' information for a given query_id.

#     Expected file structure:
#         ../queries/<query_id>/query_info.json

#     Returns:
#         (choices: dict, prefilled: list)

#     Example:
#         choices, prefilled = load_choices("0.7.1")

#     Raises:
#         - Prints clear messages for missing files, invalid JSON, or missing keys.
#     """
#     file_path = Path(base_path) / query_id / "query_info.json"

#     if not file_path.exists():
#         print(
#             f"[load_choices] ❌ File not found for query_id '{query_id}' → {file_path}"
#         )
#         return {}, []

#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         # Extract with defaults
#         choices = data.get("choices", {})
#         prefilled = data.get("prefilled", [])

#         # Validate types
#         if not isinstance(choices, dict):
#             print(f"[load_choices] ⚠️ 'choices' in {file_path} is not a dictionary.")
#             choices = {}

#         if not isinstance(prefilled, (list, tuple)):
#             print(f"[load_choices] ⚠️ 'prefilled' in {file_path} is not a list.")
#             prefilled = []

#         return choices, list(prefilled)

#     except json.JSONDecodeError as e:
#         print(f"[load_choices] ❌ JSON decode error in {file_path}: {e}")
#         return {}, []
#     except Exception as e:
#         print(f"[load_choices] ❌ Unexpected error loading {file_path}: {e}")
#         return {}, []


def map_true_answer_to_choices(true_answer: str, choices: dict):
    truth_tokens = [normalize(x) for x in re.split(r",|;", true_answer)]
    truth_letters = [
        letter for letter, label in choices.items() if normalize(label) in truth_tokens
    ]
    return sorted(truth_letters)


# def get_true_answer_from_excel(
#     excel_path: str, query_id: str, document_name: str
# ) -> str:
#     df = pd.read_excel(excel_path, index_col=0, skiprows=2)
#     if query_id not in df.index:
#         raise ValueError(f"Query ID '{query_id}' not found in Excel.")
#     if document_name not in df.columns:
#         raise ValueError(f"Document '{document_name}' not found in Excel columns.")
#     return str(df.loc[query_id, document_name])


def get_true_answer_from_dict(
    true_answers: dict, queries: dict, query_id: str, document_name: str
) -> list:
    """
    Retrieve true answers from a nested dict for a document/query,
    handling prefilled values:
      - If 'Other: ' is in prefilled and present in the value, keep what follows.
      - For any other prefilled value, discard it and anything that follows.

    Args:
        true_answers: dict like {'021': {'0.7': ['Yes','No'], ...}, ...}
        queries: dict keyed by query_id containing 'prefilled' list
        query_id: the item/query identifier
        document_name: the document column to retrieve

    Returns:
        List of cleaned answers
    """
    if document_name not in true_answers:
        raise ValueError(f"Document '{document_name}' not found in true_answers.")

    doc_dict = true_answers[document_name]

    if query_id not in doc_dict:
        raise ValueError(
            f"Query ID '{query_id}' not found for document '{document_name}'."
        )

    prefilled = queries.get(query_id, {}).get("prefilled", [])
    values = doc_dict[query_id]

    cleaned_list = []

    if not isinstance(values, list):
        values = [values]

    for v in values:
        if pd.isna(v):
            continue
        v_str = str(v).strip()
        if not v_str:
            continue

        # Handle prefilled prefixes
        for pre in prefilled:
            if pre.lower() == "other:" and v_str.startswith(pre):
                # Keep only what comes after 'Other: '
                v_str = v_str[len(pre) :].strip()
            elif v_str.startswith(pre):
                # Discard the prefilled word and anything that comes after
                v_str = ""
                break

        # Split by comma if multiple entries and add non-empty
        split_vals = [x.strip() for x in re.split(r",\s*", v_str) if x.strip()]
        cleaned_list.extend(split_vals)

    return cleaned_list


def map_values_to_letters(values_list, choices):
    """
    Map a list of values to their corresponding choice letters based on the 'choices' dict.
    - Case-insensitive matching
    - Ignores extra spaces
    - Assigns 'Y' if not found, 'Z' if value is '0'
    """
    # Build a normalized reverse map
    value_to_letter = {
        str(v["value"]).strip().lower(): k
        for k, v in choices.items()
        if isinstance(v, dict) and "value" in v
    }

    letters = []
    for v in values_list:
        v_str = str(v).strip()
        if v_str == "0":
            letters.append("Z")
            continue
        # Case-insensitive lookup
        letter = value_to_letter.get(v_str.lower(), "Y")
        letters.append(letter)

    # Remove duplicates and sort alphabetically
    return sorted(set(letters))


# ----------------------------
# NLI model for open-ended
# ----------------------------
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-small-mnli-fever-docnli-ling-2c"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
LABELS = ["entailment", "neutral", "contradiction"]


def nli_entailment(premise: str, hypothesis: str) -> bool:
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
    pred_label = LABELS[torch.argmax(probs)]
    return pred_label == "entailment"


def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation, and normalize spaces."""
    if not isinstance(s, str):
        return ""
    # Remove punctuation
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ----------------------------
# Unified evaluation
# ----------------------------
def evaluate_answer(
    llm_output: str,
    question_type: str,
    doc_name: str,
    query_id: str,
    TRUE_ANSWERS: dict,
    QUERIES: dict,
    other_callback=None,
):
    """Evaluate LLM outp
    ut based on question type."""
    print("Doc Name: ", doc_name)
    print("Query id: ", query_id)
    true_path = "../data/true_labels/"
    if isinstance(llm_output, str):
        llm_output = llm_output.strip()
    elif isinstance(llm_output, list) or isinstance(llm_output, set):
        llm_output = [str(x).strip() for x in llm_output]
    true_answer = get_true_answer_from_dict(TRUE_ANSWERS, QUERIES, query_id, doc_name)
    # Load choices if needed
    # choices = None
    # choices, prefilled = load_choices(query_id)
    choices = QUERIES[query_id].get("choices", {})

    if question_type == "single_choice":
        pred_number = clean_answer(llm_output)

        # Map 'Z' → '0' (Not reported)
        if isinstance(pred_number, str) and pred_number.strip().upper() == "Z":
            pred_number = "0"

        # Get true answer safely
        true_value_key = (
            str(true_answer[0])
            if isinstance(true_answer, (list, tuple))
            else str(true_answer)
        )

        # Compute accuracy
        acc = int(str(pred_number) == true_value_key)

        return {
            "raw_output": llm_output,
            "pred_indicator": pred_number,
            "pred_value": choices.get(str(pred_number), {}).get("value", ""),
            "true_indicator": true_value_key,
            "true_value": choices.get(true_value_key, {}).get("value", ""),
            "accuracy": acc,
        }

    elif question_type == "multiple_choice":
        # Extract uppercase letters (choices like A, B, C, Y, Z, etc.)
        pred_letters = set(re.findall(r"[A-Z]", llm_output.upper()))
        choices_norm = {k.upper(): v for k, v in choices.items()}

        # If LLM outputs "Z" (not reported), map it to 0
        if "Z" in pred_letters:
            pred_choices = ["0"]
        else:
            pred_choices = [
                choices_norm.get(str(k).upper(), {}).get("value", "")
                for k in sorted(pred_letters, key=str.upper)
            ]
            # Normalize punctuation, case, etc.
            pred_choices = [normalize_text(x) for x in pred_choices if x]

        # Normalize true choices
        true_choices = [normalize_text(x) for x in true_answer if x is not None]

        # Map true value 0 → letter Z
        true_letters = map_values_to_letters(true_choices, choices_norm)
        true_letters = ["Z" if str(x).strip() == "0" else x for x in true_letters]

        # Handle “Other” option if Y is present and callback provided
        other_value = None
        if "Y" in pred_letters and other_callback is not None:
            other_value = other_callback(llm_output)
            if other_value:
                pred_choices.append(normalize_text(other_value))

        # Compute accuracy (ignore punctuation differences)
        acc = int(set(pred_choices) == set(true_choices))

        return {
            "raw_output": llm_output,  # Original LLM output
            "pred_indicator": sorted(pred_letters),
            "pred_value": pred_choices,
            "true_indicator": true_letters,
            "true_value": true_choices,
            "accuracy": acc,
            "other_value": other_value,
        }

    elif question_type == "numeric":
        try:
            if llm_output.strip().upper() == "Z":
                pred_num = 0
            else:
                pred_num = float(llm_output)
            true_num = float(true_answer[0])
            acc = int(pred_num == true_num)
        except ValueError:
            pred_num = np.nan
            true_num = true_answer
            acc = 0

        return {
            "raw_output": llm_output,
            "pred_value": pred_num,  # return numeric value
            "true_value": true_num,
            "accuracy": acc,
        }

    elif question_type == "list":
        if isinstance(llm_output, str):
            # Split string by comma, strip whitespace, convert to set
            pred_items = set(x.strip() for x in llm_output.split(","))
        elif isinstance(llm_output, (list, set)):
            # Already a list or set, just normalize strings
            pred_items = set(str(x).strip() for x in llm_output)
        else:
            raise ValueError(f"Unexpected llm_output type: {type(llm_output)}")
        true_items = (
            set(map(str.strip, true_answer.split(",")))
            if isinstance(true_answer, str)
            else set(true_answer)
        )
        acc = int(pred_items == true_items)
        return {
            "raw_output": llm_output,
            "pred_value": sorted(pred_items),
            "true_value": sorted(true_items),
            "accuracy": acc,
        }

    elif question_type == "open_ended":
        entail1 = nli_entailment(true_answer[0], llm_output)
        entail2 = nli_entailment(llm_output, true_answer[0])
        acc = int(entail1 and entail2)
        return {
            "raw_output": llm_output,
            "true_value": true_answer,
            "accuracy": acc,
            "entail_true_to_pred": entail1,
            "entail_pred_to_true": entail2,
        }

    else:
        raise ValueError(f"Unknown question_type: {question_type}")


# if __name__ == "__main__":
# QUERIES_JSON_PATH = Path(
#     "../query_creation/queries_with_prompts.json"
# )  # Path to your JSON file

# def load_queries():
#     with open(QUERIES_JSON_PATH, "r", encoding="utf-8") as f:
#         query_dict = json.load(f)

#     queries = {}
#     for qid, q in query_dict.items():
#         queries[qid] = q  # keep original structure
#     return queries

# def load_true_answers(
#     path: str = "../data/true_labels/human_codes_test2.xlsx",
# ) -> dict:
#     df = pd.read_excel(path)

#     # Strip whitespace from column headers to avoid lookup errors
#     df.columns = [col.strip() for col in df.columns]

#     # if columns have no name, i.e. those are codings, use the entry in the second row

#     second_row = df.iloc[1]
#     new_columns = []
#     for col in df.columns:
#         if col.startswith("Unnamed"):
#             new_columns.append(str(second_row[col]))  # use second row's entry
#         else:
#             new_columns.append(col)  # keep original name

#     df.columns = new_columns

#     # ------------------------
#     # 2. Identify item codes
#     # ------------------------
#     # Extract all unique entries in the "Item" column
#     items = df["Item"].unique()

#     # Remove NaN and invalid entries
#     items = [x for x in items if x is not None and str(x) != "nan"]

#     # Keep only entries that look like hierarchical item codes (e.g. "0.7.2")
#     items = [x for x in items if not isinstance(x, int)]
#     items = [x for x in items if all(c.isdigit() or c == "." for c in x)]

#     # Exclude generic meta-rows that are not part of the coding scheme
#     to_remove = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "99"]
#     items = [x for x in items if x not in to_remove]
#     true_answers = {}
#     for doc_name in ["021", "023", "038"]:
#         item_answers = {}
#         for item in items:
#             row = df.loc[df["Item"] == item]
#             query_id = item

#             start_idx_list = df.index[df["Item"] == item].tolist()
#             if start_idx_list:
#                 start_idx = start_idx_list[0]
#                 row_ids = [start_idx]
#                 for idx in range(start_idx + 1, len(df)):
#                     if pd.isna(df.at[idx, "Item"]):
#                         row_ids.append(idx)
#                     else:
#                         break

#                 item_answers[query_id] = df.loc[row_ids, doc_name].values.tolist()

#         true_answers[doc_name] = item_answers
#     return true_answers

# QUERIES = load_queries()

# TRUE_ANSWERS = load_true_answers()

# print(
#     evaluate_answer(
#         llm_output="10",
#         question_type="numeric",
#         doc_name="021",
#         query_id="1.2",
#         TRUE_ANSWERS=TRUE_ANSWERS,
#         QUERIES=QUERIES,
#         # answer_file="human_codes_test2.xlsx",
#         # choices_file="query_choice_mapping.json",
#         # other_callback=handle_other,
#     )
# )
