import json
import re
from pathlib import Path

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
    ans = re.sub(r"^[^\w]*", "", ans)
    ans = re.sub(r"[^\w]*$", "", ans)
    first_token = ans.split()[0]
    return first_token.upper()


def normalize(text: str) -> str:
    return re.sub(r"\W+", "", text.strip().lower())


def load_choices(query_id: str) -> dict:
    try:
        with open(f"../queries/{query_id}/query_info.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        return data["choices"]
    except FileNotFoundError:
        print(f"Error: File for query_id '{query_id}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: JSON decoding failed for query_id '{query_id}'.")
        return {}
    except KeyError:
        print(
            f"Error: 'choices' key not found in query_info.json for query_id '{query_id}'."
        )
        return {}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}


def map_true_answer_to_choices(true_answer: str, choices: dict):
    truth_tokens = [normalize(x) for x in re.split(r",|;", true_answer)]
    truth_letters = [
        letter for letter, label in choices.items() if normalize(label) in truth_tokens
    ]
    return sorted(truth_letters)


def get_true_answer_from_excel(
    excel_path: str, query_id: str, document_name: str
) -> str:
    df = pd.read_excel(excel_path, index_col=0, skiprows=2)
    if query_id not in df.index:
        raise ValueError(f"Query ID '{query_id}' not found in Excel.")
    if document_name not in df.columns:
        raise ValueError(f"Document '{document_name}' not found in Excel columns.")
    return str(df.loc[query_id, document_name])


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


# ----------------------------
# Unified evaluation
# ----------------------------
def evaluate_answer(
    llm_output: str,
    question_type: str,
    doc_name: str,
    query_id: str,
    answer_file: str,
    other_callback=None,
):
    """Evaluate LLM output based on question type."""
    true_path = "../data/true_labels/"
    llm_output = llm_output.strip()
    true_answer = get_true_answer_from_excel(
        true_path + answer_file, query_id, doc_name
    )

    # Load choices if needed
    choices = None
    choices = load_choices(query_id)

    if question_type == "single_choice":
        pred_letter = clean_answer(llm_output)
        correct_letter = (
            map_true_answer_to_choices(true_answer, choices)[0]
            if choices
            else true_answer.upper()
        )
        acc = int(pred_letter == correct_letter)
        return {
            "raw_output": llm_output,
            "pred_letter": pred_letter,
            "true_letter": correct_letter,
            "accuracy": acc,
        }

    elif question_type == "multiple_choice":
        pred_letters = set(re.findall(r"[A-Z]", llm_output.upper()))
        correct_letters = (
            set(map_true_answer_to_choices(true_answer, choices))
            if choices
            else set(true_answer.upper())
        )

        # Initialize other_value
        other_value = None

        # Check if 'Z' is in the predicted letters
        if "Z" in pred_letters and other_callback is not None:
            # Call the callback using the original output
            other_value = other_callback(llm_output)

            # Use callback result for evaluation
            # Convert the returned answer to uppercase letters
            pred_letters = set(re.findall(r"[A-Z]", other_value.upper()))

        acc = int(pred_letters == correct_letters)

        return {
            "raw_output": llm_output,  # Original LLM output
            "pred_letters": sorted(pred_letters),  # Possibly corrected letters
            "true_letters": sorted(correct_letters),
            "accuracy": acc,
            "other_value": other_value,  # Callback value
        }

    elif question_type == "numeric":
        try:
            pred_num = float(llm_output)
            true_num = float(true_answer)
            acc = int(pred_num == true_num)
        except ValueError:
            acc = 0
        return {
            "raw_output": llm_output,
            "pred_value": llm_output,
            "true_value": true_answer,
            "accuracy": acc,
        }

    elif question_type == "list":
        pred_items = set(map(str.strip, llm_output.split(",")))
        true_items = (
            set(map(str.strip, true_answer.split(",")))
            if isinstance(true_answer, str)
            else set(true_answer)
        )
        acc = int(pred_items == true_items)
        return {
            "raw_output": llm_output,
            "pred_items": sorted(pred_items),
            "true_items": sorted(true_items),
            "accuracy": acc,
        }

    elif question_type == "open_ended":
        entail1 = nli_entailment(true_answer, llm_output)
        entail2 = nli_entailment(llm_output, true_answer)
        acc = int(entail1 and entail2)
        return {
            "raw_output": llm_output,
            "true_answer": true_answer,
            "accuracy": acc,
            "entail_true_to_pred": entail1,
            "entail_pred_to_true": entail2,
        }

    else:
        raise ValueError(f"Unknown question_type: {question_type}")
