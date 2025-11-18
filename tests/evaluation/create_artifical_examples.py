"""Helper script for generating artificial evaluation examples via Together API."""

import json
import os
import re
import string
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
from together import Together

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")


class SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        return kwargs.get(key, "")


safe_format = SafeFormatter().vformat
client = Together(api_key=TOGETHER_API_KEY)
QUERIES_JSON_PATH = Path(
    "../../query_creation/queries_with_prompts.json"
)  # Path to your JSON file


def load_queries():
    """Load the query catalog so we can sample prompts for artificial examples."""
    with open(QUERIES_JSON_PATH, "r", encoding="utf-8") as f:
        query_dict = json.load(f)

    queries = {}
    for qid, q in query_dict.items():
        queries[qid] = q  # keep original structure
    return queries


QUERIES = load_queries()
# query_ids = ["0.7", "0.7.1", "0.7.3", "1.2", "1.8", "1.2.1"]
# QUERIES = {qid: QUERIES.get(qid) for qid in query_ids}
#################################################
# PROMPTS #
################################################


prompt_folder = Path("unit_test_llm_prompts")

# Map of prompt types to their filenames
prompt_files = {
    "multiple_choice": "multiple_choice.txt",
    "single_choice": "single_choice.txt",
    "numeric": "numeric.txt",
    "open_ended": "open_ended.txt",
    "list": "list.txt",
}

# Load prompts from files
prompts = {
    key: (prompt_folder / filename).read_text(encoding="utf-8")
    for key, filename in prompt_files.items()
}


def generate_unit_tests_from_scheme(
    scheme_json,
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    n_examples=3,
    output_path="unit_test_examples.json",
):
    """
    Generate clean unit test examples for each coding item in a scheme.

    - Removes <think>...</think> content from outputs.
    - Structures JSON with one entry per query id and one dictionary per version/example.

    Args:
        scheme_json (dict): Your coding scheme.
        client: Initialized API client (e.g., from openai import OpenAI; client = OpenAI()).
        model (str): Model name.
        n_examples (int): Number of examples per coding variable.
        output_path (str): Path to save final JSON file.

    Returns:
        dict: {query_id: {example_1: {...}, example_2: {...}, ...}}
    """
    results = {}

    for query_id, item in scheme_json.items():
        print(f"⏳ Generating examples for {query_id}: {item.get('label', '')}")

        label = item.get("label", "")
        description = item.get("description", "")
        type_ = item.get("type", "")
        choices = item.get("choices", {})

        options_text = (
            "\n".join([f"{k}: {v['value']}" for k, v in choices.items()])
            if choices
            else "None"
        )

        instruction = prompts[type_]
        instruction = safe_format(
            instruction,
            (),
            {
                "label": label,
                "n_examples": n_examples,
                "description": description,
                "options_text": options_text,
            },
        )
        try:
            # ---- Generate from model ----
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": instruction}],
                temperature=0.7,
            )

            raw_output = response.choices[0].message.content.strip()

            # ---- Remove <think>...</think> ----
            cleaned_output = re.sub(
                r"<think>.*?</think>", "", raw_output, flags=re.DOTALL
            ).strip()

            # ---- Extract JSON safely ----
            # Match anything between ```json ... ``` or bare [ ... ]
            json_match = re.search(r"```json(.*?)```", cleaned_output, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # fallback: extract between first [ and last ]
                bracket_match = re.search(r"\[.*\]", cleaned_output, re.DOTALL)
                json_text = bracket_match.group(0) if bracket_match else cleaned_output

            # ---- Parse JSON ----
            try:
                parsed = json.loads(json_text)
            except json.JSONDecodeError:
                print(f"⚠️ JSON parse failed for {query_id}. Keeping raw text.")
                parsed = [
                    {
                        "excerpt": "PARSING_FAILED",
                        "expected_answers": [],
                        "raw": cleaned_output,
                    }
                ]

            # ---- Reformat examples ----
            examples_dict = {f"example_{i + 1}": ex for i, ex in enumerate(parsed)}
            results[query_id] = examples_dict

        except Exception as e:
            print(f"❌ Error for {query_id}: {e}")
            results[query_id] = {"error": str(e)}

    # ---- Save ----
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved cleaned unit test examples to {output_path}")
    return results


if __name__ == "__main__":
    generate_unit_tests_from_scheme(
        QUERIES,
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        n_examples=3,
    )
