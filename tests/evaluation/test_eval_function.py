"""Golden tests for the evaluation pipeline using curated prompt/answer pairs."""

from pathlib import Path
import json
import os
import sys

import pytest
from dotenv import find_dotenv, load_dotenv

# Load API keys for optional parser instantiation
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import evaluate_answer
from objective import get_llm_text
from registry import CHUNKERS, EMBEDDERS, LLMS, LLMS_META, PARSERS, RETRIEVERS

QUERIES_JSON_PATH = PROJECT_ROOT / "query_creation" / "queries_with_prompts.json"


def load_queries():
    """Load the canonical query definitions used across evaluation tests."""
    with open(QUERIES_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


QUERIES = load_queries()

UNIT_TESTS_PATH = CURRENT_DIR / "unit_test_examples.json"
with open(UNIT_TESTS_PATH, "r", encoding="utf-8") as f:
    _UNIT_TESTS_RAW = json.load(f)

KEEP_KEYS = ['1.8', '1.2.1', '3.2', '3.2.1', '3.4', '3.4.1', '3.5', '3.5.1', '4.1', '4.2', '2.6', '4.2.1', '5.3', '0.7', '0.7.1', '0.7.2', '0.7.3', '0.7.4', '0.8', '0.9', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.9', '2.2', '2.1', '2.3', '2.4', '2.5', '3.1', '4.3', '4.4', '4.6', '4.7', '5.15.2', '5.4']
UNIT_TESTS = {k: _UNIT_TESTS_RAW[k] for k in KEEP_KEYS if k in _UNIT_TESTS_RAW}

# Parameter to control whether to call real LLM or use fake outputs
EVAL_LLM = False  # Set to True to call actual LLM

@pytest.mark.parametrize(
    "query_id,ex_name,example",
    [
        (qid, ex_name, ex_data)
        for qid, examples in UNIT_TESTS.items()
        for ex_name, ex_data in examples.items()
    ],
)
def test_unit_example(query_id, ex_name, example):
    """Test unit examples either using fake LLM output or real LLM."""
    qdata = QUERIES[query_id]

    if EVAL_LLM:
        # Call LLM with the excerpt
        base_prompt = qdata["prompts"]["base_prompt"]
        prompt_filled = base_prompt.format(context=example.get("excerpt", ""))

        llm_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        ans_gen = LLMS[llm_model]

        llm_output_raw = ans_gen.generate(
            query=prompt_filled,
            chunks=[],  # no retrieval chunks for unit tests
            return_logprobs=True,
        )
        llm_output = get_llm_text(llm_output_raw)
    else:
        # Use fake LLM output from the example
        llm_output = example["llm_output"]

    # Wrap expected answer
    TRUE_ANSWERS = {"unit_test_doc": {query_id: example["true_answer"]}}

    # Evaluate
    eval_result = evaluate_answer(
        llm_output=llm_output,
        question_type=qdata["type"],
        doc_name="unit_test_doc",
        query_id=query_id,
        TRUE_ANSWERS=TRUE_ANSWERS,
        QUERIES=QUERIES,
        other_callback=None,
    )

    # Verbose output for debugging
    print(f"\n=== Query {query_id} | Example {ex_name} ===")
    print("Excerpt:", example.get("excerpt", ""))
    print("LLM Output:", llm_output)
    print("Expected:", example["true_answer"])
    print("Evaluation:", eval_result)

    # Compare evaluation accuracy to example's expected 'eval'
    expected_eval = example.get("eval", 1)  # default to 1 if not specified
    assert eval_result.get("accuracy") == expected_eval, (
        f"Failed on query {query_id}, example {ex_name}. "
        f"Expected eval={expected_eval}, got accuracy={eval_result.get('accuracy')}"
    )
