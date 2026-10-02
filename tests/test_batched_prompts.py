"""Guards on the batched prompt builder.

The expensive failure this file exists to prevent: the stored per-item prompts each
carry an "Excerpt: {context} Answer:" slot, and create_prompts.py emits that slot
once per few-shot example. Concatenating them without dereferencing would paste the
document 63 times (~934k tokens, ~7x the smallest roster window) on every paid call.
"""
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import batched  # noqa: E402

VARIANTS = [
    "base_prompt",
    "cot_prompt",
    "reasoning_prompt",
    "rewritten_prompt",
    "synthetic_few_shot_prompt",
    "true_few_shot_prompt",
]


@pytest.fixture(scope="module")
def QUERIES():
    with open(ROOT / "query_creation" / "queries_with_prompts.json") as fh:
        return json.load(fh)


def test_native_keeps_its_own_contract_wording(QUERIES):
    """batched_native must not gain the scoping block: the 26 copied Phase-1
    trials are recorded under it and must describe what would run."""
    p = batched.build_batched_prompt(QUERIES, "batched_native")
    assert "--- begin question instructions ---" not in p
    assert p.count("\n    Answer format:") == len(QUERIES)


def test_native_prompt_is_unchanged(QUERIES):
    """The 26 copied batched trials are recorded as batched_prompt_type=batched_native.
    If this byte-diff ever fails, those trials no longer describe what would run."""
    golden = _legacy_build(QUERIES)
    assert batched.build_batched_prompt(QUERIES) == golden
    assert batched.build_batched_prompt(QUERIES, "batched_native") == golden


def _legacy_build(QUERIES):
    """Verbatim copy of build_batched_prompt as of the Phase-1 study."""
    body = "\n\n".join(batched._one_item(i, it) for i, it in QUERIES.items())
    tail = (
        "\nReturn ONLY a single JSON object mapping each question id to its answer, "
        "using the per-question format. Include EVERY id. For any \"Other\" (Y) "
        'choice, add a "<id>__other" key with the free text. Example:\n'
        '{"0.7": "2", "1.1": "135", "1.5": "A,C", "0.7.1": "A,Y", '
        '"0.7.1__other": "caret", "1.3": "adolescents"}\n'
        "No explanation, no extra text."
    )
    return (
        batched._HEADER
        + "\n<document>\n{context}\n</document>\n\n<questions>\n"
        + body
        + "\n</questions>\n"
        + tail
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_exactly_one_document_insertion(QUERIES, variant):
    """THE guard. str.format substitutes every {context}, so more than one here
    multiplies the document across the prompt."""
    p = batched.build_batched_prompt(QUERIES, variant)
    assert p.count("{context}") == 1, (
        f"{variant}: {p.count('{context}')} placeholders -- each one pastes the "
        f"whole document again"
    )
    assert p.index("{context}") < p.index("<questions>")


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_item_present_with_answer_contract(QUERIES, variant):
    p = batched.build_batched_prompt(QUERIES, variant)
    for iid in QUERIES:
        assert f"[{iid}]" in p, f"{variant}: item {iid} missing"
    # our contract line names the id, so it cannot collide with the markdown
    # "**Answer format:**" that rewritten_prompt embeds in items 1.3 and 4.5.3
    assert p.count('\n    Value for "') == len(QUERIES)
    # each block is explicitly scoped, so the "answer with only..." directives the
    # stored prompts carry read as describing one field, not the whole reply
    assert p.count("--- begin question instructions ---") == len(QUERIES)


def test_true_few_shot_falls_back_like_individual_mode(QUERIES):
    """_select_prompt silently downgrades these 7 to synthetic; batched must match
    or the two modes are not comparable."""
    for iid in batched._TFS_FALLBACK:
        if iid in QUERIES:
            assert not (QUERIES[iid].get("prompts") or {}).get("true_few_shot_prompt")
    batched.build_batched_prompt(QUERIES, "true_few_shot_prompt")  # must not raise


def test_split_covers_every_item_exactly_once(QUERIES):
    a, b = batched.split_item_ids(QUERIES)
    assert set(a) | set(b) == set(QUERIES)
    assert not (set(a) & set(b))
    for half in (a, b):
        p = batched.build_batched_prompt(QUERIES, "base_prompt", item_ids=half)
        assert p.count("{context}") == 1
        assert p.count('\n    Value for "') == len(half)


def test_collapse_detector(QUERIES):
    ok, n = batched.response_is_usable("not json at all", QUERIES)
    assert not ok and n == 0
    ok, _ = batched.response_is_usable("{}", QUERIES)
    assert not ok
    full = json.dumps({i: "Z" for i in QUERIES})
    ok, n = batched.response_is_usable(full, QUERIES)
    assert ok and n == len(QUERIES)
    half = json.dumps({i: "A" for i in list(QUERIES)[: len(QUERIES) // 4]})
    ok, _ = batched.response_is_usable(half, QUERIES)
    assert not ok, "a quarter-filled response should not count as usable"
