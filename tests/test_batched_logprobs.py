"""Batched answer-token logprobs (RQ2 procedure (a)): the generator returns the
ordered token stream with top-5 alternatives and batched.locate_answer_tokens
finds each item's answer tokens by its JSON key."""
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import batched  # noqa: E402
from answer_generation.answer_generator import _ordered_logprob_content  # noqa: E402

TEXT = '{"1.1": "135", "1.5": "A,C", "1.3": "adolescents", "4.5.1": 0.85}'
TOKENS = ['{"', '1', '.', '1', '":', ' "', '135', '",', ' "', '1', '.', '5', '":', ' "', 'A', ',', 'C', '",',
          ' "', '1', '.', '3', '":', ' "', 'adoles', 'cents', '",', ' "', '4', '.', '5', '.', '1', '":', ' ', '0', '.', '85', '}']
assert "".join(TOKENS) == TEXT


def _stream(tokens=TOKENS, text=TEXT):
    return {"text": text, "tokens": list(tokens), "logprobs": [-0.01 * i for i in range(len(tokens))],
            "top_logprobs": [[{"token": t, "logprob": -0.01 * i}] for i, t in enumerate(tokens)]}


def test_locates_string_value_tokens():
    r = batched.locate_answer_tokens([_stream()], "1.1")
    assert r["tokens"] == ["135"] and r["min_logprob"] == r["logprobs"][0]
    r = batched.locate_answer_tokens([_stream()], "1.5")
    assert r["tokens"] == ["A", ",", "C"]
    r = batched.locate_answer_tokens([_stream()], "1.3")
    assert r["tokens"] == ["adoles", "cents"] and r["min_logprob"] == min(r["logprobs"])
    assert len(r["top_logprobs"]) == 2


def test_locates_numeric_value_and_key_prefix_is_not_confused():
    r = batched.locate_answer_tokens([_stream()], "4.5.1")
    assert r["tokens"] == ["0", ".", "85"]
    assert batched.locate_answer_tokens([_stream()], "4.5") == {}      # no such key
    assert batched.locate_answer_tokens([_stream()], "1") == {}


def test_tokens_merged_with_quotes_are_still_found():
    text = '{"1.1": "135"}'
    toks = ['{"1.1":', ' "135', '"}']
    r = batched.locate_answer_tokens([_stream(toks, text)], "1.1")
    assert r["tokens"] == [' "135']


def test_two_call_split_streams_and_missing_logprobs():
    s1 = _stream(['{"', '1', '.', '1', '":', ' "', '135', '"}'], '{"1.1": "135"}')
    s2 = _stream(['{"', '1', '.', '3', '":', ' "', 'x', '"}'], '{"1.3": "x"}')
    assert batched.locate_answer_tokens([s1, s2], "1.3")["tokens"] == ["x"]
    assert batched.locate_answer_tokens([], "1.1") == {}
    assert batched.logprob_stream({"text": "{}"}) is None
    assert batched.logprob_stream({"text": "{}", "logprobs_content": []}) is None


def test_ordered_content_keeps_duplicates_and_alternatives():
    T = types.SimpleNamespace
    content = [T(token="A", logprob=-0.1, top_logprobs=[T(token="A", logprob=-0.1), T(token="B", logprob=-2.5)]),
               T(token=",", logprob=-0.2, top_logprobs=None),
               T(token="A", logprob=-0.3, top_logprobs=[])]
    out = _ordered_logprob_content(content)
    assert [c["token"] for c in out] == ["A", ",", "A"]
    assert out[0]["top"] == [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.5}]
    st = batched.logprob_stream({"text": "A,A", "logprobs_content": out})
    assert st["tokens"] == ["A", ",", "A"] and st["logprobs"] == [-0.1, -0.2, -0.3]
