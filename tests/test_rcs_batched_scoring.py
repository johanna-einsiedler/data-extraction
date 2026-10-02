"""RCS re-ranker: one scoring call per item, robust parsing, usage via generate_fn."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from retrieval.rcs_retriever import RCSRetriever


class Store:
    def __init__(self, chunks): self.chunks = chunks
    def query(self, vec, top_k=5): return [(c, 1.0) for c in self.chunks[:top_k]]


def test_one_call_scores_all_candidates_and_ranks():
    calls = []
    def gen(p):
        calls.append(p); return '```json\n{"scores": {"1": 2, "2": 9, "3": 5}}\n```'
    r = RCSRetriever(llm_model="x/y", k=2, top_m=3, generate_fn=gen)
    out = r.retrieve([0.0], Store(["a", "b", "c"]), raw_query_text="q?")
    assert len(calls) == 1 and "[1] a" in calls[0] and "[3] c" in calls[0]
    assert out == [("b", 9), ("c", 5)]


def test_reasoning_prefix_and_missing_keys_are_tolerated():
    def gen(p): return 'Let me think... {"scores": {"2": "7"}} done'
    r = RCSRetriever(llm_model="x/y", k=3, top_m=3, generate_fn=gen)
    out = r.retrieve([0.0], Store(["a", "b", "c"]), raw_query_text="q?")
    assert out[0] == ("b", 7) and {s for _, s in out[1:]} == {0}


def test_unparseable_reply_keeps_store_order_after_retry():
    n = {"calls": 0}
    def gen(p): n["calls"] += 1; return "no json here"
    r = RCSRetriever(llm_model="x/y", k=2, top_m=3, generate_fn=gen)
    out = r.retrieve([0.0], Store(["a", "b", "c"]), raw_query_text="q?")
    assert n["calls"] == 2 and [c for c, _ in out] == ["a", "b"]


def test_parse_scores_clamps_and_ignores_out_of_range():
    sc = RCSRetriever._parse_scores('{"scores": {"1": 15, "2": -3, "9": 5, "x": 1}}', 3)
    assert sc == {1: 10, 2: 0}
