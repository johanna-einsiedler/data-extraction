"""Unit tests for the TPE-stage pre-flight guards (cut rules of 2026-09-28)."""
import os, pathlib, sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("SAMPLE_ONLY", "1")
sys.path.insert(0, str(ROOT / "src"))

import optuna  # noqa: E402
from objective import (project_cost_per_paper, preflight_guards,  # noqa: E402
                       COST_CAP_USD_PER_PAPER)

META = {"X": {"pricing": {"input": 2.0, "output": 8.0}},          # gpt-4.1-like
        "cheap": {"pricing": {"input": 0.05, "output": 0.2}}}


def test_batched_is_always_cheap():
    c = {"mode": "full", "llm_model": "X", "parser": "PyMuPDFParser"}
    assert project_cost_per_paper(c, META) < COST_CAP_USD_PER_PAPER


def test_big_payload_rcs_on_expensive_model_is_cut():
    # chunk_size is characters (/4 tokens): 10 chunks of 20k chars re-ranked +
    # k=10 retrieved per item on a $2/M model must exceed the cap
    c = {"mode": "rag", "llm_model": "X", "parser": "GeminiParser",
         "retriever": "RCSRetriever", "k": 10, "chunk_size": 20000, "top_m": 20}
    assert project_cost_per_paper(c, META) > COST_CAP_USD_PER_PAPER


def test_payload_capped_at_document_size():
    small = {"mode": "rag", "llm_model": "X", "parser": "GeminiParser",
             "retriever": "TopKRetriever", "k": 5, "chunk_size": 1000}
    huge = dict(small, chunk_size=400000)
    capped = dict(small, chunk_size=10**9)
    # beyond the paper, cost stops growing entirely
    assert project_cost_per_paper(huge, META) == project_cost_per_paper(capped, META)


class _T:
    def __init__(self): self.attrs = {}
    def set_user_attr(self, k, v): self.attrs[k] = v


def test_disabled_model_pruned():
    with pytest.raises(optuna.TrialPruned, match="disabled"):
        preflight_guards(_T(), {"llm_model": "deepcogito/cogito-v2-1-671b",
                                "mode": "full", "parser": "PyMuPDFParser"})


def test_nemotron_full_pruned_but_rag_allowed():
    with pytest.raises(optuna.TrialPruned, match="nemotron"):
        preflight_guards(_T(), {"llm_model": "nvidia/nemotron-3-ultra-550b-a55b",
                                "mode": "full", "parser": "PyMuPDFParser"})
    t = _T()
    preflight_guards(t, {"llm_model": "nvidia/nemotron-3-ultra-550b-a55b",
                         "mode": "rag", "parser": "PyMuPDFParser",
                         "retriever": "TopKRetriever", "k": 3, "chunk_size": 500})
    assert "projected_usd_per_paper" in t.attrs


def test_projection_recorded_even_when_kept():
    t = _T()
    preflight_guards(t, {"llm_model": "openai/gpt-oss-20b", "mode": "full",
                         "parser": "PyMuPDFParser"})
    assert t.attrs["projected_usd_per_paper"] < COST_CAP_USD_PER_PAPER


def test_preflight_is_optimistic_about_reasoning():
    # reasoning volume is guarded in-flight; pre-flight must not price it in
    c = {"mode": "rag", "llm_model": "X", "parser": "GeminiParser",
         "retriever": "TopKRetriever", "k": 3, "chunk_size": 2500,
         "disable_reasoning": False}
    m = dict(META); m["X"] = {"pricing": {"input": 2.0, "output": 12.0},
                              "reasoning": True}
    assert project_cost_per_paper(c, m) < COST_CAP_USD_PER_PAPER


def test_wall_budget_constant_exceeds_slowest_healthy_trial():
    """The wall ceiling must clear legitimate slow trials (GLM-5.2 batched ran
    201 min; DeepSeek RAG 172 min) and still catch the crawls (#53 465 min)."""
    from objective import TRIAL_WALL_BUDGET_S
    assert TRIAL_WALL_BUDGET_S > 201 * 60, "would prune healthy GLM-5.2 trials"
    assert TRIAL_WALL_BUDGET_S < 465 * 60, "would not catch the #53-class crawl"


def test_qwen_timeout_exclusion_and_its_escape_hatch(monkeypatch):
    """Qwen3.5-9B is pruned on the infrastructure criterion, and the exclusion
    is liftable by env once the provider recovers."""
    monkeypatch.setenv("QWEN_TIMEOUT_EXCLUSION", "1")
    with pytest.raises(optuna.TrialPruned, match="Qwen"):
        preflight_guards(_T(), {"llm_model": "Qwen/Qwen3.5-9B", "mode": "full",
                                "parser": "PyMuPDFParser"})
    monkeypatch.setenv("QWEN_TIMEOUT_EXCLUSION", "0")
    t = _T()
    preflight_guards(t, {"llm_model": "Qwen/Qwen3.5-9B", "mode": "full",
                         "parser": "PyMuPDFParser"})
    assert "projected_usd_per_paper" in t.attrs
