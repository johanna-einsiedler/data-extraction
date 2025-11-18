"""Integration-style tests that exercise the different retriever implementations."""

import json
import math
import os
import random
import sys

import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv

from vectorstore.numpy_store import NumpyVectorStore

# ---------- Setup ----------
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PARSERS_DIR = os.path.join(PROJECT_ROOT, "src/retrieval")
sys.path.insert(0, PARSERS_DIR)
from cross_encoder_retriever import CrossEncoderRetriever
from rcs_retriever import RCSRetriever
from token_budget_retriever import TokenBudgetRetriever
from topk_retriever import TopKRetriever


# ---- Mock Vector Store ----
class MockNumpyVectorStore:
    def __init__(self, num_docs=10, dim=4):
        self.embeddings = np.random.rand(num_docs, dim)
        self.texts = [f"Document {i}" for i in range(num_docs)]

    def query(self, query_vec, top_k=5):
        # Compute cosine similarity for reproducibility
        sims = (
            self.embeddings
            @ query_vec
            / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec))
        )
        idx_sorted = np.argsort(sims)[::-1][:top_k]
        return [(self.texts[i], float(sims[i])) for i in idx_sorted]


# ---- Fixtures ----
@pytest.fixture
def store():
    return MockNumpyVectorStore(num_docs=8, dim=4)


@pytest.fixture
def retriever():
    return TopKRetriever(k=3)


# ------------------------------
# ✅ Top-K Retriever Tests
# ------------------------------
def test_topk_retriever_returns_correct_number(store, retriever, tmp_path):
    query_vec = np.random.rand(4)
    results = retriever.retrieve(query_vec, store)

    assert isinstance(results, list)
    assert len(results) == retriever.k
    for text, score in results:
        assert isinstance(text, str)
        assert isinstance(score, float)

    # Check sorted descending by similarity
    sims = [score for _, score in results]
    assert sims == sorted(sims, reverse=True)

    # Debug save
    debug_dir = tmp_path / "retrieval_debug"
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = debug_dir / "topk_results.txt"
    with open(debug_path, "w") as f:
        for text, score in results:
            f.write(f"{text} | score={score:.4f}\n")
    assert debug_path.exists()


def test_topk_retriever_empty_store_returns_empty(tmp_path):
    class EmptyStore:
        def query(self, query_vec, top_k):
            return []

    retriever = TopKRetriever(k=3)
    results = retriever.retrieve(np.random.rand(4), EmptyStore())
    assert results == []

    debug_dir = tmp_path / "retrieval_debug"
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = debug_dir / "empty_results.txt"
    with open(debug_path, "w") as f:
        f.write("No results\n")
    assert debug_path.exists()


def test_topk_respects_token_limit():
    class MockStore:
        def query(self, query_vec, top_k):
            return [
                ("short text", 0.9),
                ("longer text " * 100, 0.8),
                ("extra chunk", 0.7),
            ]

    retriever = TopKRetriever(k=3, token_limit=10, tokenizer=lambda x: x.split())
    results = retriever.retrieve([0.1, 0.2], MockStore())
    assert len(results) == 1
    assert results[0][0] == "short text"


# ------------------------------
# ✅ Token-Budget Retriever Tests
# ------------------------------
def test_token_budget_retriever_basic(tmp_path):
    retriever = TokenBudgetRetriever(token_budget=50)

    class MockStore:
        def __init__(self):
            self.texts = [
                "short chunk one",
                "medium chunk " + "x " * 100,
                "long chunk " + "x " * 500,
            ]

        def query(self, query_vec, top_k=None, **kwargs):
            return [
                ("short chunk one", 0.95),
                ("medium chunk " + "x " * 100, 0.85),
                ("long chunk " + "x " * 500, 0.75),
            ]

    store = MockStore()
    results = retriever.retrieve([0.1, 0.2], store)

    assert len(results) > 0
    total_tokens = sum(len(text.split()) for text, _ in results)
    assert total_tokens <= 50

    debug_dir = tmp_path / "retrieval_debug"
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = debug_dir / "token_budget_results.txt"
    with open(debug_path, "w") as f:
        for text, score in results:
            f.write(f"{text} | score={score:.4f}\n")
    assert debug_path.exists()


def test_token_budget_retriever_handles_empty_store():
    class EmptyStore:
        texts = []

        def query(self, query_vec, top_k=None, **kwargs):
            return []

    retriever = TokenBudgetRetriever(token_budget=50)
    results = retriever.retrieve(np.random.rand(4), EmptyStore())
    assert results == []


def test_token_budget_retriever_truncates_large_first_chunk(tmp_path):
    """If the top-ranked chunk exceeds the token budget, it should be truncated to fit."""
    from retrieval.token_budget_retriever import TokenBudgetRetriever

    # --- Mock store with one large chunk and one smaller one ---
    class MockStore:
        def __init__(self):
            self.texts = ["word " * 100, "small text"]

        def query(self, q, top_k=None, **kwargs):
            # Return the large one as most relevant
            return [(self.texts[0], 0.95), (self.texts[1], 0.90)]

    retriever = TokenBudgetRetriever(token_budget=20)
    results = retriever.retrieve([0.1], MockStore())

    # --- Assertions ---
    assert len(results) == 1, "Should return only one truncated chunk"
    truncated_text, score = results[0]
    assert isinstance(truncated_text, str)
    assert len(truncated_text.split()) == 20, (
        "Truncated text should match the token budget"
    )
    assert math.isclose(score, 0.95, rel_tol=1e-3)

    # --- Save debug output for inspection ---
    debug_dir = tmp_path / "retrieval_debug"
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = debug_dir / "token_budget_truncated.txt"
    with open(debug_path, "w") as f:
        f.write(
            f"Returned text ({len(truncated_text.split())} tokens):\n{truncated_text}\n\nScore={score}"
        )
    assert debug_path.exists()
    print(f"Saved TokenBudgetRetriever truncation debug to {debug_path}")


def test_token_budget_retriever_custom_tokenizer():
    query_vec = "what is ai?"

    retriever = TokenBudgetRetriever(token_budget=10, tokenizer=lambda x: list(x))

    class MockStore:
        def __init__(self):
            self.texts = ["abcd efgh", "ijkl mnopqr"]

        def query(self, query_vec, top_k=None, **kwargs):
            return [(t, 1.0 - i * 0.1) for i, t in enumerate(self.texts)]

    store = MockStore()
    results = retriever.retrieve([0.1], store)
    assert len(results) >= 1


# ------------------------------------------------------------------------------------
# Mock OpenAI client for RCSRetriever
# ------------------------------------------------------------------------------------
# @pytest.mark.integration
@pytest.mark.integration
@pytest.mark.parametrize(
    "llm_model", ["gpt-4.1-2025-04-14", "meta-llama/Llama-4-Scout-17B-16E-Instruct"]
)
def test_rcs_retriever_reranking(llm_model):
    query = "what is AI?"

    class SimpleStore(NumpyVectorStore):
        def __init__(self):
            self.texts = [
                "Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals.",
                "Foxes are small-to-medium-sized omnivorous mammals belonging to several genera of the family Canidae. They have a flattened skull; upright, triangular ears; a pointed, slightly upturned snout; and a long, bushy tail ('brush').",
            ]

        def query(self, query_vec, top_k=None):
            # Return initial scores as floats 0-1
            return [(t, 1.0 - i * 0.1) for i, t in enumerate(self.texts[:top_k])]

    store = SimpleStore()

    class DummyLLMClient:
        def __init__(self):
            self.chat = self.Chat(self)
            self.completions = self.Completions(self)

        class Chat:
            def __init__(self, outer):
                self.completions = self.Completions(outer)

            class Completions:
                def __init__(self, outer):
                    self.outer = outer

                def create(self, *, model, messages):
                    prompt = messages[0]["content"]
                    return self.outer._build_response(prompt, as_text=False)

        class Completions:
            def __init__(self, outer):
                self.outer = outer

            def create(self, *, model, prompt):
                return self.outer._build_response(prompt, as_text=True)

        @staticmethod
        def _extract_chunk(prompt: str) -> str:
            marker = "Chunk:"
            return prompt.split(marker, 1)[1].strip() if marker in prompt else prompt

        def _build_response(self, prompt: str, as_text: bool):
            chunk = self._extract_chunk(prompt)
            lowered = chunk.lower()
            relevant = "artificial intelligence" in lowered or "computational systems" in lowered
            relevance = 9 if relevant else 1
            payload = json.dumps({"summary": chunk[:80], "relevance_score": relevance})

            if as_text:
                class Choice:
                    def __init__(self, text):
                        self.text = text

                class Response:
                    def __init__(self, text):
                        self.choices = [Choice(text)]

                return Response(payload)

            class Message:
                def __init__(self, content):
                    self.content = content

            class Choice:
                def __init__(self, content):
                    self.message = Message(content)

            class Response:
                def __init__(self, content):
                    self.choices = [Choice(content)]

            return Response(payload)

    client = DummyLLMClient()
    retriever = RCSRetriever(
        llm_model=llm_model,
        top_m=2,
        k=2,
        client=client,
    )

    results = retriever.retrieve(query, store)

    # Save results for debugging
    filename = f"test_rcs_rerank_results_{llm_model.replace('/', '_')}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    assert len(results) == 2

    # Extract chunks and scores
    chunk_texts = [c for c, _ in results]
    scores = [s for _, s in results]

    # Check that AI chunk is ranked higher than irrelevant chunk
    ai_chunk = store.texts[0]
    other_chunk = store.texts[1]
    ai_index = chunk_texts.index(ai_chunk)
    other_index = chunk_texts.index(other_chunk)

    assert scores[ai_index] >= scores[other_index], (
        f"Reranking failed for {llm_model}: AI chunk is not ranked higher"
    )

    # Scores should be in 0-10
    for score in scores:
        assert 0 <= score <= 10


# ---- Fixtures ----
@pytest.fixture
def cross_encoder_store():
    """Mock vector store for cross-encoder tests"""

    class MockStore:
        def __init__(self):
            self.texts = [
                "Python is a popular programming language for AI and web development.",
                "Cats are small mammals often kept as pets.",
                "JavaScript is used for web development and interactive websites.",
                "Bananas are a fruit that grows in tropical climates.",
            ]
            self.embeddings = np.random.rand(len(self.texts), 4)

        def query(self, query_vec, top_k=5):
            sims = (
                self.embeddings
                @ query_vec
                / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec))
            )
            idx_sorted = np.argsort(sims)[::-1][:top_k]
            return [(self.texts[i], float(sims[i])) for i in idx_sorted]

    return MockStore()


@pytest.fixture
def cross_encoder_retriever():
    retriever = CrossEncoderRetriever(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_m=3, k=2
    )

    class DummyModel:
        def predict(self, pairs):
            scores = []
            for _, chunk in pairs:
                lowered = chunk.lower()
                if "programming" in lowered or "web development" in lowered:
                    scores.append(0.9)
                else:
                    scores.append(0.1)
            return scores

    retriever.model = DummyModel()
    return retriever


# ---- Tests ----
def test_cross_encoder_retriever_init():
    retriever = CrossEncoderRetriever(top_m=3, k=2)
    assert retriever.top_m == 3
    assert retriever.k == 2
    assert retriever.model is None


def test_cross_encoder_retriever_returns_top_k(
    cross_encoder_store, cross_encoder_retriever, tmp_path
):
    query_vec = np.random.rand(4)
    raw_query = "Tell me about programming languages"

    results = cross_encoder_retriever.retrieve(
        query_vec, cross_encoder_store, raw_query_text=raw_query
    )

    # Should return exactly k results
    assert isinstance(results, list)
    assert len(results) == cross_encoder_retriever.k
    # Each result is a tuple (text, score)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in results)
    # Scores should be numbers
    import numbers

    assert all(isinstance(score, numbers.Real) for _, score in results)
    # Save debug output
    debug_dir = tmp_path / "retrieval_debug"
    debug_dir.mkdir(exist_ok=True)
    debug_path = debug_dir / "cross_encoder_results.txt"
    with open(debug_path, "w") as f:
        for text, score in results:
            f.write(f"{text} | score={score:.4f}\n")
    assert debug_path.exists()


def test_cross_encoder_requires_raw_query_text(
    cross_encoder_store, cross_encoder_retriever
):
    query_vec = np.random.rand(4)
    with pytest.raises(ValueError) as exc_info:
        cross_encoder_retriever.retrieve(query_vec, cross_encoder_store)
    assert "raw_query_text" in str(exc_info.value)


def test_cross_encoder_retriever_reranks_correctly(
    cross_encoder_store, cross_encoder_retriever
):
    query_vec = np.random.rand(4)
    raw_query = "Tell me about programming languages"

    results = cross_encoder_retriever.retrieve(
        query_vec, cross_encoder_store, raw_query_text=raw_query
    )

    # Extract texts and scores
    texts, scores = zip(*results)

    # We know the relevant chunks are the ones about programming
    relevant_chunks = [
        "Python is a popular programming language for AI and web development.",
        "JavaScript is used for web development and interactive websites.",
    ]

    irrelevant_chunks = [
        "Cats are small mammals often kept as pets.",
        "Bananas are a fruit that grows in tropical climates.",
    ]

    # Check that all returned chunks are relevant
    for text in texts:
        assert text in relevant_chunks, f"Unexpected irrelevant chunk returned: {text}"

    # Optional: ensure ordering is correct (highest scored chunk first)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_texts = [texts[i] for i in sorted_indices]
    assert all(t in relevant_chunks for t in sorted_texts), (
        "Reranking failed to prioritize relevant chunks"
    )
