"""Embedding tests covering local and OpenAI-backed embedders."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv

# ---------- Setup ----------
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PARSERS_DIR = os.path.join(PROJECT_ROOT, "src/embedding")
sys.path.insert(0, PARSERS_DIR)
from base_embedder import BaseEmbedder
from bge_embedder import BGEEmbedder
from e5_embedder import E5Embedder
from openai_embedder import OpenAIEmbedder


@pytest.mark.parametrize(
    "embedder_class,kwargs",
    [
        (BGEEmbedder, {"model_name": "BAAI/bge-base-en"}),  # light & fast
        (E5Embedder, {"model_name": "intfloat/e5-large-v2"}),  # light & fast
    ],
)
def test_local_embedders_output_and_similarity(embedder_class, kwargs):
    """Test local embedders for correct numpy output and semantic closeness."""
    embedder = embedder_class(**kwargs)

    texts = [
        "The cat sits on the mat.",
        "A small kitten is sitting on a mat.",
        "The stock market is up today.",
    ]

    embeddings = embedder.embed(texts)

    # 1️⃣ Check output type & shape
    assert isinstance(embeddings, np.ndarray), "Embedding output must be a numpy array"
    assert embeddings.ndim == 2, "Embeddings must be 2D"
    assert embeddings.shape[0] == len(texts), "One embedding per input text"
    assert np.isfinite(embeddings).all(), "All embedding values must be finite"

    # 2️⃣ Check semantic closeness
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_1_2 = cosine(embeddings[0], embeddings[1])
    sim_1_3 = cosine(embeddings[0], embeddings[2])

    print(f"{embedder_class.__name__}: sim(1,2)={sim_1_2:.3f}, sim(1,3)={sim_1_3:.3f}")
    assert sim_1_2 > sim_1_3, "Similar texts should be closer than unrelated ones"


@pytest.mark.requires_openai_api
def test_openai_embedder_real_api():
    """Test OpenAIEmbedder with real API call (requires OPENAI_API_KEY env var)."""
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set in environment")

    embedder = OpenAIEmbedder(api_key=api_key, model="text-embedding-3-large")

    texts = [
        "The sky is blue and clear today.",
        "It is a bright and clear blue sky.",
        "I love eating pizza with cheese.",
    ]

    embeddings = embedder.embed(texts)

    # 1️⃣ Type & shape
    assert isinstance(embeddings, np.ndarray), "Must return numpy array"
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == len(texts)
    assert np.isfinite(embeddings).all(), "All values must be finite"

    # 2️⃣ Similarity check
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_1_2 = cosine(embeddings[0], embeddings[1])
    sim_1_3 = cosine(embeddings[0], embeddings[2])

    print(f"OpenAIEmbedder: sim(1,2)={sim_1_2:.3f}, sim(1,3)={sim_1_3:.3f}")
    assert sim_1_2 > sim_1_3, "Expected similar sentences to be closer"
