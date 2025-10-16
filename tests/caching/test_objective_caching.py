import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pytest

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
OBJECTIVE_DIR = os.path.join(PROJECT_ROOT, "src/")
SRC_DIR = os.path.join(PROJECT_ROOT, "src/")

sys.path.insert(0, OBJECTIVE_DIR)

from reportlab.pdfgen import canvas


def make_dummy_pdf(path: Path, text="Hello world!"):
    """Create a minimal valid PDF."""
    c = canvas.Canvas(str(path))
    c.drawString(100, 750, text)
    c.save()


# # ---- Mock registry so imports work ----
# sys.modules["registry"] = SimpleNamespace(
#     CHUNKERS={"MockChunker": object},
#     EMBEDDERS={"MockEmbedder": object},
#     LLMS={"MockLLM": object},
#     LLMS_META={},
#     PARSERS={"MockParser": object},
#     RETRIEVERS={"MockRetriever": object},
# )

# ---- Create a temporary queries file BEFORE importing objective ----
TEST_QUERIES_PATH = Path(__file__).parent / "queries_with_prompts.json"

# Properly mock queries_with_prompts.json
mock_queries = {
    "q1": {
        "type": "mc",
        "prompts": {
            "simple": "What is 2 + 2?",
            "detailed": "Compute the sum of 2 and 2.",
        },
    }
}
TEST_QUERIES_PATH.write_text(json.dumps(mock_queries, indent=2))

# ---- Patch objective.py variable before import ----
import builtins

from vectorstore.numpy_store import NumpyVectorStore

builtins.QUERIES_JSON_PATH = str(TEST_QUERIES_PATH)

# Make sure the query_creation folder exists
queries_dir = Path(__file__).parent.parent / "query_creation"
queries_dir.mkdir(parents=True, exist_ok=True)

# Write the mock queries to the expected location too
queries_path = queries_dir / "queries_with_prompts.json"
queries_path.write_text(json.dumps(mock_queries, indent=2))

# Now you can import objective
import objective

# ---- Now import objective ----
from objective import (
    _hash_trial_key,
    ensure_chunks,
    ensure_embedding,
    ensure_parsed,
    ensure_query_embeddings,
    evaluate_answer,
    get_llm_text,
    load_cache,
    objective,
    save_cache,
)
from registry import CHUNKERS, EMBEDDERS, LLMS, LLMS_META, PARSERS, RETRIEVERS


@pytest.fixture
def fake_trial(tmp_path):
    """Simulate a minimal Optuna trial-like object."""

    class FakeTrial:
        number = 1

        def suggest_categorical(self, name, choices):
            # Deterministic: always take first option
            return choices[0]

    return FakeTrial()


@pytest.fixture
def test_dir(tmp_path):
    """Create an isolated test folder structure."""
    base = tmp_path / "test_logs"
    base.mkdir()
    (base / "trial_cache").mkdir()
    (base / "trial_logs").mkdir()
    return base


def test_hash_trial_key_consistency():
    """Same params/doc/qid → same hash."""
    params = {"a": 1, "b": "x"}
    h1 = _hash_trial_key(params, doc_idx=3, query_id="q1")
    h2 = _hash_trial_key(params, doc_idx=3, query_id="q1")
    assert h1 == h2
    # Changing one thing changes hash
    h3 = _hash_trial_key(params, doc_idx=4, query_id="q1")
    assert h1 != h3


def test_cache_roundtrip(tmp_path):
    """Ensure cache save/load consistency."""
    cache_path = tmp_path / "cache.json"
    test_cache = {"abc": {"accuracy": 0.9}}
    save_cache(test_cache)
    loaded = load_cache()
    assert loaded == test_cache


@pytest.fixture
def fake_trial(tmp_path):
    """Simulate a minimal Optuna trial-like object."""

    class FakeTrial:
        number = 1

        def suggest_categorical(self, name, choices):
            # Always pick the first choice deterministically
            return choices[0]

    return FakeTrial()


@pytest.fixture
def trial_mock():
    """Mock trial object with suggest_categorical method."""

    class TrialMock:
        def __init__(self):
            self.number = 0
            self._choices = {}

        def suggest_categorical(self, name, options):
            # Return the first option for deterministic behavior
            val = options[0]
            self._choices[name] = val
            return val

    return TrialMock()
