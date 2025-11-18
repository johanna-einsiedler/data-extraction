# embeddings/base_embedder.py
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BaseEmbedder(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Convert list of texts into embeddings (2D array)."""
        raise NotImplementedError

    def embed_query(
        self,
        query: str,
        strategy: str = "full",
        concept: Optional[str] = None,
        definition: Optional[str] = None,
        additional_information: Optional[str] = None,
        possible_answers: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Embed a query using either 'full' or 'label' strategy.
        - full: embed raw query text
        - label: embed structured concept/definition
        """
        if strategy == "full":
            return self.embed([query])[0]

        elif strategy == "label":
            parts = []
            if concept:
                parts.append(f"Concept: {concept}")
            if definition:
                parts.append(f"Definition: {definition}")
            if additional_information:
                parts.append(f"Additional info: {additional_information}")
            if possible_answers:
                parts.append("Possible answers: " + ", ".join(possible_answers))

            structured_text = "\n".join(parts)
            return self.embed([structured_text])[0]

        else:
            raise ValueError(f"Unknown query embedding strategy: {strategy}")


class DefaultEmbedder(BaseEmbedder):
    """Lightweight fallback embedder for tests and CPU-only environments."""

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            stripped = text.strip()
            length = float(len(stripped))
            words = float(len(stripped.split())) if stripped else 0.0
            # Simple checksum-style feature to add variance but deterministic
            checksum = float(sum(ord(ch) for ch in stripped) % 10_000)
            vectors.append(np.array([length, words, checksum], dtype=np.float32))
        if not vectors:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack(vectors)
