"""SentenceTransformer-based embedder using the BGE checkpoint family."""

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .base_embedder import BaseEmbedder


class BGEEmbedder(BaseEmbedder):
    """Embedder backed by BAAI's BGE models (via sentence-transformers)."""

    def __init__(self, model_name: str = "BAAI/bge-base-en", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model: Optional[SentenceTransformer] = None

    def _ensure_model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model lazily so tests can run offline."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:  # SentenceTransformer raises generic exceptions
                raise RuntimeError(
                    f"Failed to load SentenceTransformer model '{self.model_name}'. "
                    "Ensure the model name is valid and the environment has internet access or a cached copy."
                ) from exc
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        model = self._ensure_model()
        return np.array(model.encode(texts, normalize_embeddings=True))
