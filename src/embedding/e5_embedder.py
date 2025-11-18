"""SentenceTransformer-based embedder using the E5 checkpoint family."""

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .base_embedder import BaseEmbedder


class E5Embedder(BaseEmbedder):
    """Embedder backed by the E5-series models."""

    def __init__(self, model_name: str = "intfloat/e5-large-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model: Optional[SentenceTransformer] = None

    def _ensure_model(self) -> SentenceTransformer:
        """Load the E5 SentenceTransformer lazily to avoid import-time failures."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load SentenceTransformer model '{self.model_name}'. "
                    "Check the model identifier and network/cache availability."
                ) from exc
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        model = self._ensure_model()
        return np.array(model.encode(texts, normalize_embeddings=True))
