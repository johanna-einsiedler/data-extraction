from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .base_embedder import BaseEmbedder


class E5Embedder(BaseEmbedder):
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, normalize_embeddings=True))
