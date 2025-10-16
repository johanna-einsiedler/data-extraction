import os
import re
import sys
from typing import List

import numpy as np

sys.path.append(os.path.dirname(__file__))

from base_embedder import BaseEmbedder
from sentence_transformers import SentenceTransformer


class E5Embedder(BaseEmbedder):
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, normalize_embeddings=True))
