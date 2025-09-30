# embeddings/openai_embedder.py
from typing import List

import numpy as np
from openai import OpenAI

from .base_embedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([item.embedding for item in response.data])
