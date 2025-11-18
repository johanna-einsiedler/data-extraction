# embeddings/openai_embedder.py
"""OpenAI embeddings client with lazy API instantiation."""

from typing import List, Optional

import numpy as np
from openai import OpenAI

from .base_embedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """Proxy embedder that talks to OpenAI's /embeddings endpoint."""

    def __init__(self, api_key: Optional[str], model: str = "text-embedding-3-large"):
        self.api_key = api_key
        self.model = model
        self._client: Optional[OpenAI] = None

    def _ensure_client(self) -> OpenAI:
        """Instantiate the OpenAI client only when needed."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required for OpenAIEmbedder.")
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def embed(self, texts: List[str]) -> np.ndarray:
        client = self._ensure_client()
        response = client.embeddings.create(model=self.model, input=texts)
        return np.array([item.embedding for item in response.data])
