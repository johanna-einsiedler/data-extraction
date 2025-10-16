# embeddings/openai_embedder.py
import os
import re
import sys
from typing import List

import numpy as np

sys.path.append(os.path.dirname(__file__))

from base_embedder import BaseEmbedder
from openai import OpenAI


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([item.embedding for item in response.data])
