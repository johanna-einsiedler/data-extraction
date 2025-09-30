# vectorstore/numpy_store.py
from pathlib import Path
from typing import List, Tuple

import numpy as np


class NumpyVectorStore:
    def __init__(self, save_path: str = "../data/vectorstore/store.npz"):
        self.save_path = Path(save_path)
        self.vectors = None
        self.texts = []

    def add(self, embeddings: np.ndarray, chunks: List[str]):
        if self.vectors is None:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])
        self.texts.extend(chunks)

    def save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.save_path, vectors=self.vectors, texts=self.texts)

    def load(self):
        data = np.load(self.save_path, allow_pickle=True)
        self.vectors = data["vectors"]
        self.texts = list(data["texts"])

    def query(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        # cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1)
        query_norm = np.linalg.norm(query_vec)
        sims = (self.vectors @ query_vec.T) / (norms * query_norm + 1e-10)

        top_idx = np.argsort(-sims)[:top_k]
        return [(self.texts[i], float(sims[i])) for i in top_idx]
