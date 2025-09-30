# retrieval/topk_retriever.py
from typing import List, Tuple

from vectorstore.numpy_store import NumpyVectorStore

from .base_retriever import BaseRetriever


class TopKRetriever(BaseRetriever):
    def __init__(self, k: int = 5):
        self.k = k

    def retrieve(self, query_vec, store: NumpyVectorStore) -> List[Tuple[str, float]]:
        return store.query(query_vec, top_k=self.k)
