# retrieval/token_budget_retriever.py
from typing import List, Tuple

from vectorstore.numpy_store import NumpyVectorStore

from .base_retriever import BaseRetriever


class TokenBudgetRetriever(BaseRetriever):
    def __init__(self, token_budget: int = 2048):
        self.token_budget = token_budget

    def retrieve(self, query_vec, store: NumpyVectorStore) -> List[Tuple[str, float]]:
        # Retrieve all chunks sorted by similarity
        sims = store.query(query_vec, top_k=len(store.texts))

        selected = []
        total_tokens = 0

        for text, score in sims:
            tokens = len(text.split())  # crude token approximation
            if total_tokens + tokens > self.token_budget:
                break
            selected.append((text, score))
            total_tokens += tokens

        return selected
