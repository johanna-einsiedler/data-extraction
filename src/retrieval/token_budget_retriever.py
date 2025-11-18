# retrieval/token_budget_retriever.py
from typing import Callable, List, Tuple

from .base_retriever import BaseRetriever, VectorStoreProtocol, default_tokenizer


class TokenBudgetRetriever(BaseRetriever):
    def __init__(self, token_budget: int = 2048, tokenizer: Callable = None):
        """
        Args:
            token_budget (int): Maximum total token count allowed.
            tokenizer (callable, optional): Function that splits text into tokens.
                                            Defaults to simple whitespace split.
        """
        self.token_budget = token_budget
        self.tokenizer = tokenizer or default_tokenizer

    def retrieve(self, query_vec, store: VectorStoreProtocol) -> List[Tuple[str, float]]:
        if not hasattr(store, "texts") or not store.texts:
            return []

        # Retrieve all chunks sorted by similarity
        sims = store.query(query_vec, top_k=len(store.texts))

        selected = []
        total_tokens = 0

        for text, score in sims:
            tokens = self.tokenizer(text)
            token_count = len(tokens)

            if total_tokens + token_count <= self.token_budget:
                # ✅ Fits within remaining budget
                selected.append((text, score))
                total_tokens += token_count
            elif total_tokens == 0 and token_count > self.token_budget:
                # ✅ Top chunk is too large — truncate it to fit the token budget
                truncated_text = " ".join(tokens[: self.token_budget])
                selected.append((truncated_text, score))
                total_tokens = self.token_budget
                break
            else:
                # ✅ Budget full — stop selecting more chunks
                break

        return selected
