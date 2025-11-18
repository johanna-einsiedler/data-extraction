# retrieval/topk_retriever.py
from typing import Callable, List, Tuple

from .base_retriever import BaseRetriever, VectorStoreProtocol, default_tokenizer


class TopKRetriever(BaseRetriever):
    def __init__(
        self,
        k: int = 5,
        token_limit: int | None = None,
        tokenizer: Callable[[str], List[str]] | None = None,
    ):
        """
        Args:
            k (int): Max number of chunks to consider.
            token_limit (int | None): Max total token length allowed (e.g. 4096).
            tokenizer (callable): Optional tokenizer with len(tokenizer(text)) → token count.
        """
        self.k = k
        self.token_limit = token_limit
        self.tokenizer = tokenizer or default_tokenizer

    def retrieve(self, query_vec, store: VectorStoreProtocol) -> List[Tuple[str, float]]:
        results = store.query(query_vec, top_k=self.k)

        # --- Optional token budget filtering ---
        if self.token_limit is not None:
            selected, total_tokens = [], 0
            for text, score in results:
                num_tokens = len(self.tokenizer(text))
                if total_tokens + num_tokens > self.token_limit:
                    break
                selected.append((text, score))
                total_tokens += num_tokens
            results = selected

        return results
