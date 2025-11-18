# retrieval/base_retriever.py
from abc import ABC, abstractmethod
from typing import Callable, List, Protocol, Tuple

import numpy as np


class VectorStoreProtocol(Protocol):
    def query(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        ...


def default_tokenizer(text: str) -> List[str]:
    """Fallback tokenizer that splits on whitespace."""
    return text.split()


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self, query_vec: np.ndarray, store: VectorStoreProtocol
    ) -> List[Tuple[str, float]]:
        """Retrieve chunks from the vector store given a query vector."""
        raise NotImplementedError
