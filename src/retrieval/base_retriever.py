# retrieval/base_retriever.py
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from vectorstore.numpy_store import NumpyVectorStore


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self, query_vec: np.ndarray, store: NumpyVectorStore
    ) -> List[Tuple[str, float]]:
        """
        Retrieve chunks from the vector store given a query vector.
        Returns a list of tuples (chunk_text, similarity_score)
        """
        pass
