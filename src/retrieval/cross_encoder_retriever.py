# retrieval/cross_encoder_retriever.py
import os
import re
import sys
from typing import List, Tuple, Union

import numpy as np

sys.path.append(os.path.dirname(__file__))

from base_retriever import BaseRetriever
from sentence_transformers import CrossEncoder

from vectorstore.numpy_store import NumpyVectorStore


class CrossEncoderRetriever(BaseRetriever):
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m: int = 10,
        k: int = 5,
    ):
        self.top_m = top_m  # number of candidates to rerank
        self.k = k  # number of final retrieved chunks
        self.model = CrossEncoder(model_name)

    def retrieve(
        self,
        query_vec: Union[np.ndarray, List[float]],
        store: NumpyVectorStore,
        raw_query_text: str = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k chunks from a vector store using query embeddings,
        then rerank top-m chunks using a CrossEncoder with raw query text.

        Parameters:
        - query_vec: np.ndarray of shape (embedding_dim,) — the query embedding
        - store: NumpyVectorStore instance
        - raw_query_text: str — the raw text of the query for the CrossEncoder
                          (required for reranking)
        """

        if raw_query_text is None:
            raise ValueError(
                "CrossEncoder reranking requires raw_query_text for scoring."
            )

        # Step 1: Initial top-m retrieval using embeddings
        top_m_chunks = store.query(query_vec, top_k=self.top_m)
        chunk_texts = [chunk for chunk, _ in top_m_chunks]

        # Step 2: Prepare pairs for CrossEncoder: (query_text, chunk_text)
        pairs = [[raw_query_text, t] for t in chunk_texts]

        # Step 3: Get relevance scores from CrossEncoder
        scores = self.model.predict(pairs)

        # Step 4: Combine text + score, sort, and return top-k
        sorted_chunks = sorted(
            zip(chunk_texts, scores), key=lambda x: x[1], reverse=True
        )
        return sorted_chunks[: self.k]
