# retrieval/cross_encoder_retriever.py
from typing import List, Tuple, Union

import numpy as np
from sentence_transformers import CrossEncoder

from .base_retriever import BaseRetriever, VectorStoreProtocol


class CrossEncoderRetriever(BaseRetriever):
    supports_rerank = True
    supports_top_m = True
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_m: int = 10,
        k: int = 5,
        model=None,
        device: str | None = None,
        batch_size: int | None = None,
    ):
        self.top_m = top_m  # number of candidates to rerank
        self.k = k  # number of final retrieved chunks
        self.model_name = model_name
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def retrieve(
        self,
        query_vec: Union[np.ndarray, List[float]],
        store: VectorStoreProtocol,
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
        if self.model is None:
            self.model = CrossEncoder(self.model_name, device=self.device)

        if self.batch_size:
            scores = self.model.predict(pairs, batch_size=self.batch_size)
        else:
            scores = self.model.predict(pairs)

        # Step 4: Combine text + score, sort, and return top-k
        sorted_chunks = sorted(
            zip(chunk_texts, scores), key=lambda x: x[1], reverse=True
        )
        return sorted_chunks[: self.k]
