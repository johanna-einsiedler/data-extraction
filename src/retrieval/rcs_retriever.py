# retrieval/rcs_retriever.py
from typing import List, Tuple

from openai import OpenAI
from vectorstore.numpy_store import NumpyVectorStore

from .base_retriever import BaseRetriever


class RCSRetriever(BaseRetriever):
    def __init__(
        self, llm_model: str, top_m: int = 10, k: int = 5, api_key: str = None
    ):
        self.api_key = api_key
        self.top_m = top_m
        self.k = k
        self.llm_model = llm_model

    def retrieve(self, query_vec, store: NumpyVectorStore) -> List[Tuple[str, float]]:
        # 1. Initial top-m retrieval
        top_chunks = store.query(query_vec, top_k=self.top_m)

        reranked = []
        for chunk, score in top_chunks:
            prompt = f"""
                Provide a summary of the relevant information that could help answer the question based on the
                excerpt. The excerpt may be irrelevant. Do not directly answer the question - only summarize relevant
                information. Respond with the following JSON format: {{ "summary": "...",
                "relevance_score": "..." }} where "summary" is relevant information from text - 300
                words and "relevance_score" is the relevance of "summary" to answer the question (integer out of 10)
                Query: {query_vec}
                Chunk: {chunk}
                """
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.llm_model, messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            # naive parsing: extract relevance_score (integer 0-10)
            try:
                import json

                data = json.loads(content)
                score = int(data.get("relevance_score", 0))
            except Exception:
                score = 0
            reranked.append((chunk, score))

        # sort by relevance score descending
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[: self.k]
