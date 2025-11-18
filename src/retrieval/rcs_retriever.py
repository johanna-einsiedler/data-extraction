# retrieval/rcs_retriever.py
import os
import re
import logging
from typing import List, Tuple
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from together import Together

from .base_retriever import BaseRetriever, VectorStoreProtocol

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


logger = logging.getLogger(__name__)


class RCSRetriever(BaseRetriever):
    supports_rerank = True
    supports_top_m = True
    def __init__(
        self,
        llm_model: str,
        top_m: int = 10,
        k: int = 5,
        summary_length: int = 300,
        client=None,
    ):
        """
        Args:
            llm_model: Name of the model to use.
            top_m: Initial top-m retrieval.
            k: Number of chunks to return.
            api_key: API key for OpenAI if using OpenAI model.
            summary_length: Word length of the summary.
            client: Optional generic API client with `chat_completions.create`.
        """
        self.top_m = top_m
        self.k = k
        self.llm_model = llm_model

        if client is not None:
            self.client = client
        elif "gpt" in llm_model.lower():
            from openai import OpenAI

            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable is required for GPT models")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            if not TOGETHER_API_KEY:
                raise ValueError(
                    "TOGETHER_API_KEY environment variable is required for non-GPT models"
                )
            self.client = Together(api_key=TOGETHER_API_KEY)

    def retrieve(self, query_vec, store: VectorStoreProtocol) -> List[Tuple[str, float]]:
        # 1. Initial top-m retrieval
        top_chunks = store.query(query_vec, top_k=self.top_m)
        reranked = []
        for chunk, score in top_chunks:
            prompt = f"""
                Provide a summary of the relevant information that could help answer the question based on the
                excerpt. The excerpt may be irrelevant. Do not directly answer the question - only summarize relevant
                information. Respond with the following JSON format: 
                {{ "summary": "...",
                "relevance_score": "..." }} 
                where "summary" is relevant information from text - 300
                words and "relevance_score" is the relevance of "summary" to answer the question (integer out of 10)
                Query: {query_vec}
                Chunk: {chunk}
                """
            if "gpt" in self.llm_model.lower():
                response = self.client.chat.completions.create(
                    model=self.llm_model,  # or another model like "gpt-3.5-turbo"
                    messages=[{"role": "user", "content": prompt}],
                )
                # Extract the response content
                content = response.choices[0].message.content
            else:
                response = self.client.completions.create(
                    model=self.llm_model, prompt=prompt
                )
                choice = response.choices[0]
                content = choice.text

            try:
                import json

                data = json.loads(content)
                score = int(data.get("relevance_score", 0))
            except Exception:
                score = 0

            reranked.append((chunk, score))
        logger.info(
            "RCS reranked %d chunk(s) using %s; top scores: %s",
            len(reranked),
            self.llm_model,
            [s for _, s in reranked[:5]],
        )

        # 2. Rerank by relevance
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[: self.k]
