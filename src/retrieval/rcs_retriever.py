# retrieval/rcs_retriever.py
import os
import re
import sys
from typing import List, Tuple

sys.path.append(os.path.dirname(__file__))

from base_retriever import BaseRetriever
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from together import Together

from vectorstore.numpy_store import NumpyVectorStore

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class RCSRetriever(BaseRetriever):
    def __init__(
        self,
        llm_model: str,
        top_m: int = 10,
        k: int = 5,
        summary_length: int = 300,
        client=Together(api_key=TOGETHER_API_KEY),  # optional generic API client
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
        self.client = client

        # Auto-select OpenAI if model includes 'gpt'
        if "gpt" in llm_model.lower():
            from openai import OpenAI

            self.client = OpenAI(api_key=OPENAI_API_KEY)

    def retrieve(self, query_vec, store: NumpyVectorStore) -> List[Tuple[str, float]]:
        # 1. Initial top-m retrieval
        top_chunks = store.query(query_vec, top_k=self.top_m)
        print(top_chunks)
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
                print(response)
                # Extract the response content
                content = response.choices[0].message.content
            else:
                response = self.client.completions.create(
                    model=self.llm_model, prompt=prompt
                )
                print(response)
                choice = response.choices[0]
                content = choice.text

            try:
                import json

                data = json.loads(content)
                score = int(data.get("relevance_score", 0))
            except Exception:
                score = 0

            reranked.append((chunk, score))

        # 2. Rerank by relevance
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[: self.k]
