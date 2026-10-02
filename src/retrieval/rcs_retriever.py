# retrieval/rcs_retriever.py
import json
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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Route the rerank LLM to the model's ACTUAL provider (an OpenRouter model like
# qwen/qwen3-14b must not be sent to the Together client).
try:
    _ROSTER = {m["id"]: m for m in json.load(open(os.path.join(
        os.path.dirname(__file__), "..", "answer_generation", "filtered_models.json")))}
except Exception:
    _ROSTER = {}


def _provider(model):
    return _ROSTER.get(model, {}).get("api_type")


logger = logging.getLogger(__name__)


class RCSRetriever(BaseRetriever):
    supports_rerank = True
    supports_top_m = True
    needs_raw_query_text = True

    def __init__(
        self,
        llm_model: str,
        top_m: int = 10,
        k: int = 5,
        summary_length: int = 300,
        client=None,
        generate_fn=None,
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
        self.summary_length = summary_length
        self.llm_model = llm_model

        # A per-rerank timeout so a hung socket can't freeze the whole trial
        # forever (RCS makes ~top_m calls per item; without this a single stuck
        # call blocks indefinitely — the pipeline-level REQUEST_TIMEOUT does not
        # reach this retriever's own client).
        _timeout = float(os.getenv("REQUEST_TIMEOUT", "300"))
        prov = _provider(llm_model)
        # Preferred path: a callable prompt -> text supplied by the caller and
        # backed by the registry's AnswerGenerator, so the rerank call uses the
        # same provider, wire id, pinned host and client as the answer call. The
        # client-construction fallback below misroutes anything it does not know
        # ("gpt" in name -> OpenAI sent gpt-oss-120b there; unknown -> Together
        # SDK received gemini) and sends roster ids that OpenRouter rejects.
        self.generate_fn = generate_fn
        if generate_fn is not None:
            self.client = None
        elif client is not None:
            self.client = client
        elif prov == "openrouter":
            self.client = OpenAI(api_key=OPENROUTER_API_KEY,
                                 base_url="https://openrouter.ai/api/v1", timeout=_timeout)
        elif prov == "openai" or "gpt" in llm_model.lower():
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable is required for GPT models")
            self.client = OpenAI(api_key=OPENAI_API_KEY, timeout=_timeout)
        else:  # together (default for non-GPT, non-OpenRouter models)
            if not TOGETHER_API_KEY:
                raise ValueError(
                    "TOGETHER_API_KEY environment variable is required for non-GPT models"
                )
            self.client = Together(api_key=TOGETHER_API_KEY, timeout=_timeout)

    def _complete(self, prompt: str) -> str:
        if self.generate_fn is not None:
            return self.generate_fn(prompt) or ""
        response = self.client.chat.completions.create(
            model=self.llm_model, messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_scores(content: str, n: int) -> dict:
        """{index: score} from the LAST JSON object in `content` that carries a
        "scores" mapping; tolerant of code fences and of reasoning text before it."""
        best = {}
        for m in re.finditer(r"\{.*\}", content, flags=re.DOTALL):
            blob = m.group(0)
            # try progressively shorter tails so trailing prose does not break json
            for cand in (blob, blob[: blob.rfind("}") + 1]):
                try:
                    obj = json.loads(cand)
                except Exception:
                    continue
                sc = obj.get("scores", obj) if isinstance(obj, dict) else {}
                out = {}
                for k, v in (sc or {}).items():
                    try:
                        i = int(str(k).strip())
                        if 1 <= i <= n:
                            out[i] = max(0, min(10, int(float(v))))
                    except Exception:
                        continue
                if out:
                    best = out
                break
        return best

    def retrieve(
        self, query_vec, store: VectorStoreProtocol, raw_query_text: str = None
    ) -> List[Tuple[str, float]]:
        # The reranker LLM must see the QUESTION TEXT (not the embedding vector).
        question = raw_query_text if raw_query_text is not None else ""
        # 1. Initial top-m retrieval
        top_chunks = store.query(query_vec, top_k=self.top_m)
        chunks = [c for c, _ in top_chunks]
        if not chunks:
            return []
        # 2. ONE scoring call for all candidates. The Phase-1 implementation scored
        #    each candidate in its own call (up to top_m=20 per item, ~900 per
        #    paper); measured 2026-09-25 at ~13 calls/min that made a single trial a
        #    40-hour job. Scoring the numbered excerpts together costs one call per
        #    item and the same information reaches the model (question + excerpts).
        #    Changed 2026-09-25; declared in the Phase-2 prereg.
        listing = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, 1))
        prompt = (
            "You are given a question and several numbered excerpts from a research "
            "article. Rate how relevant EACH excerpt is to answering the question, as "
            "an integer from 0 (irrelevant) to 10 (directly answers it). Do not answer "
            "the question. Respond with ONLY a JSON object of the form "
            '{"scores": {"1": <0-10>, "2": <0-10>, ...}} with one entry per excerpt.\n'
            f"Question: {question}\n\nExcerpts:\n{listing}"
        )
        scores = {}
        for attempt in range(2):
            try:
                scores = self._parse_scores(self._complete(prompt), len(chunks))
            except Exception as exc:  # network / provider error surfaces here
                logger.warning("RCS scoring call failed (%s); retrying", exc)
                scores = {}
            if len(scores) >= max(1, len(chunks) // 2):
                break
        if not scores:
            logger.warning("RCS: no parseable scores for %d chunks; keeping store order", len(chunks))
        reranked = [(c, scores.get(i, 0)) for i, c in enumerate(chunks, 1)]
        # 3. Rerank by relevance (stable, so store order breaks ties)
        reranked.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            "RCS scored %d chunk(s) in one call using %s; top scores: %s",
            len(reranked), self.llm_model, [s for _, s in reranked[:5]],
        )
        return reranked[: self.k]
