import json
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from together import Together

from answer_generation.answer_generator import AnswerGenerator
from chunking import base_chunker, length_chunker, structure_chunker, text_chunker
from document_parsing import (
    grobid_parser,
    mineru_parser,
    pymupdf_parser,
    pymupdf_tesseract,
    vlm_gemini,
    vlm_qwen,
)
from embedding import base_embedder, bge_embedder, e5_embedder, openai_embedder
from retrieval import (
    base_retriever,
    cross_encoder_retriever,
    rcs_retriever,
    token_budget_retriever,
    topk_retriever,
)

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# Registry now stores classes + default kwargs
PARSERS = {
    "GROBIDParser": {
        "cls": grobid_parser.GROBIDParser,
        "kwargs": {},  # no eager server check
    },
    "PyMuPDFParser": {
        "cls": pymupdf_parser.PyMuPDFParser,
        "kwargs": {},
    },
    "PyMuPDFTesseractParser": {
        "cls": pymupdf_tesseract.PyMuPDFTesseractParser,
        "kwargs": {},
    },
    "GeminiParser": {
        "cls": vlm_gemini.GeminiParser,
        "kwargs": {"api_key": GOOGLE_API_KEY},  # set dynamically
    },
    "QwenParser": {
        "cls": vlm_qwen.QwenParser,
        "kwargs": {"api_key": TOGETHER_API_KEY},  # set dynamically
    },
    "MinerUParser": {
        "cls": mineru_parser.MinerUParser,
        "kwargs": {},
    },
}


CHUNKERS = {
    "LengthChunker": length_chunker.LengthChunker,
    "StructureChunker": structure_chunker.StructureChunker,
    "TextStructureChunker": text_chunker.TextStructureChunker,
}


class LazyEmbedder(base_embedder.BaseEmbedder):
    """Wrap embedder construction so heavy dependencies load only if needed."""

    def __init__(self, factory):
        self._factory = factory
        self._instance = None

    def _ensure(self):
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def embed(self, texts):
        return self._ensure().embed(texts)

    def __getattr__(self, item):
        # Delegate any other attribute access to the wrapped embedder
        return getattr(self._ensure(), item)


EMBEDDERS = {
    "BGEEmbedder": LazyEmbedder(lambda: bge_embedder.BGEEmbedder()),
    "E5Embedder": LazyEmbedder(lambda: e5_embedder.E5Embedder()),
}

if OPENAI_API_KEY:
    EMBEDDERS["OpenAIEmbedder"] = LazyEmbedder(
        lambda: openai_embedder.OpenAIEmbedder(api_key=OPENAI_API_KEY)
    )

RETRIEVERS = {
    "TopKRetriever": topk_retriever.TopKRetriever,
    "CrossEncoderRetriever": cross_encoder_retriever.CrossEncoderRetriever,
    "TokenBudgetRetriever": token_budget_retriever.TokenBudgetRetriever,
    "RCSRetriever": rcs_retriever.RCSRetriever,
}


client = None
models = []
if TOGETHER_API_KEY:
    try:
        client = Together(api_key=TOGETHER_API_KEY)
        models = client.models.list()
    except Exception:
        # Fail gracefully during local testing when Together API is unavailable
        models = []

base_dir = Path(__file__).resolve().parent
with open(
    base_dir / "answer_generation" / "filtered_models.json", "r", encoding="utf-8"
) as f:
    LLMS_META = json.load(f)


LLM_META_MAP = {meta["id"]: meta for meta in LLMS_META}

_API_KEYS = {
    "together": TOGETHER_API_KEY,
    "openai": OPENAI_API_KEY,
    "gemini": GOOGLE_API_KEY,
    "anthropic": ANTHROPIC_API_KEY,
    "openrouter": OPENROUTER_API_KEY,
}

LLMS = {
    meta_id: AnswerGenerator(
        api_type=meta["api_type"],
        # `api_model` lets the wire id differ from the roster key. Needed because
        # the roster key IS the Optuna categorical value: changing it would trip
        # "CategoricalDistribution does not support dynamic value space" against
        # every trial already recorded under the old name. So a model re-routed
        # from Together to OpenRouter (which uses a different namespace and case,
        # e.g. moonshotai/Kimi-K2.6 -> moonshotai/kimi-k2.6) keeps its key and
        # only changes where the call goes.
        model=meta.get("api_model", meta_id),
        api_key=_API_KEYS.get(meta["api_type"], OPENAI_API_KEY),
        exposes_logprobs=meta.get("exposes_logprobs", True),
        base_url=meta.get("base_url"),
        provider_routing=meta.get("provider_routing"),
    )
    for meta_id, meta in LLM_META_MAP.items()
}

# Models that expose usable answer-token logprobs (per the roster's
# `exposes_logprobs` flag; defaults to True when the flag is absent so the
# legacy roster is unaffected). Populated by scripts/build_filtered_models.py
# and validated by scripts/test_logprobs.py. Use this to restrict the
# calibration / uncertainty analyses (RQ2) to logprob-capable models.
LOGPROB_LLMS = {
    meta_id
    for meta_id, meta in LLM_META_MAP.items()
    if meta.get("exposes_logprobs", True)
}
