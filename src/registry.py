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
    "BaseChunker": base_chunker.BaseChunker,
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
    "BaseRetriever": base_retriever.BaseRetriever,
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

LLMS = {
    meta_id: AnswerGenerator(
        api_type=meta["api_type"],
        model=meta_id,
        api_key=TOGETHER_API_KEY if meta["api_type"] == "together" else OPENAI_API_KEY,
    )
    for meta_id, meta in LLM_META_MAP.items()
}
