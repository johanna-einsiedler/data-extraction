import os

from dotenv import find_dotenv, load_dotenv

from answer_generation.answer_generator import AnswerGenerator
from chunking import length_chunker
from document_parsing import (
    grobid_parser,
    mineru_parser,
    pymupdf_parser,
    pymupdf_tesseract,
    vlm_gemini,
    vlm_qwen,
)
from embedding import bge_embedder
from retrieval import topk_retriever

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Registry now stores classes + default kwargs
PARSERS = {
    "GROBIDParser": {
        "cls": grobid_parser.GROBIDParser,
        "kwargs": {"check_server": False},  # no eager server check
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
}

EMBEDDERS = {
    "BGEEmbedder": bge_embedder.BGEEmbedder(),
}

RETRIEVERS = {
    "TopKRetriever": topk_retriever.TopKRetriever,
}

LLMS = {
    "gpt-3.5-turbo-instruct": AnswerGenerator(
        api_type="openai", model="gpt-3.5-turbo-instruct", api_key=OPENAI_API_KEY
    ),
}
