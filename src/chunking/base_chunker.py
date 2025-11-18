# chunkers/base_chunker.py
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Chunk:
    """Lightweight container returned by chunkers."""

    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


class BaseChunker(ABC):
    """Common utilities shared by the different chunking strategies."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _sanitize(self, text: str, *, remove_tags: bool = False) -> str:
        """Normalise whitespace and optionally strip markup from the input."""
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        if remove_tags:
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\n\s+", "\n", cleaned)
        cleaned = re.sub(r"[\t ]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _tokenize(self, text: str) -> List[str]:
        """Basic whitespace tokenizer (override for specialised behaviour)."""
        return text.split()

    def _detokenize(self, tokens: List[str]) -> str:
        return " ".join(tokens).strip()

    def _token_windows(self, text: str) -> List[str]:
        """Slice text into token-length windows with optional overlap."""
        tokens = self._tokenize(text)
        if not tokens:
            return []
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: List[str] = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.chunk_size]
            if not window:
                break
            chunks.append(self._detokenize(window))
            if start + self.chunk_size >= len(tokens):
                break
        return chunks

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Split text into chunks according to a specific strategy."""
        raise NotImplementedError
