# chunkers/base_chunker.py
from abc import ABC, abstractmethod
from typing import List


class BaseChunker(ABC):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks according to strategy."""
        pass
