from abc import ABC, abstractmethod


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_path: str, out_dir: str) -> str:
        """
        Parse a document and return extracted text/structured content.
        """
        pass
