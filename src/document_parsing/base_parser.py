from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ParseResult:
    """Container for normalized parser output."""

    content: str
    format: str = "markdown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_path: str) -> ParseResult:
        """Parse a document and return extracted content wrapped in a ParseResult."""
        raise NotImplementedError
