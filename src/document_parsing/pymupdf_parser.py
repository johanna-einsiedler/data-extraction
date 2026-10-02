import fitz  # PyMuPDF

from .base_parser import BaseParser, ParseResult


class PyMuPDFParser(BaseParser):
    """Simple text extractor using PyMuPDF."""

    def parse(self, file_path: str) -> ParseResult:
        chunks = []
        with fitz.open(file_path) as doc:
            for page in doc:
                # Use 'markdown' if available (PyMuPDF4LLM), else fall back to plain text
                try:
                    chunks.append(page.get_text("markdown"))
                except (AssertionError, ValueError):
                    chunks.append(page.get_text("text"))
        content = "\n\n".join(chunks)
        return ParseResult(
            content=content,
            metadata={"parser": "pymupdf", "source_format": "text"},
        )
