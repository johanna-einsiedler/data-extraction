import fitz  # PyMuPDF

from .base_parser import BaseParser, ParseResult


class PyMuPDFParser(BaseParser):
    """Simple text extractor using PyMuPDF (Markdown output)."""

    def parse(self, file_path: str) -> ParseResult:
        chunks = []
        with fitz.open(file_path) as doc:
            for page in doc:
                chunks.append(page.get_text("markdown"))
        content = "\n\n".join(chunks)
        return ParseResult(
            content=content,
            metadata={"parser": "pymupdf", "source_format": "markdown"},
        )
