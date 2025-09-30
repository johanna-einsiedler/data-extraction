from pathlib import Path

import fitz  # PyMuPDF

from .base_parser import BaseParser


class PyMuPDFParser(BaseParser):
    def parse(self, file_path: str) -> str:
        """Parse a single PDF with PyMuPDF."""
        text = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)
