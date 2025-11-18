import io

import fitz
import pytesseract
from PIL import Image

from .base_parser import BaseParser, ParseResult


class PyMuPDFTesseractParser(BaseParser):
    """Fallback to OCR when PyMuPDF text extraction yields little content."""

    def parse(self, file_path: str) -> ParseResult:
        text = []
        with fitz.open(file_path) as doc:
            for page in doc:
                extracted = page.get_text("markdown")
                if len(extracted.strip()) < 50:  # fallback to OCR
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    extracted = pytesseract.image_to_string(img)
                text.append(extracted)
        content = "\n\n".join(text)
        return ParseResult(
            content=content,
            metadata={"parser": "pymupdf_tesseract", "source_format": "markdown"},
        )
