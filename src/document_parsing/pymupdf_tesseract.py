import io
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from .base_parser import BaseParser


class PyMuPDFTesseractParser(BaseParser):
    def parse(self, file_path: str) -> str:
        paper_name = Path(file_path).stem

        text = []
        with fitz.open(file_path) as doc:
            for page in doc:
                extracted = page.get_text()
                if len(extracted.strip()) < 50:  # fallback to OCR
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    extracted = pytesseract.image_to_string(img)
                text.append(extracted)
        return "\n".join(text)
