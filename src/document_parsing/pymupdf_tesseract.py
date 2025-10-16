import io
import os
import re
import sys
from pathlib import Path
from typing import List

import fitz
import pytesseract

sys.path.append(os.path.dirname(__file__))

from base_parser import BaseParser
from PIL import Image


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
