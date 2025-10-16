import os
import re
import sys
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

sys.path.append(os.path.dirname(__file__))

from base_parser import BaseParser


class PyMuPDFParser(BaseParser):
    def parse(self, file_path: str) -> str:
        """Parse a single PDF with PyMuPDF."""
        text = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)
