import os
import re
import sys
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image

sys.path.append(os.path.dirname(__file__))

from base_parser import BaseParser


class GeminiParser(BaseParser):
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Parser using Gemini 2.5 for image-based PDFs.
        """
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Convert each PDF page into a PIL image.
        """
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images

    def parse_page(self, page_image: Image.Image) -> str:
        """
        Parse a single page image via Gemini.
        """
        # Optionally, you can save to bytes or base64 if SDK supports
        img_path = "temp_page.png"
        page_image.save(img_path, format="PNG")

        prompt = """Convert the following scientific article page into well-structured Markdown:
- Use # and ## for headings
- Preserve bullet points, tables, math, and code blocks
- Format citations as (Author, Year)
- Output Markdown only, no explanation
Input (image):"""

        response = self.model.generate_content([prompt, img_path])
        return response.text

    def parse(self, file_path: str) -> str:
        """
        Convert a PDF to Markdown using Gemini via image OCR.
        """
        images = self.pdf_to_images(file_path)
        results = []

        for img in images:
            page_text = self.parse_page(img)
            results.append(page_text)

        return "\n\n".join(results)
