from pathlib import Path
from tempfile import NamedTemporaryFile

import google.generativeai as genai
from PIL import Image

from .base_parser import BaseParser, ParseResult
from .utils import pdf_to_images


class GeminiParser(BaseParser):
    """Run Gemini 2.5 on PDF page images, returning Markdown."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    # Scientific / clinical papers (suicide, psychiatry, etc.) false-trigger the
    # default safety filters, which return a candidate with no text part and make
    # `response.text` raise. Disable the filters for this OCR-style task.
    _SAFETY = [
        {"category": c, "threshold": "BLOCK_NONE"}
        for c in (
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        )
    ]

    def parse_page(self, page_image: Image.Image) -> str:
        """Parse a single page image via Gemini. The image (a PIL.Image) must be
        passed to generate_content directly — passing a file PATH string makes
        the model treat it as text and hallucinate a document."""
        prompt = """Convert the following scientific article page into well-structured Markdown:
- Use # and ## for headings
- Preserve bullet points, tables, math, and code blocks
- Format citations as (Author, Year)
- Output Markdown only, no explanation
Input (image):"""
        response = self.model.generate_content(
            [prompt, page_image], safety_settings=self._SAFETY
        )
        try:
            return response.text
        except Exception:
            # No text part (blocked / empty candidate) -> skip this page rather
            # than failing the whole paper.
            return ""

    def parse(self, file_path: str) -> ParseResult:
        """Convert a PDF to Markdown using Gemini via image OCR."""
        images = pdf_to_images(file_path)
        results = []

        for img in images:
            page_text = self.parse_page(img)
            results.append(page_text)

        content = "\n\n".join(results)
        return ParseResult(
            content=content,
            metadata={"parser": "gemini", "model": self.model_name},
        )
