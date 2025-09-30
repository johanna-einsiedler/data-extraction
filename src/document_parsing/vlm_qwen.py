from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image
from together import Together

from .base_parser import BaseParser


class QwenParser(BaseParser):
    def __init__(self, api_key, model_name: str = "Qwen/Qwen2.5-7B-Instruct-Turbo"):
        """
        Together API parser for Qwen2.5-VL models.
        """
        self.client = Together(api_key=api_key)
        self.model_name = model_name

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

    def parse_page(self, page_content: str) -> str:
        """
        Parse a single page via Together API.
        """
        response = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": page_content}]
        )
        return response.choices[0].message.content

    def parse(self, file_path: str) -> str:
        """
        Parse the PDF and return the full Markdown text.
        """
        images = self.pdf_to_images(file_path)
        results = []

        for i, img in enumerate(images):
            # Convert image to text via OCR first
            # Here we use pytesseract, since Together API expects text input
            from pytesseract import image_to_string

            page_text = image_to_string(img)

            # Skip empty pages
            if not page_text.strip():
                continue

            # Prompt Qwen via Together API
            prompt = f"""Convert the following scientific article text into well - structured Markdown .
                - Use # and ## headings to match the section and subsection titles .
                - Format inline citations as ( Author , Year ) if available .
                Use bullet points for lists .
                - Preserve all mathematical notation and code blocks in Markdown syntax .
                - Render tables using Markdown syntax (| ... |)
                - Output clean Markdown only no explanation .
                Input : \n
                {page_text}"""
            result = self.parse_page(prompt)
            results.append(result)

        # Combine all pages
        return "\n\n".join(results)
