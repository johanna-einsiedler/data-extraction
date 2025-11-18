"""Shared helpers reused by multiple document parsers."""

from __future__ import annotations

from typing import List

import fitz  # PyMuPDF
from PIL import Image


def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert each page of a PDF into a PIL image."""
    images: List[Image.Image] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    return images
