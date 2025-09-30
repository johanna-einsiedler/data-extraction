import os
from pathlib import Path
from typing import List, Optional

import requests

from .base_parser import BaseParser


class GROBIDParser(BaseParser):
    def __init__(self, grobid_url: str = "https://kermitt2-grobid.hf.space"):
        if not grobid_url.startswith(("http://", "https://")):
            raise ValueError("grobid_url must start with http:// or https://")

        self.grobid_url = grobid_url.rstrip("/")
        self.api_url = f"{self.grobid_url}/api/processFulltextDocument"

        # Check if server is alive
        self._check_server()

    def _check_server(self):
        service_status_url = f"{self.grobid_url}/api/isalive"
        try:
            resp = requests.get(service_status_url, timeout=5)
        except requests.RequestException as e:
            raise ConnectionError(
                f"Connection to the GROBID server failed! "
                f"Please check your connection or the URL: {self.grobid_url}"
            ) from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"GROBID server not running or unhealthy. Status: {resp.status_code}"
            )

    def parse(
        self,
        file_path: str,
        consolidate_citations: int = 0,
        consolidate_header: int = 0,
        consolidate_funders: int = 0,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> str:
        """Parse a single PDF with GROBID."""
        with open(file_path, "rb") as f:
            files = {"input": f}
            params = {
                "consolidateCitations": consolidate_citations,
                "consolidateHeader": consolidate_header,
                "consolidateFunders": consolidate_funders,
            }

            if start is not None:
                params["start"] = start
            if end is not None:
                params["end"] = end

            resp = requests.post(self.api_url, files=files, data=params)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"GROBID parsing failed with status {resp.status_code} "
                    f"for file {file_path}"
                )
            return resp.text
