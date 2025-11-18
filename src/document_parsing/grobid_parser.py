import logging
import time
from typing import Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base_parser import BaseParser, ParseResult

logger = logging.getLogger(__name__)


class GROBIDParser(BaseParser):
    """Client for a remote GROBID instance that returns TEI XML as Markdown."""

    def __init__(
        self,
        grobid_url: str = "https://kermitt2-grobid.hf.space",
        # use split timeouts: (connect, read)
        timeout: Tuple[int, int] = (5, 180),
        check_health: bool = True,
        retries: int = 5,
        backoff_factor: float = 0.5,
    ):
        """
        Args:
            grobid_url: Base URL to the GROBID service.
            timeout: (connect_timeout, read_timeout) for requests.
            check_health: If True, probe /api/isalive during init.
            retries: Max automatic retries on transient errors.
            backoff_factor: Exponential backoff factor between retries.
        """
        if not grobid_url.startswith(("http://", "https://")):
            raise ValueError("grobid_url must start with http:// or https://")

        if isinstance(timeout, (int, float)):
            # backwards compatibility if a single int is passed
            timeout = (int(timeout), int(timeout))
        self.timeout = timeout  # type: Tuple[int, int]

        self.grobid_url = grobid_url.rstrip("/")
        self.api_url = f"{self.grobid_url}/api/processFulltextDocument"
        self.health_url = f"{self.grobid_url}/api/isalive"

        # Build a session with retry + backoff for robustness
        self.session = requests.Session()
        retry_cfg = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry_cfg, pool_connections=10, pool_maxsize=20
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if check_health:
            self._check_server()

    def _check_server(self):
        """Fast health probe against /api/isalive with retries/backoff from the Session."""
        try:
            t0 = time.time()
            resp = self.session.get(
                self.health_url, timeout=self.timeout, allow_redirects=False
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"GROBID server not running or unhealthy. "
                    f"Status: {resp.status_code}; url={self.health_url}"
                )
            logger.debug(
                "GROBID server health check passed at %s in %.2fs",
                self.grobid_url,
                time.time() - t0,
            )
        except requests.RequestException as e:
            raise ConnectionError(
                f"Connection to the GROBID server failed. "
                f"Please check your connection or the URL: {self.grobid_url}"
            ) from e

    def parse(
        self,
        file_path: str,
        consolidate_citations: int = 0,
        consolidate_header: int = 0,
        consolidate_funders: int = 0,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> ParseResult:
        """Parse a single PDF with GROBID."""
        params = {
            "consolidateCitations": consolidate_citations,
            "consolidateHeader": consolidate_header,
            "consolidateFunders": consolidate_funders,
        }
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end

        try:
            t0 = time.time()
            with open(file_path, "rb") as f:
                files = {"input": f}
                resp = self.session.post(
                    self.api_url,
                    files=files,
                    data=params,
                    timeout=self.timeout,  # (connect, read). Read can be long on cold servers.
                )
            if resp.status_code != 200:
                # Let caller see a concise error; include a short snippet of response if present
                snippet = (resp.text or "").strip()
                if len(snippet) > 300:
                    snippet = snippet[:300] + "…"
                raise RuntimeError(
                    f"GROBID parsing failed with status {resp.status_code} for file {file_path}. "
                    f"Response: {snippet}"
                )
            logger.debug("GROBID parsed %s in %.2fs", file_path, time.time() - t0)
        except requests.Timeout as e:
            raise TimeoutError(
                f"GROBID request timed out (connect={self.timeout[0]}s, read={self.timeout[1]}s) "
                f"for file {file_path}"
            ) from e
        except requests.RequestException as e:
            raise RuntimeError(f"GROBID request error for file {file_path}: {e}") from e

        content = resp.text
        return ParseResult(
            content=content,
            format="tei_xml",
            metadata={
                "parser": "grobid",
                "source_format": "tei_xml",
                "grobid_url": self.grobid_url,
            },
        )
