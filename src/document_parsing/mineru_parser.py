import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(__file__))

from base_parser import BaseParser


class MinerUParser(BaseParser):
    def __init__(self, output_dir: str = "../data/_intermediate/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, file_path: str) -> str:
        # Create a unique subfolder per run
        run_dir = self.output_dir / f"mineru_{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Run mineru
            result = subprocess.run(
                ["mineru", "-p", file_path, "-o", str(run_dir)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"MinerU failed: {result.stderr}")

            # Find subdir (usually <file_stem>/auto/)
            subdirs = os.listdir(run_dir)
            if not subdirs:
                raise RuntimeError("MinerU produced no output.")

            doc_dir = run_dir / subdirs[0] / "auto"
            if not doc_dir.is_dir():
                raise RuntimeError(f"Expected auto/ directory, found: {doc_dir}")

            # Prefer Markdown, fallback to JSON
            preferred = [".md", "_content_list.json", "_middle.json", "_model.json"]
            all_text = None
            for ext in preferred:
                for f in os.listdir(doc_dir):
                    if f.endswith(ext):
                        with open(doc_dir / f, "r", encoding="utf-8") as fh:
                            all_text = fh.read()
                        break
                if all_text is not None:
                    break

            if all_text is None:
                raise RuntimeError(f"No suitable output found in {doc_dir}")

            return all_text

        finally:
            # Clean up intermediate run folder
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
