"""The MinerU wrapper must pin the backend that produced the cached search parses
(pipeline) and record version, backend and command in the parse meta.
The CLI is mocked: nothing is parsed, downloaded or started."""
import json
import pathlib
import shutil
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from document_parsing import mineru_parser as mp  # noqa: E402


@pytest.fixture
def fake_mineru(monkeypatch, tmp_path):
    """Pretend `mineru` exists and writes one Markdown file into the run dir."""
    calls = []
    monkeypatch.setattr(mp.shutil, "which", lambda name: "/fake/bin/mineru")

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        run_dir = pathlib.Path(cmd[cmd.index("-o") + 1])
        (run_dir / "paper" / "pipeline").mkdir(parents=True)
        (run_dir / "paper" / "pipeline" / "paper.md").write_text("# parsed\n", encoding="utf-8")

        class R:
            returncode, stdout, stderr = 0, "", ""

        return R()

    monkeypatch.setattr(mp.subprocess, "run", fake_run)
    return calls


def test_backend_is_pinned_to_pipeline_by_default(fake_mineru, tmp_path, monkeypatch):
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    parser = mp.MinerUParser(output_dir=str(tmp_path / "tmp"))
    assert parser.backend == "pipeline" == mp.DEFAULT_BACKEND
    res = parser.parse(str(tmp_path / "x.pdf"))
    cmd = fake_mineru[0]
    assert cmd[0] == "mineru" and cmd[cmd.index("-b") + 1] == "pipeline"
    assert res.content.startswith("# parsed")
    meta = res.metadata
    assert meta["parser"] == "mineru" and meta["source_format"] == "md"
    assert meta["mineru_backend"] == "pipeline"
    assert meta["mineru_version"] == parser.version
    assert meta["mineru_command"].endswith("-b pipeline")
    json.dumps(meta)  # meta must be JSON-serialisable for the parse meta file


def test_backend_override_is_explicit_and_recorded(fake_mineru, tmp_path, monkeypatch):
    monkeypatch.setenv("MINERU_BACKEND", "vlm-engine")
    parser = mp.MinerUParser(output_dir=str(tmp_path / "tmp"))
    res = parser.parse(str(tmp_path / "x.pdf"))
    assert fake_mineru[0][fake_mineru[0].index("-b") + 1] == "vlm-engine"
    assert res.metadata["mineru_backend"] == "vlm-engine"


def test_run_dir_is_cleaned_up(fake_mineru, tmp_path, monkeypatch):
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    out = tmp_path / "tmp"
    mp.MinerUParser(output_dir=str(out)).parse(str(tmp_path / "x.pdf"))
    assert not any(out.glob("mineru_*")), "intermediate run folder must be removed"
