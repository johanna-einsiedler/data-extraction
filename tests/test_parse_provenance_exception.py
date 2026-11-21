"""Frozen runs reject a cached parse of another parser unless the paper is named in
PARSE_PROVENANCE_EXCEPTIONS (set by run_frozen --allow-parse-fallback)."""
import json, pathlib, sys
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import objective  # noqa: E402


@pytest.fixture
def fallback_parse(tmp_path, monkeypatch):
    pdir = tmp_path / "MinerUParser"; pdir.mkdir()
    (pdir / "451_Wang_2022.md").write_text("# text\n")
    json.dump({"parser": "pymupdf", "source_format": "text"}, open(pdir / "451_Wang_2022.meta.json", "w"))
    monkeypatch.setattr(objective, "parsed_dir", lambda name: tmp_path / name)
    return tmp_path / "451_Wang_2022.pdf"


def test_strict_guard_rejects_foreign_parse(fallback_parse, monkeypatch):
    monkeypatch.delenv("PARSE_PROVENANCE_EXCEPTIONS", raising=False)
    with pytest.raises(objective.ParserProvenanceError):
        objective.ensure_parsed(fallback_parse, "MinerUParser", strict=True)


@pytest.mark.parametrize("value", ["451", "451_Wang_2022", "103, 451"])
def test_named_exception_accepts_cached_parse(fallback_parse, monkeypatch, value):
    monkeypatch.setenv("PARSE_PROVENANCE_EXCEPTIONS", value)
    out = objective.ensure_parsed(fallback_parse, "MinerUParser", strict=True)
    assert out.name == "451_Wang_2022.md"


def test_exception_is_paper_specific(fallback_parse, monkeypatch):
    monkeypatch.setenv("PARSE_PROVENANCE_EXCEPTIONS", "103")
    with pytest.raises(objective.ParserProvenanceError):
        objective.ensure_parsed(fallback_parse, "MinerUParser", strict=True)
