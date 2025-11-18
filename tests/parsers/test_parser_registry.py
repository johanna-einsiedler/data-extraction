"""Regression tests covering registry metadata for document parsers."""

import pytest

from registry import PARSERS


def _instantiate(name, entry):
    """Instantiate a parser entry, skipping gracefully if a dependency is missing."""
    cls = entry["cls"]
    kwargs = dict(entry.get("kwargs", {}))
    try:
        cls(**kwargs)
    except ValueError as exc:
        if "API key" in str(exc):
            pytest.skip(f"Skipping {name}: {exc}")
        raise
    except FileNotFoundError as exc:
        pytest.skip(f"Skipping {name}: {exc}")
    except ConnectionError as exc:
        pytest.skip(f"Skipping {name}: {exc}")


def test_registry_contains_known_parsers():
    expected = {"GROBIDParser", "PyMuPDFParser", "PyMuPDFTesseractParser"}
    missing = expected - PARSERS.keys()
    assert not missing, f"Missing expected parsers: {missing}"


def test_registry_entries_are_instantiable():
    for name, entry in PARSERS.items():
        cls = entry["cls"]
        assert callable(cls), f"Registry entry {name} has non-callable class"
        kwargs = entry.get("kwargs", {})
        if "api_key" in kwargs and kwargs["api_key"] is None:
            continue
        try:
            _instantiate(name, entry)
        except Exception as exc:
            raise AssertionError(f"Failed to instantiate {name}: {exc}") from exc
