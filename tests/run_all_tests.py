"""
Unified test runner that executes the full suite and prints a condensed summary.

Usage:
    python tests/run_all_tests.py            # run everything with default verbosity
    python tests/run_all_tests.py -q -k eval # pass flags straight to pytest
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


def _import_pytest():
    try:
        import pytest  # type: ignore
    except ImportError as exc:  # pragma: no cover - guard clause
        raise SystemExit(
            "pytest is required but not installed. "
            "Install the test dependencies (e.g. `pip install -r requirements.txt`)."
        ) from exc
    return pytest


@dataclass(eq=False)
class SummaryPlugin:
    """Collect call-phase results so we can emit a compact summary afterwards."""

    stats: Dict[str, int] = field(default_factory=lambda: {"passed": 0, "failed": 0, "skipped": 0})
    deselected: int = 0

    def pytest_runtest_logreport(self, report):  # pragma: no cover - pytest hook
        if report.when != "call":
            return
        self.stats.setdefault(report.outcome, 0)
        self.stats[report.outcome] += 1

    def pytest_deselected(self, items):  # pragma: no cover - pytest hook
        self.deselected += len(items)

    def format_summary(self) -> str:
        parts: List[str] = []
        for outcome in ("passed", "failed", "skipped"):
            count = self.stats.get(outcome, 0)
            parts.append(f"{outcome}: {count}")
        if self.deselected:
            parts.append(f"deselected: {self.deselected}")
        return " | ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full test suite with a concise summary.")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Run pytest in quiet mode (equivalent to passing -q).",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded verbatim to pytest.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pytest = _import_pytest()
    plugin = SummaryPlugin()

    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    tests_path = repo_root / "tests"
    default_args: List[str] = [str(tests_path)]

    forwarded = args.pytest_args or []
    if args.quiet and "-q" not in forwarded:
        default_args.insert(0, "-q")

    exit_code = pytest.main(default_args + forwarded, plugins=[plugin])

    print("\n=== Aggregate Test Summary ===")
    print(plugin.format_summary())
    return int(exit_code)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
