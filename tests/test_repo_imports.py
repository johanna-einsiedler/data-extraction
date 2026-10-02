"""Guard against the relocated-venv trap.

pipeline_env was editable-installed against the sibling checkout ../pipeline, so
`import registry` silently loaded OLD code (fixed 2026-10-02). Assert that every
top-level src module resolves inside THIS repo, and that a fresh interpreter with
no explicit sys.path insert never resolves `registry` anywhere else.
"""
import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODULES = ["registry", "objective", "evaluate", "batched", "binary_mode",
           "mc_rule", "text_utils", "_data_loading", "_llm_utils"]


@pytest.mark.parametrize("name", MODULES)
def test_src_module_resolves_inside_this_repo(name):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    spec = importlib.util.find_spec(name)          # resolve only; no import side effects
    assert spec is not None and spec.origin, f"{name} not found under {SRC}"
    origin = pathlib.Path(spec.origin).resolve()
    assert origin.is_relative_to(SRC), f"{name} -> {origin} (expected under {SRC})"


def test_fresh_interpreter_never_resolves_registry_elsewhere():
    code = ("import importlib.util, pathlib; s = importlib.util.find_spec('registry'); "
            "print(pathlib.Path(s.origin).resolve() if s and s.origin else 'NONE')")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "NONE" or out.startswith(str(SRC)), (
        f"registry resolves to {out}: a stale editable install / .pth is on sys.path")


def test_pipeline_cli_is_this_repo():
    spec = importlib.util.find_spec("pipeline_cli")
    assert spec and pathlib.Path(spec.origin).resolve() == ROOT / "pipeline_cli.py"
