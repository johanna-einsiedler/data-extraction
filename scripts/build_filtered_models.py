"""Build src/answer_generation/filtered_models.json from a model_entry_exam.csv.

The user re-runs the entry exam (incl. VLMs) and provides a CSV; this converts
the *included* models into the roster the pipeline consumes. We do NOT run the
exam here.

Expected CSV columns (header names are matched case-insensitively; only ``id``
is strictly required, the rest have sensible defaults):

  id                (required)  provider model-id string, e.g.
                                "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
  included          truthy -> keep this model in the roster (yes/true/1/x).
                                Defaults to True if the column is absent.
  passed            truthy -> passed the entry exam. If present, a row must be
                                truthy here to be included.
  api_type          "together" | "openai" (inferred from id if omitted:
                                gpt*/o[0-9]* -> openai, else together)
  display_name, organization                metadata (best-effort defaults)
  context_length                            int (default 0 -> unknown)
  price_input, price_output                 float $/1M tokens (default 0)
  link                                      model card URL
  type                                      free text (e.g. "R", "IT"); used
                                            only to default exposes_logprobs
  exposes_logprobs  truthy/falsy -> whether the model returns usable
                                answer-token logprobs. If omitted, defaults to
                                False for rows whose ``type`` looks like a
                                reasoning/thinking model, else True. ALWAYS
                                re-validate empirically with test_logprobs.py.

Usage:
  python scripts/build_filtered_models.py path/to/model_entry_exam.csv
  python scripts/build_filtered_models.py exam.csv --out src/answer_generation/filtered_models.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "src" / "answer_generation" / "filtered_models.json"

_TRUTHY = {"1", "true", "yes", "y", "x", "included", "pass", "passed", "t"}
_FALSY = {"0", "false", "no", "n", "", "nan", "excluded", "fail", "failed", "f"}


def _truthy(val, default: bool) -> bool:
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    return default


def _infer_api_type(model_id: str) -> str:
    mid = model_id.strip().lower()
    if re.match(r"^(gpt-?|o[0-9]|chatgpt)", mid.split("/")[-1]):
        return "openai"
    return "together"


def _looks_reasoning(type_str: str) -> bool:
    s = str(type_str).strip().lower()
    return ("reason" in s) or ("think" in s) or s in {"r", "r?"}


def _get(row, *names, default=None):
    """Case-insensitive column lookup with fallbacks."""
    for n in names:
        for col in row.index:
            if col.strip().lower() == n:
                v = row[col]
                if pd.isna(v):
                    return default
                return v
    return default


def build(csv_path: Path, out_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    has_included = any(c.strip().lower() == "included" for c in df.columns)
    has_passed = any(c.strip().lower() in ("passed", "pass") for c in df.columns)

    models: list[dict] = []
    skipped = []
    for _, row in df.iterrows():
        model_id = str(_get(row, "id", "model", "model_id", "model name",
                            default="")).strip()
        if not model_id:
            continue
        included = _truthy(_get(row, "included", "include", "included?"),
                           default=not has_included)
        passed = _truthy(_get(row, "passed", "pass"), default=True)
        if not (included and (passed or not has_passed)):
            skipped.append(model_id)
            continue

        api_type = str(_get(row, "api_type", "provider_api", default="")).strip().lower()
        if api_type not in ("together", "openai"):
            api_type = _infer_api_type(model_id)

        type_str = str(_get(row, "type", "model_type", default=""))
        exposes = _truthy(
            _get(row, "exposes_logprobs", "logprobs"),
            default=not _looks_reasoning(type_str),
        )

        def _num(x, cast, default):
            try:
                return cast(str(x).replace("$", "").replace(",", "").strip())
            except (TypeError, ValueError):
                return default

        meta = {
            "id": model_id,
            "api_type": api_type,
            "display_name": str(_get(row, "display_name", "name", "model name",
                                     default=model_id)).strip(),
            "organization": str(_get(row, "organization", "provider", "org",
                                     default="")).strip(),
            "context_length": _num(_get(row, "context_length", "context", "ctx",
                                        default=0), int, 0),
            "pricing": {
                "input": _num(_get(row, "price_input", "input", "cost_input",
                                   default=0), float, 0.0),
                "output": _num(_get(row, "price_output", "output", "cost_output",
                                    default=0), float, 0.0),
            },
            "link": str(_get(row, "link", "url", default="")).strip(),
            "type": type_str.strip(),
            "exposes_logprobs": exposes,
        }
        models.append(meta)

    out_path.write_text(json.dumps(models, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}  ({len(models)} models; skipped {len(skipped)})")
    n_lp = sum(m["exposes_logprobs"] for m in models)
    n_together = sum(m["api_type"] == "together" for m in models)
    print(f"  together={n_together}  openai={len(models)-n_together}  "
          f"exposes_logprobs={n_lp}/{len(models)}")
    if skipped:
        print(f"  skipped (not included/passed): {', '.join(skipped[:12])}"
              + (" ..." if len(skipped) > 12 else ""))
    print("  NOTE: re-validate exposes_logprobs empirically with test_logprobs.py")
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="path to model_entry_exam.csv")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    build(Path(args.csv), Path(args.out))


if __name__ == "__main__":
    main()
