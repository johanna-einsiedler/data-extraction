"""JSON reading that tolerates the artifacts we actually have on disk.

15 trial_summary.json files contain bare `NaN` literals in `query_aggregates`
(written by json.dump for float('nan') — invalid JSON, which orjson rejects and
the stdlib accepts). Read fast, fall back when needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orjson


def load(path: str | Path) -> Any:
    b = Path(path).read_bytes()
    try:
        return orjson.loads(b)
    except orjson.JSONDecodeError:
        return json.loads(b)  # tolerates NaN / Infinity
