"""Per-call provenance ledger (added 2026-10-02, before the frozen test run).

One JSONL line per API call: UTC time, requested and resolved model, routed
provider, the provider's response id / created / fingerprint, usage, finish
reason, and SHA-256 of the prompt material and of the output. Lines are
hash-chained (each carries the SHA-256 of the previous line), so the file is
tamper-evident: editing or dropping any line breaks every later hash. The
chain head is stamped into the Optuna trial record; committing it to a
timestamped repository attests that the ledger existed in that state then.
Provider response ids are the externally checkable part: OpenRouter serves
its own record of any generation id at GET /api/v1/generation?id=…

Path: $CALL_LEDGER_PATH (objective.py sets it per trial to
<trial_folder>/calls.jsonl); unset -> trial_cache/call_ledger.jsonl.
Never raises into the caller: a ledger failure must not fail an API call."""
from __future__ import annotations
import hashlib, json, os, threading, datetime
from pathlib import Path
from typing import Any, Dict, Optional

_LOCK = threading.Lock()
_HEAD: Dict[str, str] = {}          # path -> hash of the last line written by this process
GENESIS = "0" * 64


def ledger_path() -> Path:
    p = os.getenv("CALL_LEDGER_PATH")
    if p:
        return Path(p)
    return Path(__file__).resolve().parents[2] / "trial_cache" / "call_ledger.jsonl"


def sha256(text: Optional[str]) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _g(obj: Any, *names, default=None):
    """getattr/getitem chain tolerant of SDK objects and plain dicts."""
    for n in names:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(n)
        else:
            obj = getattr(obj, n, None)
    return default if obj is None else obj


def provenance(response: Any, api_type: str) -> Dict[str, Any]:
    """Provider-issued identifiers, by response shape.
    OpenAI/Together/OpenRouter: id, created, model, system_fingerprint (+ OpenRouter
    `provider`); Anthropic: id, model; Gemini: response_id, model_version."""
    return {
        "response_id": _g(response, "id") or _g(response, "response_id"),
        "created": _g(response, "created"),
        "model_resolved": _g(response, "model") or _g(response, "model_version"),
        "provider_routed": _g(response, "provider") if api_type == "openrouter" else None,
        "system_fingerprint": _g(response, "system_fingerprint"),
    }


def _last_hash_on_disk(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            back = min(size, 65536)
            f.seek(size - back)
            tail = f.read().splitlines()
        for line in reversed(tail):
            line = line.strip()
            if line:
                return json.loads(line)["hash"]
    except Exception:
        pass
    return GENESIS


def record(entry: Dict[str, Any], path: Optional[Path] = None) -> Optional[str]:
    """Append one hash-chained line; returns the new head hash (None on failure)."""
    try:
        path = Path(path) if path else ledger_path()
        with _LOCK:
            key = str(path)
            prev = _HEAD.get(key) or _last_hash_on_disk(path)
            body = dict(entry)
            body["prev_hash"] = prev
            payload = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
            h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            body["hash"] = h
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str) + "\n")
                f.flush()
            _HEAD[key] = h
            return h
    except Exception:
        return None


def verify(path: Path) -> Dict[str, Any]:
    """Recompute the chain. Returns {ok, n, head, first_bad_line}."""
    prev = GENESIS
    n = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            h = obj.pop("hash", None)
            if obj.get("prev_hash") != prev:
                return {"ok": False, "n": n, "head": prev, "first_bad_line": i, "why": "prev_hash mismatch"}
            payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
            if hashlib.sha256(payload.encode("utf-8")).hexdigest() != h:
                return {"ok": False, "n": n, "head": prev, "first_bad_line": i, "why": "line hash mismatch"}
            prev = h
            n += 1
    return {"ok": True, "n": n, "head": prev, "first_bad_line": None}


def entry_for_call(*, api_type: str, model_requested: str, response: Any,
                   usage: Dict[str, int], finish_reason: Optional[str],
                   prompt_material: str, output_text: Optional[str],
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    e = {
        "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
        "api_type": api_type,
        "model_requested": model_requested,
        **provenance(response, api_type),
        "usage": usage,
        "finish_reason": finish_reason,
        "prompt_sha256": sha256(prompt_material),
        "output_sha256": sha256(output_text),
        "output_chars": len(output_text or ""),
    }
    if extra:
        e.update(extra)
    return e
