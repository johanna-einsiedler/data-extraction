"""Call-provenance ledger (2026-10-02): hash chain, tamper detection, provider
provenance extraction, and the generate() hook's entry builder."""
import json, os, pathlib, sys, types
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from answer_generation import call_ledger as cl  # noqa: E402


def _entry(i):
    return {"ts_utc": f"2026-10-02T00:00:0{i}Z", "api_type": "openrouter", "model_requested": "m",
            "response_id": f"gen-{i}", "usage": {"prompt_tokens": 10 * i}, "prompt_sha256": "a" * 64}


def test_chain_verifies_and_detects_tampering(tmp_path):
    p = tmp_path / "calls.jsonl"
    heads = [cl.record(_entry(i), p) for i in range(1, 4)]
    assert all(heads) and len(set(heads)) == 3
    v = cl.verify(p)
    assert v["ok"] and v["n"] == 3 and v["head"] == heads[-1]
    # tamper with the middle line's usage -> chain breaks at that line
    lines = p.read_text().splitlines()
    obj = json.loads(lines[1]); obj["usage"]["prompt_tokens"] = 999
    lines[1] = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")
    v = cl.verify(p)
    assert not v["ok"] and v["first_bad_line"] == 2
    # dropping a line also breaks the chain
    p.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert not cl.verify(p)["ok"]


def test_chain_continues_across_processes(tmp_path):
    """A fresh process (empty in-memory head) must chain onto the last line on disk."""
    p = tmp_path / "calls.jsonl"
    h1 = cl.record(_entry(1), p)
    cl._HEAD.clear()
    h2 = cl.record(_entry(2), p)
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert rows[1]["prev_hash"] == h1 and cl.verify(p)["ok"] and cl.verify(p)["head"] == h2


def test_provenance_shapes():
    oai = types.SimpleNamespace(id="chatcmpl-1", created=1727800000, model="gpt-4.1-2025-04-14",
                                system_fingerprint="fp_abc", provider="DeepInfra")
    pv = cl.provenance(oai, "openrouter")
    assert pv == {"response_id": "chatcmpl-1", "created": 1727800000, "model_resolved": "gpt-4.1-2025-04-14",
                  "provider_routed": "DeepInfra", "system_fingerprint": "fp_abc"}
    assert cl.provenance(oai, "openai")["provider_routed"] is None
    gem = types.SimpleNamespace(response_id="resp-9", model_version="gemini-3.1-pro-preview-0924")
    pg = cl.provenance(gem, "gemini")
    assert pg["response_id"] == "resp-9" and pg["model_resolved"] == "gemini-3.1-pro-preview-0924"
    assert cl.provenance({"id": "x", "model": "m"}, "together")["response_id"] == "x"   # dict shape


def test_entry_for_call_hashes_prompt_and_output():
    r = types.SimpleNamespace(id="id1", created=1, model="m")
    e = cl.entry_for_call(api_type="together", model_requested="m", response=r,
                          usage={"prompt_tokens": 5, "completion_tokens": 2, "reasoning_tokens": 0},
                          finish_reason="stop", prompt_material="P", output_text="O")
    assert e["prompt_sha256"] == cl.sha256("P") and e["output_sha256"] == cl.sha256("O") and e["output_chars"] == 1
    assert e["response_id"] == "id1" and e["ts_utc"].endswith("+00:00")


def test_record_never_raises(tmp_path, monkeypatch):
    """A ledger failure must not propagate into the API call."""
    bad = tmp_path / "nodir" / "x" / "calls.jsonl"
    monkeypatch.setattr(cl.Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    assert cl.record(_entry(1), bad) is None
