"""Binary question mode, collapsed to ONE call per item.

The original binary mode asked a separate yes/no prompt per choice (item 3.1 with
22 options -> 22 calls). This keeps the per-choice binary framing but in a SINGLE
prompt: the model is shown every option and returns ONE JSON object mapping each
option letter to true/false. That answer set is then scored exactly like any
other multiple-choice prediction.

  build_binary_prompt(qid, qdata)      -> combined prompt (with {context} slot)
  parse_binary_response(text, qdata)   -> "A,C" letters string ("Z" if none)
"""
import json
import re

_HEADER = (
    "You are coding one research article for a systematic review of machine learning "
    "in psychology. For the item below, decide for EACH option INDEPENDENTLY whether "
    "it applies to the study, based strictly on the document. Do not guess or infer; "
    "if an option is not explicitly supported, mark it false.\n"
)


def _opt_text(v):
    return v.get("value") if isinstance(v, dict) else v


def build_binary_prompt(qid, qdata):
    """One prompt listing every option as an independent yes/no decision, asking
    for a single JSON object {letter: true/false}."""
    choices = qdata.get("choices", {}) or {}
    label = qdata.get("label", "") or qdata.get("variable", "")
    desc = (qdata.get("description") or "").strip()
    opts = "\n".join(f"{k}) {_opt_text(v)}" for k, v in choices.items())
    body = f"\nItem [{qid}] {label}"
    if desc:
        body += f"\n{desc[:200]}"
    body += "\nOptions:\n" + opts
    tail = (
        "\n\nFor EVERY option letter above, answer true if it applies to this study "
        "or false if it does not. Return ONLY a single JSON object mapping each option "
        'letter to true or false — include every letter. Example: {"A": false, '
        '"B": true, "C": false}. No explanation, no extra text.'
    )
    return _HEADER + "\n<document>\n{context}\n</document>\n" + body + tail


def _extract_json(text):
    if not text:
        return {}
    t = re.sub(r"```(?:json)?", "", text.strip())
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _truthy(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("true", "yes", "y", "1", "a", "applies", "present", "selected")


def parse_binary_response(text, qdata):
    """JSON of per-option booleans -> comma-separated selected letters ("Z" if
    none). Tolerant of missing letters and A/B-style yes/no values."""
    obj = _extract_json(text)
    sel = []
    for letter in (qdata.get("choices", {}) or {}):
        v = obj.get(letter, obj.get(letter.lower(), obj.get(letter.upper())))
        if _truthy(v):
            sel.append(letter)
    return ",".join(sel) if sel else "Z"
