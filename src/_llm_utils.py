"""LLM text post-processing utilities: cleaning reasoning markers and normalising outputs."""

import math
import re

_LLM_STRIP_PATTERNS = [
    (r"<(think|reason|thought)>.*?</\1>", dict(flags=re.DOTALL | re.IGNORECASE)),
    (
        r"\((thinking|reasoning|step-by-step reasoning):.*?\)",
        dict(flags=re.DOTALL | re.IGNORECASE),
    ),
    (r"<!--\s*(think|reasoning|steps).*?-->", dict(flags=re.DOTALL | re.IGNORECASE)),
    (
        r"^(Thoughts?:|Reasoning:|Step\s*\d*:|Internal monologue:|Chain-of-thought:)\s*",
        dict(flags=re.IGNORECASE | re.MULTILINE),
    ),
]


def _clean_llm_text(text):
    """Remove internal reasoning markers frequently returned by LLM APIs."""
    if not isinstance(text, str):
        return text
    cleaned = text
    for pattern, kwargs in _LLM_STRIP_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, **kwargs)
    return cleaned.strip()


def get_llm_text(output):
    """
    Normalize assorted LLM responses (string/dict/list) into a cleaned string.
    Handles direct strings, OpenAI-style choices, Together responses, and
    nested structures while stripping out internal reasoning markers.
    """
    if output is None:
        return ""

    if isinstance(output, str):
        return _clean_llm_text(output)

    if isinstance(output, (list, tuple, set)):
        parts = [get_llm_text(item) for item in output]
        return "\n".join([p for p in parts if p])

    if isinstance(output, dict):
        if "text" in output:
            return get_llm_text(output["text"])
        if "message" in output:
            return get_llm_text(output["message"])
        if "content" in output:
            return get_llm_text(output["content"])
        if "choices" in output:
            parts = [get_llm_text(choice) for choice in output.get("choices", [])]
            return "\n".join([p for p in parts if p])

        parts = []
        for key, value in output.items():
            if key in {"logprobs", "usage"}:
                continue
            parts.append(get_llm_text(value))
        return "\n".join([p for p in parts if p])

    if hasattr(output, "content"):
        return get_llm_text(getattr(output, "content"))

    return _clean_llm_text(str(output))


def get_cleaned_text_and_logprobs(llm_output):
    """
    Returns cleaned text and a dict of token -> logprob.
    Only includes logprobs for tokens that appear in the cleaned text.

    Handles multiple logprob formats:
      - dict  {token: logprob}          (OpenAI chat completions)
      - object with .tokens / .token_logprobs  (Together SDK)
      - np.nan / None                   (o3-mini, fallback)
    """
    text = llm_output["text"]
    logprobs_obj = llm_output.get("logprobs")

    cleaned_text = _clean_llm_text(text)

    # ---- Already a plain dict (OpenAI path) ----
    if isinstance(logprobs_obj, dict):
        # Filter to tokens that appear in cleaned text
        filtered = {}
        idx = 0
        for tok, lp in logprobs_obj.items():
            if lp is None:
                continue
            pos = cleaned_text.find(tok, idx)
            if pos != -1:
                filtered[tok] = lp
                idx = pos + len(tok)
        return cleaned_text, filtered

    # ---- np.nan or None — no logprobs available ----
    if logprobs_obj is None:
        return cleaned_text, {}
    try:
        if math.isnan(logprobs_obj):
            return cleaned_text, {}
    except (TypeError, ValueError):
        pass

    # ---- Fallback: try to treat as a dict-like or object with .content ----
    # (Handles any unexpected wrapper objects)
    content_list = getattr(logprobs_obj, "content", None)
    if content_list and isinstance(content_list, list):
        filtered = {}
        idx = 0
        for entry in content_list:
            tok = (
                entry.get("token")
                if isinstance(entry, dict)
                else getattr(entry, "token", None)
            )
            lp = (
                entry.get("logprob")
                if isinstance(entry, dict)
                else getattr(entry, "logprob", None)
            )
            if tok is None or lp is None:
                continue
            pos = cleaned_text.find(tok, idx)
            if pos != -1:
                filtered[tok] = lp
                idx = pos + len(tok)
        return cleaned_text, filtered

    # ---- Legacy Together-style object with .tokens / .token_logprobs ----
    tokens = getattr(logprobs_obj, "tokens", []) or []
    token_logprobs = getattr(logprobs_obj, "token_logprobs", []) or []

    logprobs_dict = {}
    if tokens and token_logprobs:
        idx = 0
        cleaned_len = len(cleaned_text)
        for tok, lp in zip(tokens, token_logprobs):
            tok_len = len(tok)
            if idx >= cleaned_len:
                break

            if cleaned_text[idx : idx + tok_len] == tok:
                logprobs_dict[tok] = lp
                idx += tok_len
            elif tok.strip() != "" and tok in cleaned_text[idx:]:
                next_idx = cleaned_text.find(tok, idx)
                if next_idx != -1:
                    logprobs_dict[tok] = lp
                    idx = next_idx + tok_len

    return cleaned_text, logprobs_dict
