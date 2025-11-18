"""Helpers for invoking LangExtract with OpenAI and Together providers."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import langextract as lx

DEFAULT_LANGEXTRACT_MODEL = "gemini-2.5-flash"

PROVIDER_MAP = {
    "openai": ("OpenAILanguageModel", "OPENAI_API_KEY"),
    "together": ("TogetherProvider", "TOGETHER_API_KEY"),
}


def _build_examples(examples: List[Dict[str, str]]):
    payload = []
    for example in examples or []:
        context = example.get("context", "")
        answer = example.get("answer")
        if not context:
            continue
        extraction = lx.data.Extraction(
            extraction_class="answer",
            extraction_text=context,
            attributes={"answer": answer},
        )
        payload.append(lx.data.ExampleData(text=context, extractions=[extraction]))
    return payload or None


def run_lang_extract(
    prompt_description: str,
    examples: List[Dict[str, str]],
    context_text: str,
    model_id: Optional[str] = None,
    api_type: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, object]:
    """Execute LangExtract, routing requests to OpenAI or Together providers."""

    if not context_text.strip():
        return {"text": "", "extractions": []}

    lx.providers.load_plugins_once()

    chosen_model = model_id or DEFAULT_LANGEXTRACT_MODEL
    provider = None
    provider_kwargs = {}

    if api_type in PROVIDER_MAP:
        provider, env_var = PROVIDER_MAP[api_type]
        key = api_key or os.getenv(env_var)
        if not key:
            raise ValueError(f"LangExtract requires {env_var} for {api_type} models.")
        provider_kwargs["api_key"] = key
        if api_type == "openai":
            provider_kwargs.setdefault("fence_output", True)
            provider_kwargs.setdefault("use_schema_constraints", False)
    elif api_key:
        provider_kwargs["api_key"] = api_key

    extract_kwargs = dict(
        text_or_documents=context_text,
        prompt_description=prompt_description,
        examples=_build_examples(examples),
        model_id=chosen_model,
    )
    if provider:
        extract_kwargs["provider"] = provider
    if provider_kwargs:
        extract_kwargs["provider_kwargs"] = provider_kwargs

    result = lx.extract(**extract_kwargs)

    answers: List[str] = []
    raw_records: List[Dict[str, object]] = []

    for extraction in getattr(result, "extractions", []):
        attrs = getattr(extraction, "attributes", None) or {}
        raw_records.append(
            {
                "class": getattr(extraction, "extraction_class", ""),
                "text": getattr(extraction, "extraction_text", ""),
                "attributes": attrs,
            }
        )
        answer_val = attrs.get("answer")
        if answer_val:
            answers.append(str(answer_val))

    return {
        "text": ",".join(answers),
        "extractions": raw_records,
    }
