# answer_generator.py
import base64
import io
import json
import logging
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI
from together import Together

logger = logging.getLogger(__name__)


def _pil_to_data_url(image) -> str:
    """Serialize a PIL image to a PNG data URL for the OpenAI-style `image_url`
    payload (works for OpenAI, OpenRouter, and Together multimodal models)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

_MAX_RETRIES = 12
# seconds between retries; caps at 300 for later attempts -> ~35 min total, so a
# transient WiFi/DNS outage is ridden out rather than failing the whole trial.
_RETRY_BACKOFF = (5, 15, 30, 60, 120, 180, 300)

# Transient error strings / types that warrant a retry
_RETRYABLE_STRINGS = (
    "503",
    "502",
    "504",
    "429",
    "overloaded",
    "rate limit",
    "timeout",
    "timed out",
    "server error",
    "service unavailable",
    "too many requests",
    # network / DNS blips (e.g. laptop briefly loses WiFi) — ride them out
    "connection error",
    "connection aborted",
    "connection reset",
    "connection refused",
    "max retries exceeded",
    "failed to resolve",
    "nameresolution",
    "temporarily unavailable",
    "apiconnectionerror",
    "network",
)

# Regex to pull "Error code: <NNN>" from Together-style error messages
_HTTP_CODE_RE = re.compile(r"Error code:\s*(\d{3})")


class InsufficientCreditsError(RuntimeError):
    """Raised when a provider rejects a call for lack of funds (out of credits /
    quota / billing / 402). Distinct type so the objective can skip the WHOLE
    trial immediately (rather than hammering every remaining paper) while the
    trial's config is still saved in the study for a later --rerun."""


# Substrings that mean "you are out of money" (NOT a transient rate-limit). These
# are permanent until the user tops up, so we fail the trial fast instead of
# burning the retry budget or grinding through 35 papers that will all fail.
_INSUFFICIENT_FUNDS_STRINGS = (
    "insufficient_quota",
    "exceeded your current quota",
    "check your plan and billing",
    "billing details",
    "credit limit exceeded",
    "credit_limit",
    "add credits",
    "insufficient credit",
    "insufficient balance",
    "payment required",
    "402",
)


def escape_braces_keep_context(template: str, placeholder: str = "context") -> str:
    """
    Escape all braces in the template except the one for {context}.
    """
    # Temporarily replace the placeholder with a token
    token = "___PLACEHOLDER___"
    template = template.replace(f"{{{placeholder}}}", token)

    # Escape all remaining braces
    template = template.replace("{", "{{").replace("}", "}}")

    # Put the placeholder back
    template = template.replace(token, f"{{{placeholder}}}")
    return template


class AnswerGenerator:
    def __init__(
        self,
        api_type: str = "openai",
        api_key: str = None,
        model: str = None,
        exposes_logprobs: bool = True,
        base_url: Optional[str] = None,
        provider_routing: Optional[Dict[str, Any]] = None,
    ):
        """
        api_type: "openai", "together", "gemini", "anthropic", or "openrouter"
        api_key: API key for the corresponding provider
        model: model name, e.g., "gpt-4.1" or "claude-opus-4-8"
        exposes_logprobs: whether the model returns usable token logprobs. When
            False (e.g. reasoning models, Gemini frontier, Claude), logprobs are
            not requested and are returned as NaN.
        base_url: override the API base URL. Lets the OpenAI-compatible path point
            at a self-hosted server (vLLM/Ollama) or an aggregator (OpenRouter);
            ignored by the together/gemini/anthropic branches. None = provider
            default (unchanged for the existing OpenAI models).
        provider_routing: OpenRouter-only `provider` routing object (e.g.
            {"order": [...], "allow_fallbacks": False}). `require_parameters` is
            forced on so OpenRouter only routes to a provider that actually
            supports the params we send (logprobs / response_format) — making
            logprobs reliable and, with an explicit order, runs reproducible.
        """
        self.api_type = api_type.lower()
        self.api_key = api_key
        self.model = model
        self.exposes_logprobs = exposes_logprobs
        self.base_url = base_url
        self.provider_routing = provider_routing
        self._client = None

        if self.api_type not in {
            "openai",
            "together",
            "gemini",
            "anthropic",
            "openrouter",
        }:
            raise ValueError(f"Unsupported api_type: {api_type}")

    def _ensure_client(self):
        if not self.api_key:
            raise ValueError(
                f"API key is required for {self.api_type.title()} AnswerGenerator."
            )
        if self._client is None:
            # A per-request timeout is essential on EVERY provider: without one a
            # hung socket (flaky network, a stalled big VLM/batched payload) blocks
            # the call forever, and because these calls run under the global
            # _MODEL_LOCK a single stall freezes the whole trial (observed: a run
            # wedged for 6h+ at 0% CPU). The Together SDK in particular defaults to
            # NO timeout. On timeout the SDK raises, so _call_with_retry backs off
            # and retries instead of stalling. Override via REQUEST_TIMEOUT (s).
            # 300s: generous enough for a heavy VLM/batched reasoning call (many
            # page images / all items at once) yet bounds a true hang. Timeout
            # errors are retryable, so a genuinely slow provider is retried.
            _timeout = float(os.getenv("REQUEST_TIMEOUT", "300"))
            if self.api_type in ("openai", "openrouter"):
                base_url = self.base_url
                if self.api_type == "openrouter" and not base_url:
                    base_url = "https://openrouter.ai/api/v1"
                # base_url=None -> OpenAI SDK default (existing models unchanged).
                self._client = OpenAI(
                    api_key=self.api_key, base_url=base_url, timeout=_timeout
                )
            elif self.api_type == "together":
                # Together's OpenAI-compatible endpoint rather than the `together`
                # SDK. Same infrastructure, same weights -- but the SDK types
                # `logprobs` as an int and serialises True as `1`, which the newer
                # strict models reject with
                #   400 "invalid type: integer `1`, expected a boolean"
                # and the drop-logprobs fallback then silently returns none.
                # Measured 2026-09-23: MiniMax-M3 and GLM-5.2 yield NO logprobs
                # through the SDK and full logprobs through this client; gpt-oss
                # and Qwen3.5 work either way. Swapping clients is a bug fix, not
                # a provider change, so it introduces no serving confound.
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.together.xyz/v1",
                    timeout=_timeout,
                )
            elif self.api_type == "gemini":
                from google import genai
                from google.genai import types as _gtypes

                # A per-request timeout (ms) is essential: gemini-3.1-pro-preview
                # is a reasoning model, and without a timeout a hung connection
                # (flaky network) blocks the call — and the whole trial — forever
                # (observed: an 8-hour stall on one paper). On timeout the SDK
                # raises, so _call_with_retry can back off and retry instead.
                # Shares the REQUEST_TIMEOUT knob with the other providers.
                self._client = genai.Client(
                    api_key=self.api_key,
                    http_options=_gtypes.HttpOptions(timeout=int(_timeout * 1000)),
                )
            elif self.api_type == "anthropic":
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate(
        self,
        query: str,
        chunks: List[str],
        # prompt_template: Optional[str] = None,
        max_tokens: Optional[int] = None,
        return_logprobs: bool = False,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_enabled: Optional[bool] = None,
        images: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer using retrieved chunks and a prompt.
        Returns a dict with:
            - 'text': generated answer
            - 'logprobs': optional dict with 'tokens' and 'token_logprobs'

        images: optional list of PIL.Image page renders. When given (VLM mode,
            multimodal models only), the pages are attached to the request so the
            model reads the article directly from its images instead of parsed
            text; the {context} placeholder then carries only a short pointer.
        """
        context_text = "\n".join([str(chunk[0]) for chunk in chunks])

        # if prompt_template is None:
        #     prompt_template = (
        #         "Use the following context to answer the question.\n\n"
        #         "Context:\n{context}\n\n"
        #         "Question: {query}\n\n"
        #         "Answer concisely:"
        #     )
        query = escape_braces_keep_context(query, "context")
        prompt = query.format(context=context_text)
        result: Dict[str, Any] = {}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # VLM mode: build a multimodal user turn (text + page images) for the
        # OpenAI-compatible providers (openai/openrouter/together). Gemini takes
        # PIL images directly in its own `contents` list (handled in its branch).
        if images and self.api_type in ("openai", "openrouter", "together"):
            _content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for _im in images:
                _content.append(
                    {"type": "image_url", "image_url": {"url": _pil_to_data_url(_im)}}
                )
            messages.append({"role": "user", "content": _content})
        else:
            messages.append({"role": "user", "content": prompt})

        client = self._ensure_client()

        if self.api_type in ("openai", "openrouter"):
            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=messages,
            #     top_logprobs=20,
            #     # max_tokens=max_tokens,
            # )
            if self.model == "o3-mini-2025-01-31":
                response = client.responses.create(
                    instructions=system_prompt,
                    model=self.model,
                    input=prompt,  # top_logprobs=20
                )
                text = response.output_text
                token_logprobs = np.nan

            else:
                oai_kwargs: Dict[str, Any] = dict(
                    model=self.model,
                    messages=messages,
                )
                # Deterministic decoding for reproducibility. temperature=0 is
                # sent to every model. Until 2026-10-02 the OpenAI reasoning
                # family (gpt-5.x / o-series) was skipped on the assumption that
                # it rejects a non-default temperature; a live probe showed
                # gpt-5.2 accepts it (and repeats exactly with it, not without),
                # so the Phase-1/2 gpt-5.2 search trials ran at the provider
                # default -- declared in the Phase-2 deviations addendum.
                oai_kwargs["temperature"] = 0
                if max_tokens is not None:
                    oai_kwargs["max_tokens"] = int(max_tokens)
                # Reasoning models (gpt-5.x, o-series) reject logprobs.
                if self.exposes_logprobs:
                    oai_kwargs["logprobs"] = True
                if (
                    response_format is not None
                    and response_format.get("type") == "json_schema"
                ):
                    oai_kwargs["response_format"] = response_format
                # OpenRouter: send its `provider`/`reasoning` controls via
                # extra_body. `require_parameters` makes OpenRouter route only to
                # a provider that supports what we asked for (logprobs /
                # response_format), so requested logprobs actually come back;
                # an explicit `order` (from the roster) pins the provider for
                # reproducibility. reasoning uses OpenRouter's own schema (not
                # Together's), so the thinking-off toggle works here too.
                if self.api_type == "openrouter":
                    prov = dict(self.provider_routing or {})
                    prov.setdefault("require_parameters", True)
                    extra_body: Dict[str, Any] = {"provider": prov}
                    if reasoning_enabled is not None:
                        extra_body["reasoning"] = {"enabled": reasoning_enabled}
                    oai_kwargs["extra_body"] = extra_body
                # Route through the robust retry wrapper (same as Together /
                # Anthropic / Gemini) so transient connection/429/5xx errors are
                # retried with back-off instead of failing the trial on the first
                # blip. fallback_kwargs enables the drop-logprobs retry on a
                # non-retryable logprobs error.
                response = self._call_with_retry(
                    lambda kw=oai_kwargs: client.chat.completions.create(**kw),
                    fallback_kwargs=oai_kwargs,
                    client=client,
                )
                text = response.choices[0].message.content

                # Unwrap {"answer": "..."} from json_schema structured output
                if (
                    response_format is not None
                    and response_format.get("type") == "json_schema"
                ):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict) and "answer" in parsed:
                            text = str(parsed["answer"])
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Build dict: token -> logprob (None when not requested/available)
                lp = getattr(response.choices[0], "logprobs", None)
                if self.exposes_logprobs and lp is not None and lp.content:
                    token_logprobs = {t.token: t.logprob for t in lp.content}
                else:
                    token_logprobs = np.nan

            result["text"] = text
            result["logprobs"] = token_logprobs

        elif self.api_type == "together":
            kwargs: Dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                temperature=0,  # deterministic decoding for reproducibility
            )
            # Only the Anthropic branch ever forwarded max_tokens; here the provider
            # default (4096 on Together) silently capped the batched reply, which a
            # reasoning model can exhaust before emitting any JSON. Sent only when
            # the caller sets it, so per-item calls behave exactly as in Phase 1.
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            # Only request logprobs from models that support them. Models flagged
            # exposes_logprobs=False (e.g. Kimi-K2.6/K2.7-Code) reject `logprobs`
            # with a 400 whose error body has a dict `message`, which the Together
            # SDK itself fails to parse (pydantic ValidationError) — so avoid it
            # up front instead of relying on the drop-logprobs fallback per call.
            if self.exposes_logprobs:
                # Together now follows the OpenAI schema: `logprobs` is a BOOLEAN
                # (an integer `1` -> 400 "invalid type: integer, expected boolean").
                # logprobs=True returns the chosen token's logprob (what we use).
                kwargs["logprobs"] = True
            # Reasoning models only return token logprobs in non-thinking mode.
            # reasoning_enabled=False disables chain-of-thought so logprobs flow;
            # None leaves the model's default (thinking on for reasoning models).
            if reasoning_enabled is not None:
                # Together's own `reasoning` field; the OpenAI client only
                # forwards non-standard params through extra_body.
                kwargs["extra_body"] = {"reasoning": {"enabled": reasoning_enabled}}
            if response_format is not None:
                kwargs["response_format"] = response_format
            response = self._call_with_retry(
                lambda kw=kwargs: client.chat.completions.create(**kw),
                fallback_kwargs=kwargs,
                client=client,
            )
            result["text"] = response.choices[0].message.content

            # --- Extract logprobs into a plain {token: logprob} dict ---
            logprob_payload = getattr(response.choices[0], "logprobs", None)
            token_logprobs: Dict[str, float] = {}
            if logprob_payload is not None:
                # New format (OpenAI-compatible): .content is a list of
                # {"token": str, "logprob": float, ...} dicts
                content_list = getattr(logprob_payload, "content", None)
                if content_list:
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
                        if tok is not None and lp is not None:
                            token_logprobs[tok] = lp
                else:
                    # Legacy format: .tokens / .token_logprobs parallel lists
                    tokens = getattr(logprob_payload, "tokens", None) or []
                    lps = getattr(logprob_payload, "token_logprobs", None) or []
                    for tok, lp in zip(tokens, lps):
                        if tok is not None and lp is not None:
                            token_logprobs[tok] = lp
            result["logprobs"] = token_logprobs

        elif self.api_type == "gemini":
            # Google Gemini (frontier models do not expose logprobs).
            from google.genai import types as _gtypes

            # Disable automatic function calling: we pass no tools, so the AFC
            # loop ("max remote calls: 10") is pure overhead and one more way for
            # a reasoning model's multi-step response to hang.
            cfg_kwargs = {
                "automatic_function_calling": _gtypes.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                "temperature": 0,  # deterministic decoding for reproducibility
            }
            if system_prompt:
                cfg_kwargs["system_instruction"] = system_prompt
            cfg = _gtypes.GenerateContentConfig(**cfg_kwargs)
            # VLM mode: pass the page images (PIL) alongside the text prompt. The
            # google-genai SDK accepts PIL.Image objects directly in `contents`.
            gemini_contents = ([prompt] + list(images)) if images else prompt
            response = self._call_with_retry(
                lambda: client.models.generate_content(
                    model=self.model, contents=gemini_contents, config=cfg
                )
            )
            # A reasoning response can carry only non-text 'thought_signature'
            # parts, which makes response.text raise; treat that as empty.
            try:
                result["text"] = response.text or ""
            except Exception:
                result["text"] = ""
            result["logprobs"] = np.nan

        elif self.api_type == "anthropic":
            # Anthropic Claude (no logprobs; quality-only baseline).
            anthropic_kwargs: Dict[str, Any] = dict(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # deterministic decoding for reproducibility
            )
            if system_prompt:
                anthropic_kwargs["system"] = system_prompt
            response = self._call_with_retry(
                lambda: client.messages.create(**anthropic_kwargs)
            )
            result["text"] = "".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            )
            result["logprobs"] = np.nan

        # Token usage for cost accounting (captured for every provider; cost is
        # computed later from the roster's per-model $/M prices).
        result["usage"] = self._usage(response)
        result["finish_reason"] = self._finish_reason(response)
        # Provenance ledger (2026-10-02): provider-issued response id / created /
        # resolved model / routed provider, usage and prompt+output hashes, one
        # hash-chained JSONL line per call. Never allowed to fail the call.
        try:
            from answer_generation import call_ledger as _cl
            result["provenance"] = _cl.provenance(response, self.api_type)
            _cl.record(_cl.entry_for_call(
                api_type=self.api_type, model_requested=self.model, response=response,
                usage=result["usage"], finish_reason=result.get("finish_reason"),
                prompt_material="\x00".join([system_prompt or "", query or "", context_text or ""]),
                output_text=result.get("text"),
                extra={"reasoning_enabled": reasoning_enabled, "n_images": len(images) if images else 0},
            ))
        except Exception:
            pass
        return result

    # -----------------------------------------------------------------
    @staticmethod
    def _finish_reason(response) -> str:
        """Normalised stop reason, or "" when the provider does not report one.

        Batched mode needs to tell "the model answered badly" from "the model ran
        out of output budget": a truncated reply yields unparseable JSON, which
        parse_batched_response silently turns into 43 abstentions. Returns "length"
        for every provider's truncation signal so callers can branch on one value.

        Shapes: OpenAI/Together/OpenRouter `choices[0].finish_reason` ("length"),
        Anthropic `stop_reason` ("max_tokens"), Gemini `candidates[0].finish_reason`
        (an enum whose name is "MAX_TOKENS").
        """
        raw = None
        ch = getattr(response, "choices", None)
        if ch:
            raw = getattr(ch[0], "finish_reason", None)
        if raw is None:
            raw = getattr(response, "stop_reason", None)
        if raw is None:
            cands = getattr(response, "candidates", None)
            if cands:
                raw = getattr(cands[0], "finish_reason", None)
        if raw is None:
            return ""
        raw = getattr(raw, "name", raw)  # unwrap enum
        norm = str(raw).strip().lower()
        return "length" if norm in ("length", "max_tokens", "maxtokens") else norm

    @staticmethod
    def _usage(response) -> Dict[str, int]:
        """Best-effort {prompt_tokens, completion_tokens} across provider response
        shapes: OpenAI/Together `usage.prompt/completion_tokens`, Anthropic
        `usage.input/output_tokens`, Gemini `usage_metadata.*token_count`."""
        u = getattr(response, "usage", None)
        if u is not None:
            pt = getattr(u, "prompt_tokens", None)
            ct = getattr(u, "completion_tokens", None)
            if pt is None:
                pt = getattr(u, "input_tokens", 0)
            if ct is None:
                ct = getattr(u, "output_tokens", 0)
            det = getattr(u, "completion_tokens_details", None)
            rt = getattr(det, "reasoning_tokens", None) if det is not None else None
            return {"prompt_tokens": int(pt or 0), "completion_tokens": int(ct or 0),
                    "reasoning_tokens": int(rt or 0)}
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            return {
                "prompt_tokens": int(getattr(um, "prompt_token_count", 0) or 0),
                "completion_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
                "reasoning_tokens": int(getattr(um, "thoughts_token_count", 0) or 0),
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}

    @staticmethod
    def _extract_error_details(exc: Exception) -> Dict[str, Any]:
        """Pull as much diagnostic info as possible from an API exception."""
        details: Dict[str, Any] = {
            "exception_class": type(exc).__qualname__,
            "exception_module": type(exc).__module__,
            "message": str(exc),
        }

        # --- Together SDK exceptions (TogetherException) ---
        # Attributes: http_status, headers, request_id, api_response
        if hasattr(exc, "http_status") and exc.http_status is not None:
            details["http_status"] = exc.http_status
        if hasattr(exc, "request_id") and exc.request_id is not None:
            details["request_id"] = exc.request_id
        if hasattr(exc, "headers") and exc.headers:
            hdrs = exc.headers
            # Extract the most useful headers for debugging
            interesting = {}
            for key in (
                "x-request-id",
                "retry-after",
                "x-ratelimit-remaining",
                "x-ratelimit-limit",
                "x-ratelimit-reset",
                "cf-ray",
                "x-together-request-id",
            ):
                val = hdrs.get(key) if isinstance(hdrs, dict) else None
                if val is not None:
                    interesting[key] = val
            if interesting:
                details["headers"] = interesting
        if hasattr(exc, "api_response"):
            details["api_response"] = str(exc.api_response)

        # --- OpenAI SDK exceptions (APIStatusError) ---
        # Attributes: response (httpx.Response), body, status_code, code, param
        if hasattr(exc, "status_code"):
            details["http_status"] = exc.status_code
        if hasattr(exc, "body") and exc.body is not None:
            details["body"] = exc.body
        if hasattr(exc, "code") and exc.code is not None:
            details["error_code"] = exc.code
        if hasattr(exc, "param") and exc.param is not None:
            details["param"] = exc.param
        if hasattr(exc, "response"):
            try:
                resp = exc.response
                if hasattr(resp, "headers"):
                    interesting = {}
                    for key in (
                        "x-request-id",
                        "retry-after",
                        "x-ratelimit-remaining-requests",
                        "x-ratelimit-remaining-tokens",
                        "x-ratelimit-reset-requests",
                        "cf-ray",
                    ):
                        val = resp.headers.get(key)
                        if val is not None:
                            interesting[key] = val
                    if interesting:
                        details["headers"] = interesting
                if hasattr(resp, "url"):
                    details["url"] = str(resp.url)
            except Exception:
                pass

        # Fallback: parse HTTP code from the string representation
        if "http_status" not in details:
            m = _HTTP_CODE_RE.search(str(exc))
            if m:
                details["http_status"] = int(m.group(1))

        return details

    # A 429 for exhausted quota / billing is PERMANENT — retrying it just burns
    # the full back-off budget (~35 min) per call before failing. Treat these as
    # non-retryable even though the generic "429" substring is retryable.
    _NONRETRYABLE_STRINGS = (
        "insufficient_quota",
        "exceeded your current quota",
        "check your plan and billing",
        "billing details",
    )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True if the exception looks transient."""
        msg = str(exc).lower()
        if any(s in msg for s in AnswerGenerator._NONRETRYABLE_STRINGS):
            return False
        return any(s in msg for s in _RETRYABLE_STRINGS)

    @staticmethod
    def _as_clean_error(exc, details):
        """Re-wrap a provider exception as a plain RuntimeError whose repr/str is
        JSON-serializable.

        Together's error.__repr__ calls json.dumps() on a CaseInsensitiveDict
        (response headers) and raises `TypeError: Object of type
        CaseInsensitiveDict is not JSON serializable`. Optuna records a failed
        trial via repr(func_err) OUTSIDE the objective's try/except, so that
        TypeError escapes `catch=(Exception,)` and kills the ENTIRE study on the
        first bad-model/400 error. A RuntimeError with a string message reprs
        safely; the full traceback is already logged above. `from None` severs
        the un-serializable original so nothing can repr it later."""
        clean = (
            f"{details.get('exception_module', '?')}.{details.get('exception_class', '?')}: "
            f"HTTP {details.get('http_status', '?')} - {details.get('message', str(exc))}"
        )
        return RuntimeError(clean)

    @staticmethod
    def _call_with_retry(call_fn, fallback_kwargs=None, client=None):
        """Call *call_fn* with exponential back-off on transient errors.

        If the first call fails with a non-retryable error (e.g. Pydantic
        validation from bad logprob payload), retry once without logprobs.
        """
        last_exc = None
        t0 = time.time()
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return call_fn()
            except Exception as exc:
                last_exc = exc
                details = AnswerGenerator._extract_error_details(exc)
                elapsed = time.time() - t0

                # Out of money -> fail IMMEDIATELY with a distinct type so the
                # objective skips the whole trial (config stays saved for rerun).
                # No retry, no drop-logprobs fallback: neither fixes an empty wallet.
                if any(s in str(exc).lower() for s in _INSUFFICIENT_FUNDS_STRINGS):
                    logger.error(
                        "Insufficient funds (HTTP %s) — skipping trial, keeping config: %s",
                        details.get("http_status", "???"), details.get("message", str(exc))[:200],
                    )
                    raise InsufficientCreditsError(
                        AnswerGenerator._as_clean_error(exc, details).args[0]
                    ) from None

                if AnswerGenerator._is_retryable(exc):
                    wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                    logger.warning(
                        "Transient API error (attempt %d/%d, %.1fs elapsed)\n"
                        "  HTTP %s | %s.%s\n"
                        "  Message : %s\n"
                        "  Headers : %s\n"
                        "  RequestID: %s\n"
                        "  Retrying in %ds …",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        elapsed,
                        details.get("http_status", "???"),
                        details.get("exception_module", "?"),
                        details.get("exception_class", "?"),
                        details.get("message", ""),
                        details.get("headers", {}),
                        details.get("request_id")
                        or details.get("headers", {}).get("x-request-id", "n/a"),
                        wait,
                    )
                    # Log full traceback only on the first attempt
                    if attempt == 0:
                        logger.debug(
                            "Full traceback (first attempt):\n%s",
                            traceback.format_exc(),
                        )
                    time.sleep(wait)
                    continue

                # Non-retryable (e.g. Pydantic validation) — try without logprobs
                logger.warning(
                    "Non-retryable API error:\n"
                    "  HTTP %s | %s.%s\n"
                    "  Message : %s\n"
                    "  Details : %s\n%s",
                    details.get("http_status", "???"),
                    details.get("exception_module", "?"),
                    details.get("exception_class", "?"),
                    details.get("message", ""),
                    {
                        k: v
                        for k, v in details.items()
                        if k
                        not in (
                            "message",
                            "http_status",
                            "exception_module",
                            "exception_class",
                        )
                    },
                    traceback.format_exc(),
                )
                # `reasoning` now travels inside extra_body on the Together and
                # OpenRouter paths, so the fallback has to strip it from there too
                # -- otherwise a model that rejects a disabled reasoning mode would
                # keep sending it and fail the retry for the same reason.
                _eb = (fallback_kwargs or {}).get("extra_body") or {}
                if (
                    fallback_kwargs is not None
                    and client is not None
                    and (
                        "logprobs" in fallback_kwargs
                        or "reasoning" in fallback_kwargs
                        or "reasoning" in _eb
                    )
                ):
                    logger.warning("Retrying without logprobs/reasoning …")
                    fallback_kwargs.pop("logprobs", None)
                    fallback_kwargs.pop("reasoning", None)
                    if _eb:
                        _eb.pop("reasoning", None)
                        if _eb:
                            fallback_kwargs["extra_body"] = _eb
                        else:
                            fallback_kwargs.pop("extra_body", None)
                    try:
                        return client.chat.completions.create(**fallback_kwargs)
                    except Exception as exc2:
                        # e.g. a 400 non-serverless model still 400s without
                        # logprobs — clean it so Optuna can record the failure.
                        raise AnswerGenerator._as_clean_error(
                            exc2, AnswerGenerator._extract_error_details(exc2)
                        ) from None
                raise AnswerGenerator._as_clean_error(exc, details) from None

        # All retries exhausted — log and raise so the caller can skip this study
        details = AnswerGenerator._extract_error_details(last_exc)
        elapsed = time.time() - t0
        msg = (
            "\n" + "=" * 60 + "\n"
            f"  API unavailable after {_MAX_RETRIES + 1} attempts ({elapsed:.0f}s total).\n"
            f"  HTTP status  : {details.get('http_status', '???')}\n"
            f"  Error class  : {details.get('exception_module')}.{details.get('exception_class')}\n"
            f"  Request ID   : {details.get('request_id') or details.get('headers', {}).get('x-request-id', 'n/a')}\n"
            f"  Headers      : {details.get('headers', {})}\n"
            f"  Last error   : {last_exc}\n"
            "  Skipping this study and continuing.\n" + "=" * 60
        )
        logger.error(msg)
        raise AnswerGenerator._as_clean_error(last_exc, details) from None
