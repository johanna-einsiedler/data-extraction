# Temperature / determinism probe — 2026-10-02 (`analysis/hpo_analysis/temperature_probe.py`)

Same short prompt, `temperature=0`, `max_tokens=4000` (`max_completion_tokens` for gpt-5.2).
"distinct" = number of different completions among identical calls.

| model (provider) | config | T=0 accepted | distinct outputs | note |
|---|---|---|---|---|
| gemini-3.1-pro-preview (Google) | thinking on (proprietary champion) | yes | 1 / 3 | deterministic; `thinking_budget=0` rejected ("only works in thinking mode") |
| zai-org/GLM-5.2 (Together) | reasoning OFF (open champion) | yes | 3 / 3 | non-deterministic |
| zai-org/GLM-5.2 (Together) | reasoning on | yes | 3 / 3 | |
| google/gemma-4-31B-it (OpenRouter) | default (consumer champion) | yes | 3 / 3 | reasoning off: 2 / 3 |
| gpt-5.2 (OpenAI) | default = reasoning_effort none (0 reasoning tokens) | **yes, only with reasoning off** | 1 / 2 | effort low/medium + temperature=0 -> 400 "Only the default (1) value is supported"; effort=medium: 3 / 3 distinct, seed=42 does not fix it (second probe, 2026-10-02) |
| openai/gpt-oss-120b (Together) | default | yes | 2 / 2 | |
| openai/gpt-oss-20b (OpenRouter) | default | yes | 2 / 2 | |
| MiniMaxAI/MiniMax-M3 (Together) | default | yes | 2 / 2 | |
| nvidia/nemotron-3-ultra (OpenRouter) | default | yes | 2 / 2 | |
| moonshotai/Kimi-K2.6 (OpenRouter) | default | yes | 2 / 2 | |
| moonshotai/Kimi-K2.7-Code (OpenRouter) | default | yes | 2 / 2 | |
| qwen/qwen3-14b (OpenRouter) | default | yes | 2 / 2 | |
| Qwen/Qwen3.5-9B (Together) | default | yes | inconclusive | empty content even at 4,000 tokens (reasoning-only reply) |
| deepseek-ai/DeepSeek-V4-Pro (OpenRouter) | — | — | — | "not a valid model" at probe time |

Conclusion: every roster model accepts T=0 in the state the pipeline runs it; for the OpenAI reasoning family that state is reasoning OFF (with reasoning active the API fixes temperature at 1). For the open-weight models and gemini, T=0 is accepted with reasoning on. Determinism is a serving-stack property. Of the frozen
pipelines only the proprietary one repeats exactly. Replicability is measured (RQ2(c) reruns), not assumed.

## gpt-5.2: the search trials ran without reasoning (2026-10-02)

The client sends no `reasoning_effort` to gpt-5.2, and the API default is `none`. Completion tokens per call
in the Phase-1 study (per-item modes, `completion_tokens / llm_calls`):

| model | label `effective_reasoning` | completion tokens per call (median over trials) |
|---|---|---|
| ministral-8b | off | 8 |
| **gpt-5.2** (#481, #503) | **on (wrong)** | **26** |
| qwen3-14b | off | 40 |
| gemini-3.1-pro | on | 293 |
| MiniMax-M3 | on | 346 |
| Kimi-K2.7-Code | on | 445 |
| qwen3-14b | on | 680 |

The four completed gpt-5.2 trials (Phase-1 #27, #346, #481, #503; seed copies phase2 #12, #18, #53) are labelled
`effective_reasoning=on` because the label derives from the roster capability flag. They ran reasoning-off at
temperature 1. The DB label is left as is; the Spec A reasoning coefficient is unaffected (model dummies absorb
an always-on model; only the gpt-5.2 dummy would shift by the reasoning coefficient).
