# Accuracy definition (per-variable scoring) — for the Phase-2 preregistration

> **Status:** reconciled against the code on 2026-09-04 (`src/mc_rule.py`, `src/evaluate.py`,
> `src/answer_generation/filtered_models.json`). Earlier revisions of this file described strict
> exact set-match for multiple choice and Phi-3-mini as the open-ended judge; both were superseded
> by the implementation and are corrected here.

This documents how each codebook item is scored in the tuning objective and the downstream analyses.
It refines Phase-1 §5.1 with (a) the gold-construction procedure supplied by the data owner, and (b) the
open-ended metric change (NLI → LLM-as-a-judge) already noted in Phase-2 §1.2. **No partial credit is
used** — every structured item follows `Acc_i = 1(Ŷ_i = Y_i)`; only normalization of surface forms and
the semantic matchers (judge for "Other"/open-ended) define *when* two answers count as equal.

## Ground truth
The finalized gold (`09_data_final`) is built per variable as follows:
- **Single-choice**: manually adjudicated consensus (rater conflicts flagged and resolved; in rare cases
  both raters were wrong and corrected).
- **Multiple-choice, curated** (`3.1 algo`, `4.1 compare`, `4.2 valid`, `4.3 perfmetric`): manually
  adjudicated final solutions, treated as ground truth analogously to single-choice.
- **Multiple-choice, remaining**: pragmatic rule-based gold = the **union** of the two raters' selections.
- **"Other" entries** (any MC with an "Other" category): open free-text entries were manually mapped to
  existing codebook levels or to newly-defined levels; the dataset stores the concrete new level.
- **Dropped** (`1.10 rationale`, `2.6 splitprep`, `4.4 perftrain`, `5.3 interpret`, `5.4 general`): too
  many rater conflicts / not objectively adjudicable → excluded from the finalized gold.

## Accuracy by question type
| Type | Items (n) | Scoring rule |
|------|-----------|--------------|
| **Single-choice** | 10 | Exact match of the selected option key. `Z` (not reported) is an option. |
| **Multiple-choice** | 21 | **Agreement band `I ⊆ P ⊆ U`** for the 17 non-curated items: the prediction must contain the two raters' intersection `I` and stay inside their union `U` (option letters only, taken from the per-rater raw file). The 4 curated items {3.1, 4.1, 4.2, 4.3} keep **strict exact set-match** against the adjudicated gold. Implemented in `src/mc_rule.py:score_mc`. |
| — "Other" element | — | The "Other" (Y) option counts as shared **only if** the model's follow-up free-text and the gold's "Other" entry are judged equivalent by the LLM-as-a-judge (`openai/gpt-oss-120b` via Together). This stricter variant is **logged as `accuracy_secondary`, not the objective** (opt-in via `MC_JUDGE_SECONDARY=1`), so the production objective never calls the judge in the hot path. LLM-Other vs a gold *existing* option (or vice-versa) → mismatch. |
| **Numeric entry** | 9 | Parse the value (strip `%`, units, thousands separators); `Z`/absent maps to the not-reported code, and matches a not-applicable (`-99`) gold. Otherwise compare **rounded to the lesser reported decimal precision** (e.g. `0.85` ≈ gold `0.849`; `90%` → `90`); integer counts are exact. |
| **List entry** | 1 | Normalize each item (lowercase; strip version numbers, parentheticals, `package`/`library` noise) then require an **exact set-match** of the normalized item sets. |
| **Open-ended** | 2 | **LLM-as-a-judge** equivalence (`openai/gpt-oss-120b` via Together; set by `JUDGE_BACKEND` / `EQUIV_JUDGE_MODEL`): correct iff the model's answer and the gold are judged to convey the same meaning. Replaces the Phase-1 bidirectional DeBERTa-NLI (judge 97% vs NLI 73% on a small labeled set; Phase-2 §1.2). The NLI metric remains selectable via `OPEN_ENDED_METRIC=nli` using the original Phase-1 checkpoint. |

**Objective** (Phase-1 §6.2.1): mean of `Acc_i` over the 35 tuning papers × **43 evaluable items**.

## Per-item scoring
| Item | Variable | Type | Rule |
|------|----------|------|------|
| 0.7 | code | single_choice | exact |
| 0.7.1 | codelang | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 0.7.2 | codevers | single_choice | exact |
| 0.7.3 | packages | list | normalized exact set-match |
| 0.7.4 | packagesvers | single_choice | exact |
| 0.8 | data | single_choice | exact |
| 0.9 | prereg | single_choice | exact |
| 1.1 | n | numeric | round-to-precision |
| 1.2 | p | numeric | round-to-precision |
| 1.2.1 | cases | numeric | round-to-precision |
| 1.3 | sample | open_ended | LLM-judge |
| 1.4 | country | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 1.5 | predtype | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 1.6 | outtype | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 1.7 | predtask | open_ended | LLM-judge |
| 1.8 | outcome | single_choice | exact |
| 1.9 | long | single_choice | exact |
| 2.1 | miss | single_choice | exact |
| 2.2 | misshand | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 2.3 | enrich | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 2.4 | engineer | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 2.5 | select | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 3.1 | algo | multiple_choice | exact set-match (**curated** gold) |
| 3.2 | tunevalues | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 3.2.1 | tunemethod | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 3.3 | resampling | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 3.4 | nested | single_choice | exact |
| 3.4.1 | nesthand | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 3.5 | imbalhand | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 3.5.1 | sampling | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 4.1 | compare | multiple_choice | exact set-match (**curated** gold) |
| 4.2 | valid | multiple_choice | exact set-match (**curated** gold) |
| 4.2.1 | ratio | numeric | round-to-precision |
| 4.3 | perfmetric | multiple_choice | exact set-match (**curated** gold) |
| 4.5.1–4.5.5 | highestperf | numeric | round-to-precision; `-99` = not-applicable |
| 4.6 | ci | single_choice | exact |
| 4.7 | sig | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 5.1 | iml | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| 5.2 | limit | multiple_choice | band `I ⊆ P ⊆ U` (raw rater sets) |
| **Dropped** | 1.10 rationale, 2.6 splitprep, 4.4 perftrain, 5.3 interpret, 5.4 general | — | not evaluated (no adjudicated gold) |
| **No gold column** | 4.1.1 bestmodel, 4.5.6 (R² version) | open_ended | not evaluated |

Implementation: `src/evaluate.py` (per-type scoring), `src/_data_loading.py` (gold mapping, `DROPPED_ITEMS`,
`CURATED_MC_ITEMS`, Unicode-unescape). The open-ended judge = `equivalency_detection/evaluate_llms.py`.

## Procedural notes & deviations from Phase-1

### Model roster (RQ1.5): 17 entries, 16 effective

Phase-1 named a 19-model roster. Two models were retired from the Together **serverless** tier before the
tuning run and could no longer be called reproducibly (HTTP 400 `model_not_available` / "non-serverless";
using them would require a paid dedicated endpoint), so they were dropped:

- **`zai-org/GLM-5.1`** — the GLM family remains represented by `zai-org/GLM-5.2` (serverless).
- **`Qwen/Qwen3-235B-A22B-Instruct-2507`** — the Qwen family remains represented by `Qwen/Qwen3.5-9B`.

`src/answer_generation/filtered_models.json` holds **17 entries**: 12 Together, 2 OpenRouter (local-tier:
`mistralai/ministral-8b-2512`, `qwen/qwen3-14b`), 2 OpenAI (`gpt-4.1-2025-04-14`, `gpt-5.2`) and 1 Gemini
(`gemini-3.1-pro-preview`) — i.e. **14 open-weight and 3 proprietary**, of which 5 are image-capable.
Phase-1's small-reasoning OpenAI entry (o3-mini-high, since retired) is represented by **gpt-5.2**.

`MiniMaxAI/MiniMax-M2.7` is flagged `disabled` (its provider moved it off the serverless tier mid-study). It
is never seeded but is deliberately **kept in the sampling distribution**, because removing a category from a
live Optuna study invalidates every trial already recorded against it. The roster is therefore
**17 sampleable / 16 effective**.

`claude-opus-4-8` [Anthropic] is **deferred** (its API account was out of credit at run time) and is *not*
present in the roster file. `pearl-ai/gemma-4-31b-it` was dropped as a near-duplicate of
`google/gemma-4-31B-it` and is likewise absent.

**Logprob availability.** 12 of the 17 expose token logprobs; the 5 that do not are `Kimi-K2.7-Code`,
`Kimi-K2.6`, `gpt-5.2`, `gemini-3.1-pro-preview` and `mistralai/ministral-8b-2512`. They contribute to the
accuracy comparison but cannot enter any logprob-based confidence/calibration analysis.

### No trial pruning — every configuration evaluated on the full tuning set
Phase-1 §6.2 specifies that each sampled configuration is scored on the **full 35-paper tuning set**. The
Optuna study therefore runs with a **NopPruner** (no per-trial early stopping): every trial's reported
accuracy is a complete 35-paper mean, so the model comparison is unbiased. The only early-termination is
the study-level **patience rule** (stop after 15 trials with no improvement, minimum 100 trials). An
earlier configuration used Optuna's default MedianPruner, which was found to terminate slower-starting
models' trials before completion (partial, non-comparable scores) and was removed for this reason.

### Run robustness (no effect on scoring)
Two reliability hardenings, neither of which changes any score: (a) provider API errors are re-wrapped as
plain exceptions before Optuna records a failed trial — a raw provider error object's `repr` was not
JSON-serializable and could abort the entire study when Optuna logged the failure; (b) transient network
errors are retried with exponential back-off. Both affect only whether a trial completes, not how it is
scored.
