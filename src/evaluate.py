import hashlib
import json
import os
import re
from pathlib import Path

from text_utils import normalize, normalize_text
import mc_rule  # shared revised MC/list/numeric rules (also used by the offline re-scorer)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ----------------------------
# Utilities
# ----------------------------
def clean_answer(ans: str) -> str:
    if not ans:
        return ""

    ans = ans.strip()

    # Remove any HTML/XML-like tags such as <...> or </>
    ans = re.sub(r"<[^>]*>", "", ans)

    # Remove leading and trailing non-word characters
    ans = re.sub(r"^[^\w]*", "", ans)
    ans = re.sub(r"[^\w]*$", "", ans)

    # Extract first token (word or number)
    tokens = ans.split()
    if not tokens:
        return ""

    first_token = tokens[0]
    return first_token.upper()



# def load_choices(query_id: str, base_path: str = "../queries") -> tuple[dict, list]:
#     """
#     Load 'choices' and 'prefilled' information for a given query_id.

#     Expected file structure:
#         ../queries/<query_id>/query_info.json

#     Returns:
#         (choices: dict, prefilled: list)

#     Example:
#         choices, prefilled = load_choices("0.7.1")

#     Raises:
#         - Prints clear messages for missing files, invalid JSON, or missing keys.
#     """
#     file_path = Path(base_path) / query_id / "query_info.json"

#     if not file_path.exists():
#         print(
#             f"[load_choices] ❌ File not found for query_id '{query_id}' → {file_path}"
#         )
#         return {}, []

#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         # Extract with defaults
#         choices = data.get("choices", {})
#         prefilled = data.get("prefilled", [])

#         # Validate types
#         if not isinstance(choices, dict):
#             print(f"[load_choices] ⚠️ 'choices' in {file_path} is not a dictionary.")
#             choices = {}

#         if not isinstance(prefilled, (list, tuple)):
#             print(f"[load_choices] ⚠️ 'prefilled' in {file_path} is not a list.")
#             prefilled = []

#         return choices, list(prefilled)

#     except json.JSONDecodeError as e:
#         print(f"[load_choices] ❌ JSON decode error in {file_path}: {e}")
#         return {}, []
#     except Exception as e:
#         print(f"[load_choices] ❌ Unexpected error loading {file_path}: {e}")
#         return {}, []


def map_true_answer_to_choices(true_answer: str, choices: dict):
    truth_tokens = [normalize(x) for x in re.split(r",|;", true_answer)]
    truth_letters = [
        letter for letter, label in choices.items() if normalize(label) in truth_tokens
    ]
    return sorted(truth_letters)


# def get_true_answer_from_excel(
#     excel_path: str, query_id: str, document_name: str
# ) -> str:
#     df = pd.read_excel(excel_path, index_col=0, skiprows=2)
#     if query_id not in df.index:
#         raise ValueError(f"Query ID '{query_id}' not found in Excel.")
#     if document_name not in df.columns:
#         raise ValueError(f"Document '{document_name}' not found in Excel columns.")
#     return str(df.loc[query_id, document_name])


def get_true_answer_from_dict(
    true_answers: dict, queries: dict, query_id: str, document_name: str
) -> list:
    """
    Retrieve true answers from a nested dict for a document/query,
    handling prefilled values:
      - If 'Other: ' is in prefilled and present in the value, keep what follows.
      - For any other prefilled value, discard it and anything that follows.

    Args:
        true_answers: dict like {'021': {'0.7': ['Yes','No'], ...}, ...}
        queries: dict keyed by query_id containing 'prefilled' list
        query_id: the item/query identifier
        document_name: the document column to retrieve

    Returns:
        List of cleaned answers
    """
    if document_name not in true_answers:
        raise ValueError(f"Document '{document_name}' not found in true_answers.")

    doc_dict = true_answers[document_name]

    if query_id not in doc_dict:
        raise ValueError(
            f"Query ID '{query_id}' not found for document '{document_name}'."
        )

    prefilled = queries.get(query_id, {}).get("prefilled", [])
    values = doc_dict[query_id]

    cleaned_list = []

    if not isinstance(values, list):
        values = [values]

    for v in values:
        if pd.isna(v):
            continue
        v_str = str(v).strip()
        if not v_str:
            continue

        # Handle prefilled prefixes
        for pre in prefilled:
            if pre.lower() == "other:" and v_str.startswith(pre):
                # Keep only what comes after 'Other: '
                v_str = v_str[len(pre) :].strip()
            elif v_str.startswith(pre):
                # Discard the prefilled word and anything that comes after
                v_str = ""
                break

        # Split by comma if multiple entries and add non-empty
        split_vals = [x.strip() for x in re.split(r",\s*", v_str) if x.strip()]
        cleaned_list.extend(split_vals)

    return cleaned_list


def _option_alias_keys(value: str) -> set:
    """Normalized alias keys for one answer-option value, so gold that is stored
    as an abbreviation / a fragment still resolves to the right letter. Includes
    the full text, the text with parentheticals removed, each parenthetical token,
    and each '/'-separated part (with and without parentheticals)."""
    keys = set()
    v = str(value)
    keys.add(normalize_text(v))
    keys.add(normalize_text(re.sub(r"\([^)]*\)", " ", v)))  # drop parentheticals
    for p in re.findall(r"\(([^)]*)\)", v):                 # parenthetical tokens
        keys.add(normalize_text(p))
    for part in v.split("/"):                               # "Log Loss / Cross-Entropy"
        keys.add(normalize_text(part))
        keys.add(normalize_text(re.sub(r"\([^)]*\)", " ", part)))
    keys.discard("")
    return keys


_GOLD_VALUE_MAP = None


def _load_gold_value_map():
    """Load the reviewed gold-value -> option-letter map (analysis/translation_list.csv)
    so scoring maps each gold phrasing exactly as decided (incl. aliases and 4.7's
    C->Y). Keyed (item_id, normalized_value). Missing file -> empty (fall back to
    the automatic matcher)."""
    global _GOLD_VALUE_MAP
    if _GOLD_VALUE_MAP is None:
        _GOLD_VALUE_MAP = {}
        p = Path(__file__).resolve().parent.parent / "analysis" / "translation_list.csv"
        if p.exists():
            import csv

            with open(p, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    k = (str(row["item"]).strip(), normalize_text(row["gold_value"]))
                    _GOLD_VALUE_MAP[k] = str(row["maps_to"]).strip().upper()
    return _GOLD_VALUE_MAP


def map_values_to_letters(values_list, choices, item_id=None):
    """
    Map a list of gold values to their choice letters, robustly.
    - First consults the reviewed map (analysis/translation_list.csv) keyed by
      (item_id, value) when ``item_id`` is given — the authoritative decision.
    - Else matches on normalized full text, parenthetical abbreviations,
      paren-stripped text, and '/'-split parts of each option (:func:`_option_alias_keys`).
    - Adds a small manual override for abbreviations absent from the option text
      (``f1`` -> "F1 Score"; ``r2``/``r²`` -> "R-squared").
    - Assigns 'Z' if value is '0', 'Y' (Other) if nothing matches.
    """
    gmap = _load_gold_value_map() if item_id is not None else {}
    value_to_letter = {}
    for k, v in choices.items():
        if not (isinstance(v, dict) and "value" in v):
            continue
        for key in _option_alias_keys(v["value"]):
            value_to_letter.setdefault(key, k)

    # Manual overrides for abbreviations that do not appear in the option text.
    def _letter_where(pred):
        for k, v in choices.items():
            if isinstance(v, dict) and pred(normalize_text(str(v.get("value", "")))):
                return k
        return None

    f1_letter = _letter_where(lambda n: n == "f1 score" or n.startswith("f1 "))
    r2_letter = _letter_where(lambda n: n.replace(" ", "") == "rsquared")
    for alias, letter in {"f1": f1_letter, "r2": r2_letter, "r²": r2_letter}.items():
        if letter:
            value_to_letter.setdefault(alias, letter)

    def _resolve(nv_local):
        """One normalized value -> its option letter, or None if it matches no
        single option (reviewed map first, then the alias matcher)."""
        if item_id is not None and (str(item_id), nv_local) in gmap:
            return gmap[(str(item_id), nv_local)]
        return value_to_letter.get(nv_local)

    letters = []
    for v in values_list:
        v_str = str(v).strip()
        if v_str == "0":
            letters.append("Z")
            continue
        nv = normalize_text(v_str)
        direct = _resolve(nv)
        if direct is not None:
            letters.append(direct)
            continue
        # Split-and-match fallback: a value that matches no single option but whose
        # delimiter-split parts ALL match real options is a concatenated multi-select
        # (a gold-storage artifact, e.g. "<option B text>, <option E text>"), not a
        # genuine "Other". Only applied when EVERY part resolves, so free-text "Other"
        # and comma-containing option labels are never mis-split.
        split = None
        for _d in (", ", "; ", " and "):
            parts = [p.strip() for p in v_str.split(_d) if p.strip()]
            if len(parts) < 2:
                continue
            cand = [_resolve(normalize_text(p)) for p in parts]
            if all(c is not None for c in cand):
                split = cand
                break
        letters.extend(split if split else ["Y"])

    # Remove duplicates and sort alphabetically
    return sorted(set(letters))


# ----------------------------
# Open-ended equivalence scoring
# ----------------------------
# Phase-1 prereg opted for an LLM-as-a-judge (Phi-3-mini) over DeBERTa NLI:
# on the validation set the judge scored 97% vs the NLI's 73% (the NLI misjudged
# ~half of the truly-equivalent pairs). See equivalency_detection/evaluate_llms.py.
# Both models load lazily. Set OPEN_ENDED_METRIC=nli to use the old NLI instead.
OPEN_ENDED_METRIC = os.getenv("OPEN_ENDED_METRIC", "judge").lower()

# --- DeBERTa NLI (lazy; fallback path only) ---
_NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-small-mnli-fever-docnli-ling-2c"
_NLI = {"tok": None, "model": None}
_NLI_LABELS = ["entailment", "neutral", "contradiction"]


def _load_nli():
    if _NLI["model"] is None:
        _NLI["tok"] = AutoTokenizer.from_pretrained(_NLI_MODEL_NAME, use_fast=False)
        _NLI["model"] = AutoModelForSequenceClassification.from_pretrained(_NLI_MODEL_NAME)
    return _NLI["tok"], _NLI["model"]


def nli_entailment(premise: str, hypothesis: str) -> bool:
    tok, mdl = _load_nli()
    inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        probs = F.softmax(mdl(**inputs).logits, dim=-1)
    return _NLI_LABELS[torch.argmax(probs)] == "entailment"


# --- LLM-as-a-judge (Phi-3-mini; the prereg method, default) ---
JUDGE_MODEL_NAME = os.getenv("EQUIV_JUDGE_MODEL", "microsoft/Phi-3-mini-4k-instruct")
# Bump when the judge prompt/logic changes so stale cached verdicts are invalidated.
_JUDGE_VERSION = "v2-supersetrelaxed"
# Judge backend: "local" loads JUDGE_MODEL_NAME via HF (heavy, Phi-3 fp32 ~15GB — can
# OOM a laptop); "together" runs it as a Together API chat completion (no local load).
# For "together", set EQUIV_JUDGE_MODEL to a serverless model, e.g.
# openai/gpt-oss-120b or meta-llama/Llama-3.3-70B-Instruct-Turbo.
JUDGE_BACKEND = os.getenv("JUDGE_BACKEND", "local")
_JUDGE_CLIENT = None
_JUDGE = {"tok": None, "model": None}

# Persistent verdict cache (the judge is greedy/deterministic), keyed on
# (model, gold, pred). Avoids re-running Phi-3 on repeated re-scores / study trials.
_JUDGE_CACHE_PATH = Path(
    os.getenv(
        "JUDGE_CACHE",
        str(Path(__file__).resolve().parent.parent / "trial_cache" / "judge_cache.json"),
    )
)
_judge_cache = None


def _judge_cache_get():
    global _judge_cache
    if _judge_cache is None:
        try:
            _judge_cache = json.loads(_JUDGE_CACHE_PATH.read_text())
        except Exception:
            _judge_cache = {}
    return _judge_cache


def _judge_cache_save():
    """Atomic merge-and-replace so several worker processes can share the cache.

    Each process holds its own in-memory dict; a plain write_text would let the
    last writer discard the others' new verdicts and could expose a half-written
    file to a concurrent reader. Re-read the file, merge, write a temp file and
    os.replace it (atomic on POSIX)."""
    global _judge_cache
    try:
        if _JUDGE_CACHE_PATH.exists():
            on_disk = json.loads(_JUDGE_CACHE_PATH.read_text())
            if isinstance(on_disk, dict):
                on_disk.update(_judge_cache)
                _judge_cache = on_disk
    except Exception:
        pass  # unreadable/partial file: our in-memory copy is the best we have
    try:
        _JUDGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _JUDGE_CACHE_PATH.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(_judge_cache))
        os.replace(tmp, _JUDGE_CACHE_PATH)
    except Exception:
        pass

_JUDGE_SYSTEM = (
    "You are a careful scientific evaluator.\n"
    "Output MUST be valid, standalone JSON. No commentary, no code fences, no explanations."
)
_JUDGE_USER_TMPL = (
    "Task: Determine if candidate description B matches reference description A.\n\n"
    "Description A (reference / gold):\n{a}\n\n"
    "Description B (candidate / model):\n{b}\n\n"
    "Rules:\n"
    "- Ignore differences in case, punctuation, word order, and spelling variants"
    " (e.g. behavior/behaviour).\n"
    "- Synonyms/paraphrases of the same task or construct are the SAME"
    " (e.g. predict/forecast/classify/identify a target; sample vs participants).\n"
    "- If B adds MORE correct detail that A lacks (a superset: extra population, site,"
    " time-window, method, or specificity), it is STILL the same, as long as it does"
    " not contradict A.\n"
    "- Mark as different ONLY if B contradicts A, or changes/omits the CORE construct,"
    " population, outcome, or unit of analysis.\n\n"
    "Respond strictly as JSON:\n"
    '{{"equivalent": true|false, "confidence": 0..1, "reason": "<short>"}}'
)


def _load_judge():
    if _JUDGE["model"] is None:
        from transformers import AutoModelForCausalLM

        tok = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        _JUDGE["tok"] = tok
        _JUDGE["model"] = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL_NAME,
            # JUDGE_DTYPE=bfloat16 halves memory (~7.6 GB) so two workers fit in
            # 34 GB; default float32 keeps Phase-1 numerics. Greedy verdicts can
            # differ between dtypes in rare edge cases -- declare if used.
            torch_dtype=getattr(torch, os.getenv("JUDGE_DTYPE", "float32"))
        ).eval()
    return _JUDGE["tok"], _JUDGE["model"]


def _judge_parse_json(text: str):
    """Return (equivalent, confidence, reason) from the last valid JSON object."""
    last = None
    for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            last = json.loads(m.group(0))
        except Exception:
            continue
    if last is None:
        return False, 0.0, "no-json"
    try:
        conf = max(0.0, min(1.0, float(last.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    return bool(last.get("equivalent")), conf, str(last.get("reason", "")).strip()


def _norm_for_judge(s: str) -> str:
    """Case/punctuation/whitespace-insensitive normal form for the judge fast-path."""
    s = str(s).strip().casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _judge_via_together(user: str) -> str:
    """Run the judge as a Together chat completion (no local model load). Simple
    exponential back-off on transient errors."""
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        from together import Together
        _JUDGE_CLIENT = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    import time
    last = None
    for attempt in range(6):
        try:
            resp = _JUDGE_CLIENT.chat.completions.create(
                model=JUDGE_MODEL_NAME,
                messages=[{"role": "system", "content": _JUDGE_SYSTEM},
                          {"role": "user", "content": user}],
                max_tokens=200,
                temperature=0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:               # noqa: BLE001
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"judge Together call failed after retries: {last}")


def judge_equivalent(a: str, b: str):
    """LLM-as-a-judge (Phi-3) equivalence of two open-ended answers.
    Returns (equivalent: bool, confidence: float, reason: str)."""
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    # An empty/whitespace answer can never be equivalent to a real description.
    # (Fixes the judge false-positive where a BLANK extraction was scored correct
    #  with a hallucinated rationale.) Checked before the cache so stale cached
    #  verdicts can't reintroduce it.
    if not a.strip() or not b.strip():
        return False, 0.0, "empty-answer"
    # Pure case/punctuation/word-spacing differences are equivalent by construction
    # (the judge was observed to flip verdicts on capitalization alone).
    if _norm_for_judge(a) == _norm_for_judge(b):
        return True, 1.0, "normalized-exact"
    cache = _judge_cache_get()
    key = hashlib.sha1(
        f"{JUDGE_MODEL_NAME}\x00{_JUDGE_VERSION}\x00{a}\x00{b}".encode("utf-8")
    ).hexdigest()
    if key in cache:
        c = cache[key]
        return bool(c[0]), float(c[1]), str(c[2])

    user = _JUDGE_USER_TMPL.format(a=a, b=b)
    if JUDGE_BACKEND == "together":
        reply = _judge_via_together(user)
    else:
        tok, mdl = _load_judge()
        if hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(
                [
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"<|system|>\n{_JUDGE_SYSTEM}\n<|end|>\n<|user|>\n{user}\n<|end|>\n<|assistant|>\n"

        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
        end_id = None
        try:
            end_id = tok.convert_tokens_to_ids("<|end|>")
            if isinstance(end_id, list):
                end_id = end_id[0]
            if end_id == tok.unk_token_id:
                end_id = None
        except Exception:
            end_id = None
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=end_id or tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=False)
        start = text.rfind("<|assistant|>")
        if start != -1:
            seg = text[start + len("<|assistant|>") :]
            e = seg.find("<|end|>")
            reply = (seg[:e] if e != -1 else seg).strip()
        else:
            reply = text[len(prompt) :].strip()
    verdict = _judge_parse_json(reply.replace("<|end|>", "").strip())
    cache[key] = list(verdict)
    _judge_cache_save()
    return verdict



# ----------------------------
# Unified evaluation
# ----------------------------
def evaluate_answer(
    llm_output: str,
    question_type: str,
    doc_name: str,
    query_id: str,
    TRUE_ANSWERS: dict,
    QUERIES: dict,
    other_callback=None,
    curated_mc_items=None,
):
    """Evaluate LLM output based on question type.

    ``curated_mc_items``: optional set of multiple-choice item ids whose gold is
    manually adjudicated. For those, MC accuracy is strict exact set-match. For
    every other MC item (rule-based rater-union gold), accuracy is the Jaccard
    set-overlap share, which gives partial credit and is not biased by the
    union's over-inclusion. ``None`` (default) keeps the legacy exact behaviour
    for all MC items. Exact, Jaccard, and per-level agreement are always all
    returned; only which one populates ``accuracy`` changes."""
    print("Doc Name: ", doc_name)
    print("Query id: ", query_id)
    true_path = "../data/true_labels/"
    if isinstance(llm_output, str):
        llm_output = llm_output.strip()
    elif isinstance(llm_output, list) or isinstance(llm_output, set):
        llm_output = [str(x).strip() for x in llm_output]
    true_answer = get_true_answer_from_dict(TRUE_ANSWERS, QUERIES, query_id, doc_name)
    # Load choices if needed
    # choices = None
    # choices, prefilled = load_choices(query_id)
    choices = QUERIES[query_id].get("choices", {})

    if question_type == "single_choice":
        # The option code is whatever precedes the first ';'. reasoning_prompt asks
        # for "<code>; <1-2 sentences>", and models add "; explanation" unprompted
        # (338 of 9,044 Phase-1 base_prompt answers). clean_answer keeps the first
        # whitespace token, so without this split "1; ..." became "1;" and could
        # never equal "1" -- zeroing every well-formed reasoning_prompt answer
        # (measured 2026-09-24: batched reasoning_prompt single-choice 0.00 -> 0.79).
        pred_number = clean_answer(str(llm_output).split(";")[0])

        # Map 'Z' → '0' (Not reported)
        if isinstance(pred_number, str) and pred_number.strip().upper() == "Z":
            pred_number = "0"

        # Get true answer safely
        true_value_key = (
            str(true_answer[0])
            if isinstance(true_answer, (list, tuple))
            else str(true_answer)
        )

        # Compute accuracy
        acc = int(str(pred_number) == true_value_key)

        return {
            "raw_output": llm_output,
            "pred_indicator": pred_number,
            "pred_value": choices.get(str(pred_number), {}).get("value", ""),
            "true_indicator": true_value_key,
            "true_value": choices.get(true_value_key, {}).get("value", ""),
            "accuracy": acc,
        }

    elif question_type == "multiple_choice":
        choices_norm = {k.upper(): v for k, v in choices.items()}
        valid_keys = set(choices_norm.keys()) | {"Z"}

        # Extract predicted letters robustly: a compact run of valid letters
        # ("IKMNQR"), else delimiter-separated single letters — so prose or a
        # spelled-out abbreviation cannot explode into spurious selections.
        raw = str(llm_output).upper()
        compact = re.sub(r"[^A-Z]", "", raw)
        if compact and len(compact) <= 8 and all(c in valid_keys for c in compact):
            pred_letters = set(compact)
        else:
            pred_letters = {
                t
                for t in re.findall(r"(?<![A-Za-z])[A-Z](?![A-Za-z])", raw)
                if t in valid_keys
            }

        # Recognize the spelled-out option ("Other" -> Y) that the letter scan misses.
        if "Y" in choices_norm and re.search(r"\bOTHER\b", raw):
            pred_letters.add("Y")

        # If LLM outputs "Z" (not reported), map it to 0
        if "Z" in pred_letters:
            pred_letters = {"Z"}
            pred_choices = ["0"]
        else:
            pred_choices = [
                normalize_text(choices_norm.get(k, {}).get("value", ""))
                for k in sorted(pred_letters)
            ]
            pred_choices = [x for x in pred_choices if x]

        # Normalize true choices (for display) and map gold -> letters. Pass the
        # RAW gold values to the mapper — it normalizes internally, and keeping the
        # raw delimiters lets its split-and-match fallback recover concatenated
        # multi-selects (normalize_text strips the ", " so pre-normalizing would hide them).
        true_choices = [normalize_text(x) for x in true_answer if x is not None]
        true_letters = map_values_to_letters(
            [str(x) for x in true_answer if x is not None], choices_norm, item_id=query_id)
        true_letters = ["Z" if str(x).strip() == "0" else x for x in true_letters]

        # Handle “Other” (Y): fetch the model's free-text via the follow-up callback.
        other_value = None
        if "Y" in pred_letters and other_callback is not None:
            other_value = other_callback(llm_output)
            if other_value:
                pred_choices.append(normalize_text(other_value))

        # STRICT EXACT set-match on letters (prereg §5.1) for ALL MC items; the
        # curated-vs-union distinction affects only gold construction, not scoring.
        pset, tset = set(pred_letters), set(true_letters)

        # Revised metric (mc_rule, shared with the offline re-scorer): non-curated MC
        # uses I ⊆ P ⊆ U from the per-rater data; curated {3.1,4.1,4.2,4.3} uses exact
        # set-match; both LETTERS-ONLY as the PRIMARY objective (fast, no judge).
        item_label = (QUERIES.get(query_id, {}) or {}).get("label", "")
        acc_primary = mc_rule.score_mc(
            query_id, pset, tset, pred_choices, doc_name, item_label, choices_norm
        )
        # Dependency-gated MC (3.2.1, 3.4.1, 3.5, 3.5.1): when the parent condition
        # makes the item not-applicable, the finalized gold is NA and a model
        # abstention should count correct — mirrors the DEP_NUMERIC rule below.
        # (score_mc scores against the raw-rater agreement band, which doesn't see
        # the resolved NA gold, so correct abstentions were being marked wrong.)
        if query_id in mc_rule.DEP_MC and mc_rule.na_pair(llm_output, true_answer):
            acc_primary = 1

        # SECONDARY (opt-in, MC_JUDGE_SECONDARY=1): the "Other" (Y) element counts as
        # shared only if the free-texts are judge-equivalent. Skipped by default so the
        # production objective never invokes the judge in the hot path.
        other_ok = True
        if os.getenv("MC_JUDGE_SECONDARY", "0") == "1" and "Y" in pset and "Y" in tset:
            gold_other = "; ".join(
                tc
                for tc in true_choices
                if tc and (map_values_to_letters([tc], choices_norm, item_id=query_id)
                           or ["Y"])[0] == "Y"
            )
            llm_other = normalize_text(other_value) if other_value else ""
            other_ok = bool(
                gold_other and llm_other and judge_equivalent(gold_other, llm_other)[0]
            )
        acc_secondary = int(acc_primary and other_ok)
        acc_exact = int(pset == tset and other_ok)   # legacy strict metric (reference)

        # Jaccard + per-level are logged for RQ1.3 analysis only (not the objective).
        if not pset and not tset:
            acc_jaccard = 1.0
        else:
            union = len(pset | tset)
            acc_jaccard = (len(pset & tset) / union) if union else 1.0
        levels = set(choices_norm.keys()) | pset | tset
        acc_per_level = (
            sum(int((lvl in pset) == (lvl in tset)) for lvl in levels) / len(levels)
            if levels
            else 1.0
        )

        return {
            "raw_output": llm_output,  # Original LLM output
            "pred_indicator": sorted(pset),
            "pred_value": pred_choices,
            "true_indicator": sorted(tset),
            "true_value": true_choices,
            "accuracy": acc_primary,  # revised: I⊆P⊆U (non-curated) / exact (curated), letters-only
            "accuracy_secondary": acc_secondary,  # + Other free-text judged (opt-in)
            "accuracy_exact": acc_exact,  # legacy strict exact set-match
            "accuracy_jaccard": acc_jaccard,  # logged for analysis only
            "accuracy_per_level": acc_per_level,
            "mc_metric": "I<=P<=U | curated exact (letters-only)",
            "other_value": other_value,
        }

    elif question_type == "numeric":
        def _parse_num(s):
            """(value, decimals) from the first numeric token, after stripping %,
            units, thousands commas. 'Z'/unparseable -> abstention (0.0)."""
            s = str(s).strip()
            if s.upper() == "Z":
                return 0.0, 0
            s = s.replace(",", "")             # thousands separators
            m = re.search(r"-?\d*\.?\d+", s)   # first numeric token; drops %, units
            if not m:
                return 0.0, 0
            tok = m.group()
            dec = len(tok.split(".")[1]) if "." in tok else 0
            return float(tok), dec

        pred_num, pred_dec = _parse_num(llm_output)
        try:
            true_str = str(true_answer[0]).strip()
            true_num = float(true_str)
            true_dec = len(true_str.split(".")[1]) if "." in true_str else 0
        except (ValueError, TypeError, IndexError):
            true_num, true_dec = np.nan, 0

        if true_num == -99:
            # Not-applicable gold: abstention (absent -> 0) is correct; a real value is wrong.
            acc = int(pred_num == 0)
        elif np.isnan(true_num):
            acc = 0
        else:
            # Compare rounded to the lesser reported precision (0.849 vs 0.85 -> match;
            # 90% -> 90; integer counts stay exact). No partial credit.
            ndp = min(pred_dec, true_dec)
            acc = int(round(pred_num, ndp) == round(true_num, ndp))

        # dependency-gated numerics (1.2.1, 4.2.1): gold-NA + model-abstain -> correct
        # (reconciles -99/-96/0/Z between coder and model conventions).
        if query_id in mc_rule.DEP_NUMERIC and mc_rule.na_pair(llm_output, true_answer):
            acc = 1

        return {
            "raw_output": llm_output,
            "pred_value": pred_num,  # return numeric value
            "true_value": true_num,
            "accuracy": acc,
        }

    elif question_type == "list":
        def _norm_items(raw):
            """Split on newlines/;/,/bullets and normalize each item: lowercase,
            strip bullets, parentheticals (e.g. '(R stats)'), version numbers, and
            'R package'/'library' noise -> comparable package/library names."""
            if isinstance(raw, (list, set, tuple)):
                parts = []
                for x in raw:
                    parts.extend(re.split(r"[\n;,]", str(x)))
            else:
                parts = re.split(r"[\n;,]", str(raw))
            out = []
            for p in parts:
                p = p.strip().lstrip("-*•·").strip().lower()
                if not p or p == "z":
                    continue
                p = re.sub(r"\([^)]*\)", " ", p)                      # (R stats), (v1.2)
                p = re.sub(r"\bv?\d+(?:\.\d+)+\b", " ", p)            # version numbers
                p = re.sub(r"\b(?:r|python)\s+package\b|\bpackage\b|\blibrary\b", " ", p)
                p = re.sub(r"[^\w\s]", " ", p)
                p = re.sub(r"\s+", " ", p).strip()
                if p:
                    out.append(p)
            return out

        # Strict EXACT set-match (prereg §5.1) after normalization — no partial credit.
        pred_items = set(_norm_items(llm_output))
        true_items = set(_norm_items(true_answer))
        acc = int(pred_items == true_items)
        # gold ['0']/NA (no packages) + model Z/[]/abstain -> correct
        if mc_rule.na_pair(llm_output, true_answer):
            acc = 1

        return {
            "raw_output": llm_output,
            "pred_value": sorted(pred_items),
            "true_value": sorted(true_items),
            "accuracy": acc,
        }

    elif question_type == "open_ended":
        if OPEN_ENDED_METRIC == "nli":
            entail1 = nli_entailment(true_answer[0], llm_output)
            entail2 = nli_entailment(llm_output, true_answer[0])
            return {
                "raw_output": llm_output,
                "pred_value": llm_output,
                "true_value": true_answer,
                "accuracy": int(entail1 and entail2),
                "entail_true_to_pred": entail1,
                "entail_pred_to_true": entail2,
            }
        # LLM-as-a-judge (Phi-3), per Phase-1 preregistration.
        equiv, conf, reason = judge_equivalent(true_answer[0], llm_output)
        return {
            "raw_output": llm_output,
            "pred_value": llm_output,  # so the model's answer is visible in reports
            "true_value": true_answer,
            "accuracy": int(equiv),
            "judge_equivalent": bool(equiv),
            "judge_confidence": conf,
            "judge_reason": reason,
        }

    else:
        raise ValueError(f"Unknown question_type: {question_type}")


# if __name__ == "__main__":
# QUERIES_JSON_PATH = Path(
#     "../query_creation/queries_with_prompts.json"
# )  # Path to your JSON file

# def load_queries():
#     with open(QUERIES_JSON_PATH, "r", encoding="utf-8") as f:
#         query_dict = json.load(f)

#     queries = {}
#     for qid, q in query_dict.items():
#         queries[qid] = q  # keep original structure
#     return queries

# def load_true_answers(
#     path: str = "../data/true_labels/human_codes_test2.xlsx",
# ) -> dict:
#     df = pd.read_excel(path)

#     # Strip whitespace from column headers to avoid lookup errors
#     df.columns = [col.strip() for col in df.columns]

#     # if columns have no name, i.e. those are codings, use the entry in the second row

#     second_row = df.iloc[1]
#     new_columns = []
#     for col in df.columns:
#         if col.startswith("Unnamed"):
#             new_columns.append(str(second_row[col]))  # use second row's entry
#         else:
#             new_columns.append(col)  # keep original name

#     df.columns = new_columns

#     # ------------------------
#     # 2. Identify item codes
#     # ------------------------
#     # Extract all unique entries in the "Item" column
#     items = df["Item"].unique()

#     # Remove NaN and invalid entries
#     items = [x for x in items if x is not None and str(x) != "nan"]

#     # Keep only entries that look like hierarchical item codes (e.g. "0.7.2")
#     items = [x for x in items if not isinstance(x, int)]
#     items = [x for x in items if all(c.isdigit() or c == "." for c in x)]

#     # Exclude generic meta-rows that are not part of the coding scheme
#     to_remove = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "99"]
#     items = [x for x in items if x not in to_remove]
#     true_answers = {}
#     for doc_name in ["021", "023", "038"]:
#         item_answers = {}
#         for item in items:
#             row = df.loc[df["Item"] == item]
#             query_id = item

#             start_idx_list = df.index[df["Item"] == item].tolist()
#             if start_idx_list:
#                 start_idx = start_idx_list[0]
#                 row_ids = [start_idx]
#                 for idx in range(start_idx + 1, len(df)):
#                     if pd.isna(df.at[idx, "Item"]):
#                         row_ids.append(idx)
#                     else:
#                         break

#                 item_answers[query_id] = df.loc[row_ids, doc_name].values.tolist()

#         true_answers[doc_name] = item_answers
#     return true_answers

# QUERIES = load_queries()

# TRUE_ANSWERS = load_true_answers()

# print(
#     evaluate_answer(
#         llm_output="10",
#         question_type="numeric",
#         doc_name="021",
#         query_id="1.2",
#         TRUE_ANSWERS=TRUE_ANSWERS,
#         QUERIES=QUERIES,
#         # answer_file="human_codes_test2.xlsx",
#         # choices_file="query_choice_mapping.json",
#         # other_callback=handle_other,
#     )
# )
