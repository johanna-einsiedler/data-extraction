"""Batched question mode: ask ALL evaluable items in one prompt per paper (vs the
default one-call-per-item). ~40x fewer calls. Gated to full-document mode, since a
single shared context can't do per-question RAG retrieval.

  build_batched_prompt(QUERIES, prompt_type, item_ids) -> combined user prompt
  parse_batched_response(text, QUERIES) -> (answers {qid:str}, others {qid:str})
  response_is_usable(text, QUERIES)     -> (ok, n_parsed) -- collapse detector

`prompt_type` selects how each item block is written:

  "batched_native"  the synthesized block (type/label/description/choices). This is
                    what every batched trial before 2026-09 used, so it is preserved
                    byte-for-byte and is the level those trials are recorded under.
  <variant name>    the stored per-item prompt from queries_with_prompts.json --
                    base_prompt, cot_prompt, reasoning_prompt, rewritten_prompt,
                    synthetic_few_shot_prompt, true_few_shot_prompt.

Porting the stored prompts needs one non-obvious fix. They were written for
one-call-per-item, so each ends with an "Excerpt: {context} Answer:" slot, and
create_prompts.py appends that slot *inside* the few-shot example loop -- so a
2-example true_few_shot item carries TWO {context} placeholders (27 of 43 items;
rewritten_prompt reaches three). AnswerGenerator.generate() calls
`query.format(context=...)`, which substitutes EVERY occurrence. Concatenating the
stored prompts naively would therefore paste the whole document 63 times: ~934k
tokens, about 7x the smallest roster context window.

So in batched mode the document is hoisted into a single <document> block and every
per-item {context} is dereferenced (see _deref_context).

The model returns one JSON object {qid: answer, "<qid>__other": free_text}. Each
answer is then scored per-item exactly as in individual mode (evaluate_answer), so
all existing scoring rules still apply. The inline "__other" is for parity/records
only (the primary objective scores Other at the letter level, not its free text).
"""
import json
import re

_HEADER = (
    "You are coding one research article for a systematic review of machine learning "
    "in psychology. Read the full article below, then answer EVERY question strictly "
    "from the article. Do not guess or infer; if a detail is not explicitly reported, "
    "answer Z.\n"
)

_FMT = {
    "multiple_choice": "letters of ALL options that apply, comma-separated (e.g. A,C); Z if not stated",
    "single_choice": "the single option code; Z if not stated",
    "numeric": "the number only; Z if not stated",
    "list": "comma-separated list of items; Z if none",
    "open_ended": "a short factual phrase; Z if not stated",
}


# --- Turning a one-call-per-item prompt into a batched item block ------------
#
# The stored prompts assume they are sent alone with the document substituted into
# their own {context} slot. In a batch the document is hoisted into a single
# <document> block, so every placeholder has to go -- str.format substitutes ALL of
# them, and 27 of 43 items carry more than one (create_prompts.py appends the slot
# once per few-shot example). Left alone that pastes the document up to 63 times.
#
# The placeholders come in three kinds, which need three different treatments:
#
#   1. ANSWER SLOTS   "Excerpt: {context} Answer:", and the markdown and fenced
#                     variants rewritten_prompt uses. Deleted outright.
#   2. DIRECTIVES     "Replace {context} with the provided excerpt." Instructions
#                     to the *prompt assembler*, meaningless once the document is
#                     hoisted. Deleted as whole sentences.
#   3. META-REFERENCES "the excerpt (provided in the placeholder {context})".
#                     These carry meaning, so they are reworded to point at the
#                     hoisted block rather than cut.

# 1. Answer slots. Tolerates markdown decoration and fenced blocks, because
#    rewritten_prompt is an LLM paraphrase and formats the slot freely:
#      "**Excerpt:**  \n{context}\n\n**Answer:**"   "```\nExcerpt:\n{context}\n```"
_CTX_SLOT = re.compile(
    r"""
    [\s*_#>`-]*                      # leading decoration
    (?:Excerpt|Context|Input|Source\ Material)\s*:?   # the label
    [^\n{}]{0,60}?                   # a short parenthetical, e.g. "(the text ...)"
    [\s*_:`>-]*
    (?:```[a-z]*\s*)?                # optional opening fence
    \{context\}
    (?:\s*```)?                      # optional closing fence
    [\s*_#>`-]*
    (?:Answer|Your\ answer)?\s*:?[\s*_:`-]*
    """,
    re.I | re.X,
)

# 1b. A bare fenced placeholder, no label. Appears where the label sat inside a
#     directive that rule 2 removed, e.g. item 3.1's
#     "**Excerpt (replace `{context}` with the paper excerpt)**\n\n```\n{context}\n```".
_CTX_FENCED = re.compile(r"\n?[ \t]*```[a-z]*[ \t]*\n?\s*\{context\}\s*\n?[ \t]*```[ \t]*", re.I)

# 2. Whole-sentence directives about substituting the placeholder.
_CTX_DIRECTIVE = re.compile(
    r"[^.\n]*\bReplace\b[^.\n]*\{context\}[^.\n]*[.\n]?", re.I
)

# 3. Meta-references, longest phrasing first so the specific ones win.
_CTX_META = [
    (re.compile(r"\(\s*(?:provided|supplied)\s+in\s+the\s+placeholder\s*[`*]*\{context\}[`*]*\s*\)", re.I),
     "(the document above)"),
    # absorb a determiner/participle so we do not emit "the provided the document above"
    (re.compile(r"\b(?:the|this)\s+(?:provided|supplied|given)\s+[`*]*\{context\}[`*]*", re.I),
     "the document above"),
    (re.compile(r"(?:provided|supplied)\s+in\s+the\s+placeholder\s*[`*]*\{context\}[`*]*", re.I),
     "provided in the document above"),
    (re.compile(r"the\s+(?:text|passage|excerpt)\s+(?:that\s+replaces|inside)\s*[`*]*\{context\}[`*]*", re.I),
     "the document above"),
    (re.compile(r"(?:a\s+)?passage\s*\(\s*[`*]*\{context\}[`*]*\s*\)", re.I),
     "the document above"),
    (re.compile(r"[`*]*\{context\}[`*]*\s*\([^)]{0,60}\)", re.I),
     "the document above"),
    (re.compile(r"the\s+placeholder\s*[`*]*\{context\}[`*]*", re.I),
     "the document above"),
    (re.compile(r"[`*]+\{context\}[`*]+", re.I),
     "the document above"),
]
_CTX_BACKREF = "the document above"

# true_few_shot has no real examples for these 7 items; _select_prompt silently
# falls back to synthetic there, and batched must make the same substitution or the
# two modes would not be comparable.
_TFS_FALLBACK = ("1.2.1", "3.4.1", "3.5", "3.5.1", "0.7", "0.7.2", "0.7.4")


def _deref_context(text):
    """Remove the per-item document slot so the hoisted <document> is the only copy.

    Returns text guaranteed to contain no "{context}" -- verified across all six
    variants by tests/test_batched_prompts.py, which is the check that keeps the
    63x-document blow-up from ever reaching a paid call.
    """
    out = _CTX_DIRECTIVE.sub(" ", text)
    out = _CTX_SLOT.sub("\n", out)
    out = _CTX_FENCED.sub("\n", out)
    for pat, repl in _CTX_META:
        out = pat.sub(repl, out)
    out = out.replace("{context}", _CTX_BACKREF)
    # tidy the seams the deletions leave behind
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n[ \t]*(?:```|---|\|)?[ \t]*\n{2,}", "\n\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _answer_contract(iid, it):
    """The id + output-format lines that make a per-item prompt usable in a batch."""
    t = it.get("type")
    lines = [f'    Value for "{iid}": {_FMT.get(t, "Z if not stated")}']
    if t in ("multiple_choice", "single_choice") and "Y" in (it.get("choices") or {}):
        lines.append(
            f'    If you select "Y" (Other), also give its free text under key "{iid}__other".'
        )
    return "\n".join(lines)


# Every stored prompt was written to be sent alone, so it tells the model to reply
# with one value and nothing else -- "a single line", "only the letter", "no
# explanation". Measured across the 43 items: cot_prompt says so in 43 of 43,
# rewritten in 24, base and synthetic in 19, true_few_shot in 15. Pasted into a
# batch that asks for ONE JSON object, those 43 instructions each contradict the
# reply format.
#
# They are not deleted: "only the letter, no explanation" is exactly the right
# instruction for that item's VALUE, and cutting it would change the prompt being
# tested. Instead each block is explicitly scoped, so the directives inside are
# read as describing one field rather than the whole response.
_SCOPE = (
    "    (The instructions below define the VALUE for this one question. "
    'Any "answer with only..." / "no explanation" / single-line or code-block '
    "instruction inside them describes that value, NOT the shape of your reply -- "
    "the reply is always the single JSON object described at the end.)"
)


def _variant_item(iid, it, prompt_type):
    """One item block built from its stored prompt variant."""
    prompts = it.get("prompts") or {}
    ptype = prompt_type
    if ptype == "true_few_shot_prompt" and not prompts.get(ptype):
        ptype = "synthetic_few_shot_prompt"
    body = prompts.get(ptype)
    if not body:
        raise ValueError(f"item {iid}: no prompt '{prompt_type}' (nor a fallback)")
    t = it.get("type")
    return (
        f"[{iid}] ({t}) {it.get('label') or it.get('variable', '')}\n"
        + _SCOPE
        + "\n--- begin question instructions ---\n"
        + _deref_context(body)
        + "\n--- end question instructions ---\n"
        + _answer_contract(iid, it)
    )


def _one_item(iid, it):
    t = it.get("type")
    lines = [f"[{iid}] ({t}) {it.get('label') or it.get('variable', '')}"]
    desc = (it.get("description") or "").strip()
    if desc:
        lines.append(f"    {desc[:160]}")
    ch = it.get("choices") or {}
    if ch:
        opts = "; ".join(
            f"{k}) {v.get('value') if isinstance(v, dict) else v}" for k, v in ch.items()
        )
        lines.append(f"    Options: {opts}")
    lines.append(f"    Answer format: {_FMT.get(t, 'Z if not stated')}")
    if t in ("multiple_choice", "single_choice") and "Y" in ch:
        lines.append(
            f'    If you select "Y" (Other), also give its free text under key "{iid}__other".'
        )
    return "\n".join(lines)


def build_batched_prompt(QUERIES, prompt_type="batched_native", item_ids=None):
    """Assemble the single combined prompt over all items. Keeps exactly one
    {context} placeholder, which answer_generator fills with the full document.

    item_ids restricts the block to a subset -- used by the two-call split fallback
    when a model cannot return all 43 answers in one response.
    """
    items = (
        QUERIES
        if item_ids is None
        else {k: v for k, v in QUERIES.items() if k in set(item_ids)}
    )
    if not items:
        raise ValueError("build_batched_prompt: empty item set")
    if prompt_type in (None, "batched_native"):
        body = "\n\n".join(_one_item(iid, it) for iid, it in items.items())
    else:
        body = "\n\n".join(
            _variant_item(iid, it, prompt_type) for iid, it in items.items()
        )
    tail = (
        "\nReturn ONLY a single JSON object mapping each question id to its answer, "
        "using the per-question format. Include EVERY id. For any \"Other\" (Y) "
        'choice, add a "<id>__other" key with the free text. Example:\n'
        '{"0.7": "2", "1.1": "135", "1.5": "A,C", "0.7.1": "A,Y", '
        '"0.7.1__other": "caret", "1.3": "adolescents"}\n'
        "No explanation, no extra text."
    )
    return (
        _HEADER
        + "\n<document>\n{context}\n</document>\n\n<questions>\n"
        + body
        + "\n</questions>\n"
        + tail
    )


def split_item_ids(QUERIES):
    """Halve the item set for the two-call fallback, preserving order."""
    ids = list(QUERIES)
    mid = (len(ids) + 1) // 2
    return ids[:mid], ids[mid:]


def _stringify(v):
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    return str(v).strip()


def _extract_json(text):
    """Best-effort: pull the first {...} object from the model output (tolerating
    ``` fences and surrounding prose)."""
    if not text:
        return {}
    t = text.strip()
    t = re.sub(r"```(?:json)?", "", t)
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def parse_batched_response(text, QUERIES):
    """Split the JSON response into per-item answers. Missing/unparseable items
    default to 'Z' (so they score as abstentions, not crashes)."""
    obj = _extract_json(text)
    answers, others = {}, {}
    for iid in QUERIES:
        v = obj.get(iid)
        answers[iid] = _stringify(v) if v is not None and str(v).strip() != "" else "Z"
        o = obj.get(f"{iid}__other")
        if o:
            others[iid] = _stringify(o)
    return answers, others


def response_is_usable(text, QUERIES, min_frac=0.5):
    """Did the model actually answer, or did the whole paper collapse?

    parse_batched_response() maps an unparseable response to "Z" for every item,
    which scores the all-abstain floor (18 of 43 items are correctly NA) instead of
    raising. That silent failure hit 7 of 28 batched trials in the Phase-1 study and
    was only ever caught post-hoc in stage2_analyses/. Callers use this to decide
    whether to retry with the two-call split.

    Returns (ok, n_parsed). `ok` is False when the JSON did not parse at all, or when
    fewer than `min_frac` of the requested ids came back with a non-empty value.
    """
    obj = _extract_json(text)
    if not obj:
        return False, 0
    n = sum(
        1
        for iid in QUERIES
        if obj.get(iid) is not None and str(obj.get(iid)).strip() != ""
    )
    return (n >= max(1, int(min_frac * len(QUERIES)))), n
