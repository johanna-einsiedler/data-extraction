import ast
import json
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from together import Together

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
BASE_DIR = Path(__file__).resolve().parent

os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")

TEMPLATE_SOURCES = {
    "base": BASE_DIR / "base_prompt",
    "base_reasoning": BASE_DIR / "base_prompt_with_reasoning",
    "system_single_outcome": BASE_DIR / "system_prompt_single_outcome.txt",
    "system_multiple_outcomes": BASE_DIR / "system_prompt_multiple_outcomes.txt",
    "follow_up": BASE_DIR / "follow_up_prompt_other.txt",
}
client = Together()


def load_templates():
    """Load prompt templates from configured sources and return them as a dict."""
    templates = {}
    for category, source in TEMPLATE_SOURCES.items():
        source_path = Path(source)
        if source_path.is_dir():
            templates[category] = {}
            for template_file in sorted(source_path.glob("*.txt")):
                templates[category][template_file.stem] = template_file.read_text(
                    encoding="utf-8"
                ).strip()
        elif source_path.is_file():
            templates[category] = source_path.read_text(encoding="utf-8").strip()
        else:
            templates[category] = {}
    return templates


def load_prompt(file_name="rewriting_prompt.txt"):
    """Load a single prompt file relative to this module, if it exists."""
    path = BASE_DIR / file_name
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def load_few_shot_templates(folder_path):
    """Load all few-shot templates from the given folder (relative to this module)."""
    templates = {}
    folder = BASE_DIR / folder_path
    if not folder.exists():
        return templates
    for template_file in sorted(folder.glob("*.txt")):
        query_type = template_file.stem
        templates[query_type] = template_file.read_text(encoding="utf-8")
    return templates


def format_choices(choices):
    """Convert a choices mapping into a multi-line string suitable for prompt templates."""
    formatted = []
    for key, info in choices.items():
        value = info.get("value", "")
        desc = info.get("description", "").strip()
        if desc:
            formatted.append(f"{key}: {value} - {desc}")
        else:
            formatted.append(f"{key}: {value}")
    return "\n".join(formatted)


def create_prompts(record, templates):
    """Combine a record with each available template and return the rendered prompts."""
    q_type = record.get("type")
    prompts = {}

    for category, template_source in templates.items():
        if isinstance(template_source, dict):
            template = template_source.get(q_type)
        else:
            template = template_source
        if not template:
            continue

        if isinstance(template_source, dict) and q_type in [
            "multiple_choice",
            "single_choice",
        ]:
            # Multiple/single choice prompts expect a rendered list of answer options.
            choices_str = (
                format_choices(record.get("choices", {}))
                if record.get("choices")
                else "{choices}"
            )
            prompt = template.format(
                concept=record.get("description", ""),
                description=record.get("description_detailed", ""),
                instructions=record.get("instructions", ""),
                choices=choices_str,
                context="{context}",
            )
        elif isinstance(template_source, dict):
            prompt = template.format(
                concept=record.get("description", ""),
                description=record.get("description_detailed", ""),
                instructions=record.get("instructions", ""),
                context="{context}",
            )
        else:
            prompt = template
        # Store the final prompt keyed by its template category.
        prompts[category] = prompt

    return prompts


def build_excel_few_shot_examples(record):
    """
    Build few-shot examples from the Excel-derived record.
    Each record may have up to two examples based on the columns:
    - correct_answer_100 / paragraph_100
    - correct_answer_108 / paragraph_108

    Examples without a paragraph are skipped.
    """
    examples = []

    for suffix in ["100", "108"]:
        answer_key = f"correct_answer_{suffix}"
        paragraph_key = f"paragraph_{suffix}"

        paragraph = record.get(paragraph_key)
        correct_answer = record.get(answer_key)

        # Include only if paragraph exists and is non-empty
        if paragraph and str(paragraph).strip():
            example = (
                f"Example {suffix}:\n"
                f"Context: {paragraph.strip()}\n"
                f"Answer: {correct_answer if correct_answer else 'N/A'}"
            )
            examples.append(example)

    return "\n\n".join(examples) if examples else None


def map_values_to_letters(values_list, choices):
    """
    Map a list of values to their corresponding choice letters based on the 'choices' dict.
    - Case-insensitive matching
    - Ignores extra spaces
    - Assigns 'Y' if not found, 'Z' if value is '0'
    """
    # Build a normalized reverse map
    value_to_letter = {
        str(v["value"]).strip().lower(): k
        for k, v in choices.items()
        if isinstance(v, dict) and "value" in v
    }

    letters = []
    for v in values_list:
        v_str = str(v).strip()
        if v_str == "0":
            letters.append("Z")
            continue
        # Case-insensitive lookup
        letter = value_to_letter.get(v_str.lower(), "Y")
        letters.append(letter)

    # Remove duplicates and sort alphabetically
    return sorted(set(letters))


def generate_prompts_with_rewrite(
    records, templates, rewriting_prompt, model="openai/gpt-oss-120b", only_base=False
):
    results = {}  # use dict keyed by query_id
    few_shot_templates = load_few_shot_templates("synthetic_few_shot")

    for record in records:
        qid = record.get("query_id")
        print(qid)
        if not qid:
            raise ValueError("Record missing 'query_id'")

        prompts = create_prompts(record, templates)
        base_prompt = prompts.get("base")
        reasoning_prompt = prompts.get("base_reasoning")
        if only_base:
            # Skip all further steps and return only the base prompt
            record_with_prompts = record.copy()
            record_with_prompts["prompts"] = {"base_prompt": base_prompt}
            results[qid] = record_with_prompts
            continue
        rewritten_prompt = None
        if base_prompt:
            rewritten_prompt = (
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": rewriting_prompt},
                        {"role": "user", "content": base_prompt},
                    ],
                )
                .choices[0]
                .message.content
            )

        few_shot_template = few_shot_templates.get(record.get("type"))
        synthetic_few_shot_examples = None

        if few_shot_template:
            few_shot_prompt = few_shot_template.format(
                concept=record.get("description"),
                choices=record.get("choices"),
                description=record.get("description_detailed", ""),
                instructions=record.get("instructions", ""),
            )
            synthetic_few_shot_examples = (
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": few_shot_prompt},
                        {"role": "user", "content": base_prompt},
                    ],
                )
                .choices[0]
                .message.content
            )

            # synthetic few shot

            base_prompt_stem = re.sub(
                r"\s*Excerpt:\s*\{context\}\s*Answer:\s*$",
                "",
                base_prompt,
                flags=re.DOTALL,
            )

            synthetic_few_shot_prompt = f"{base_prompt_stem}\n\n# Examples:\n"
            if synthetic_few_shot_examples:
                synthetic_few_shot_prompt = (
                    synthetic_few_shot_prompt + f"{synthetic_few_shot_examples}"
                )
            synthetic_few_shot_prompt = (
                synthetic_few_shot_prompt + "Excerpt: {context} Answer:"
            )

            if len(record["examples"]) > 0:
                true_few_shot_prompt = f"{base_prompt_stem}\n\n# Examples:\n"
                for example in record["examples"]:
                    true_few_shot_prompt = (
                        true_few_shot_prompt
                        + f"Example: {example['context']}\n True Answer: {map_values_to_letters(example['answer'], record['choices'])}\n"
                    )
                    true_few_shot_prompt = (
                        true_few_shot_prompt + " Excerpt: {context} Answer:"
                    )
            else:
                true_few_shot_prompt = []
            # ---- Step 3: Binarization (for choice questions only) ----
            binary_prompts = None
            if record.get("type") in ["multiple_choice", "single_choice"]:
                num_options = len(record.get("choices", {}))
                if num_options > 2:
                    max_retries = 3
                    for attempt in range(1, max_retries + 1):
                        try:
                            binarized_output = (
                                client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": BINARIZATION_PROMPT,
                                        },
                                        {"role": "user", "content": base_prompt},
                                    ],
                                )
                                .choices[0]
                                .message.content
                            )

                            # Convert the string representation of a dict into an actual dict
                            binarized_output = ast.literal_eval(binarized_output)

                            binary_prompts = []
                            for key, entry in binarized_output.items():
                                if entry.get("letter") == "Z":
                                    continue  # skip 'Z' entries (e.g., "Not reported")
                                binary_prompt = BINARY_BASE_PROMPT.format(
                                    concept=record.get("description", ""),
                                    question=entry.get("question"),
                                    context="{context}",
                                )
                                binary_prompts.append(binary_prompt)

                            # ✅ If we got here, the operation succeeded
                            break

                        except Exception as e:
                            print(
                                f"⚠️ Attempt {attempt}/{max_retries} — binarization failed for {qid}: {e}"
                            )
                            if attempt == max_retries:
                                print(
                                    f"❌ Giving up on {qid} after {max_retries} failed attempts."
                                )
                                binary_prompts = None

            # ---- Step 4: Create follow up prompt ----
            follow_up_prompt = None
            if record.get("type") == "multiple_choice":
                template = FOLLOW_UP_PROMPT
                choices_dict = record.get("choices", {})

                # Format the choices into a readable string
                def format_choices(choices_dict):
                    lines = []
                    for key, val in sorted(choices_dict.items()):
                        value = val.get("value") or val.get("description") or str(val)
                        # Escape any braces to avoid .format() issues
                        value = value.replace("{", "{{").replace("}", "}}")
                        lines.append(f"{key}: {value}")
                    return "\n".join(lines)

                choices_str = format_choices(choices_dict) if choices_dict else ""

                follow_up_prompt = template.format(
                    label=record.get("label", ""),
                    description=record.get("description", ""),
                    instructions=record.get("instructions", ""),
                    choices=choices_str,
                    context="{context}",  # keep placeholder for LLM input
                )
        # ---- Step 5: Store results ----
        record_with_prompts = record.copy()
        record_with_prompts["prompts"] = {
            "base_prompt": base_prompt,
            "reasoning_prompt": reasoning_prompt,
            "rewritten_prompt": rewritten_prompt,
            "synthetic_few_shot_examples": synthetic_few_shot_examples,
            "synthetic_few_shot_prompt": synthetic_few_shot_prompt,
            "binary_prompts": binary_prompts,
            "follow_up_prompt": follow_up_prompt,
            "true_few_shot_prompt": true_few_shot_prompt,
        }

        results[qid] = record_with_prompts
    # reorder to take into account dependencies
    evaluation_order = [
        "1.8",
        "1.2.1",
        "3.2",
        "3.2.1",
        "3.4",
        "3.4.1",
        "3.5",
        "3.5.1",
        "4.1",
        "4.1.1",
        "4.2",
        "2.6",
        "4.2.1",
        "4.5",
        "5.3",
    ]

    ordered = {}
    for qid in evaluation_order:
        if qid in results.keys():  # <-- call the method
            ordered[qid] = results[qid]
    for qid in results.keys():  # <-- call the method
        if qid not in ordered:
            ordered[qid] = results[qid]
    return ordered


if __name__ == "__main__":
    PROMPT_TEMPLATES = load_templates()

    with open("query_info.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    REWRITING_PROMPT = load_prompt("rewriting_prompt.txt")
    BINARIZATION_PROMPT = load_prompt("binarization_prompt.txt")
    FOLLOW_UP_PROMPT = load_prompt("follow_up_prompt_other.txt")
    BINARY_BASE_PROMPT = load_prompt("binary_base_prompt.txt")

    # indices = [0, 1, 2, 3, 7, 9, 10, 15]
    final_results = generate_prompts_with_rewrite(
        # [data[i] for i in indices]
        data,
        PROMPT_TEMPLATES,
        REWRITING_PROMPT,
        only_base=False,
    )

    with open("queries_with_prompts.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
