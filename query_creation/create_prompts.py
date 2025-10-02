import json
import os

from dotenv import find_dotenv, load_dotenv
from together import Together

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")

TEMPLATE_DIRS = {
    "base": "base_prompt",
    # "few_shot": "few_shot_prompt",
    "base_reasoning": "base_prompt_with_reasoning",
    # add more types like "cot": "chain_of_thought_prompt_templates" if needed
}

client = Together()


def load_templates():
    """Load all templates from the defined directories into a dict of dicts."""
    templates = {}
    for category, dir_path in TEMPLATE_DIRS.items():
        templates[category] = {}
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(".txt"):
                    template_type = filename.replace(".txt", "")
                    with open(
                        os.path.join(dir_path, filename), "r", encoding="utf-8"
                    ) as f:
                        templates[category][template_type] = f.read().strip()
    return templates


def load_rewriting_prompt(file_name="rewriting_prompt.txt"):
    """Read the rewriting prompt from the same folder as this script."""
    path = os.path.join(os.path.dirname(__file__), file_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def load_few_shot_templates(folder_path):
    """
    Reads all few-shot prompt templates from text files in a folder.
    Each file should be named after the query_type, e.g., 'single_choice.txt'.

    Returns:
        dict: {query_type: template_string}
    """
    templates = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            query_type = os.path.splitext(filename)[0]  # remove .txt extension
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                templates[query_type] = f.read()
    return templates


def format_choices(choices):
    """Format choices for multiple/single choice questions."""
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
    """Create base and reasoning prompts for a single record."""
    q_type = record.get("type")
    prompts = {}

    for category, templates_dict in templates.items():
        template = templates_dict.get(q_type)
        if not template:
            continue  # skip if no template for this type in this category

        if q_type in ["multiple_choice", "single_choice"]:
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
        else:  # open_ended, list, numeric
            prompt = template.format(
                concept=record.get("description", ""),
                description=record.get("description_detailed", ""),
                instructions=record.get("instructions", ""),
                context="{context}",
            )

        prompts[category] = prompt

    return prompts


def generate_prompts_with_rewrite(
    records, templates, rewriting_prompt, model="openai/gpt-oss-120b"
):
    """Generate prompts and allow for rewriting via LLM later."""
    results = []
    for record in records:
        print(record.get("query_id"))
        print(record.get("type"))
        prompts = create_prompts(record, templates)

        # Store base & reasoning prompts separately for later rewriting
        base_prompt = prompts.get("base")
        reasoning_prompt = prompts.get("base_reasoning")

        # rewrite Base prompt using LLM
        rewritten_prompt = None
        if base_prompt:
            # Example: rewritten_prompt = call_llm_to_rewrite(base_prompt)
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
        few_shot_template = load_few_shot_templates("synthetic_few_shot/")[
            record["type"]
        ]

        synthetic_few_shot_examples = None
        if few_shot_template:
            few_shot_prompt = few_shot_template.format(
                concept=record.get("concept"),
                choices=record.get("choices"),
                description=record.get("description", ""),
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

        # 3. Append the few-shot examples to the base (or rewritten) prompt
        combined_prompt = base_prompt or ""
        if synthetic_few_shot_examples:
            combined_prompt = (
                f"{base_prompt}\n\n# Examples:\n{synthetic_few_shot_examples}"
            )

        # 4. Save all relevant outputs
        results.append(
            {
                "query_id": record.get("query_id"),
                "base_prompt": base_prompt,
                "reasoning_prompt": reasoning_prompt,
                "rewritten_prompt": rewritten_prompt,
                "synthetic_few_shot_examples": synthetic_few_shot_examples,
                "synthetic_few_shot_prompt": combined_prompt,
            }
        )
    return results


if __name__ == "__main__":
    # Load templates
    PROMPT_TEMPLATES = load_templates()

    # Load your JSON data
    with open("query_info.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    REWRITING_PROMPT = load_rewriting_prompt("rewriting_prompt.txt")

    # Generate prompts and optionally prepare rewritten versions
    final_results = generate_prompts_with_rewrite(
        data, PROMPT_TEMPLATES, REWRITING_PROMPT
    )

    # Save results
    with open("queries_with_prompts.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
