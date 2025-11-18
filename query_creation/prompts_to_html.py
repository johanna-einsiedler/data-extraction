"""Render the generated prompts JSON into an HTML table for quick inspection."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

BASE_DIR = Path(__file__).resolve().parent
PROMPT_JSON_PATH = BASE_DIR / "queries_with_prompts.json"
OUTPUT_HTML_PATH = BASE_DIR / "research_prompts.html"


HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Research Prompt Table</title>
<style>
  body { font-family: Arial, sans-serif; margin: 20px; }
  .table-container { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; table-layout: fixed; min-width: 1200px; }
  th, td { border: 1px solid #ccc; padding: 8px; vertical-align: top; word-wrap: break-word; min-width: 200px; }
  th { background-color: #f2f2f2; text-align: left; }
  tr:nth-child(even) { background-color: #fafafa; }
  pre { white-space: pre-wrap; margin: 0; }
</style>
</head>
<body>
<h1>Research Prompt Table</h1>
<div class="table-container">
<table>
"""

HTML_FOOTER = """\
</tbody>
</table>
</div>
</body>
</html>
"""


def load_prompts(path: Path) -> Dict[str, Mapping[str, str]]:
    """Load the JSON file containing prompts grouped by query identifier."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def collect_prompt_keys(prompt_data: Mapping[str, Mapping[str, str]]) -> Iterable[str]:
    """Return a stable, human-friendly order of all prompt variants present in the data."""
    discovered = set()
    for item in prompt_data.values():
        discovered.update(item.get("prompts", {}).keys())

    preferred_order = [
        "base_prompt",
        "reasoning_prompt",
        "rewritten_prompt",
        "synthetic_few_shot_examples",
        "synthetic_few_shot_prompt",
        "true_few_shot_prompt",
        "binary_prompts",
        "follow_up_prompt",
    ]

    ordered = [key for key in preferred_order if key in discovered]
    ordered.extend(sorted(discovered - set(ordered)))
    return ordered


def normalise_value(value) -> str:
    """Convert prompt values of various types to a readable string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n\n".join(str(item) for item in value)
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_table(prompt_data: Mapping[str, Mapping[str, str]]) -> str:
    """Render the full HTML table with all prompt variants."""
    prompt_keys = list(collect_prompt_keys(prompt_data))

    header_cells = [
        "<th>Query ID</th>",
        "<th>Label</th>",
        "<th>Type</th>",
    ] + [
        f"<th>{html.escape(key.replace('_', ' ').title())}</th>"
        for key in prompt_keys
    ]

    rows = []
    for qid, item in prompt_data.items():
        prompts = item.get("prompts", {})
        row_cells = [
            f"<td>{html.escape(qid)}</td>",
            f"<td>{html.escape(str(item.get('label', '')))}</td>",
            f"<td>{html.escape(str(item.get('type', '')))}</td>",
        ]

        for key in prompt_keys:
            value = normalise_value(prompts.get(key, ""))
            row_cells.append(f"<td><pre>{html.escape(value)}</pre></td>")

        row_html = "\n".join(["<tr>"] + row_cells + ["</tr>"])
        rows.append(row_html)

    table_html = "\n".join(
        [
            "<thead>",
            "<tr>",
            "\n".join(header_cells),
            "</tr>",
            "</thead>",
            "<tbody>",
            "\n".join(rows),
            "</tbody>",
        ]
    )
    return table_html


def write_html(content: str, path: Path) -> None:
    """Persist the rendered HTML to disk."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def main() -> None:
    """Entry point for the CLI script: render HTML overview of all prompts."""
    data = load_prompts(PROMPT_JSON_PATH)
    table_html = render_table(data)
    html_content = HTML_HEAD + table_html + HTML_FOOTER
    write_html(html_content, OUTPUT_HTML_PATH)
    print(f"HTML file generated: {OUTPUT_HTML_PATH}")


if __name__ == "__main__":
    main()
