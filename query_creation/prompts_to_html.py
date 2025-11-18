"""Render the generated prompts JSON into an HTML table for quick inspection."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, Mapping

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
<thead>
<tr>
<th>Query ID</th>
<th>Base Prompt</th>
<th>Reasoning Prompt</th>
<th>Rewritten Prompt</th>
<th>Synthetic Few-Shot Examples</th>
<th>Synthetic Few-Shot Prompt</th>
</tr>
</thead>
<tbody>
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


def render_rows(prompt_data: Mapping[str, Mapping[str, str]]) -> str:
    """Render a table row for each query with the key prompt variants."""
    rows = []
    for qid, item in prompt_data.items():
        prompts = item.get("prompts", {})
        rows.append(
            "\n".join(
                [
                    "<tr>",
                    f"<td>{html.escape(qid)}</td>",
                    f"<td><pre>{html.escape(prompts.get('base_prompt', '') or '')}</pre></td>",
                    f"<td><pre>{html.escape(prompts.get('reasoning_prompt', '') or '')}</pre></td>",
                    f"<td><pre>{html.escape(prompts.get('rewritten_prompt', '') or '')}</pre></td>",
                    f"<td><pre>{html.escape(prompts.get('synthetic_few_shot_examples', '') or '')}</pre></td>",
                    f"<td><pre>{html.escape(prompts.get('synthetic_few_shot_prompt', '') or '')}</pre></td>",
                    "</tr>",
                ]
            )
        )
    return "\n".join(rows)


def write_html(content: str, path: Path) -> None:
    """Persist the rendered HTML to disk."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def main() -> None:
    """Entry point for the CLI script."""
    data = load_prompts(PROMPT_JSON_PATH)
    rows = render_rows(data)
    html_content = HTML_HEAD + rows + HTML_FOOTER
    write_html(html_content, OUTPUT_HTML_PATH)
    print(f"HTML file generated: {OUTPUT_HTML_PATH}")


if __name__ == "__main__":
    main()
