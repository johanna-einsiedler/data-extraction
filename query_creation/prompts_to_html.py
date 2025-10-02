import json

with open("queries_with_prompts.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)


# HTML template with scrollable table
html_head = """
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

html_footer = """
</tbody>
</table>
</div>
</body>
</html>
"""

# Build table rows
rows = ""
for item in json_data:
    rows += "<tr>\n"
    rows += f"<td>{item.get('query_id', '')}</td>\n"
    rows += f"<td><pre>{item.get('base_prompt', '')}</pre></td>\n"
    rows += f"<td><pre>{item.get('reasoning_prompt', '')}</pre></td>\n"
    rows += f"<td><pre>{item.get('rewritten_prompt', '')}</pre></td>\n"
    rows += f"<td><pre>{item.get('synthetic_few_shot_examples', '')}</pre></td>\n"
    rows += f"<td><pre>{item.get('synthetic_few_shot_prompt', '')}</pre></td>\n"
    rows += "</tr>\n"

# Combine everything
html_content = html_head + rows + html_footer

# Save to file
with open("research_prompts.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("HTML file generated: research_prompts.html")
