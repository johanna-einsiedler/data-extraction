import json
from pathlib import Path

# Base folder for queries
queries_folder = Path("synthetic_queries")
queries_folder.mkdir(exist_ok=True)

# Example queries
query_ids = ["1.1", "1.2"]

for qid in query_ids:
    q_folder = queries_folder / qid
    q_folder.mkdir(exist_ok=True)

    # --- 1️⃣ system prompt ---
    system_prompt = f"This is the system prompt for query {qid}."
    with open(q_folder / "system_prompt.txt", "w") as f:
        f.write(system_prompt)

    # --- 2️⃣ base prompt ---
    base_prompt = f"Answer the question using the provided context for query {qid}."
    with open(q_folder / "base_prompt.txt", "w") as f:
        f.write(base_prompt)

    # --- 3️⃣ few-shot prompt ---
    few_shot_prompt = (
        f"Here are some examples. Then answer the question for query {qid}."
    )
    with open(q_folder / "few_shot_prompt.txt", "w") as f:
        f.write(few_shot_prompt)

    # --- 4️⃣ label-definition pair ---
    label_def_pair = {
        "label": f"Label for {qid}",
        "definition": f"This is the definition of the label for query {qid}.",
        "choices": [f"Option {i}" for i in range(1, 4)],  # optional choices
    }
    with open(q_folder / "label_definition_pair.json", "w") as f:
        json.dump(label_def_pair, f, indent=2)

print(f"Synthetic queries folder created at: {queries_folder}")
