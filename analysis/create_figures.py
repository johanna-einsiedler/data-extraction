import json
import os
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import specification_curve as sc

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def plot_overall_accuracy(trial_logs_folder):
    """
    Reads trial_summary.json from each trial subfolder inside `trial_logs_folder`
    and plots overall average accuracy vs. trial number.
    """
    trial_ids = []
    accuracies = []

    # Go through each subfolder
    for trial_dir in os.listdir(trial_logs_folder):
        trial_path = os.path.join(trial_logs_folder, trial_dir)
        summary_path = os.path.join(trial_path, "trial_summary.json")

        if not os.path.isdir(trial_path):
            continue  # skip non-directories
        if not os.path.exists(summary_path):
            print(f"⚠️ No trial_summary.json found in {trial_dir}")
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            trial_id = data.get("trial_number", trial_dir)
            acc = data.get("overall_average_accuracy")

            if acc is not None:
                trial_ids.append(int(trial_id))
                accuracies.append(acc)
        except Exception as e:
            print(f"⚠️ Error reading {trial_dir}: {e}")

    if not trial_ids:
        print("❌ No valid trial summaries found.")
        return

    # Sort by trial id
    sorted_pairs = sorted(zip(trial_ids, accuracies), key=lambda x: x[0])
    trial_ids, accuracies = zip(*sorted_pairs)

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(trial_ids, accuracies, marker="o", linestyle="-", linewidth=1)
    plt.title("Overall Accuracy per Trial")
    plt.xlabel("Trial ID")
    plt.ylabel("Overall Average Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


from collections import defaultdict

import numpy as np


def plot_query_accuracies(trial_logs_folder):
    """
    Plots per-query accuracies across all trials.
    Each point = one trial's accuracy for that query.
    Adds a line for the mean accuracy per query.
    X-axis: query IDs
    Y-axis: accuracy (0–1)
    """
    query_accs = defaultdict(list)

    # Loop through all trial subfolders
    for trial_dir in os.listdir(trial_logs_folder):
        trial_path = os.path.join(trial_logs_folder, trial_dir)
        summary_path = os.path.join(trial_path, "trial_summary.json")

        if not os.path.isdir(trial_path) or not os.path.exists(summary_path):
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            query_aggregates = data.get("query_aggregates", {})
            for q_id, q_data in query_aggregates.items():
                acc = q_data.get("average_accuracy")
                if acc is not None:
                    query_accs[q_id].append(acc)
        except Exception as e:
            print(f"⚠️ Skipping {trial_dir}: {e}")

    if not query_accs:
        print("❌ No query accuracies found.")
        return

    # --- Prepare data for plotting ---
    queries = sorted(
        query_accs.keys(),
        key=lambda x: [
            float(p) if p.replace(".", "", 1).isdigit() else p for p in x.split(".")
        ],
    )
    x = []
    y = []

    # Compute mean accuracies for the line
    mean_accs = []
    for i, q_id in enumerate(queries):
        accs = query_accs[q_id]
        for acc in accs:
            x.append(i)
            y.append(acc)
        mean_accs.append(np.mean(accs))

    # --- Plot ---
    plt.figure(figsize=(14, 6))

    # Scatter plot for individual trial accuracies
    plt.scatter(x, y, alpha=0.4, s=40, color="blue", label="Trial Accuracy")

    # Line plot for mean accuracy
    plt.plot(
        range(len(queries)),
        mean_accs,
        color="red",
        linewidth=2.5,
        label="Mean Accuracy",
    )

    plt.xticks(range(len(queries)), queries, rotation=90)
    plt.xlabel("Query ID")
    plt.ylabel("Accuracy")
    plt.title("Per-Query Accuracy Across Trials")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


import pandas as pd
import seaborn as sns


def plot_document_accuracies(trial_logs_folder):
    """
    Plots per-document accuracies across all trials as a jitter plot,
    and connects dots belonging to the same trial with lines.
    """
    records = []

    # Loop through all trial folders
    for trial_dir in os.listdir(trial_logs_folder):
        trial_path = os.path.join(trial_logs_folder, trial_dir)
        summary_path = os.path.join(trial_path, "trial_summary.json")

        if not os.path.isdir(trial_path) or not os.path.exists(summary_path):
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            trial_id = data.get("trial_number", trial_dir)
            doc_aggregates = data.get("document_aggregates", {})
            for _, doc_data in doc_aggregates.items():
                doc_name = doc_data.get("doc_name", f"unknown_{_}")
                acc = doc_data.get("average_accuracy")
                if acc is not None:
                    records.append(
                        {"trial_id": trial_id, "document": doc_name, "accuracy": acc}
                    )
        except Exception as e:
            print(f"⚠️ Skipping {trial_dir}: {e}")

    if not records:
        print("❌ No document accuracies found.")
        return

    df = pd.DataFrame(records)

    # Sort documents naturally
    doc_order = sorted(df["document"].unique(), key=lambda x: (x.isdigit(), x))
    df["document"] = pd.Categorical(df["document"], categories=doc_order, ordered=True)

    plt.figure(figsize=(12, 6))

    # Connect dots for each trial
    for trial_id, trial_df in df.groupby("trial_id"):
        trial_df_sorted = trial_df.sort_values("document")
        plt.plot(
            trial_df_sorted["document"],
            trial_df_sorted["accuracy"],
            color="gray",
            alpha=0.3,
            linewidth=1,
        )

    # Jittered dots
    sns.stripplot(
        data=df,
        x="document",
        y="accuracy",
        jitter=False,
        size=6,
        alpha=0.7,
        color="darkorange",
    )

    # Mean accuracy per document
    sns.pointplot(
        data=df,
        x="document",
        y="accuracy",
        estimator=np.mean,
        ci=None,
        color="black",
        markers="o",
        scale=0.8,
    )

    plt.xticks(rotation=45)
    plt.xlabel("Document Name / ID")
    plt.ylabel("Accuracy")
    plt.title("Per-Document Accuracies Across Trials (Connected by Trial)")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def load_trial_summaries(trial_logs_folder):
    """
    Reads all trial_summary.json files and returns a DataFrame with:
      - one row per trial
      - columns: trial_id, overall_average_accuracy, and all trial parameters
    """
    records = []

    for trial_dir in os.listdir(trial_logs_folder):
        trial_path = os.path.join(trial_logs_folder, trial_dir)
        summary_path = os.path.join(trial_path, "trial_summary.json")

        if not os.path.isdir(trial_path) or not os.path.exists(summary_path):
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            record = {
                "trial_id": int(data.get("trial_number", trial_dir)),
                "overall_average_accuracy": data.get("overall_average_accuracy", None),
            }

            # Add parameters as columns
            params = data.get("parameters", {})
            for key, val in params.items():
                record[key] = val

            records.append(record)

        except Exception as e:
            print(f"⚠️ Skipping {trial_dir}: {e}")

    if not records:
        print("❌ No valid trial_summary.json files found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("trial_id").reset_index(drop=True)

    print(f"✅ Loaded {len(df)} trials with {len(df.columns)} columns.")
    return df


from matplotlib.lines import Line2D

label_map = {
    "embedder": {"BGEEmbedder": "BGE"},
    "retriever": {"TopKRetriever": "TopK"},
    "query_embedding": {"raw": "Raw", "label_def": "Def."},
    "chunker": {"LengthChunker": "Length"},
    "llm_model": {
        "arcee-ai/AFM-4.5B": "Arcee AFM 4.5",
        "arcee-ai/maestro-reasoning": "Arcee Maestro",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free": "DeepSeek R1 Llama 70B Distill",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "DeepSeek R1 Qwen 14B Distill",
        "gpt-4.1-2025-04-14": "GPT 4.1",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama 4 Maverick",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": "Llama 4 Scout",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "Llama 3.1 405B Instruct",
        "mistralai/Mistral-Small-24B-Instruct-2501": "Mistral 24B Instruct",
        "moonshotai/Kimi-K2-Instruct": "Moonshot Kimi K2 Instruct",
        "openai/gpt-oss-120b": "GPT-oss 120b",
        "openai/gpt-oss-20b": "GPT-oss 20B",
        "zai-org/GLM-4.5-Air-FP8": "GLM 4.5 Air",
    },
    "default_prompt_type": {
        "base_prompt": "Base",
        "reasoning_prompt": "Reasoning",
        "rewritten_prompt": "Rewritten",
        "synthetic_few_shot_prompt": "Synthetic Few Shot",
        "true_few_shot_prompt": "Few Shot",
    },
    "mode": {"rag": "RAG"},
}


param_name_map = {
    "use_default_prompt": "Binary Questions",
    "query_embedding": "Query Embedding Type",
    "default_prompt_type": "Default Prompt Type",
    "llm_model": "LLM",
    "k": "# of chunks retrieved",
    "retriever": "Retriever",
    "embedder": "Embedding Model",
    "chunker": "Chunker",
    "chunk_overlap": "Chunk Overlap",
    "chunk_size": "Chunk Size",
    "mode": "Mode",
    "parser": "Parser",
}
from matplotlib.patches import Patch


def plot_horizontal_legend(
    colored_params, color_map_global, max_rows_per_col=8, fig_width=14
):
    import matplotlib.pyplot as plt

    legend_ax = plt.gca()  # use current axes
    legend_ax.axis("off")

    # Compute total sub-columns
    total_subcols = sum(
        ((len(vals) - 1) // max_rows_per_col + 1) for vals in colored_params.values()
    )
    x_start, x_end = 0.02, 0.98
    x_total = x_end - x_start
    x_step = x_total / total_subcols
    x_cursor = x_start

    for param_name, vals in colored_params.items():
        n_vals = len(vals)
        n_subcols = (n_vals - 1) // max_rows_per_col + 1
        rows_per_subcol = (n_vals - 1) // n_subcols + 1

        # parameter label centered above its sub-columns
        center_x = x_cursor + (x_step * n_subcols) / 2
        legend_ax.text(
            center_x,
            1.02,
            param_name,
            ha="center",
            va="bottom",
            fontsize=9,
            transform=legend_ax.transAxes,
        )

        for subcol in range(n_subcols):
            for row in range(rows_per_subcol):
                val_idx = subcol * rows_per_subcol + row
                if val_idx >= n_vals:
                    continue
                v = vals[val_idx]
                # x position
                x_pos = x_cursor + subcol * x_step
                # y position
                y_pos = 1 - (row + 0.5) / (rows_per_subcol + 1)
                # dot
                legend_ax.scatter(x_pos, y_pos, color=color_map_global[v], s=50)
                # label close to dot
                legend_ax.text(x_pos + 0.02, y_pos, str(v), va="center", fontsize=8)

        # advance cursor to next parameter block
        x_cursor += (
            n_subcols * x_step + x_step * 0.1
        )  # small padding between parameters


def plot_spec_curve(df, param_keys=None, label_map=None, param_name_map=None):
    """
    df: pandas DataFrame with trial info
    param_keys: list of parameter columns (optional)
    label_map: dict to map values to shorter labels
    param_name_map: dict to map column names to human-readable names
    """
    if param_keys is None:
        param_keys = [
            c for c in df.columns if c not in ["trial_id", "overall_average_accuracy"]
        ]
    df = df.sort_values("overall_average_accuracy", ascending=False).reset_index(
        drop=True
    )

    fig_height = 2 + len(param_keys) * 0.6 + 1.5
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, fig_height),
        gridspec_kw={"height_ratios": [2, len(param_keys) / 2]},
    )
    fig.subplots_adjust(bottom=0.3)

    # --- Top plot: Accuracy ---
    sns.lineplot(
        data=df,
        x=df.index,
        y="overall_average_accuracy",
        marker="o",
        color="black",
        ax=ax1,
    )
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Bottom heatmap of parameters ---
    y_positions = {p: i for i, p in enumerate(param_keys)}
    color_palette = sns.color_palette("tab20", n_colors=20)
    color_map_global = {}
    colored_params = {}

    for param in param_keys:
        values = df[param]
        if label_map and param in label_map:
            values = values.map(lambda v: label_map[param].get(v, v))
        uniques = sorted(values.dropna().unique(), key=lambda x: str(x))
        y = y_positions[param]
        display_name = (
            param_name_map[param]
            if param_name_map and param in param_name_map
            else param
        )

        # Decide colored vs text
        if len(uniques) > 2 and not pd.api.types.is_numeric_dtype(values):
            for v in uniques:
                if v not in color_map_global:
                    color_map_global[v] = color_palette[
                        len(color_map_global) % len(color_palette)
                    ]
            for x, v in enumerate(values):
                ax2.scatter(
                    x, y, color=color_map_global[v], s=50, alpha=0.9, edgecolor="none"
                )
            colored_params[display_name] = uniques
        else:
            for x, v in enumerate(values):
                ax2.text(x, y, str(v), ha="center", va="center", fontsize=8)

    ax2.set_yticks(list(y_positions.values()))
    ax2.set_yticklabels(
        [param_name_map.get(p, p) if param_name_map else p for p in param_keys]
    )
    ax2.set_xlabel("Trial Rank")
    ax2.set_ylabel("Parameter")
    ax2.set_xlim(-0.5, len(df) - 0.5)
    ax2.set_ylim(-0.5, len(param_keys) - 0.5)
    fig.subplots_adjust(bottom=0.4)  # make room
    legend_ax = fig.add_axes([0.1, -0.05, 0.7, 0.2])  # full width, 20% of figure height
    plot_horizontal_legend(colored_params, color_map_global)

    plt.tight_layout(rect=[0, 0.2, 1, 1])
    plt.show()


# Example usage:
# plot_readable_spec_curve("path/to/trial_logs")


# Example usage:
# plot_specification_curve_dots("path/to/trial_logs")


# Example usage:
# plot_specification_curve("path/to/trial_logs")

# Example usage:
# df = load_trial_summaries("path/to/trial_logs")
# display(df.head())

# Example usage:
# plot_mean_document_accuracies("path/to/trial_logs")

# Example usage:
# plot_query_accuracies("path/to/trial_logs")

# Example usage:
# plot_overall_accuracy_from_folders("path/to/trial_logs")

if __name__ == "__main__":
    path = PROJECT_ROOT / "trial_logs"
    # Example usage:
    plot_overall_accuracy(path)
    plot_query_accuracies(path)
    plot_document_accuracies(path)
    summaries = load_trial_summaries(path)
    # summaries.to_csv("summaries.csv")
    plot_spec_curve(summaries, label_map=label_map, param_name_map=param_name_map)
