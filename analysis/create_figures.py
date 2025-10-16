import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

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


def plot_query_accuracies(trial_logs_folder):
    """
    Plots per-query accuracies across all trials.
    Each point = one trial's accuracy for that query.
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

    for i, q_id in enumerate(queries):
        for acc in query_accs[q_id]:
            x.append(i)
            y.append(acc)

    # --- Plot ---
    plt.figure(figsize=(14, 6))
    plt.scatter(x, y, alpha=0.4, s=40, color="blue")
    plt.xticks(range(len(queries)), queries, rotation=90)
    plt.xlabel("Query ID")
    plt.ylabel("Accuracy")
    plt.title("Per-Query Accuracy Across Trials")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


import numpy as np


def plot_mean_document_accuracies(trial_logs_folder):
    """
    Computes and plots the mean accuracy for each document across all trials.
    X-axis: document name
    Y-axis: mean accuracy (0–1)
    """
    doc_accs = defaultdict(list)

    # Loop through all trial folders
    for trial_dir in os.listdir(trial_logs_folder):
        trial_path = os.path.join(trial_logs_folder, trial_dir)
        summary_path = os.path.join(trial_path, "trial_summary.json")

        if not os.path.isdir(trial_path) or not os.path.exists(summary_path):
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            doc_aggregates = data.get("document_aggregates", {})
            for _, doc_data in doc_aggregates.items():
                doc_name = doc_data.get("doc_name", f"unknown_{_}")
                acc = doc_data.get("average_accuracy")
                if acc is not None:
                    doc_accs[doc_name].append(acc)
        except Exception as e:
            print(f"⚠️ Skipping {trial_dir}: {e}")

    if not doc_accs:
        print("❌ No document accuracies found.")
        return

    # Sort document names naturally (e.g., "2" before "10")
    sorted_docs = sorted(doc_accs.keys(), key=lambda x: (x.isdigit(), x))

    # Compute mean accuracies
    mean_accuracies = [np.mean(doc_accs[d]) for d in sorted_docs]

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.bar(sorted_docs, mean_accuracies, color="darkorange", alpha=0.8)
    plt.xticks(rotation=45)
    plt.xlabel("Document Name / ID")
    plt.ylabel("Mean Accuracy")
    plt.title("Mean Per-Document Accuracy Across Trials")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


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
    plot_mean_document_accuracies(path)
