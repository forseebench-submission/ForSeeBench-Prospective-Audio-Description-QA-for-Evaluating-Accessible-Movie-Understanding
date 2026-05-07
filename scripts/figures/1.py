#!/usr/bin/env python3
"""Generate the adaptive-context source comparison figure.

Macro movie accuracy is the unweighted mean of per-movie accuracies. It is
derived from per-example predictions and movie identifiers when not stored in a
metrics file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


SOURCES = [
    ("Human AD", "outputs/evaluation/ground_truth/all_10_movies_run/metrics.json", "outputs/evaluation/ground_truth/all_10_movies_run/predictions.jsonl"),
    ("NarrAD", "outputs/evaluation/narrad/all_10_movies_run/metrics.json", "outputs/evaluation/narrad/all_10_movies_run/predictions.jsonl"),
    ("AutoAD-Zero", "outputs/evaluation/autoad_zero/all_10_movies_run/metrics.json", "outputs/evaluation/autoad_zero/all_10_movies_run/predictions.jsonl"),
]
COLORS = {
    "Human AD": "#4C78A8",
    "NarrAD": "#F58518",
    "AutoAD-Zero": "#54A24B",
}


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def per_movie_accuracy(dataset_path: Path, predictions_path: Path) -> list[dict[str, object]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset file for macro accuracy: {dataset_path}")
    dataset = [json.loads(line) for line in dataset_path.open("r", encoding="utf-8") if line.strip()]
    predictions = {row["id"]: row for row in read_jsonl(predictions_path)}
    by_movie: dict[str, list[bool]] = {}
    for item in dataset:
        pred = predictions.get(item["id"])
        by_movie.setdefault(item["movie"], []).append(bool(pred and pred.get("correct") is True))
    if not by_movie:
        raise ValueError("Cannot derive macro movie accuracy from an empty dataset.")
    return [
        {
            "movie": movie,
            "correct": sum(values),
            "total": len(values),
            "accuracy": sum(values) / len(values),
        }
        for movie, values in sorted(by_movie.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/all_movies/eval_all10.jsonl")
    parser.add_argument("--csv", default="figures/context_ladder_data.csv")
    parser.add_argument("--per-movie-csv", default="figures/context_ladder_per_movie_data.csv")
    parser.add_argument("--pdf", default="figures/context_ladder.pdf")
    parser.add_argument("--png", default="figures/context_ladder.png")
    args = parser.parse_args()

    rows = []
    per_movie_rows = []
    for source, metrics_file, predictions_file in SOURCES:
        metrics = read_json(Path(metrics_file))
        per_movie = per_movie_accuracy(Path(args.dataset), Path(predictions_file))
        macro_acc = sum(row["accuracy"] for row in per_movie) / len(per_movie)
        for row in per_movie:
            per_movie_rows.append(
                {
                    "source": source,
                    "movie": row["movie"],
                    "correct": row["correct"],
                    "total": row["total"],
                    "accuracy_percent": row["accuracy"] * 100,
                }
            )
        rows.append(
            {
                "source": source,
                "accuracy_percent": metrics["accuracy"] * 100,
                "macro_movie_accuracy_percent": macro_acc * 100,
                "num_examples": metrics["num_total_examples"],
                "num_correct": metrics["num_correct"],
                "num_invalid_predictions": metrics["num_invalid_predictions"],
                "metrics_file": metrics_file,
                "predictions_file": predictions_file,
            }
        )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    per_movie_csv_path = Path(args.per_movie_csv)
    with per_movie_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_movie_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_movie_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.55), sharey=True, constrained_layout=True)
    metrics_to_plot = [
        ("accuracy_percent", "Accuracy"),
        ("macro_movie_accuracy_percent", "Macro movie acc."),
    ]
    labels = [row["source"] for row in rows]
    colors = [COLORS[label] for label in labels]
    for ax, (key, title) in zip(axes, metrics_to_plot):
        values = [float(row[key]) for row in rows]
        ax.barh(range(len(rows)), values, color=colors, height=0.54)
        ax.set_title(title, fontsize=9, pad=6)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 65)
        ax.grid(axis="x", color="#E7E7E7", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for idx, value in enumerate(values):
            ax.text(value + 1.1, idx, f"{value:.1f}", ha="left", va="center", fontsize=8)
    axes[0].set_xlabel("QA accuracy (%)")
    axes[1].set_xlabel("Unweighted mean over movies (%)")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    Path(args.pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, bbox_inches="tight")
    fig.savefig(args.png, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
