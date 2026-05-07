#!/usr/bin/env python3
"""Generate a readable prior-AD source comparison figure.

This figure compares AD sources while holding the QA item and selected prior
context positions fixed. Bars report overall accuracy and movie-balanced
accuracy, where movie-balanced accuracy is the unweighted average of per-movie
accuracies.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SOURCES = [
    (
        "MAD-eval Human AD",
        "outputs/evaluation/ground_truth/all_10_movies_adaptive_source_neutral/metrics.json",
        "outputs/evaluation/ground_truth/all_10_movies_adaptive_source_neutral/predictions.jsonl",
    ),
    (
        "NarrAD",
        "outputs/evaluation/narrad/all_10_movies_adaptive_source_neutral/metrics.json",
        "outputs/evaluation/narrad/all_10_movies_adaptive_source_neutral/predictions.jsonl",
    ),
    (
        "AutoAD-Zero",
        "outputs/evaluation/autoad_zero/all_10_movies_adaptive_source_neutral/metrics.json",
        "outputs/evaluation/autoad_zero/all_10_movies_adaptive_source_neutral/predictions.jsonl",
    ),
]


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def derive_movie_balanced_accuracy(dataset_path: Path, predictions_path: Path) -> tuple[float, list[dict[str, object]]]:
    """Return movie-balanced accuracy and per-movie rows."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset file for movie-balanced accuracy: {dataset_path}")

    dataset = read_jsonl(dataset_path)
    predictions = {row["id"]: row for row in read_jsonl(predictions_path)}

    by_movie: dict[str, list[bool]] = {}
    for item in dataset:
        movie = item.get("movie") or item.get("movie_id") or item.get("movie_name")
        if movie is None:
            raise KeyError("Dataset item is missing a movie/movie_id/movie_name field.")
        pred = predictions.get(item["id"])
        by_movie.setdefault(str(movie), []).append(bool(pred and pred.get("correct") is True))

    per_movie_rows = []
    for movie, values in sorted(by_movie.items()):
        acc = sum(values) / len(values)
        per_movie_rows.append(
            {
                "movie": movie,
                "correct": sum(values),
                "total": len(values),
                "accuracy_percent": acc * 100,
            }
        )

    if not per_movie_rows:
        raise ValueError("Cannot compute movie-balanced accuracy from an empty dataset.")

    movie_balanced = sum(float(row["accuracy_percent"]) for row in per_movie_rows) / len(per_movie_rows)
    return movie_balanced, per_movie_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/all_movies/eval_all10.jsonl")
    parser.add_argument("--csv", default="figures/context_ladder_data.csv")
    parser.add_argument("--per-movie-csv", default="figures/context_ladder_per_movie_data.csv")
    parser.add_argument("--pdf", default="figures/context_ladder.pdf")
    parser.add_argument("--png", default="figures/context_ladder.png")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    per_movie_rows: list[dict[str, object]] = []

    for source, metrics_file, predictions_file in SOURCES:
        metrics = read_json(Path(metrics_file))
        movie_balanced_acc, source_per_movie = derive_movie_balanced_accuracy(
            Path(args.dataset), Path(predictions_file)
        )

        for row in source_per_movie:
            per_movie_rows.append({"source": source, **row})

        rows.append(
            {
                "source": source,
                "overall_accuracy_percent": metrics["accuracy"] * 100,
                "movie_balanced_accuracy_percent": movie_balanced_acc,
                "num_examples": metrics.get("num_total_examples", ""),
                "num_correct": metrics.get("num_correct", ""),
                "num_invalid_predictions": metrics.get("num_invalid_predictions", ""),
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
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels = [str(row["source"]) for row in rows]
    overall = [float(row["overall_accuracy_percent"]) for row in rows]
    balanced = [float(row["movie_balanced_accuracy_percent"]) for row in rows]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    fig.subplots_adjust(top=0.82, bottom=0.24, left=0.10, right=0.98)

    bars1 = ax.bar(x - width / 2, overall, width, label="Overall accuracy", color="#6B8FB8")
    bars2 = ax.bar(x + width / 2, balanced, width, label="Movie-balanced accuracy", color="#D8A24A")

    ax.set_ylabel("QA accuracy (%)")
    ax.set_ylim(0, max(max(overall), max(balanced)) + 14)
    ax.set_xticks(x)
    ax.set_xticklabels(["MAD-eval\nHuman AD", "NarrAD", "AutoAD-Zero"])
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.0,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
            )

    fig.suptitle("Prior AD Source Comparison", fontsize=14, fontweight="bold", y=0.96)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.6,
    )

    Path(args.pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, bbox_inches="tight")
    fig.savefig(args.png, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
