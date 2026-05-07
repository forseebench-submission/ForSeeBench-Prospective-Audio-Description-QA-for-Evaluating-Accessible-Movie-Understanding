#!/usr/bin/env python3
"""Generate the fixed-window context-scaling figure from PrediCC results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "ground_truth": "MAD-eval Human AD",
    "Human AD": "MAD-eval Human AD",
    "MAD-eval Human AD": "MAD-eval Human AD",
    "AutoAD-Zero": "AutoAD-Zero",
    "autoad_zero": "AutoAD-Zero",
    "NarrAD": "NarrAD",
    "narrad": "NarrAD",
}
SOURCE_ORDER = ["MAD-eval Human AD", "NarrAD", "AutoAD-Zero"]
COLORS = {
    "MAD-eval Human AD": "#4E79A7",
    "NarrAD": "#59A14F",
    "AutoAD-Zero": "#E15759",
}
MARKERS = {
    "MAD-eval Human AD": "o",
    "NarrAD": "s",
    "AutoAD-Zero": "^",
}


def load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing PrediCC summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_source(source: str, payload: dict) -> str:
    return payload.get("label") or LABELS.get(source, source)


def rows_from_summary(summary: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source, payload in summary.get("sources", {}).items():
        label = normalize_source(source, payload)
        for row in payload.get("rows", []):
            rows.append(
                {
                    "source": label,
                    "source_key": source,
                    "k": int(row["k"]),
                    "accuracy_percent": float(row["accuracy_percent"]),
                    "predicc_points": float(row["predicc_points"]),
                    "num_examples": int(row["num_examples"]),
                    "num_correct": int(row["num_correct"]),
                    "num_invalid_predictions": int(row.get("num_invalid_predictions", 0)),
                }
            )
    if not rows:
        raise ValueError("PrediCC summary contains no source rows.")
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = {source: idx for idx, source in enumerate(SOURCE_ORDER)}
    fieldnames = [
        "source",
        "source_key",
        "k",
        "accuracy_percent",
        "predicc_points",
        "num_examples",
        "num_correct",
        "num_invalid_predictions",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (order.get(str(r["source"]), 99), int(r["k"]))))


def plot(rows: list[dict[str, object]], pdf_path: Path, png_path: Path) -> None:
    sources = sorted(
        {str(row["source"]) for row in rows},
        key=lambda s: SOURCE_ORDER.index(s) if s in SOURCE_ORDER else 99,
    )
    k_values = sorted({int(row["k"]) for row in rows})

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.4,
            "figure.titlesize": 11.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax_acc = plt.subplots(figsize=(3.8, 3.2))

    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.18, top=0.88)

    fig.suptitle(
        "Fixed-Window Context Contribution",
        fontweight="bold",
        y=0.97,
    )

    for source in sources:
        source_rows = sorted(
            [row for row in rows if row["source"] == source],
            key=lambda r: int(r["k"])
        )

        x = [int(row["k"]) for row in source_rows]
        acc = [float(row["accuracy_percent"]) for row in source_rows]

        color = COLORS.get(source, "#333333")
        marker = MARKERS.get(source, "o")

        ax_acc.plot(
            x,
            acc,
            marker=marker,
            color=color,
            linewidth=2.0,
            markersize=5.0,
            label=source,
        )

        ax_acc.annotate(
            f"{acc[-1]:.1f}%",
            (x[-1], acc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontsize=7.7,
        )

    ax_acc.set_xlabel("Prior AD clips shown (k)")
    ax_acc.set_ylabel("QA accuracy (%)")
    ax_acc.set_xticks(k_values)
    ax_acc.set_ylim(40, 62)

    ax_acc.grid(axis="y", color="#E6E6E6", linewidth=0.8)

    ax_acc.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        handlelength=1.5,
    )

    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)
    ax_acc.set_axisbelow(True)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="outputs/evaluation/predicc/all_10_movies_run_fixed_k0/predicc_summary.json")
    parser.add_argument("--csv", default="figures/context_scaling_data.csv")
    parser.add_argument("--pdf", default="figures/context_scaling.pdf")
    parser.add_argument("--png", default="figures/context_scaling.png")
    args = parser.parse_args()

    summary = load_summary(Path(args.summary))
    rows = rows_from_summary(summary)
    write_csv(rows, Path(args.csv))
    plot(rows, Path(args.pdf), Path(args.png))


if __name__ == "__main__":
    main()
