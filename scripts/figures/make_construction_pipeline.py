#!/usr/bin/env python3
"""Generate the ForSeeBench construction pipeline figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt


BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#F4F6F8"
TEXT = "#1F2933"


def box(ax, xy, wh, title, lines, edge=BLUE, face=GRAY):
    x, y = xy
    w, h = wh
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(rect)
    ax.text(x + 0.04 * w, y + h - 0.14 * h, title, fontsize=8.8, fontweight="bold", color=TEXT, va="top")
    for idx, line in enumerate(lines):
        ax.text(x + 0.06 * w, y + h - (0.36 + idx * 0.16) * h, line, fontsize=7.4, color=TEXT, va="top")


def arrow(ax, start, end, color="#6B7280"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color=color, shrinkA=4, shrinkB=4),
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(
        ax,
        (0.02, 0.58),
        (0.18, 0.32),
        "MAD-eval input",
        ["movie clips", "human AD", "timestamps"],
        edge=BLUE,
    )
    box(
        ax,
        (0.255, 0.58),
        (0.19, 0.32),
        "Ordered blocks",
        ["10-clip search regions", "oldest to newest", "target not fixed"],
        edge=BLUE,
    )
    box(
        ax,
        (0.505, 0.58),
        (0.20, 0.32),
        "Qwen curation",
        ["select target", "choose prior evidence", "label target type"],
        edge=GREEN,
    )
    box(
        ax,
        (0.765, 0.58),
        (0.215, 0.32),
        "QA item",
        ["hidden AD target", "question + 4 options", "typed distractors"],
        edge=GREEN,
    )

    box(
        ax,
        (0.25, 0.12),
        (0.30, 0.28),
        "Validity gates",
        ["nontrivial target", "strictly prior context", "exact evidence spans"],
        edge=ORANGE,
        face="#FFF7ED",
    )
    box(
        ax,
        (0.64, 0.12),
        (0.30, 0.28),
        "Audit metadata",
        ["target / reasoning type", "validation output", "leakage-risk checks"],
        edge=RED,
        face="#FFF5F5",
    )

    arrow(ax, (0.20, 0.74), (0.255, 0.74))
    arrow(ax, (0.445, 0.74), (0.505, 0.74))
    arrow(ax, (0.705, 0.74), (0.765, 0.74))
    arrow(ax, (0.62, 0.58), (0.45, 0.40))
    arrow(ax, (0.55, 0.26), (0.64, 0.26))
    arrow(ax, (0.79, 0.40), (0.86, 0.58))

    ax.text(0.02, 0.07, "Implemented pipeline: parse clips -> build search blocks -> select target/context -> generate QA -> validate/filter -> export metadata.", fontsize=7.2, color="#4B5563")

    fig.tight_layout(pad=0.2)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/construction_pipeline.pdf", bbox_inches="tight")
    fig.savefig("figures/construction_pipeline.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
