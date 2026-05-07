#!/usr/bin/env python3
"""Generate the ForSeeBench teaser/task figure from a selected real example."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt


BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
INK = "#1F2933"
MUTED = "#6B7280"


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def extract_frame(video_path: Path, output_path: Path) -> bool:
    if not video_path.exists():
        return False
    try:
        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 2:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
        ok, frame = capture.read()
        capture.release()
        if ok and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image.thumbnail((420, 260))
            image.save(output_path)
            return True
    except Exception:
        pass
    ffmpeg = shutil.which("ffmpeg") or "/thayerfs/apps/tecplot_360_2012_R1/bin/ffmpeg"
    if not Path(ffmpeg).exists():
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        "select=eq(n\\,0),scale=420:-1",
        "-frames:v",
        "1",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        return False
    return output_path.exists()


def make_placeholder(path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (420, 240), "#E5E7EB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, 414, 234), outline="#9CA3AF", width=3)
    draw.text((24, 94), "Frame unavailable", fill="#374151")
    draw.text((24, 124), label[:55], fill="#6B7280")
    image.save(path)


def box(ax, xy, wh, title, edge=BLUE, face="#FFFFFF", title_size=8.6):
    x, y = xy
    w, h = wh
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.3,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(rect)
    ax.text(x + 0.03 * w, y + h - 0.07 * h, title, fontsize=title_size, fontweight="bold", color=INK, va="top")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", default="figures/teaser_example.json")
    parser.add_argument("--pdf", default="figures/teaser_figure.pdf")
    parser.add_argument("--png", default="figures/teaser_figure.png")
    args = parser.parse_args()

    example_path = Path(args.example)
    if not example_path.exists():
        raise FileNotFoundError(f"Missing teaser example JSON: {example_path}")
    example = json.loads(example_path.read_text(encoding="utf-8"))

    for frame in example["frames"]:
        output = Path(frame["frame_path"])
        video = Path(frame["video_clip_path"])
        if not extract_frame(video, output):
            # Replace this fallback by providing a valid local frame path in figures/teaser_example.json.
            make_placeholder(output, frame["label"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.02, 0.60), (0.24, 0.34), "Source clips", edge=BLUE, face="#F8FAFC")
    for idx, frame in enumerate(example["frames"][:3]):
        img = mpimg.imread(frame["frame_path"])
        x0 = 0.04 + idx * 0.067
        x1 = x0 + 0.058
        ax.imshow(img, extent=(x0, x1, 0.735, 0.855), aspect="auto", zorder=2)
        ax.add_patch(patches.Rectangle((x0, 0.735), 0.058, 0.12, fill=False, linewidth=0.8, edgecolor="#111827", zorder=3))
        ax.text(x0, 0.712, frame["label"], fontsize=5.7, color=MUTED)
    ax.text(0.04, 0.645, "Source frames only.\nThe answerer sees text, not video.", fontsize=6.3, color=MUTED)

    box(ax, (0.29, 0.10), (0.38, 0.84), "Prior AD context available to answerer", edge=BLUE, face="#F8FAFC", title_size=8.2)
    prior = example["prior_ad_sentences"]
    highlighted = set(example["highlighted_evidence_sentence_indices"])
    y = 0.855
    for idx, sentence in enumerate(prior, start=1):
        is_hi = idx in highlighted
        if is_hi:
            ax.add_patch(
                patches.FancyBboxPatch(
                    (0.315, y - 0.035),
                    0.325,
                    0.038,
                    boxstyle="round,pad=0.006,rounding_size=0.008",
                    linewidth=0,
                    facecolor="#FFF3D8",
                    zorder=0,
                )
            )
        ax.text(0.315, y, f"{idx:02d}", fontsize=5.9, color=ORANGE if is_hi else MUTED, fontweight="bold" if is_hi else "normal", va="top")
        ax.text(0.350, y, wrap(sentence, 52), fontsize=5.9, color=INK, va="top")
        y -= 0.064 if len(sentence) < 75 else 0.078

    box(ax, (0.70, 0.63), (0.28, 0.31), "Hidden future AD target", edge=RED, face="#FFF5F5")
    ax.text(0.725, 0.81, "withheld during evaluation", fontsize=6.9, color=RED, fontweight="bold")
    ax.text(0.725, 0.735, wrap(example["hidden_future_ad"], 34), fontsize=8.2, color=INK, va="top")

    box(ax, (0.70, 0.10), (0.28, 0.47), "Prospective QA", edge=GREEN, face="#F7FCF7")
    ax.text(0.725, 0.50, wrap(example["question"], 36), fontsize=8.4, fontweight="bold", color=INK, va="top")
    y = 0.425
    for idx, option in enumerate(example["options"]):
        letter = chr(ord("A") + idx)
        is_gold = option == example["gold_answer"]
        if is_gold:
            ax.add_patch(
                patches.FancyBboxPatch(
                    (0.72, y - 0.036),
                    0.228,
                    0.044,
                    boxstyle="round,pad=0.006,rounding_size=0.010",
                    linewidth=0,
                    facecolor="#DCF2D6",
                    zorder=0,
                )
            )
        ax.text(0.73, y, f"{letter}.", fontsize=7.4, color=GREEN if is_gold else INK, fontweight="bold", va="top")
        ax.text(0.765, y, wrap(option, 28), fontsize=7.1, color=INK, va="top")
        if is_gold:
            ax.text(0.948, y, "✓", fontsize=9.0, color=GREEN, fontweight="bold", va="top", ha="right")
        y -= 0.083

    ax.annotate("", xy=(0.70, 0.35), xytext=(0.67, 0.48), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#6B7280"))
    ax.annotate("", xy=(0.70, 0.74), xytext=(0.67, 0.78), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#6B7280", linestyle="--"))
    ax.text(0.02, 0.045, "Task: answer from prior AD only. Future AD and video are hidden from the evaluation answerer.", fontsize=7.4, color=INK)

    fig.tight_layout(pad=0.2)
    Path(args.pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, bbox_inches="tight")
    fig.savefig(args.png, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
