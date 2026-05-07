#!/usr/bin/env python3
"""Extract frame contact sheets for selected real teaser candidates."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


CANDIDATE_IDS = [
    "lsmdc::1005_Signs::block::00296_00305::w10::1005_Signs__2293p421_2299p864",
    "lsmdc::1005_Signs::block::00166_00175::w10::1005_Signs__1362p42_1367p91",
    "lsmdc::3015_CHARLIE_ST_CLOUD::block::00269_00278::w10::3015_CHARLIE_ST_CLOUD__2936p934_2941p36",
    "lsmdc::3015_CHARLIE_ST_CLOUD::block::00208_00217::w10::3015_CHARLIE_ST_CLOUD__2336p549_2338p866",
    "lsmdc::3074_THE_ROOMMATE::block::00535_00544::w10::3074_THE_ROOMMATE__5019p608_5022p029",
    "lsmdc::3032_HOW_DO_YOU_KNOW::block::00070_00079::w10::3032_HOW_DO_YOU_KNOW__1082p723_1091p655",
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def extract_mid_frame(video_path: Path, out_path: Path) -> bool:
    if not video_path.exists():
        return False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frame = max(frame_count // 2, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, frame)
    return True


def short_label(text: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(text, width=width, max_lines=2, placeholder="..."))


def make_sheet(example: dict, out_dir: Path) -> Path:
    clips = list(example.get("context", [])) + [example.get("target", {})]
    labels = [f"prior {idx + 1}" for idx in range(len(example.get("context", [])))] + ["hidden target"]
    frames = []
    for idx, clip in enumerate(clips):
        video_path = Path(clip.get("video_path", ""))
        frame_path = out_dir / f"{example['movie']}__{example['target_clip_id']}__{idx}.png"
        ok = extract_mid_frame(video_path, frame_path)
        frames.append((frame_path if ok else None, labels[idx], clip.get("audio_description", "")))

    fig, axes = plt.subplots(1, len(frames), figsize=(2.05 * len(frames), 2.25))
    if len(frames) == 1:
        axes = [axes]
    for ax, (frame_path, label, ad_text) in zip(axes, frames):
        ax.set_axis_off()
        if frame_path is not None:
            ax.imshow(plt.imread(frame_path))
        else:
            ax.text(0.5, 0.55, "missing\nframe", ha="center", va="center", fontsize=8)
        ax.set_title(label, fontsize=8, pad=3)
        ax.text(0.0, -0.08, short_label(ad_text), transform=ax.transAxes, fontsize=6.2, va="top")
    fig.suptitle(f"{example['movie']} | {example['target_type']} | {example['target_clip_id']}", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    sheet_path = out_dir / f"{example['movie']}__{example['target_clip_id']}__sheet.png"
    fig.savefig(sheet_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return sheet_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default="data/interim/kept_examples_all_movies.jsonl")
    parser.add_argument("--out-dir", default="figures/teaser_candidates")
    args = parser.parse_args()

    examples = {row["id"]: row for row in read_jsonl(Path(args.examples))}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for candidate_id in CANDIDATE_IDS:
        if candidate_id not in examples:
            raise KeyError(f"Candidate not found in examples file: {candidate_id}")
        sheet = make_sheet(examples[candidate_id], out_dir)
        print(sheet)


if __name__ == "__main__":
    main()
