#!/usr/bin/env python3
"""Run Qwen target-action extraction over temporal windows."""

from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.generation.extract_actions import extract_target_action
from forseebench.io.write_jsonl import read_jsonl, write_jsonl
from forseebench.qwen.qwen_client import QwenClient
from forseebench.utils.config import load_yaml
from forseebench.utils.logging import log
from forseebench.utils.resume import append_jsonl, load_existing_ids, progress_payload, reset_file, write_progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--movie", default=None, help="Optional exact movie ID to keep, e.g. 1005_Signs")
    parser.add_argument("--resume", action="store_true", help="Append only rows whose ids are not already present in the output JSONL.")
    parser.add_argument("--log_every", type=int, default=10, help="Print progress every N processed rows.")
    args = parser.parse_args()

    windows = read_jsonl(args.input)
    if args.movie is not None:
        windows = [window for window in windows if window["movie"] == args.movie]
    input_total = len(windows)
    existing_ids = load_existing_ids(args.output) if args.resume else set()
    if existing_ids:
        windows = [window for window in windows if window["id"] not in existing_ids]
    if args.limit is not None:
        windows = windows[: args.limit]
    pending_total = len(windows)
    log(
        f"extract input rows={input_total} pending={pending_total} movie={args.movie or 'all'} "
        f"existing_skipped={len(existing_ids)} limit={args.limit} input={args.input}"
    )
    client = QwenClient.from_config(load_yaml(args.config))
    dataset_config = load_yaml("configs/dataset_generation.yaml")
    progress_output = dataset_config.get(
        "progress_output_extract_actions",
        "data/interim/progress_extract_actions.json",
    )
    if not args.resume:
        reset_file(args.output)

    counts = Counter()
    total = pending_total
    for window in windows:
        target_action = extract_target_action(window, client)
        row = window | {
            "expectedness": target_action.get("expectedness"),
            "target_action": target_action,
        }
        if target_action.get("expectedness_warning") is not None:
            row["expectedness_warning"] = target_action["expectedness_warning"]
        append_jsonl(args.output, [row])
        counts["processed"] += 1
        write_progress(
            progress_output,
            progress_payload(
                total_rows=total,
                processed_rows=counts["processed"],
                completed_rows=min(input_total, len(existing_ids) + counts["processed"]),
                original_total_rows=input_total,
                remaining_rows=max(0, total - counts["processed"]),
                last_row_id=row["id"],
                output_path=args.output,
            ),
        )
        if counts["processed"] % args.log_every == 0 or counts["processed"] == total:
            log(
                f"extract progress processed={counts['processed']}/{total} "
                f"completed={min(input_total, len(existing_ids) + counts['processed'])}/{input_total} "
                f"last_id={row['id']}"
            )
    log(
        f"extract output rows={counts['processed']}/{total} "
        f"completed={min(input_total, len(existing_ids) + counts['processed'])}/{input_total} output={args.output}"
    )


if __name__ == "__main__":
    main()
