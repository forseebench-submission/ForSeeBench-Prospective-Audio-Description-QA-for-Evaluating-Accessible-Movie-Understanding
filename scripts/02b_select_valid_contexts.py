#!/usr/bin/env python3
"""Select one target clip plus supporting buildup from each search block."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.generation.select_context import (
    iterate_target_level_context_decisions,
    route_target_level_decision,
    select_valid_contexts_fixed_candidate_windows,
    target_level_passes_selection,
)
from forseebench.io.write_jsonl import read_jsonl, write_jsonl
from forseebench.qwen.qwen_client import QwenClient
from forseebench.utils.config import load_yaml
from forseebench.utils.logging import log
from forseebench.utils.resume import load_existing_ids


def _append_jsonl(path: str | Path, row: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _reset_output(path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")


def _write_progress(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Search blocks JSONL, typically from 02_build_windows.py")
    parser.add_argument("--qwen_config", required=True)
    parser.add_argument("--context_config", default="configs/context_selection.yaml")
    parser.add_argument("--selected_output", default=None)
    parser.add_argument("--rejected_output", default=None)
    parser.add_argument("--challenge_output", default=None)
    parser.add_argument("--movie", default=None, help="Optional exact movie ID to keep, e.g. 1005_Signs")
    parser.add_argument("--log_every", type=int, default=10, help="Print target-level progress every N targets")
    parser.add_argument("--resume", action="store_true", help="Continue from existing selected/rejected/challenge outputs and progress file.")
    args = parser.parse_args()

    windows = read_jsonl(args.input)
    if args.movie is not None:
        windows = [window for window in windows if window["movie"] == args.movie]
    input_total = len(windows)
    log(f"context selection input blocks={input_total} movie={args.movie or 'all'} input={args.input}")
    context_config = load_yaml(args.context_config)
    qwen_client = QwenClient.from_config(load_yaml(args.qwen_config))
    selected_output = args.selected_output or context_config["output_selected"]
    rejected_output = args.rejected_output or context_config["output_rejected"]
    challenge_output = args.challenge_output or context_config["output_challenge"]
    progress_output = context_config["progress_output"]

    selection_mode = context_config.get("selection_mode", "target_level_qwen_minimal_evidence")
    if selection_mode == "fixed_candidate_windows":
        results = select_valid_contexts_fixed_candidate_windows(
            windows,
            qwen_client=qwen_client,
            config=context_config,
            progress_callback=None,
        )
        write_jsonl(selected_output, results["selected"])
        write_jsonl(rejected_output, results["rejected"])
        write_jsonl(challenge_output, results["challenge"])
        log(
            "selected contexts: "
            f"selected={len(results['selected'])} rejected={len(results['rejected'])} challenge={len(results['challenge'])} "
            f"stats={results['stats']}"
        )
        return

    if selection_mode != "target_level_qwen_minimal_evidence":
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    if context_config.get("stream_outputs", True) and not args.resume:
        _reset_output(selected_output)
        _reset_output(rejected_output)
        _reset_output(challenge_output)

    counts = Counter()
    if args.resume:
        counts["selected_targets"] = len(load_existing_ids(selected_output))
        counts["rejected_targets"] = len(load_existing_ids(rejected_output))
        counts["challenge_targets"] = len(load_existing_ids(challenge_output))
        done_ids = (
            load_existing_ids(selected_output)
            | load_existing_ids(rejected_output)
            | load_existing_ids(challenge_output)
        )
    else:
        done_ids = set()
    skipped = 0
    pending_total = max(0, input_total - len(done_ids))
    last_target_id = None
    for decision in iterate_target_level_context_decisions(windows, qwen_client=qwen_client, config=context_config):
        if decision["id"] in done_ids:
            skipped += 1
            continue
        last_target_id = decision["id"]
        counts["processed_targets"] += 1
        status = route_target_level_decision(decision)
        if status == "selected":
            _append_jsonl(selected_output, decision)
            counts["selected_targets"] += 1
        elif status == "challenge":
            _append_jsonl(challenge_output, decision)
            counts["challenge_targets"] += 1
        else:
            _append_jsonl(rejected_output, decision)
            counts["rejected_targets"] += 1

        progress_payload = {
            "total_targets": input_total,
            "pending_targets": pending_total,
            "processed_targets": counts["processed_targets"],
            "completed_targets": skipped + counts["processed_targets"],
            "remaining_targets": max(0, pending_total - counts["processed_targets"]),
            "selected_targets": counts["selected_targets"],
            "rejected_targets": counts["rejected_targets"],
            "challenge_targets": counts["challenge_targets"],
            "last_target_id": last_target_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_progress(progress_output, progress_payload)

        if counts["processed_targets"] % args.log_every == 0:
            log(
                f"context progress processed={counts['processed_targets']}/{pending_total} "
                f"completed={skipped + counts['processed_targets']}/{input_total} "
                f"selected={counts['selected_targets']} "
                f"rejected={counts['rejected_targets']} challenge={counts['challenge_targets']} "
                f"qwen_calls={decision.get('qwen_call_index', 0)} status={status} "
                f"target={decision.get('target_clip_id')} selected_w={decision.get('selected_window_size')}"
            )
        if status == "rejected":
            log(
                f"target rejected movie={decision['movie']} target={decision.get('target_clip_id')} "
                f"reason={decision.get('rejection_reason')}"
            )

    log(
        "selected contexts: "
        f"processed={counts['processed_targets']}/{pending_total} completed={skipped + counts['processed_targets']}/{input_total} skipped={skipped} "
        f"selected={counts['selected_targets']} rejected={counts['rejected_targets']} challenge={counts['challenge_targets']} "
        f"selected_output={selected_output} rejected_output={rejected_output} challenge_output={challenge_output}"
    )


if __name__ == "__main__":
    main()
