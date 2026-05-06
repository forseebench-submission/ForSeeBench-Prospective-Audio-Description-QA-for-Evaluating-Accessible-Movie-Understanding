#!/usr/bin/env python3
"""Validate candidate examples and split into kept, rejected, and challenge sets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.generation.build_dataset import candidate_passes_main_gate, candidate_routes_to_challenge
from forseebench.generation.normalization import (
    normalize_bool,
    normalize_candidate_example,
    normalize_float,
    normalize_validation_payload,
)
from forseebench.generation.validate_examples import validate_candidate
from forseebench.io.write_jsonl import read_jsonl, write_jsonl
from forseebench.qwen.qwen_client import QwenClient
from forseebench.utils.config import load_yaml
from forseebench.utils.logging import log
from forseebench.utils.resume import append_jsonl, load_existing_ids, progress_payload, reset_file, write_progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--kept", required=True)
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_config", default="configs/dataset_generation.yaml")
    parser.add_argument("--movie", default=None, help="Optional exact movie ID to keep, e.g. 1005_Signs")
    parser.add_argument("--resume", action="store_true", help="Append only rows whose ids are not already present in the output JSONLs.")
    parser.add_argument("--log_every", type=int, default=10, help="Print progress every N processed rows.")
    args = parser.parse_args()

    candidates = read_jsonl(args.input)
    if args.movie is not None:
        candidates = [row for row in candidates if row["source"]["movie"] == args.movie]
    input_total = len(candidates)
    if args.resume:
        done_ids = load_existing_ids(args.kept) | load_existing_ids(args.rejected) | load_existing_ids(args.challenge)
        if done_ids:
            candidates = [row for row in candidates if row["id"] not in done_ids]
    else:
        done_ids = set()
    pending_total = len(candidates)
    log(
        f"validate input rows={input_total} pending={pending_total} movie={args.movie or 'all'} "
        f"existing_skipped={len(done_ids)} input={args.input}"
    )
    qwen_config = load_yaml(args.config)
    dataset_config = load_yaml(args.dataset_config)
    client = QwenClient.from_config(qwen_config)
    if not args.resume:
        reset_file(args.kept)
        reset_file(args.rejected)
        reset_file(args.challenge)
    thresholds = dataset_config["validation_thresholds"]
    progress_output = dataset_config.get(
        "progress_output_validate_filter",
        "data/interim/progress_validate_filter.json",
    )
    counts = Counter()
    total = pending_total
    for example in candidates:
        normalize_candidate_example(example)
        validation = normalize_validation_payload(
            validate_candidate(example, client),
            existing_quality=example.get("quality", {}),
        )
        example["quality"].update(
            {
                "qwen_confidence": normalize_float(validation.get("qwen_confidence"), 0.0),
                "evidence_sufficiency": normalize_float(validation.get("evidence_sufficiency"), 0.0),
                "distractor_quality": normalize_float(validation.get("distractor_quality"), 0.0),
                "should_keep": normalize_bool(validation.get("should_keep"), False),
            }
        )
        example["validation"] = validation
        quality_pass = (
            example["quality"]["qwen_confidence"] >= thresholds["qwen_confidence"]
            and example["quality"]["evidence_sufficiency"] >= thresholds["evidence_sufficiency"]
            and example["quality"]["distractor_quality"] >= thresholds["distractor_quality"]
        )
        expectedness_filter_enabled = dataset_config.get("filter_by_expectedness", False)
        expectedness_pass = True
        if expectedness_filter_enabled:
            expectedness = example.get("expectedness")
            expectedness_pass = (
                expectedness is not None
                and dataset_config["expectedness_main_min"] <= float(expectedness) <= dataset_config["expectedness_main_max"]
            )
        main_benchmark_pass = candidate_passes_main_gate(
            example,
            quality_pass=quality_pass,
            expectedness_pass=expectedness_pass,
        )
        if main_benchmark_pass and counts["kept"] < dataset_config["pilot"]["max_examples"]:
            append_jsonl(args.kept, [example])
            counts["kept"] += 1
            status = "kept"
        elif candidate_routes_to_challenge(example):
            append_jsonl(args.challenge, [example])
            counts["challenge"] += 1
            status = "challenge"
        else:
            append_jsonl(args.rejected, [example])
            counts["rejected"] += 1
            status = "rejected"
        counts["processed"] += 1
        write_progress(
            progress_output,
            progress_payload(
                total_rows=total,
                processed_rows=counts["processed"],
                completed_rows=min(input_total, len(done_ids) + counts["processed"]),
                original_total_rows=input_total,
                kept_rows=counts["kept"],
                rejected_rows=counts["rejected"],
                challenge_rows=counts["challenge"],
                remaining_rows=max(0, total - counts["processed"]),
                last_row_id=example["id"],
            ),
        )
        if counts["processed"] % args.log_every == 0 or counts["processed"] == total:
            log(
                f"validate progress processed={counts['processed']}/{total} "
                f"completed={min(input_total, len(done_ids) + counts['processed'])}/{input_total} "
                f"kept={counts['kept']} "
                f"rejected={counts['rejected']} challenge={counts['challenge']} "
                f"status={status} last_id={example['id']}"
            )
    log(
        "validation stats: "
        + json.dumps(
            {
                "processed": counts["processed"],
                "pending_total": total,
                "completed": min(input_total, len(done_ids) + counts["processed"]),
                "input_total": input_total,
                "kept": counts["kept"],
                "rejected": counts["rejected"],
                "challenge": counts["challenge"],
                "kept_output": args.kept,
                "rejected_output": args.rejected,
                "challenge_output": args.challenge,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
