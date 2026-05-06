#!/usr/bin/env python3
"""Generate candidate benchmark examples from extracted target actions."""

from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.generation.build_dataset import build_candidate_examples
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

    client = QwenClient.from_config(load_yaml(args.config))
    extraction_records = read_jsonl(args.input)
    if args.movie is not None:
        extraction_records = [row for row in extraction_records if row["movie"] == args.movie]
    input_total = len(extraction_records)
    existing_ids = load_existing_ids(args.output) if args.resume else set()
    if existing_ids:
        extraction_records = [row for row in extraction_records if row["id"] not in existing_ids]
    if args.limit is not None:
        extraction_records = extraction_records[: args.limit]
    pending_total = len(extraction_records)
    log(
        f"generate input rows={input_total} pending={pending_total} movie={args.movie or 'all'} "
        f"existing_skipped={len(existing_ids)} limit={args.limit} input={args.input}"
    )
    dataset_config = load_yaml("configs/dataset_generation.yaml")
    progress_output = dataset_config.get(
        "progress_output_generate_examples",
        "data/interim/progress_generate_examples.json",
    )
    if not args.resume:
        reset_file(args.output)

    counts = Counter()
    total = pending_total
    for row in extraction_records:
        try:
            examples = build_candidate_examples([row], qwen_client=client)
        except Exception as exc:
            counts["failed"] += 1
            counts["processed"] += 1
            log(f"generate failed id={row.get('id')} error={type(exc).__name__}: {exc}")
            write_progress(
                progress_output,
                progress_payload(
                    total_rows=total,
                    processed_rows=counts["processed"],
                    completed_rows=min(input_total, len(existing_ids) + counts["processed"]),
                    original_total_rows=input_total,
                    failed_rows=counts["failed"],
                    remaining_rows=max(0, total - counts["processed"]),
                    last_row_id=row.get("id"),
                    output_path=args.output,
                ),
            )
            continue
        if not examples:
            counts["processed"] += 1
            continue
        example = examples[0]
        append_jsonl(args.output, [example])
        counts["processed"] += 1
        counts["written"] += 1
        write_progress(
            progress_output,
            progress_payload(
                total_rows=total,
                processed_rows=counts["processed"],
                completed_rows=min(input_total, len(existing_ids) + counts["processed"]),
                original_total_rows=input_total,
                failed_rows=counts["failed"],
                remaining_rows=max(0, total - counts["processed"]),
                last_row_id=example["id"],
                output_path=args.output,
            ),
        )
        if counts["processed"] % args.log_every == 0 or counts["processed"] == total:
            log(
                f"generate progress processed={counts['processed']}/{total} "
                f"completed={min(input_total, len(existing_ids) + counts['processed'])}/{input_total} "
                f"failed={counts['failed']} "
                f"last_id={example['id']}"
            )
    log(
        f"generate output rows={counts['written']} processed={counts['processed']}/{total} "
        f"completed={min(input_total, len(existing_ids) + counts['processed'])}/{input_total} "
        f"failed={counts['failed']} output={args.output}"
    )


if __name__ == "__main__":
    main()
