#!/usr/bin/env python3
"""Parse raw LSMDC-like clip descriptions into normalized JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.io.load_lsmdc import load_lsmdc_records
from forseebench.io.video_paths import find_mad_video_path
from forseebench.io.write_jsonl import write_jsonl
from forseebench.utils.logging import log
from forseebench.utils.resume import file_has_content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--movie", default=None, help="Optional exact movie ID to keep, e.g. 1005_Signs")
    parser.add_argument("--video_root", default=None, help="Optional MAD video root; attaches video_path to matching clip rows.")
    parser.add_argument("--resume", action="store_true", help="Skip this stage if the output file already exists and is non-empty.")
    args = parser.parse_args()

    if args.resume and file_has_content(args.output):
        log(f"resume: skipping parse because output already exists at {args.output}")
        return

    records = load_lsmdc_records(args.input)
    if args.movie is not None:
        records = [record for record in records if record.movie == args.movie]
    if args.limit is not None:
        records = records[: args.limit]
    if args.video_root is not None:
        for record in records:
            record.video_path = find_mad_video_path(
                video_root=args.video_root,
                movie=record.movie,
                timestamp_start=record.timestamp_start,
                timestamp_end=record.timestamp_end,
            )
    count = write_jsonl(args.output, (record.to_dict() for record in records))
    log(f"parsed {count} normalized clip rows to {args.output}")


if __name__ == "__main__":
    main()
