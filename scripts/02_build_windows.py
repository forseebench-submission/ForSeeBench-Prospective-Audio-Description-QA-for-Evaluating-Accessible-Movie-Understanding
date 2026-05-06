#!/usr/bin/env python3
"""Build max-lookahead search blocks from parsed clip rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.io.load_lsmdc import ClipRecord
from forseebench.io.write_jsonl import read_jsonl, write_jsonl
from forseebench.parsing.build_temporal_windows import build_search_blocks
from forseebench.utils.logging import log
from forseebench.utils.resume import file_has_content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_window_clips", type=int, default=3)
    parser.add_argument("--window_size", type=int, default=None, help="Deprecated alias for --max_window_clips")
    parser.add_argument("--max_blocks", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None, help="Deprecated alias for --max_blocks")
    parser.add_argument("--movie", default=None, help="Optional exact movie ID to keep, e.g. 1005_Signs")
    parser.add_argument("--resume", action="store_true", help="Skip this stage if the output file already exists and is non-empty.")
    args = parser.parse_args()

    if args.resume and file_has_content(args.output):
        log(f"resume: skipping window build because output already exists at {args.output}")
        return

    rows = [ClipRecord(**row) for row in read_jsonl(args.input)]
    if args.movie is not None:
        rows = [row for row in rows if row.movie == args.movie]
    log(f"window build input rows={len(rows)} movie={args.movie or 'all'} input={args.input}")
    max_window_clips = args.window_size if args.window_size is not None else args.max_window_clips
    max_blocks = args.max_windows if args.max_windows is not None else args.max_blocks
    blocks = build_search_blocks(rows, max_window_clips=max_window_clips, max_blocks=max_blocks)
    count = write_jsonl(args.output, (block.to_dict() for block in blocks))
    log(
        f"window build output blocks={count} max_window_clips={max_window_clips} "
        f"max_blocks={max_blocks} output={args.output}"
    )


if __name__ == "__main__":
    main()
