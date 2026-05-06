#!/usr/bin/env python3
"""Split kept examples by movie without leakage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.generation.build_dataset import make_private_row, make_public_row, split_examples_by_movie
from forseebench.io.write_jsonl import read_jsonl, write_jsonl
from forseebench.utils.logging import log
from forseebench.utils.resume import file_has_content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--challenge_input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--split_by", default="movie")
    parser.add_argument("--movie", default=None, help="Optional exact movie ID to keep, e.g. 1005_Signs")
    parser.add_argument("--resume", action="store_true", help="Skip this stage if all output files already exist and are non-empty.")
    args = parser.parse_args()

    if args.split_by != "movie":
        raise ValueError("ForSeeBench only supports split_by=movie")
    kept = read_jsonl(args.input)
    challenge = read_jsonl(args.challenge_input)
    if args.movie is not None:
        kept = [row for row in kept if row["source"]["movie"] == args.movie]
        challenge = [row for row in challenge if row["source"]["movie"] == args.movie]
    splits = split_examples_by_movie(
        kept,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        expected = [
            out_dir / "train.jsonl",
            out_dir / "val.jsonl",
            out_dir / "test.jsonl",
            out_dir / "challenge_unpredictable.jsonl",
        ]
        if all(file_has_content(path) for path in expected):
            log(f"resume: skipping split because output files already exist in {out_dir}")
            return
    for split_name, rows in splits.items():
        write_jsonl(out_dir / f"{split_name}.jsonl", rows)
        write_jsonl(out_dir / f"{split_name}_public.jsonl", [make_public_row(row, split_name) for row in rows])
        write_jsonl(out_dir / f"{split_name}_private.jsonl", [make_private_row(row, split_name) for row in rows])
        log(f"{split_name}: {len(rows)} examples")
    write_jsonl(out_dir / "challenge_unpredictable.jsonl", challenge)
    write_jsonl(out_dir / "challenge_public.jsonl", [make_public_row(row, "challenge") for row in challenge])
    write_jsonl(out_dir / "challenge_private.jsonl", [make_private_row(row, "challenge") for row in challenge])
    log(f"challenge_unpredictable: {len(challenge)} examples")


if __name__ == "__main__":
    main()
