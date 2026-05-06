#!/usr/bin/env python3
"""Export a minimal dataset card from processed splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.io.write_jsonl import read_jsonl
from forseebench.utils.resume import file_has_content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true", help="Skip this stage if the output file already exists and is non-empty.")
    args = parser.parse_args()

    if args.resume and file_has_content(args.output):
        return

    train = read_jsonl(args.train)
    val = read_jsonl(args.val)
    test = read_jsonl(args.test)
    challenge = read_jsonl(args.challenge)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                "# ForSeeBench Dataset Card",
                "",
                "## Splits",
                "",
                f"- Train: {len(train)}",
                f"- Val: {len(val)}",
                f"- Test: {len(test)}",
                f"- Challenge: {len(challenge)}",
                "",
                "## Task",
                "",
                "Predict the actual next action from temporally ordered LSMDC context.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
