#!/usr/bin/env python3
"""Evaluate multiple-choice predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.evaluation.evaluate_mcq import evaluate_mcq_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    metrics = evaluate_mcq_file(args.input, args.predictions)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
