#!/usr/bin/env python3
"""Evaluate ForSeeBench MCQs using generated AD from Results.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.evaluation.autoad_mcq import evaluate_autoad_mcq
from forseebench.qwen.qwen_client import QwenClient
from forseebench.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/per_movie/1005_Signs/train.jsonl")
    parser.add_argument("--results-csv", default="Results.csv")
    parser.add_argument("--output-dir", default="outputs/evaluation/autoad_zero/1005_Signs")
    parser.add_argument("--qwen-config", default="configs/qwen.yaml")
    parser.add_argument("--ad-column", default="AutoAD-Zero")
    parser.add_argument("--timestamp-tolerance", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run-prompts",
        type=int,
        default=0,
        help="Write this many prompt previews. If --no-qwen is set, do not call Qwen.",
    )
    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument("--no-shuffle-options", action="store_true")
    args = parser.parse_args()

    client = None if args.no_qwen else QwenClient.from_config(load_yaml(args.qwen_config))
    metrics = evaluate_autoad_mcq(
        dataset_path=args.dataset,
        results_csv=args.results_csv,
        output_dir=args.output_dir,
        qwen_client=client,
        ad_column=args.ad_column,
        tolerance=args.timestamp_tolerance,
        limit=args.limit,
        dry_run_prompts=args.dry_run_prompts,
        shuffle_options=not args.no_shuffle_options,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
