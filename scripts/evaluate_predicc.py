#!/usr/bin/env python3
"""Evaluate PrediCC@k for generated audio-description sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.evaluation.autoad_mcq import evaluate_predicc_mcq
from forseebench.qwen.qwen_client import QwenClient
from forseebench.utils.config import load_yaml


def parse_k_values(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/all_movies/eval_all10.jsonl")
    parser.add_argument("--results-csv", default="Results.csv")
    parser.add_argument("--output-dir", default="outputs/evaluation/predicc/all_10_movies_run")
    parser.add_argument("--qwen-config", default="configs/qwen.yaml")
    parser.add_argument("--ad-columns", nargs="+", default=["AutoAD-Zero", "NarrAD"])
    parser.add_argument("--context-lengths", default="0,1,2,4,8")
    parser.add_argument("--parsed-sequences-dir", default="data/interim/per_movie")
    parser.add_argument("--timestamp-tolerance", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run-prompts", type=int, default=0)
    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument("--no-shuffle-options", action="store_true")
    parser.add_argument("--oracle-accuracy", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    client = None if args.no_qwen else QwenClient.from_config(load_yaml(args.qwen_config))
    if client is not None:
        client.max_tokens = args.max_tokens
    summary = evaluate_predicc_mcq(
        dataset_path=args.dataset,
        results_csv=args.results_csv,
        output_dir=args.output_dir,
        qwen_client=client,
        ad_columns=args.ad_columns,
        context_lengths=parse_k_values(args.context_lengths),
        parsed_sequences_dir=args.parsed_sequences_dir,
        tolerance=args.timestamp_tolerance,
        limit=args.limit,
        dry_run_prompts=args.dry_run_prompts,
        shuffle_options=not args.no_shuffle_options,
        oracle_accuracy=args.oracle_accuracy,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
