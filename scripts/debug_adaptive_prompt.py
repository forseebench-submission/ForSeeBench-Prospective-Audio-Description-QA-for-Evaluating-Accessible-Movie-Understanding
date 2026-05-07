#!/usr/bin/env python3
"""Print representative adaptive-context prompts for each AD source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.evaluation.autoad_mcq import (
    build_generated_ad_index,
    build_ground_truth_index,
    load_generated_ad_rows,
    make_mcq_prompt,
    make_option_order,
    materialize_autoad_context,
    read_jsonl,
    selected_prior_context_clips,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/all_movies/eval_all10.jsonl")
    parser.add_argument("--results-csv", default="Results.csv")
    parser.add_argument("--item-index", type=int, default=0)
    parser.add_argument(
        "--ad-columns",
        default="ground_truth,NarrAD,AutoAD-Zero",
        help="Comma-separated Results.csv columns to print.",
    )
    args = parser.parse_args()

    examples = read_jsonl(args.dataset)
    example = examples[args.item_index]
    selected_clips = selected_prior_context_clips(example)
    option_order = make_option_order(example["id"], len(example["options"]))

    print("DEBUG METADATA")
    print(f"item id: {example['id']}")
    print(f"target sequence index: {(example.get('target') or {}).get('sequence_index')}")
    print(f"selected context clip ids: {[clip.get('clip_id') for clip in selected_clips]}")
    print(f"selected context sequence indices: {[clip.get('sequence_index') for clip in selected_clips]}")
    print(f"hidden target AD: {example.get('target_sentence') or (example.get('target') or {}).get('audio_description')}")
    print(f"gold answer idx: {example.get('answer_idx')}")
    print(f"gold answer: {example.get('answer_text')}")
    print(f"option order: {option_order}")

    for column in [value.strip() for value in args.ad_columns.split(",") if value.strip()]:
        rows = load_generated_ad_rows(args.results_csv, column=column)
        context, unmatched = materialize_autoad_context(
            example,
            build_generated_ad_index(rows),
            text_index=build_ground_truth_index(rows),
        )
        print()
        print("=" * 80)
        print(f"SOURCE COLUMN: {column}")
        print(f"materialized adaptive context: {context}")
        print(f"unmatched selected clips: {unmatched}")
        print("-" * 80)
        print(make_mcq_prompt(example, context, source_name=column, option_order=option_order))


if __name__ == "__main__":
    main()
