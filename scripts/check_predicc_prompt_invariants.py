#!/usr/bin/env python3
"""Lightweight invariant checks for fixed-window PrediCC prompts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.evaluation.autoad_mcq import fixed_prior_context, make_mcq_prompt, make_option_order
from forseebench.evaluation.autoad_mcq import load_sequence_clip_index
from forseebench.io.write_jsonl import read_jsonl


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/all_movies/eval_all10.jsonl")
    parser.add_argument("--parsed-sequences-dir", default="data/interim/per_movie")
    args = parser.parse_args()

    examples = read_jsonl(args.dataset)
    clip_index = load_sequence_clip_index(args.parsed_sequences_dir)
    counts: dict[str, int] = {
        "k0_prompt_contains_source_name": 0,
        "k0_prompt_contains_real_id": 0,
        "k0_prompt_contains_prior_context_header": 0,
        "k0_prompt_differs_across_sources": 0,
        "prompt_contains_dummy_prediction_idx_zero": 0,
        "context_includes_target_or_later": 0,
        "k1_not_immediate_prior": 0,
        "k_not_last_k_prior": 0,
    }
    sample_failures: dict[str, list[str]] = {key: [] for key in counts}

    def record(name: str, example_id: str) -> None:
        counts[name] += 1
        if len(sample_failures[name]) < 5:
            sample_failures[name].append(example_id)

    for example in examples:
        option_order = make_option_order(example["id"], len(example["options"]))
        no_context_prompts = [
            make_mcq_prompt(example, [], source_name=source_name, option_order=option_order)
            for source_name in ["ground_truth", "NarrAD", "AutoAD-Zero"]
        ]
        k0_prompt = no_context_prompts[0]
        if len(set(no_context_prompts)) != 1:
            record("k0_prompt_differs_across_sources", example["id"])
        for source_name in ["ground_truth", "Human AD", "NarrAD", "AutoAD-Zero"]:
            if source_name in k0_prompt:
                record("k0_prompt_contains_source_name", example["id"])
                break
        if example["id"] in k0_prompt:
            record("k0_prompt_contains_real_id", example["id"])
        if "Prior" in k0_prompt or "audio descriptions" in k0_prompt:
            record("k0_prompt_contains_prior_context_header", example["id"])
        if '"prediction_idx": 0' in k0_prompt:
            record("prompt_contains_dummy_prediction_idx_zero", example["id"])

        target = example.get("target") or {}
        target_sequence_index = int(target["sequence_index"])
        prior = [
            clip
            for clip in clip_index[example["movie"]]
            if int(clip["sequence_index"]) < target_sequence_index
        ]
        for k in [1, 2, 4, 8]:
            clips = fixed_prior_context(example, clip_index, k=k)
            seqs = [int(clip["sequence_index"]) for clip in clips]
            if any(seq >= target_sequence_index for seq in seqs):
                record("context_includes_target_or_later", example["id"])
            if k == 1:
                expected = [int(prior[-1]["sequence_index"])] if prior else []
                if seqs != expected:
                    record("k1_not_immediate_prior", example["id"])
            else:
                expected = [int(clip["sequence_index"]) for clip in prior[-k:]]
                if seqs != expected:
                    record("k_not_last_k_prior", example["id"])

    result: dict[str, Any] = {
        "dataset": args.dataset,
        "num_examples": len(examples),
        "counts": counts,
        "sample_failures": sample_failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if any(counts.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
