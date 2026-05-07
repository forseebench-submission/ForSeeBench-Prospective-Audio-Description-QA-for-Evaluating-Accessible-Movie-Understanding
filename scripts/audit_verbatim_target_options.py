#!/usr/bin/env python3
"""Audit answer options that exactly match hidden target AD text."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.io.write_jsonl import read_jsonl


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def target_text(example: dict[str, Any]) -> str:
    target = example.get("target") or {}
    return target.get("audio_description") or example.get("target_sentence") or ""


def target_clip_id(example: dict[str, Any]) -> str | None:
    target = example.get("target") or {}
    return example.get("target_clip_id") or target.get("clip_id")


def target_sequence_index(example: dict[str, Any]) -> int | None:
    target = example.get("target") or {}
    value = target.get("sequence_index")
    return int(value) if value is not None else None


def audit_examples(examples: list[dict[str, Any]], *, source_file: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for example in examples:
        hidden_target = target_text(example)
        if not hidden_target:
            continue
        normalized_target = normalize_text(hidden_target)
        for option_idx, option in enumerate(example.get("options", [])):
            if normalize_text(option) != normalized_target:
                continue
            cases.append(
                {
                    "id": example.get("id"),
                    "movie": example.get("movie"),
                    "target_clip_id": target_clip_id(example),
                    "target_sequence_index": target_sequence_index(example),
                    "hidden_target_ad": hidden_target,
                    "question": example.get("question"),
                    "options": example.get("options", []),
                    "matching_option_idx": option_idx,
                    "correct_answer_idx": example.get("answer_idx"),
                    "matching_option_is_correct": option_idx == example.get("answer_idx"),
                    "selected_prior_context": example.get("context", []),
                    "evidence_spans": example.get("evidence", []),
                    "target_type": example.get("target_type"),
                    "reasoning_type": example.get("reasoning_type"),
                    "source_file_path": source_file,
                }
            )
    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "movie",
        "target_clip_id",
        "target_sequence_index",
        "hidden_target_ad",
        "question",
        "options",
        "matching_option_idx",
        "correct_answer_idx",
        "matching_option_is_correct",
        "selected_prior_context",
        "evidence_spans",
        "target_type",
        "reasoning_type",
        "source_file_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for key in ["options", "selected_prior_context", "evidence_spans", "reasoning_type"]:
                csv_row[key] = json.dumps(csv_row.get(key), ensure_ascii=False)
            writer.writerow(csv_row)


def write_readme(path: Path, *, dataset_path: str, num_examples: int, num_cases: int) -> None:
    path.write_text(
        "\n".join(
            [
                "# Verbatim Target Option Audit",
                "",
                "These examples are cases where an answer option exactly matches the hidden target AD. "
                "They are stored for optional manual paraphrasing. They are not removed from the main "
                "benchmark by default because answer options are generated from the hidden target in a "
                "multiple-choice QA setting.",
                "",
                f"- Source benchmark: `{dataset_path}`",
                f"- Source benchmark size: {num_examples}",
                f"- Exact verbatim target-option cases: {num_cases}",
                "",
                "The match criterion is case-insensitive after whitespace normalization.",
                "",
                "Files:",
                "- `verbatim_target_option_cases.jsonl`: one JSON object per matching case.",
                "- `verbatim_target_option_cases.csv`: spreadsheet-friendly copy of the same cases.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/all_movies/eval_all10.jsonl")
    parser.add_argument("--output-dir", default="data/audits/verbatim_target_options")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = read_jsonl(dataset_path)
    cases = audit_examples(examples, source_file=str(dataset_path))
    write_jsonl(output_dir / "verbatim_target_option_cases.jsonl", cases)
    write_csv(output_dir / "verbatim_target_option_cases.csv", cases)
    write_readme(
        output_dir / "README.md",
        dataset_path=str(dataset_path),
        num_examples=len(examples),
        num_cases=len(cases),
    )
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "num_examples": len(examples),
                "num_verbatim_target_option_cases": len(cases),
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
