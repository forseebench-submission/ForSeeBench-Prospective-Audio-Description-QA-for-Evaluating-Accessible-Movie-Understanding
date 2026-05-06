#!/usr/bin/env python3
"""Export release-facing ForSeeBench Q/A JSONL files from the internal benchmark file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.io.write_jsonl import read_jsonl, write_jsonl
from forseebench.utils.schema import validate_release_public_example, validate_release_with_answers_example


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "data/processed/all_movies/eval_all10.jsonl"
DEFAULT_PUBLIC = "hf_dataset/data/qna_test.jsonl"
DEFAULT_WITH_ANSWERS = "hf_dataset/data/qna_with_answers.jsonl"


def make_prior_context(row: dict[str, Any]) -> list[dict[str, str]]:
    clip_ids = row.get("context_clip_ids", [])
    sentences = row.get("context_sentences", [])
    if not isinstance(clip_ids, list) or not isinstance(sentences, list):
        raise ValueError(f"{row.get('id', '<missing id>')}: context_clip_ids/context_sentences must be lists")
    if len(clip_ids) != len(sentences):
        raise ValueError(f"{row.get('id', '<missing id>')}: context_clip_ids and context_sentences do not align")
    return [{"clip_id": str(clip_id), "text": str(text)} for clip_id, text in zip(clip_ids, sentences)]


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    prior_context = make_prior_context(row)
    output = {
        "id": row["id"],
        "source_id": row["movie"],
        "prior_context": prior_context,
        "question": row["question"],
        "options": list(row["options"]),
        "question_type": row["question_type"],
        "target_type": row["target_type"],
        "context_length": len(prior_context),
    }
    if "expectedness" in row:
        output["expectedness"] = row["expectedness"]
    return output


def evidence_rows(row: dict[str, Any]) -> list[dict[str, str]]:
    evidence = row.get("evidence")
    if not isinstance(evidence, list):
        return []
    clean: list[dict[str, str]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        clip_id = item.get("clip_id")
        span = item.get("span")
        if not clip_id or not span:
            continue
        clean_item = {"clip_id": str(clip_id), "span": str(span)}
        evidence_type = item.get("evidence_type")
        if evidence_type:
            clean_item["evidence_type"] = str(evidence_type)
        clean.append(clean_item)
    return clean


def with_answers_row(row: dict[str, Any]) -> dict[str, Any]:
    output = public_row(row)
    output.update(
        {
            "answer_idx": row["answer_idx"],
            "answer_text": row["answer_text"],
            "hidden_target_ad": row["target_sentence"],
        }
    )
    evidence = evidence_rows(row)
    if evidence:
        output["evidence"] = evidence
    distractors = row.get("distractor_metadata")
    if isinstance(distractors, list):
        output["distractor_metadata"] = [str(item) for item in distractors]
    return output


def validate_or_raise(rows: list[dict[str, Any]], schema: str) -> None:
    validator = validate_release_public_example if schema == "public" else validate_release_with_answers_example
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        row_errors = validator(row)
        errors.extend(f"row {index}: {message}" for message in row_errors)
    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:20])
        raise ValueError(f"{schema} export failed validation with {len(errors)} error(s):\n{preview}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Internal benchmark JSONL. Default: {DEFAULT_INPUT}")
    parser.add_argument("--public-output", default=DEFAULT_PUBLIC, help=f"Public/no-answer output. Default: {DEFAULT_PUBLIC}")
    parser.add_argument(
        "--with-answers-output",
        default=DEFAULT_WITH_ANSWERS,
        help=f"Answer-bearing scoring output. Default: {DEFAULT_WITH_ANSWERS}",
    )
    args = parser.parse_args()

    input_path = ROOT / args.input
    if not input_path.exists():
        print(f"ERROR: input file does not exist: {input_path}", file=sys.stderr)
        return 2
    rows = read_jsonl(input_path)
    public_rows = [public_row(row) for row in rows]
    answer_rows = [with_answers_row(row) for row in rows]
    try:
        validate_or_raise(public_rows, "public")
        validate_or_raise(answer_rows, "with_answers")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    public_count = write_jsonl(ROOT / args.public_output, public_rows)
    answer_count = write_jsonl(ROOT / args.with_answers_output, answer_rows)
    print(f"Wrote {public_count} rows to {args.public_output}")
    print(f"Wrote {answer_count} rows to {args.with_answers_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
