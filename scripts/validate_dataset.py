#!/usr/bin/env python3
"""Validate ForSeeBench sample JSONL/JSON/CSV/parquet files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.utils.schema import (
    validate_example,
    validate_private_example,
    validate_public_example,
    validate_release_public_example,
    validate_release_with_answers_example,
)


DEFAULT_INPUT = "hf_dataset/data/qna_with_answers.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object, got {type(payload).__name__}")
            rows.append(payload)
    return rows


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc


def validate_csv(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return 0, [f"{path}: missing CSV header"]
        required = {"movie", "clip_id", "audio_description"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            errors.append(f"{path}: missing required CSV columns: {', '.join(missing)}")
        count = 0
        for count, row in enumerate(reader, start=1):
            for key in required & set(reader.fieldnames):
                if not row.get(key):
                    errors.append(f"{path}: row {count}: empty required column {key}")
                    break
        return count, errors


def infer_schema(row: dict[str, Any]) -> str:
    if "prior_context" in row:
        if "answer_idx" in row or "answer_text" in row:
            return "with_answers"
        return "public"
    if "selection_metadata" in row or "validation_metadata" in row:
        return "internal"
    if "answer_idx" in row or "answer_text" in row:
        return "private"
    return "public"


def validate_rows(rows: list[dict[str, Any]], schema: str) -> list[str]:
    errors: list[str] = []
    validators = {
        "public": validate_release_public_example,
        "legacy_public": validate_public_example,
        "private": validate_private_example,
        "legacy_private": validate_private_example,
        "with_answers": validate_release_with_answers_example,
        "internal": validate_example,
    }
    if schema == "auto":
        inferred = {infer_schema(row) for row in rows}
        if len(inferred) > 1:
            return [f"mixed inferred schemas in one file: {', '.join(sorted(inferred))}"]
    for index, row in enumerate(rows, start=1):
        row_schema = infer_schema(row) if schema == "auto" else schema
        row_errors = validators[row_schema](row)
        errors.extend(f"row {index}: {message}" for message in row_errors)
    return errors


def validate_file(path: Path, schema: str) -> tuple[str, int, list[str]]:
    if not path.exists():
        return schema, 0, [f"{path}: file does not exist"]
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
        if not rows:
            return schema, 0, [f"{path}: no rows found"]
        used_schema = infer_schema(rows[0]) if schema == "auto" else schema
        return used_schema, len(rows), validate_rows(rows, schema)
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
            rows = payload["data"]
        else:
            return schema, 0, [f"{path}: expected list of rows or object with data list"]
        if not all(isinstance(row, dict) for row in rows):
            return schema, 0, [f"{path}: all rows must be objects"]
        used_schema = infer_schema(rows[0]) if rows and schema == "auto" else schema
        return used_schema, len(rows), validate_rows(rows, schema)
    if path.suffix.lower() == ".csv":
        count, errors = validate_csv(path)
        return "csv", count, errors
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            return "parquet", 0, [f"{path}: parquet validation requires pandas/pyarrow; install optional dependencies"]
        frame = pd.read_parquet(path)
        return "parquet", len(frame), []
    return schema, 0, [f"{path}: unsupported file extension {path.suffix!r}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Dataset file to validate. Default: {DEFAULT_INPUT}")
    parser.add_argument(
        "--schema",
        choices=("auto", "public", "with_answers", "private", "legacy_public", "legacy_private", "internal"),
        default="auto",
        help="Expected ForSeeBench JSON schema. Use public for no-answer rows and with_answers for scoring rows.",
    )
    args = parser.parse_args()

    path = Path(args.input)
    try:
        used_schema, count, errors = validate_file(path, args.schema)
    except OSError as exc:
        print(f"MISSING {path}: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"MISSING {exc}", file=sys.stderr)
        return 2

    if errors:
        print(f"MISSING {path}: validation failed with {len(errors)} error(s)")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  - ... {len(errors) - 20} more")
        return 1
    print(f"PASS {path}: {count} rows validated as {used_schema} schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
