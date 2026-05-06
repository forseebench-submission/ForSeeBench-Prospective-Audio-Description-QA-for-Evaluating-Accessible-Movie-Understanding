#!/usr/bin/env python3
"""Print compact diagnostics for per-movie ForSeeBench pipeline outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_malformed_json": True})
    return rows


def _counts(movie_dir: Path, names: list[str]) -> dict[str, int]:
    return {name: len(_read_jsonl(movie_dir / name)) for name in names}


def _top(counter: Counter, n: int = 8) -> dict[str, int]:
    return dict(counter.most_common(n))


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter = Counter()
    for row in rows:
        value: Any = row
        for part in key.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if isinstance(value, list):
            for item in value:
                counter[str(item)] += 1
        else:
            counter[str(value)] += 1
    return _top(counter)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie_dir", required=True)
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--stage", default="all")
    args = parser.parse_args()

    movie_dir = Path(args.movie_dir)
    processed_dir = Path(args.processed_dir) if args.processed_dir else None
    print(f"[ForSeeBench] summary stage={args.stage} movie_dir={movie_dir}")

    context_counts = _counts(
        movie_dir,
        ["selected_contexts.jsonl", "rejected_contexts.jsonl", "challenge_contexts.jsonl"],
    )
    print("[ForSeeBench] contexts " + json.dumps(context_counts, sort_keys=True))

    example_counts = _counts(
        movie_dir,
        ["candidate_examples.jsonl", "kept_examples.jsonl", "rejected_examples.jsonl", "challenge_examples.jsonl"],
    )
    print("[ForSeeBench] examples " + json.dumps(example_counts, sort_keys=True))

    candidates = _read_jsonl(movie_dir / "candidate_examples.jsonl")
    rejected = _read_jsonl(movie_dir / "rejected_examples.jsonl")
    if candidates:
        for key in ("target_type", "question_type", "predictability", "continuity_type", "selection_metadata.target_triviality"):
            print(f"[ForSeeBench] candidate_{key} " + json.dumps(_distribution(candidates, key), sort_keys=True))
    if rejected:
        failures: Counter = Counter()
        for row in rejected:
            for reason in row.get("validation", {}).get("failure_reasons", []) or []:
                failures[str(reason)] += 1
        print("[ForSeeBench] rejected_failure_reasons " + json.dumps(_top(failures), sort_keys=True))

    if processed_dir is not None:
        processed_counts = _counts(
            processed_dir,
            ["train.jsonl", "val.jsonl", "test.jsonl", "challenge_unpredictable.jsonl"],
        )
        print("[ForSeeBench] processed " + json.dumps(processed_counts, sort_keys=True))


if __name__ == "__main__":
    main()
