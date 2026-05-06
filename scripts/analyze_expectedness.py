#!/usr/bin/env python3
"""Summarize expectedness values in a JSONL file of benchmark-style examples."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forseebench.io.write_jsonl import read_jsonl


def summarize_expectedness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate expectedness statistics for benchmark examples."""

    numeric_values = [
        float(row["expectedness"])
        for row in rows
        if row.get("expectedness") is not None and isinstance(row.get("expectedness"), (int, float)) and not isinstance(row.get("expectedness"), bool)
    ]
    null_count = sum(1 for row in rows if row.get("expectedness") is None)
    breakdown: dict[str, dict[str, Any]] = {}

    grouped: dict[str, list[float]] = {}
    grouped_nulls: Counter[str] = Counter()
    grouped_counts: Counter[str] = Counter()
    for row in rows:
        reasoning_types = row.get("reasoning_type") or ["unknown"]
        for label in reasoning_types:
            grouped_counts[label] += 1
            value = row.get("expectedness")
            if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
                grouped_nulls[label] += 1
            else:
                grouped.setdefault(label, []).append(float(value))
    for label, count in grouped_counts.items():
        values = grouped.get(label, [])
        breakdown[label] = {
            "count": count,
            "null": grouped_nulls[label],
            "mean": statistics.mean(values) if values else None,
        }

    return {
        "count": len(rows),
        "mean_expectedness": statistics.mean(numeric_values) if numeric_values else None,
        "median_expectedness": statistics.median(numeric_values) if numeric_values else None,
        "min_expectedness": min(numeric_values) if numeric_values else None,
        "max_expectedness": max(numeric_values) if numeric_values else None,
        "above_0_8": sum(1 for value in numeric_values if value > 0.8),
        "between_0_4_and_0_8": sum(1 for value in numeric_values if 0.4 <= value <= 0.8),
        "below_0_4": sum(1 for value in numeric_values if value < 0.4),
        "null_count": null_count,
        "by_reasoning_type": breakdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSONL file of benchmark-style examples")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    summary = summarize_expectedness(rows)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
