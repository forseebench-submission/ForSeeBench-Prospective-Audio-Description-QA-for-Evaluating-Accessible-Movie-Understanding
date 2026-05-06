"""CLI-facing evaluation helpers."""

from __future__ import annotations

from typing import Any

from forseebench.evaluation.metrics import compute_mcq_metrics
from forseebench.io.write_jsonl import read_jsonl


def load_predictions(path: str) -> dict[str, str]:
    """Load prediction rows containing `id` and `prediction`."""

    return {row["id"]: row["prediction"] for row in read_jsonl(path)}


def evaluate_mcq_file(dataset_path: str, predictions_path: str) -> dict[str, Any]:
    """Evaluate predictions against a processed dataset file."""

    examples = read_jsonl(dataset_path)
    predictions = load_predictions(predictions_path)
    return compute_mcq_metrics(examples, predictions)
