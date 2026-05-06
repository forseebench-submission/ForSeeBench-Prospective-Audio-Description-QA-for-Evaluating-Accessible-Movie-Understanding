"""Evaluation metrics for multiple-choice predictions."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _gold_label(example: dict[str, Any]) -> Any:
    if "answer_idx" in example:
        return example["answer_idx"]
    if "answer" in example:
        return example["answer"]
    raise KeyError("example must contain answer_idx or answer")


def compute_mcq_metrics(examples: list[dict[str, Any]], predictions: dict[str, Any]) -> dict[str, Any]:
    """Compute accuracy and label counts for MCQ predictions."""

    total = len(examples)
    correct = 0
    missing = 0
    gold_counts = Counter()
    pred_counts = Counter()
    for example in examples:
        gold = _gold_label(example)
        pred = predictions.get(example["id"])
        gold_counts[gold] += 1
        if pred is None:
            missing += 1
        else:
            pred_counts[pred] += 1
        if pred == gold:
            correct += 1
    accuracy = correct / total if total else 0.0
    return {
        "num_examples": total,
        "num_predictions": len(predictions),
        "num_missing_predictions": missing,
        "num_correct": correct,
        "accuracy": accuracy,
        "gold_label_distribution": dict(gold_counts),
        "prediction_distribution": dict(pred_counts),
    }
