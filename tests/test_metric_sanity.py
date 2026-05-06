from __future__ import annotations

from forseebench.evaluation.metrics import compute_mcq_metrics


def test_compute_mcq_metrics_counts_missing_as_incorrect() -> None:
    examples = [
        {"id": "a", "answer_idx": 0},
        {"id": "b", "answer_idx": 1},
        {"id": "c", "answer_idx": 2},
    ]
    predictions = {"a": 0, "b": 2}

    metrics = compute_mcq_metrics(examples, predictions)

    assert metrics["num_examples"] == 3
    assert metrics["num_predictions"] == 2
    assert metrics["num_missing_predictions"] == 1
    assert metrics["num_correct"] == 1
    assert metrics["accuracy"] == 1 / 3


def test_compute_mcq_metrics_accepts_release_answer_rows() -> None:
    examples = [
        {"id": "a", "answer_idx": 0, "answer_text": "A"},
        {"id": "b", "answer_idx": 1, "answer_text": "B"},
    ]
    predictions = {"a": 0, "b": 1}

    metrics = compute_mcq_metrics(examples, predictions)

    assert metrics["num_correct"] == 2
    assert metrics["accuracy"] == 1.0
