from __future__ import annotations

import csv
import json
from pathlib import Path

from forseebench.evaluation.autoad_mcq import (
    build_generated_ad_index,
    build_ground_truth_index,
    compute_autoad_metrics,
    find_generated_ad,
    fixed_prior_context,
    load_generated_ad_rows,
    make_option_order,
    materialize_autoad_context,
    summarize_predicc,
    normalize_prediction,
)


def test_load_and_match_generated_ad_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "Results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["start", "end", "AutoAD-Zero", "ground_truth", "NarrAD", "movie", "vid"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "start": "144.104",
                "end": "148.433",
                "AutoAD-Zero": "He looks around the room.",
                "ground_truth": "His eyes dart around the room.",
                "NarrAD": "",
                "movie": "SIGNS",
                "vid": "tt0286106",
            }
        )

    index = build_generated_ad_index(load_generated_ad_rows(csv_path, column="AutoAD-Zero"))
    match = find_generated_ad(index, movie="1005_Signs", start=144.104, end=148.433)

    assert match is not None
    assert match.text == "He looks around the room."


def test_materialize_autoad_context_reports_unmatched() -> None:
    index = build_generated_ad_index([])
    example = {
        "id": "ex1",
        "context": [
            {
                "clip_id": "1005_Signs__1p0_2p0",
                "movie": "1005_Signs",
                "timestamp_start": "1.0",
                "timestamp_end": "2.0",
            }
        ],
    }

    context, unmatched = materialize_autoad_context(example, index)

    assert context == []
    assert unmatched[0]["clip_id"] == "1005_Signs__1p0_2p0"


def test_materialize_autoad_context_falls_back_to_ground_truth_text(tmp_path: Path) -> None:
    csv_path = tmp_path / "Results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["start", "end", "AutoAD-Zero", "ground_truth", "NarrAD", "movie", "vid"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "start": "101.901",
                "end": "102.796",
                "AutoAD-Zero": "The moon is covered by clouds.",
                "ground_truth": "Dark clouds cover half of the bright moon.",
                "NarrAD": "",
                "movie": "LEGION",
                "vid": "tt1038686",
            }
        )
    rows = load_generated_ad_rows(csv_path, column="AutoAD-Zero")
    example = {
        "id": "ex1",
        "context": [
            {
                "clip_id": "1026_Legion__102p101_102p996",
                "movie": "1026_Legion",
                "timestamp_start": "102.101",
                "timestamp_end": "102.996",
                "audio_description": "Dark clouds cover half of the bright moon.",
            }
        ],
    }

    context, unmatched = materialize_autoad_context(
        example,
        build_generated_ad_index(rows),
        text_index=build_ground_truth_index(rows),
    )

    assert unmatched == []
    assert context == ["The moon is covered by clouds."]


def test_compute_autoad_metrics_uses_answer_idx() -> None:
    examples = [
        {"id": "a", "answer_idx": 0},
        {"id": "b", "answer_idx": 2},
    ]
    predictions = [
        {"id": "a", "prediction_idx": 0},
        {"id": "b", "prediction_idx": 1},
        {"id": "bad", "prediction_idx": None},
    ]

    metrics = compute_autoad_metrics(examples, predictions)

    assert metrics["num_examples"] == 2
    assert metrics["num_correct"] == 1
    assert metrics["accuracy"] == 0.5


def test_shuffled_prediction_maps_back_to_original_option() -> None:
    options = ["correct", "wrong a", "wrong b", "wrong c"]
    option_order = [2, 0, 3, 1]

    original_idx, original_text, displayed_idx, displayed_text = normalize_prediction(
        {"prediction_idx": 1},
        options,
        option_order=option_order,
    )

    assert displayed_idx == 1
    assert displayed_text == "correct"
    assert original_idx == 0
    assert original_text == "correct"


def test_option_order_is_stable_and_nontrivial() -> None:
    first = make_option_order("example-id", 4)
    second = make_option_order("example-id", 4)

    assert first == second
    assert sorted(first) == [0, 1, 2, 3]


def test_fixed_prior_context_uses_last_k_clips_before_target() -> None:
    clips = [
        {"movie": "M", "clip_id": f"c{i}", "sequence_index": i}
        for i in range(5)
    ]
    example = {
        "id": "ex1",
        "movie": "M",
        "target_clip_id": "c4",
        "target": {"clip_id": "c4", "sequence_index": 4},
    }

    context = fixed_prior_context(example, {"M": clips}, k=2)

    assert [clip["clip_id"] for clip in context] == ["c2", "c3"]


def test_summarize_predicc_computes_context_gain() -> None:
    summary = summarize_predicc(
        {
            "NarrAD": {
                0: {"accuracy": 0.25, "num_correct": 1, "num_examples": 4},
                1: {"accuracy": 0.50, "num_correct": 2, "num_examples": 4},
                2: {"accuracy": 0.75, "num_correct": 3, "num_examples": 4},
            }
        }
    )

    rows = summary["sources"]["NarrAD"]["rows"]
    assert rows[0]["predicc"] == 0.0
    assert rows[1]["predicc"] == 0.25
    assert rows[2]["predicc"] == 0.5
