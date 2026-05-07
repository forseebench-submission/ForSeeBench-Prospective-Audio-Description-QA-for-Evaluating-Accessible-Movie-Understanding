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
    make_mcq_prompt,
    make_option_order,
    materialize_autoad_context,
    summarize_predicc,
    normalize_prediction,
    selected_prior_context_clips,
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


def test_selected_prior_context_clips_drops_target_and_later_clips() -> None:
    example = {
        "id": "ex1",
        "target": {"clip_id": "c3", "sequence_index": 3},
        "context": [
            {"clip_id": "c1", "sequence_index": 1},
            {"clip_id": "c3", "sequence_index": 3},
            {"clip_id": "c4", "sequence_index": 4},
        ],
    }

    context = selected_prior_context_clips(example)

    assert [clip["clip_id"] for clip in context] == ["c1"]


def test_materialize_autoad_context_uses_only_selected_prior_clips(tmp_path: Path) -> None:
    csv_path = tmp_path / "Results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["start", "end", "AutoAD-Zero", "ground_truth", "NarrAD", "movie", "vid"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "start": "1.0",
                "end": "2.0",
                "AutoAD-Zero": "Prior source text.",
                "ground_truth": "Prior human AD.",
                "NarrAD": "",
                "movie": "MOVIE",
                "vid": "vid",
            }
        )
        writer.writerow(
            {
                "start": "3.0",
                "end": "4.0",
                "AutoAD-Zero": "Target or later source text.",
                "ground_truth": "Target or later human AD.",
                "NarrAD": "",
                "movie": "MOVIE",
                "vid": "vid",
            }
        )
    rows = load_generated_ad_rows(csv_path, column="AutoAD-Zero")
    example = {
        "id": "ex1",
        "target": {"clip_id": "c3", "sequence_index": 3},
        "context": [
            {
                "clip_id": "c1",
                "movie": "Movie",
                "sequence_index": 1,
                "timestamp_start": "1.0",
                "timestamp_end": "2.0",
            },
            {
                "clip_id": "c3",
                "movie": "Movie",
                "sequence_index": 3,
                "timestamp_start": "3.0",
                "timestamp_end": "4.0",
            },
        ],
    }

    context, unmatched = materialize_autoad_context(
        example,
        build_generated_ad_index(rows),
    )

    assert unmatched == []
    assert context == ["Prior source text."]


def test_adaptive_context_uses_selected_context_not_last_k_context(tmp_path: Path) -> None:
    csv_path = tmp_path / "Results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["start", "end", "AutoAD-Zero", "ground_truth", "NarrAD", "movie", "vid"],
        )
        writer.writeheader()
        for idx in range(1, 5):
            writer.writerow(
                {
                    "start": f"{idx}.0",
                    "end": f"{idx}.5",
                    "AutoAD-Zero": f"source c{idx}",
                    "ground_truth": f"human c{idx}",
                    "NarrAD": f"narrad c{idx}",
                    "movie": "MOVIE",
                    "vid": "vid",
                }
            )
    rows = load_generated_ad_rows(csv_path, column="AutoAD-Zero")
    example = {
        "id": "ex1",
        "movie": "Movie",
        "target": {"clip_id": "c5", "sequence_index": 5},
        "context": [
            {
                "clip_id": "c1",
                "movie": "Movie",
                "sequence_index": 1,
                "timestamp_start": "1.0",
                "timestamp_end": "1.5",
            },
            {
                "clip_id": "c3",
                "movie": "Movie",
                "sequence_index": 3,
                "timestamp_start": "3.0",
                "timestamp_end": "3.5",
            },
        ],
    }

    adaptive_context, unmatched = materialize_autoad_context(
        example,
        build_generated_ad_index(rows),
    )
    fixed_window_context = fixed_prior_context(
        example,
        {
            "Movie": [
                {"movie": "Movie", "clip_id": f"c{idx}", "sequence_index": idx}
                for idx in range(1, 6)
            ]
        },
        k=2,
    )

    assert unmatched == []
    assert adaptive_context == ["source c1", "source c3"]
    assert [clip["clip_id"] for clip in fixed_window_context] == ["c3", "c4"]


def test_selected_adaptive_positions_are_identical_across_sources_and_only_text_differs(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "Results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["start", "end", "AutoAD-Zero", "ground_truth", "NarrAD", "movie", "vid"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "start": "1.0",
                "end": "2.0",
                "AutoAD-Zero": "AutoAD text at c1.",
                "ground_truth": "Human text at c1.",
                "NarrAD": "NarrAD text at c1.",
                "movie": "MOVIE",
                "vid": "vid",
            }
        )
        writer.writerow(
            {
                "start": "3.0",
                "end": "4.0",
                "AutoAD-Zero": "AutoAD text at c3.",
                "ground_truth": "Human text at c3.",
                "NarrAD": "NarrAD text at c3.",
                "movie": "MOVIE",
                "vid": "vid",
            }
        )
    example = {
        "id": "ex1",
        "target": {"clip_id": "c4", "sequence_index": 4},
        "context": [
            {
                "clip_id": "c1",
                "movie": "Movie",
                "sequence_index": 1,
                "timestamp_start": "1.0",
                "timestamp_end": "2.0",
            },
            {
                "clip_id": "c3",
                "movie": "Movie",
                "sequence_index": 3,
                "timestamp_start": "3.0",
                "timestamp_end": "4.0",
            },
        ],
    }
    selected_positions = [
        (clip["clip_id"], clip["sequence_index"], clip["timestamp_start"], clip["timestamp_end"])
        for clip in selected_prior_context_clips(example)
    ]

    contexts = {}
    for column in ["ground_truth", "NarrAD", "AutoAD-Zero"]:
        rows = load_generated_ad_rows(csv_path, column=column)
        contexts[column], unmatched = materialize_autoad_context(
            example,
            build_generated_ad_index(rows),
            text_index=build_ground_truth_index(rows),
        )
        assert unmatched == []
        assert [
            (clip["clip_id"], clip["sequence_index"], clip["timestamp_start"], clip["timestamp_end"])
            for clip in selected_prior_context_clips(example)
        ] == selected_positions

    assert contexts["ground_truth"] == ["Human text at c1.", "Human text at c3."]
    assert contexts["NarrAD"] == ["NarrAD text at c1.", "NarrAD text at c3."]
    assert contexts["AutoAD-Zero"] == ["AutoAD text at c1.", "AutoAD text at c3."]


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


def test_no_context_prompt_is_source_neutral_and_has_no_metadata() -> None:
    example = {
        "id": "lsmdc::Movie::block::00001_00010::w10::Movie__1p0_2p0",
        "question": "What happens next?",
        "target_sentence": "Hidden target AD must not appear.",
        "video_path": "data/raw/movie/hidden-target.avi",
        "options": ["correct option", "wrong a", "wrong b", "wrong c"],
    }
    option_order = [0, 1, 2, 3]

    prompt_ground_truth = make_mcq_prompt(
        example,
        [],
        source_name="ground_truth",
        option_order=option_order,
    )
    prompt_narrad = make_mcq_prompt(
        example,
        [],
        source_name="NarrAD",
        option_order=option_order,
    )
    prompt_autoad = make_mcq_prompt(
        example,
        [],
        source_name="AutoAD-Zero",
        option_order=option_order,
    )

    assert prompt_ground_truth == prompt_narrad == prompt_autoad
    assert "ground_truth" not in prompt_ground_truth
    assert "NarrAD" not in prompt_ground_truth
    assert "AutoAD-Zero" not in prompt_ground_truth
    assert "Prior audio descriptions" not in prompt_ground_truth
    assert example["id"] not in prompt_ground_truth
    assert example["target_sentence"] not in prompt_ground_truth
    assert example["video_path"] not in prompt_ground_truth
    assert "video" not in prompt_ground_truth.lower()
    assert "frame" not in prompt_ground_truth.lower()
    assert "image" not in prompt_ground_truth.lower()
    assert '"prediction_idx": 0' not in prompt_ground_truth
    assert "Choose the best answer." in prompt_ground_truth
    assert 'Return strict JSON with keys "prediction_idx" and "prediction_text".' in prompt_ground_truth
    assert '"prediction_idx" must be one of 0, 1, 2, or 3.' in prompt_ground_truth
    assert '"prediction_text" must exactly match the selected option.' in prompt_ground_truth


def test_context_prompt_keeps_allowed_prior_context_and_neutral_schema() -> None:
    example = {
        "id": "lsmdc::Movie::block::00001_00010::w10::Movie__1p0_2p0",
        "question": "What happens next?",
        "target_sentence": "Hidden target AD must not appear.",
        "target": {"video_path": "data/raw/movie/hidden-target.avi"},
        "answer_idx": 0,
        "answer_text": "correct option",
        "options": ["correct option", "wrong a", "wrong b", "wrong c"],
    }

    prompt = make_mcq_prompt(
        example,
        ["Earlier context."],
        source_name="NarrAD",
        option_order=[0, 1, 2, 3],
    )

    assert (
        "Base your answer on the prior audio descriptions below. First identify "
        "which answer option is best supported by the prior descriptions, "
        "then return only the final JSON answer."
    ) in prompt
    assert "Prior audio descriptions:" in prompt
    assert "1. Earlier context." in prompt
    assert "NarrAD" not in prompt
    assert "ground_truth" not in prompt
    assert "AutoAD-Zero" not in prompt
    assert example["id"] not in prompt
    assert example["target_sentence"] not in prompt
    assert "answer_idx" not in prompt
    assert "answer_text" not in prompt
    assert "gold" not in prompt.lower()
    assert "correct answer" not in prompt.lower()
    assert "video_path" not in prompt
    assert "data/raw/movie/hidden-target.avi" not in prompt
    assert "frame" not in prompt.lower()
    assert "image" not in prompt.lower()
    assert '"prediction_idx": 0' not in prompt
    assert 'Return strict JSON with keys "prediction_idx" and "prediction_text".' in prompt
    assert '"prediction_idx" must be one of 0, 1, 2, or 3.' in prompt
    assert '"prediction_text" must exactly match the selected option.' in prompt


def test_empty_adaptive_context_uses_no_context_prompt(tmp_path: Path) -> None:
    csv_path = tmp_path / "Results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["start", "end", "AutoAD-Zero", "ground_truth", "NarrAD", "movie", "vid"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "start": "3.0",
                "end": "4.0",
                "AutoAD-Zero": "Filtered target source text.",
                "ground_truth": "Filtered target human AD.",
                "NarrAD": "",
                "movie": "MOVIE",
                "vid": "vid",
            }
        )
    rows = load_generated_ad_rows(csv_path, column="AutoAD-Zero")
    example = {
        "id": "lsmdc::Movie::block::00001_00010::w10::Movie__3p0_4p0",
        "question": "What happens next?",
        "target_sentence": "Hidden target AD.",
        "target": {"clip_id": "c3", "sequence_index": 3},
        "context": [
            {
                "clip_id": "c3",
                "movie": "Movie",
                "sequence_index": 3,
                "timestamp_start": "3.0",
                "timestamp_end": "4.0",
                "audio_description": "Filtered target human AD.",
                "video_path": "data/raw/movie/target.avi",
            }
        ],
        "options": ["correct option", "wrong a", "wrong b", "wrong c"],
    }

    context, unmatched = materialize_autoad_context(
        example,
        build_generated_ad_index(rows),
    )
    prompt = make_mcq_prompt(
        example,
        context,
        source_name="AutoAD-Zero",
        option_order=[0, 1, 2, 3],
    )

    assert context == []
    assert unmatched == []
    assert "Prior audio descriptions" not in prompt
    assert "Filtered target source text." not in prompt
    assert "Filtered target human AD." not in prompt
    assert "AutoAD-Zero" not in prompt
    assert example["id"] not in prompt
    assert example["target_sentence"] not in prompt
    assert "data/raw/movie/target.avi" not in prompt
    assert "Choose the best answer." in prompt


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


def test_summarize_predicc_uses_shared_no_context_baseline() -> None:
    summary = summarize_predicc(
        {
            "ground_truth": {
                1: {"accuracy": 0.50, "num_correct": 2, "num_examples": 4},
            },
            "NarrAD": {
                1: {"accuracy": 0.75, "num_correct": 3, "num_examples": 4},
            },
        },
        shared_acc0_metrics={"accuracy": 0.25, "num_correct": 1, "num_examples": 4},
    )

    assert summary["sources"]["ground_truth"]["acc0"] == 0.25
    assert summary["sources"]["NarrAD"]["acc0"] == 0.25
    assert summary["sources"]["ground_truth"]["shared_acc0"] is True
    assert summary["sources"]["NarrAD"]["rows"][0]["shared_no_context"] is True
    assert summary["sources"]["ground_truth"]["rows"][1]["predicc"] == 0.25
    assert summary["sources"]["NarrAD"]["rows"][1]["predicc"] == 0.5
