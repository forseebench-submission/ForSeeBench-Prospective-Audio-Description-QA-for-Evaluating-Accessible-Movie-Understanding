from __future__ import annotations

from forseebench.generation.select_context import (
    MINIMAL_REJECTION_REASONS,
    iterate_target_level_context_decisions,
    route_target_level_decision,
    target_level_passes_selection,
)
from helpers import FakeQwenClient


def _make_blocks() -> list[dict]:
    clips = [
        {"clip_id": "a1", "timestamp_start": "0", "timestamp_end": "1", "audio_description": "A man studies the closed cabinet.", "sequence_index": 0},
        {"clip_id": "a2", "timestamp_start": "1", "timestamp_end": "2", "audio_description": "He reaches toward the cabinet handle.", "sequence_index": 1},
        {"clip_id": "a3", "timestamp_start": "2", "timestamp_end": "3", "audio_description": "The cabinet door swings open to reveal stacked plates.", "sequence_index": 2},
        {"clip_id": "a4", "timestamp_start": "3", "timestamp_end": "4", "audio_description": "He lifts one of the plates from the shelf.", "sequence_index": 3},
        {"clip_id": "a5", "timestamp_start": "4", "timestamp_end": "5", "audio_description": "He turns toward the sink with the plate.", "sequence_index": 4},
        {"clip_id": "a6", "timestamp_start": "5", "timestamp_end": "6", "audio_description": "Water starts running into the sink basin.", "sequence_index": 5},
    ]
    return [
        {
            "id": "lsmdc::MovieA::block::00000_00004::w5",
            "movie": "MovieA",
            "block_start_index": 0,
            "block_end_index": 4,
            "max_window_clips": 5,
            "clips": clips[:5],
        },
        {
            "id": "lsmdc::MovieA::block::00001_00005::w5",
            "movie": "MovieA",
            "block_start_index": 1,
            "block_end_index": 5,
            "max_window_clips": 5,
            "clips": clips[1:6],
        },
        {
            "id": "lsmdc::MovieA::block::00002_00005::w4",
            "movie": "MovieA",
            "block_start_index": 2,
            "block_end_index": 5,
            "max_window_clips": 4,
            "clips": clips[2:6],
        },
        {
            "id": "lsmdc::MovieA::block::00003_00005::w3",
            "movie": "MovieA",
            "block_start_index": 3,
            "block_end_index": 5,
            "max_window_clips": 3,
            "clips": clips[3:6],
        },
    ]


def _config() -> dict:
    return {
        "selection_mode": "target_level_qwen_minimal_evidence",
        "max_qwen_calls": None,
    }


def test_block_search_uses_earliest_nontrivial_target_and_advances_after_target() -> None:
    client = FakeQwenClient(
        {
            "select_block_target_context": [
                {
                    "found_valid_target": True,
                    "target_clip_id": "a3",
                    "target_position_in_block": 3,
                    "target_type": "object_reveal",
                    "target_triviality": "nontrivial",
                    "target_validity_reason": "Earliest informative reveal with clear buildup.",
                    "selected_context_clip_ids": ["a1", "a2"],
                    "selected_context_spans": [{"clip_id": "a2", "span": "He reaches toward the cabinet handle.", "evidence_type": "physical_precondition"}],
                    "evidence_clip_ids": ["a2"],
                    "predictability": "predictable",
                    "evidence_sufficiency": 0.81,
                    "reasoning_type": ["physical_precondition"],
                    "should_keep": True,
                },
                    {
                        "found_valid_target": True,
                        "target_clip_id": "a6",
                        "target_position_in_block": 3,
                        "target_type": "state_change",
                        "target_triviality": "nontrivial",
                        "target_validity_reason": "A visible environment change follows the sink setup.",
                    "selected_context_clip_ids": ["a4", "a5"],
                    "selected_context_spans": [{"clip_id": "a5", "span": "He turns toward the sink with the plate.", "evidence_type": "physical_precondition"}],
                    "evidence_clip_ids": ["a5"],
                    "predictability": "predictable",
                    "evidence_sufficiency": 0.77,
                    "reasoning_type": ["physical_precondition"],
                    "should_keep": True,
                },
            ]
        }
    )

    decisions = list(iterate_target_level_context_decisions(_make_blocks(), qwen_client=client, config=_config()))

    assert [row["block_start_index"] for row in decisions] == [0, 3]
    assert decisions[0]["target_type"] == "object_reveal"
    assert decisions[0]["target_triviality"] == "nontrivial"
    assert decisions[0]["selected_context_clip_ids"] == ["a1", "a2"]
    assert decisions[0]["evidence_clip_ids"] == ["a2"]
    assert target_level_passes_selection(decisions[0]) is True
    assert route_target_level_decision(decisions[0]) == "selected"


def test_trivial_target_is_rejected() -> None:
    client = FakeQwenClient(
        {
            "select_block_target_context": {
                "found_valid_target": True,
                "target_clip_id": "a5",
                "target_position_in_block": 5,
                "target_type": "action_transition",
                "target_triviality": "trivial_generic_motion",
                "target_validity_reason": "Generic continuation with low information value.",
                "selected_context_clip_ids": ["a4"],
                "selected_context_spans": [{"clip_id": "a4", "span": "He lifts one of the plates from the shelf.", "evidence_type": "motion_trajectory"}],
                "predictability": "predictable",
                "evidence_sufficiency": 0.7,
                "reasoning_type": ["motion_trajectory"],
                "should_keep": False,
            }
        }
    )

    decision = list(iterate_target_level_context_decisions([_make_blocks()[0]], qwen_client=client, config=_config()))[0]

    assert target_level_passes_selection(decision) is False
    assert route_target_level_decision(decision) == "rejected"
    assert decision["rejection_reason"] == "trivial_target"
    assert decision["rejection_reason"] in MINIMAL_REJECTION_REASONS


def test_meaningful_but_noninferable_routes_to_challenge() -> None:
    client = FakeQwenClient(
        {
            "select_block_target_context": {
                "found_valid_target": True,
                "target_clip_id": "a6",
                "target_position_in_block": 5,
                "target_type": "state_change",
                "target_triviality": "meaningful_but_noninferable",
                "target_validity_reason": "Meaningful update, but prior context does not support it well enough.",
                "selected_context_clip_ids": ["a5"],
                "selected_context_spans": [{"clip_id": "a5", "span": "He turns toward the sink with the plate.", "evidence_type": "other"}],
                "predictability": "underdetermined",
                "evidence_sufficiency": 0.2,
                "reasoning_type": ["other"],
                "should_keep": False,
            }
        }
    )

    decision = list(iterate_target_level_context_decisions([_make_blocks()[1]], qwen_client=client, config=_config()))[0]

    assert target_level_passes_selection(decision) is False
    assert route_target_level_decision(decision) == "challenge"
    assert decision["rejection_reason"] == "meaningful_but_noninferable"


def test_missing_predictability_still_routes_to_selected_when_otherwise_valid() -> None:
    client = FakeQwenClient(
        {
            "select_block_target_context": {
                "found_valid_target": True,
                "target_clip_id": "a3",
                "target_position_in_block": 3,
                "target_type": "object_reveal",
                "target_triviality": "nontrivial",
                "target_validity_reason": "Supported reveal.",
                "selected_context_clip_ids": ["a2"],
                "selected_context_spans": [{"clip_id": "a2", "span": "He reaches toward the cabinet handle.", "evidence_type": "physical_precondition"}],
                "predictability": None,
                "evidence_sufficiency": 0.8,
                "reasoning_type": ["physical_precondition"],
                "should_keep": True,
            }
        }
    )

    decision = list(iterate_target_level_context_decisions([_make_blocks()[0]], qwen_client=client, config=_config()))[0]

    assert target_level_passes_selection(decision) is True
    assert route_target_level_decision(decision) == "selected"
