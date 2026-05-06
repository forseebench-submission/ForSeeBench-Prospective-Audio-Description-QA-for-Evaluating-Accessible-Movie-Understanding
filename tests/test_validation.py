from __future__ import annotations

from forseebench.generation.build_dataset import (
    build_action_extraction_records,
    build_candidate_examples,
    candidate_passes_main_gate,
    make_private_row,
    make_public_row,
    validate_and_filter_examples,
)
from forseebench.generation.extract_actions import extract_target_action
from forseebench.generation.normalization import (
    normalize_bool,
    normalize_candidate_example,
    normalize_evidence_type,
    normalize_question_type,
    normalize_reasoning_types,
    normalize_validation_payload,
)
from helpers import FakeQwenClient


def test_extract_target_action_scores_expectedness_in_stage03() -> None:
    window = {
        "context": [
            {"clip_id": "a1", "timestamp_start": "0", "timestamp_end": "2", "audio_description": "A woman reaches toward a closed door with a key."},
        ],
        "target": {"clip_id": "a2", "timestamp_start": "2", "timestamp_end": "4", "audio_description": "She unlocks the door and enters."},
        "target_type": "action_transition",
        "expectedness": None,
    }
    client = FakeQwenClient(
        {
            "extract_target_action": {
                "target_sentence": "She unlocks the door and enters.",
                "target_type": "action_transition",
                "normalized_target": "She unlocks the door and enters.",
                "notes": "",
            },
            "score_expectedness": {
                "expectedness": "0.85",
                "notes": "Highly expected from the key setup.",
            },
        }
    )

    target_action = extract_target_action(window, client)

    assert target_action["expectedness"] == 0.85
    assert target_action["target_sentence"] == "She unlocks the door and enters."


def test_build_action_extraction_records_defaults_missing_expectedness_to_null() -> None:
    windows = [
        {
            "id": "MovieA__a2__q1",
            "movie": "MovieA",
            "context": [
                {"clip_id": "a1", "timestamp_start": "0", "timestamp_end": "2", "audio_description": "A woman looks at a closed door."},
            ],
            "target": {"clip_id": "a2", "timestamp_start": "2", "timestamp_end": "4", "audio_description": "The door swings open."},
            "target_type": "state_change",
            "target_clip_id": "a2",
            "expectedness": None,
        }
    ]
    client = FakeQwenClient(
        {
            "extract_target_action": {
                "target_sentence": "The door swings open.",
                "target_type": "state_change",
                "normalized_target": "The door swings open.",
                "notes": "",
            },
            "score_expectedness": {
                "expectedness": "",
                "notes": "Missing numeric output.",
            },
        }
    )

    records = build_action_extraction_records(windows, qwen_client=client)

    assert records[0]["expectedness"] is None
    assert records[0]["target_action"]["expectedness"] is None
    assert records[0]["expectedness_warning"] == "invalid_expectedness_output"


def test_build_candidate_examples_preserves_expectedness_and_target_type() -> None:
    extraction_record = {
        "id": "MovieA__a4__q1",
        "movie": "MovieA",
        "context": [
            {"clip_id": "a2", "timestamp_start": "2", "timestamp_end": "4", "audio_description": "She pulls out a key and steps toward the lock."},
            {"clip_id": "a3", "timestamp_start": "4", "timestamp_end": "6", "audio_description": "She raises the key toward the door."},
        ],
        "target": {"clip_id": "a4", "timestamp_start": "6", "timestamp_end": "8", "audio_description": "She unlocks the door and pushes it open."},
        "target_clip_id": "a4",
        "target_type": "action_transition",
        "target_triviality": "nontrivial",
        "target_validity_reason": "Supported action update.",
        "selected_context_clip_ids": ["a2", "a3"],
        "selected_window_size": 2,
        "context_selection_strategy": "qwen_block_search",
        "evidence_clip_ids": ["a3"],
        "qwen_selection_output": {
            "predictability": "predictable",
            "continuity_type": "continuous_physical",
            "selected_context_spans": [{"clip_id": "a3", "span": "She raises the key toward the door.", "evidence_type": "motion_trajectory"}],
            "evidence_sufficiency": 0.84,
            "reasoning_type": ["motion_trajectory"],
            "should_keep": True,
        },
        "expectedness": 0.76,
        "target_action": {
            "raw_description": "She unlocks the door and pushes it open.",
            "target_sentence": "She unlocks the door and pushes it open.",
            "target_type": "action_transition",
            "normalized_target": "She unlocks the door and pushes it open.",
            "expectedness": 0.76,
        },
    }
    generation_client = FakeQwenClient(
        {
            "generate_question": {
                "question_type": "what_happens_next",
                "question": "What visible update happens next?",
                "options": [
                    "She unlocks the door and pushes it open.",
                    "She drops the key and leaves.",
                    "A neighbor walks into the room.",
                    "The ceiling fan stops spinning.",
                ],
                "answer_idx": 0,
                "distractor_metadata": ["correct", "contradicts_context", "unrelated", "unrelated"],
            }
        }
    )

    candidates = build_candidate_examples([extraction_record], qwen_client=generation_client)

    assert candidates[0]["expectedness"] == 0.76
    assert candidates[0]["target_type"] == "action_transition"
    assert candidates[0]["question_type"] == "what_happens_next"
    assert candidates[0]["answer_text"] == "She unlocks the door and pushes it open."


def test_build_candidate_examples_normalizes_qwen_labels() -> None:
    extraction_record = {
        "id": "MovieA__a4__q1",
        "movie": "MovieA",
        "context": [
            {"clip_id": "a3", "timestamp_start": "4", "timestamp_end": "6", "audio_description": "She raises the key toward the door."},
        ],
        "target": {"clip_id": "a4", "timestamp_start": "6", "timestamp_end": "8", "audio_description": "She unlocks the door and pushes it open."},
        "target_clip_id": "a4",
        "target_type": "participant_update",
        "target_triviality": "nontrivial",
        "target_validity_reason": "Supported participant update.",
        "selected_context_clip_ids": ["a3"],
        "selected_window_size": 1,
        "context_selection_strategy": "qwen_block_search",
        "evidence_clip_ids": ["a3"],
        "qwen_selection_output": {
            "predictability": "predictable",
            "selected_context_spans": [{"clip_id": "a3", "span": "She raises the key toward the door.", "evidence_type": "emotion_cue"}],
            "evidence_sufficiency": "0.84",
            "reasoning_type": ["social_cue", "hazard_cue"],
            "should_keep": "false",
        },
        "expectedness": 0.76,
        "target_action": {
            "raw_description": "She unlocks the door and pushes it open.",
            "target_sentence": "She unlocks the door and pushes it open.",
            "target_type": "participant_update",
            "normalized_target": "She unlocks the door and pushes it open.",
            "expectedness": 0.76,
        },
    }
    generation_client = FakeQwenClient(
        {
            "generate_question": {
                "question_type": "participant_update",
                "question": "What development involving her happens next?",
                "options": [
                    "She unlocks the door and pushes it open.",
                    "She drops the key and leaves.",
                    "A neighbor walks into the room.",
                    "The ceiling fan stops spinning.",
                ],
                "answer_idx": 0,
                "distractor_metadata": ["correct", "contradicts_context", "unrelated", "unrelated"],
            }
        }
    )

    candidate = build_candidate_examples([extraction_record], qwen_client=generation_client)[0]

    assert candidate["question_type"] == "what_happens_next"
    assert candidate["evidence"][0]["evidence_type"] == "emotional_cue"
    assert candidate["reasoning_type"] == ["social_interaction", "hazard_prediction"]
    assert candidate["continuity_type"] == "continuous_physical"
    assert candidate["quality"]["should_keep"] is False


def test_validate_and_filter_examples_keeps_predictable_high_quality_item() -> None:
    candidate = {
        "id": "MovieA__a4__q1",
        "movie": "MovieA",
        "target_clip_id": "a4",
        "target_sentence": "She unlocks the door and pushes it open.",
        "context_clip_ids": ["a3"],
        "context_sentences": ["She raises the key toward the door."],
        "question_type": "what_happens_next",
        "question": "What visible update happens next?",
        "options": [
            "She unlocks the door and pushes it open.",
            "She drops the key and leaves.",
            "A neighbor walks into the room.",
            "The ceiling fan stops spinning.",
        ],
        "answer_idx": 0,
        "answer_text": "She unlocks the door and pushes it open.",
        "predictability": "predictable",
        "expectedness": 0.81,
        "target_type": "action_transition",
        "evidence_clip_ids": ["a3"],
        "selection_metadata": {
            "target_triviality": "nontrivial",
            "target_validity_reason": "Supported action update.",
            "selected_window_size": 1,
        },
        "validation_metadata": {"oracle_pass": None, "grounding_pass": None, "distractor_quality": 0.0},
        "distractor_metadata": ["correct", "contradicts_context", "unrelated", "unrelated"],
        "split": None,
        "context": [],
        "target": {"clip_id": "a4"},
        "evidence": [{"clip_id": "a3", "span": "She raises the key toward the door.", "evidence_type": "motion_trajectory"}],
        "reasoning_type": ["motion_trajectory"],
        "continuity_type": "continuous_physical",
        "continuity_scores": {},
        "semantic_similarity_last": 0.0,
        "semantic_similarity_mean": 0.0,
        "timestamp_gap": 0.0,
        "entity_overlap": 0.0,
        "action_overlap": 0.0,
        "location_overlap": 0.0,
        "selected_window_size": 1,
        "context_selection_strategy": "qwen_block_search",
        "full_prior_context_clip_ids": ["a3"],
        "selected_context_clip_ids": ["a3"],
        "rejection_reason": None,
        "qwen_selection_output": {},
        "qwen_continuity_output": {},
        "quality": {"qwen_confidence": 0.0, "evidence_sufficiency": 0.92, "distractor_quality": 0.0, "should_keep": True},
    }
    validator_client = FakeQwenClient(
        {
            "validate_example": {
                "should_keep": True,
                "qwen_confidence": 0.95,
                "evidence_sufficiency": 0.92,
                "distractor_quality": 0.88,
                "failure_reasons": [],
                "recommended_fix": "",
            }
        }
    )

    filtered = validate_and_filter_examples(
        [candidate],
        qwen_client=validator_client,
        thresholds={"qwen_confidence": 0.7, "evidence_sufficiency": 0.7, "distractor_quality": 0.7},
        max_kept=500,
    )
    assert len(filtered["kept"]) == 1
    assert filtered["kept"][0]["validation_metadata"]["distractor_quality"] == 0.88


def test_main_gate_allows_all_target_types_without_physical_social_continuity() -> None:
    for target_type in (
        "action_transition",
        "state_change",
        "participant_update",
        "spatial_consequence",
        "object_reveal",
        "visible_text_update",
    ):
        candidate = {
            "target_type": target_type,
            "predictability": "predictable",
            "evidence": [{"clip_id": "a3", "span": "setup", "evidence_type": "other"}],
            "evidence_clip_ids": ["a3"],
            "continuity_type": "dialogue_continuous",
            "selection_metadata": {"target_triviality": "nontrivial"},
            "quality": {"should_keep": True},
        }
        assert candidate_passes_main_gate(candidate, quality_pass=True, expectedness_pass=True)


def test_normalization_helpers_cover_common_qwen_variants() -> None:
    assert normalize_bool("false") is False
    assert normalize_bool("true") is True
    assert normalize_question_type("object_reveal", "object_reveal") == "what_is_revealed_next"
    assert normalize_evidence_type("emotion_cue") == "emotional_cue"
    assert normalize_reasoning_types(["social_cue", "hazard_cue", "visible_text_update"]) == [
        "social_interaction",
        "hazard_prediction",
        "other",
    ]


def test_normalize_candidate_example_repairs_schema_level_labels() -> None:
    example = {
        "target_type": "state_change",
        "question_type": "state_change",
        "evidence": [{"clip_id": "a1", "span": "setup", "evidence_type": "action_transition"}],
        "reasoning_type": ["emotion_driven"],
        "continuity_type": "",
        "quality": {
            "qwen_confidence": "0.8",
            "evidence_sufficiency": "0.75",
            "distractor_quality": "0.7",
            "should_keep": "false",
        },
    }

    normalize_candidate_example(example)

    assert example["question_type"] == "what_changes_next"
    assert example["evidence"][0]["evidence_type"] == "other"
    assert example["reasoning_type"] == ["social_interaction"]
    assert example["continuity_type"] == "continuous_social"
    assert example["quality"]["should_keep"] is False


def test_normalize_validation_payload_fills_positive_scores_only_without_failures() -> None:
    validation = normalize_validation_payload(
        {
            "should_keep": True,
            "qwen_confidence": 0.0,
            "evidence_sufficiency": 0.0,
            "distractor_quality": 0.0,
            "failure_reasons": [],
        },
        existing_quality={"evidence_sufficiency": 0.76},
    )
    failed_validation = normalize_validation_payload(
        {
            "should_keep": True,
            "qwen_confidence": 0.0,
            "evidence_sufficiency": 0.0,
            "distractor_quality": 0.0,
            "failure_reasons": ["oracle_target_answerability"],
        },
        existing_quality={"evidence_sufficiency": 0.76},
    )

    assert validation["qwen_confidence"] == 0.7
    assert validation["evidence_sufficiency"] == 0.76
    assert validation["distractor_quality"] == 0.7
    assert failed_validation["qwen_confidence"] == 0.0


def test_make_public_and_private_rows_omit_track_and_raw_traces() -> None:
    example = {
        "id": "MovieA__a4__q1",
        "movie": "MovieA",
        "target_clip_id": "a4",
        "target_sentence": "She unlocks the door and pushes it open.",
        "context_clip_ids": ["a2", "a3"],
        "context_sentences": ["She pulls out a key.", "She raises the key toward the door."],
        "question_type": "what_happens_next",
        "question": "What visible update happens next?",
        "options": [
            "She unlocks the door and pushes it open.",
            "She drops the key and leaves.",
            "A neighbor walks into the room.",
            "The ceiling fan stops spinning.",
        ],
        "answer_idx": 0,
        "answer_text": "She unlocks the door and pushes it open.",
        "predictability": "predictable",
        "expectedness": 0.81,
        "target_type": "action_transition",
        "evidence_clip_ids": ["a3"],
        "selection_metadata": {
            "target_triviality": "nontrivial",
            "target_validity_reason": "Supported action update.",
            "selected_window_size": 2,
        },
        "validation_metadata": {"oracle_pass": True, "grounding_pass": True, "distractor_quality": 0.88},
    }

    public_row = make_public_row(example, "test")
    private_row = make_private_row(example, "test")

    assert "answer_idx" not in public_row
    assert "selection_metadata" not in public_row
    assert public_row["split"] == "test_public"
    assert private_row["split"] == "test_private"
    assert private_row["answer_idx"] == 0
    assert private_row["answer_text"] == "She unlocks the door and pushes it open."
    assert private_row["answer_source"] == {
        "source": "target_sentence",
        "target_clip_id": "a4",
        "target_sentence": "She unlocks the door and pushes it open.",
    }
    assert "selection_metadata" not in private_row
    assert "validation_metadata" not in private_row
    assert "track" not in public_row
    assert "track" not in private_row
