from __future__ import annotations

from forseebench.utils.schema import validate_private_example, validate_public_example


def test_validate_public_example_accepts_simple_clip_native_row() -> None:
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
            "She drops the key and walks away.",
            "A delivery person enters the hallway.",
            "The lights in another room switch off.",
        ],
        "predictability": "predictable",
        "expectedness": 0.72,
        "target_type": "action_transition",
        "evidence_clip_ids": ["a3"],
        "split": "test_public",
    }
    assert validate_public_example(example) == []


def test_validate_private_example_accepts_public_fields_plus_answer_key() -> None:
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
            "She drops the key and walks away.",
            "A delivery person enters the hallway.",
            "The lights in another room switch off.",
        ],
        "answer_idx": 0,
        "answer_text": "She unlocks the door and pushes it open.",
        "answer_source": {
            "source": "target_sentence",
            "target_clip_id": "a4",
            "target_sentence": "She unlocks the door and pushes it open.",
        },
        "predictability": "predictable",
        "expectedness": 0.72,
        "target_type": "action_transition",
        "evidence_clip_ids": ["a3"],
        "split": "test_private",
    }
    assert validate_private_example(example) == []


def test_validate_public_example_rejects_track_and_bad_expectedness() -> None:
    example = {
        "id": "MovieA__a4__q1",
        "movie": "MovieA",
        "target_clip_id": "a4",
        "target_sentence": "She unlocks the door and pushes it open.",
        "context_clip_ids": ["a2"],
        "context_sentences": ["She pulls out a key."],
        "question_type": "what_happens_next",
        "question": "What visible update happens next?",
        "options": ["A", "B", "C", "D"],
        "predictability": "predictable",
        "expectedness": 1.2,
        "target_type": "action_transition",
        "evidence_clip_ids": ["a2"],
        "track": "named_ad",
    }
    errors = validate_public_example(example)
    assert "expectedness must be in [0.0, 1.0]" in errors
    assert "track must not be present" in errors
