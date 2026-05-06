"""Normalize Qwen labels and scalar values before schema validation."""

from __future__ import annotations

import re
from typing import Any

from forseebench.utils.schema import (
    CONTINUITY_VALUES,
    EVIDENCE_TYPES,
    QUESTION_TYPE_VALUES,
    REASONING_TYPES,
    TARGET_TYPE_VALUES,
)


TARGET_TYPE_TO_QUESTION_TYPE = {
    "action_transition": "what_happens_next",
    "participant_update": "what_happens_next",
    "spatial_consequence": "what_happens_next",
    "state_change": "what_changes_next",
    "object_reveal": "what_is_revealed_next",
    "visible_text_update": "what_text_appears_next",
}

EVIDENCE_ALIASES = {
    "emotion_cue": "emotional_cue",
    "emotion_driven": "emotional_cue",
    "social_interaction": "social_cue",
    "hazard_prediction": "hazard_cue",
    "intent_driven": "scene_script",
}

REASONING_ALIASES = {
    "social_cue": "social_interaction",
    "emotional_cue": "social_interaction",
    "emotion_cue": "social_interaction",
    "emotion_driven": "social_interaction",
    "hazard_cue": "hazard_prediction",
    "dialogue_cue": "dialogue_grounded",
    "spatial_setup": "physical_precondition",
}


def normalize_bool(value: Any, default: bool = False) -> bool:
    """Return a real bool for common Qwen boolean encodings."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1", "keep"}:
            return True
        if text in {"false", "no", "0", "", "reject"}:
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return default


def normalize_float(value: Any, default: float = 0.0) -> float:
    """Return a float from numbers or score-like strings."""

    if value in {None, ""}:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match is None:
            return default
        try:
            return float(match.group(0))
        except ValueError:
            return default
    return default


def normalize_question_type(value: Any, target_type: Any = None) -> str:
    """Map Qwen question labels into the public schema labels."""

    label = str(value or "").strip()
    if label in QUESTION_TYPE_VALUES:
        return label
    target_label = str(target_type or "").strip()
    if label in TARGET_TYPE_VALUES:
        return TARGET_TYPE_TO_QUESTION_TYPE[label]
    if target_label in TARGET_TYPE_TO_QUESTION_TYPE:
        return TARGET_TYPE_TO_QUESTION_TYPE[target_label]
    return "what_happens_next"


def normalize_evidence_type(value: Any) -> str:
    label = str(value or "").strip()
    label = EVIDENCE_ALIASES.get(label, label)
    if label in EVIDENCE_TYPES:
        return label
    return "other"


def normalize_reasoning_type(value: Any) -> str:
    label = str(value or "").strip()
    label = REASONING_ALIASES.get(label, label)
    if label in REASONING_TYPES:
        return label
    return "other"


def normalize_reasoning_types(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    elif value in {None, ""}:
        raw_items = []
    else:
        raw_items = [value]
    normalized = [normalize_reasoning_type(item) for item in raw_items]
    return normalized or ["other"]


def normalize_continuity_type(value: Any, reasoning_type: Any = None) -> str:
    """Normalize continuity, deriving a conservative value for legacy rows."""

    label = str(value or "").strip()
    if label in CONTINUITY_VALUES:
        return label
    reasoning = set(normalize_reasoning_types(reasoning_type))
    if reasoning & {"physical_precondition", "motion_trajectory", "object_affordance", "scene_script", "hazard_prediction"}:
        return "continuous_physical"
    if reasoning & {"social_interaction", "intent_driven"}:
        return "continuous_social"
    if "dialogue_grounded" in reasoning:
        return "dialogue_continuous"
    return "discontinuous_unpredictable"


def normalize_evidence_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(
            {
                **row,
                "evidence_type": normalize_evidence_type(row.get("evidence_type")),
            }
        )
    return normalized


def normalize_candidate_example(example: dict[str, Any]) -> dict[str, Any]:
    """Normalize an internal candidate example in place and return it."""

    target_type = example.get("target_type")
    example["question_type"] = normalize_question_type(example.get("question_type"), target_type)
    example["evidence"] = normalize_evidence_rows(example.get("evidence", []))
    example["reasoning_type"] = normalize_reasoning_types(example.get("reasoning_type", ["other"]))
    example["continuity_type"] = normalize_continuity_type(
        example.get("continuity_type"),
        example.get("reasoning_type"),
    )
    quality = example.get("quality")
    if isinstance(quality, dict):
        quality["qwen_confidence"] = normalize_float(quality.get("qwen_confidence"), 0.0)
        quality["evidence_sufficiency"] = normalize_float(quality.get("evidence_sufficiency"), 0.0)
        quality["distractor_quality"] = normalize_float(quality.get("distractor_quality"), 0.0)
        quality["should_keep"] = normalize_bool(quality.get("should_keep"), False)
    return example


def normalize_validation_payload(
    validation: dict[str, Any],
    *,
    existing_quality: dict[str, Any] | None = None,
    positive_default: float = 0.7,
) -> dict[str, Any]:
    """Normalize validator scores, filling omitted positive scores conservatively."""

    existing_quality = existing_quality or {}
    failure_reasons = validation.get("failure_reasons", [])
    if not isinstance(failure_reasons, list):
        failure_reasons = [str(failure_reasons)]
    should_keep = normalize_bool(validation.get("should_keep"), False)
    qwen_confidence = normalize_float(validation.get("qwen_confidence"), 0.0)
    evidence_sufficiency = normalize_float(
        validation.get("evidence_sufficiency"),
        normalize_float(existing_quality.get("evidence_sufficiency"), 0.0),
    )
    distractor_quality = normalize_float(validation.get("distractor_quality"), 0.0)
    if should_keep and not failure_reasons:
        qwen_confidence = max(qwen_confidence, positive_default)
        evidence_sufficiency = max(
            evidence_sufficiency,
            normalize_float(existing_quality.get("evidence_sufficiency"), 0.0),
        )
        distractor_quality = max(distractor_quality, positive_default)
    return {
        **validation,
        "should_keep": should_keep,
        "qwen_confidence": qwen_confidence,
        "evidence_sufficiency": evidence_sufficiency,
        "distractor_quality": distractor_quality,
        "failure_reasons": failure_reasons,
    }
