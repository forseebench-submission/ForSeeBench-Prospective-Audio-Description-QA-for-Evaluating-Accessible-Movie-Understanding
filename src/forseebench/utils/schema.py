"""ForSeeBench schema validation."""

from __future__ import annotations

from typing import Any

PREDICTABILITY_VALUES = {"predictable", "underdetermined", "unpredictable"}
QUESTION_TYPE_VALUES = {
    "what_happens_next",
    "what_changes_next",
    "what_is_revealed_next",
    "what_text_appears_next",
}
TARGET_TYPE_VALUES = {
    "action_transition",
    "state_change",
    "participant_update",
    "spatial_consequence",
    "object_reveal",
    "visible_text_update",
}
CONTINUITY_VALUES = {
    "continuous_physical",
    "continuous_social",
    "dialogue_continuous",
    "scene_cut_semantically_linked",
    "discontinuous_unpredictable",
}
EVIDENCE_TYPES = {
    "physical_precondition",
    "motion_trajectory",
    "object_affordance",
    "social_cue",
    "dialogue_cue",
    "emotional_cue",
    "scene_script",
    "hazard_cue",
    "other",
}
REASONING_TYPES = {
    "physical_precondition",
    "motion_trajectory",
    "object_affordance",
    "social_interaction",
    "dialogue_grounded",
    "intent_driven",
    "scene_script",
    "hazard_prediction",
    "other",
}
DISTRACTOR_TYPES = {
    "correct",
    "already_happened",
    "entity_swapped",
    "plausible_unsupported",
    "contradicts_context",
    "unrelated",
}


def _validate_expectedness(expectedness: Any, errors: list[str]) -> None:
    if expectedness is None:
        return
    if isinstance(expectedness, bool) or not isinstance(expectedness, (int, float)):
        errors.append("expectedness must be numeric or null")
    elif not 0.0 <= float(expectedness) <= 1.0:
        errors.append("expectedness must be in [0.0, 1.0]")


def _validate_core_row(example: dict[str, Any], errors: list[str]) -> None:
    required = (
        "id",
        "movie",
        "target_clip_id",
        "target_sentence",
        "context_clip_ids",
        "context_sentences",
        "question_type",
        "question",
        "options",
        "predictability",
        "expectedness",
        "target_type",
        "evidence_clip_ids",
    )
    for key in required:
        if key not in example:
            errors.append(f"missing top-level key: {key}")
    if errors:
        return
    if not isinstance(example["movie"], str) or not example["movie"]:
        errors.append("movie must be non-empty string")
    if not isinstance(example["target_clip_id"], str) or not example["target_clip_id"]:
        errors.append("target_clip_id must be non-empty string")
    if not isinstance(example["target_sentence"], str) or not example["target_sentence"]:
        errors.append("target_sentence must be non-empty string")
    if not isinstance(example["context_clip_ids"], list):
        errors.append("context_clip_ids must be a list")
    if not isinstance(example["context_sentences"], list):
        errors.append("context_sentences must be a list")
    if len(example.get("context_clip_ids", [])) != len(example.get("context_sentences", [])):
        errors.append("context_clip_ids and context_sentences must align")
    if example.get("question_type") not in QUESTION_TYPE_VALUES:
        errors.append("invalid question_type")
    if not isinstance(example.get("question"), str) or not example["question"]:
        errors.append("question must be non-empty string")
    if not isinstance(example.get("options"), list) or len(example["options"]) != 4:
        errors.append("options must contain exactly four strings")
    else:
        for option in example["options"]:
            if not isinstance(option, str) or not option.strip():
                errors.append("each option must be a non-empty string")
                break
    if example.get("predictability") not in PREDICTABILITY_VALUES:
        errors.append("invalid predictability label")
    if example.get("target_type") not in TARGET_TYPE_VALUES:
        errors.append("invalid target_type")
    if not isinstance(example.get("evidence_clip_ids"), list):
        errors.append("evidence_clip_ids must be a list")
    _validate_expectedness(example.get("expectedness"), errors)
    if "track" in example:
        errors.append("track must not be present")


def validate_public_example(example: dict[str, Any]) -> list[str]:
    """Validate the compact public export schema."""

    errors: list[str] = []
    _validate_core_row(example, errors)
    if "answer_idx" in example or "answer_text" in example:
        errors.append("public example must not contain answer key")
    split = example.get("split")
    if split is not None and (not isinstance(split, str) or not split.endswith("_public")):
        errors.append("public split must end with _public")
    return errors


def validate_release_public_example(example: dict[str, Any]) -> list[str]:
    """Validate the release-facing public/no-answer benchmark schema."""

    errors: list[str] = []
    required = (
        "id",
        "source_id",
        "prior_context",
        "question",
        "options",
        "question_type",
        "target_type",
        "context_length",
    )
    for key in required:
        if key not in example:
            errors.append(f"missing top-level key: {key}")
    for forbidden in ("answer_idx", "answer_text", "hidden_target_ad", "target_sentence", "target_clip_id"):
        if forbidden in example:
            errors.append(f"public example must not contain {forbidden}")
    if errors:
        return errors
    if not isinstance(example["id"], str) or not example["id"]:
        errors.append("id must be non-empty string")
    if not isinstance(example["source_id"], str) or not example["source_id"]:
        errors.append("source_id must be non-empty string")
    prior_context = example["prior_context"]
    if not isinstance(prior_context, list):
        errors.append("prior_context must be a list")
    else:
        for item in prior_context:
            if not isinstance(item, dict):
                errors.append("each prior_context item must be an object")
                break
            if not isinstance(item.get("clip_id"), str) or not item["clip_id"]:
                errors.append("each prior_context item must have non-empty clip_id")
                break
            if not isinstance(item.get("text"), str) or not item["text"]:
                errors.append("each prior_context item must have non-empty text")
                break
    if not isinstance(example.get("question"), str) or not example["question"]:
        errors.append("question must be non-empty string")
    if not isinstance(example.get("options"), list) or len(example["options"]) != 4:
        errors.append("options must contain exactly four strings")
    else:
        for option in example["options"]:
            if not isinstance(option, str) or not option.strip():
                errors.append("each option must be a non-empty string")
                break
    if example.get("question_type") not in QUESTION_TYPE_VALUES:
        errors.append("invalid question_type")
    if example.get("target_type") not in TARGET_TYPE_VALUES:
        errors.append("invalid target_type")
    context_length = example.get("context_length")
    if not isinstance(context_length, int) or context_length < 0:
        errors.append("context_length must be a non-negative integer")
    elif isinstance(prior_context, list) and context_length != len(prior_context):
        errors.append("context_length must equal len(prior_context)")
    _validate_expectedness(example.get("expectedness"), errors)
    return errors


def validate_release_with_answers_example(example: dict[str, Any]) -> list[str]:
    """Validate the release-facing answer-bearing/scoring schema."""

    public_view = {
        key: value
        for key, value in example.items()
        if key not in {"answer_idx", "answer_text", "hidden_target_ad", "evidence"}
    }
    errors = validate_release_public_example(public_view)
    for key in ("answer_idx", "answer_text"):
        if key not in example:
            errors.append(f"missing top-level key: {key}")
    if errors:
        return errors
    answer_idx = example["answer_idx"]
    if not isinstance(answer_idx, int) or not 0 <= answer_idx < 4:
        errors.append("answer_idx must be in [0, 3]")
    elif example["options"][answer_idx] != example["answer_text"]:
        errors.append("answer_text must match options[answer_idx]")
    if "hidden_target_ad" in example and (not isinstance(example["hidden_target_ad"], str) or not example["hidden_target_ad"]):
        errors.append("hidden_target_ad must be non-empty string when present")
    if "evidence" in example:
        evidence = example["evidence"]
        if not isinstance(evidence, list):
            errors.append("evidence must be a list")
        else:
            for item in evidence:
                if not isinstance(item, dict):
                    errors.append("each evidence item must be an object")
                    break
                if not isinstance(item.get("clip_id"), str) or not item["clip_id"]:
                    errors.append("each evidence item must have non-empty clip_id")
                    break
                if not isinstance(item.get("span"), str) or not item["span"]:
                    errors.append("each evidence item must have non-empty span")
                    break
    return errors


def validate_private_example(example: dict[str, Any]) -> list[str]:
    """Validate the compact private export schema."""

    errors: list[str] = []
    _validate_core_row(example, errors)
    for key in ("answer_idx", "answer_text", "answer_source"):
        if key not in example:
            errors.append(f"missing top-level key: {key}")
    if errors:
        return errors
    answer_idx = example["answer_idx"]
    if not isinstance(answer_idx, int) or not 0 <= answer_idx < 4:
        errors.append("answer_idx must be in [0, 3]")
    elif example["options"][answer_idx] != example["answer_text"]:
        errors.append("answer_text must match options[answer_idx]")
    answer_source = example.get("answer_source")
    if not isinstance(answer_source, dict):
        errors.append("answer_source must be an object")
    else:
        if answer_source.get("source") != "target_sentence":
            errors.append("answer_source.source must be target_sentence")
        if answer_source.get("target_clip_id") != example["target_clip_id"]:
            errors.append("answer_source.target_clip_id must match target_clip_id")
        if answer_source.get("target_sentence") != example["target_sentence"]:
            errors.append("answer_source.target_sentence must match target_sentence")
    split = example.get("split")
    if split is not None and (not isinstance(split, str) or not split.endswith("_private")):
        errors.append("private split must end with _private")
    return errors


def validate_example(example: dict[str, Any]) -> list[str]:
    """Validate an internal candidate example row used before final export."""

    errors: list[str] = []
    _validate_core_row(example, errors)
    for key in ("answer_idx", "answer_text"):
        if key not in example:
            errors.append(f"missing top-level key: {key}")
    if errors:
        return errors
    answer_idx = example["answer_idx"]
    if not isinstance(answer_idx, int) or not 0 <= answer_idx < 4:
        errors.append("answer_idx must be in [0, 3]")
    elif example["options"][answer_idx] != example["answer_text"]:
        errors.append("answer_text must match options[answer_idx]")
    for key in ("selection_metadata", "validation_metadata"):
        if key not in example:
            errors.append(f"missing top-level key: {key}")
    if errors:
        return errors
    if not isinstance(example["selection_metadata"], dict):
        errors.append("selection_metadata must be an object")
    if not isinstance(example["validation_metadata"], dict):
        errors.append("validation_metadata must be an object")
    if not isinstance(example.get("distractor_metadata"), list) or len(example["distractor_metadata"]) != 4:
        errors.append("distractor_metadata must be a list of four labels")
    else:
        for label in example["distractor_metadata"]:
            if label not in DISTRACTOR_TYPES:
                errors.append(f"invalid distractor type: {label}")
    if example["distractor_metadata"][example["answer_idx"]] != "correct":
        errors.append("correct answer must be labeled correct in distractor metadata")
    if example.get("continuity_type") not in CONTINUITY_VALUES:
        errors.append("invalid continuity_type")
    if not isinstance(example.get("context"), list):
        errors.append("context must be a list")
    if example.get("target") is not None and not isinstance(example.get("target"), dict):
        errors.append("target must be an object or null")
    if not isinstance(example.get("evidence"), list):
        errors.append("evidence must be a list")
    if not isinstance(example.get("quality"), dict):
        errors.append("quality must be an object")
    if not isinstance(example.get("continuity_scores"), dict):
        errors.append("continuity_scores must be an object")
    for key in ("semantic_similarity_last", "semantic_similarity_mean", "entity_overlap", "action_overlap", "location_overlap"):
        if not isinstance(example.get(key), (int, float)):
            errors.append(f"{key} must be numeric")
    if example.get("timestamp_gap") is not None and not isinstance(example.get("timestamp_gap"), (int, float)):
        errors.append("timestamp_gap must be numeric or null")
    if not isinstance(example.get("selected_window_size"), int):
        errors.append("selected_window_size must be int")
    if not isinstance(example.get("context_selection_strategy"), str) or not example["context_selection_strategy"]:
        errors.append("context_selection_strategy must be non-empty string")
    if not isinstance(example.get("full_prior_context_clip_ids"), list):
        errors.append("full_prior_context_clip_ids must be a list")
    if not isinstance(example.get("selected_context_clip_ids"), list):
        errors.append("selected_context_clip_ids must be a list")
    if example.get("rejection_reason") is not None and not isinstance(example.get("rejection_reason"), str):
        errors.append("rejection_reason must be string or null")
    if example.get("qwen_selection_output") is not None and not isinstance(example.get("qwen_selection_output"), dict):
        errors.append("qwen_selection_output must be object or null")
    if example.get("qwen_continuity_output") is not None and not isinstance(example.get("qwen_continuity_output"), dict):
        errors.append("qwen_continuity_output must be object or null")
    for item in example.get("reasoning_type", []):
        if item not in REASONING_TYPES:
            errors.append(f"invalid reasoning_type: {item}")
    for evidence_item in example.get("evidence", []):
        if evidence_item.get("evidence_type") not in EVIDENCE_TYPES:
            errors.append(f"invalid evidence_type: {evidence_item.get('evidence_type')}")
        if not evidence_item.get("clip_id"):
            errors.append("evidence missing clip_id")
        if not evidence_item.get("span"):
            errors.append("evidence missing span")
    quality = example.get("quality", {})
    for key in ("qwen_confidence", "evidence_sufficiency", "distractor_quality"):
        value = quality.get(key)
        if not isinstance(value, (int, float)):
            errors.append(f"quality.{key} must be numeric")
    if not isinstance(quality.get("should_keep"), bool):
        errors.append("quality.should_keep must be boolean")
    return errors
