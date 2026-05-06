"""Block-level target and context selection for ForSeeBench."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Iterable

from forseebench.generation.continuity_filter import build_continuity_features
from forseebench.generation.normalization import normalize_continuity_type, normalize_reasoning_types
from forseebench.qwen.prompts import make_block_selection_prompt, make_block_selection_video_prompt, make_continuity_prompt
from forseebench.qwen.qwen_client import QwenClient


ALLOWED_CONTINUITY = {"continuous_physical", "continuous_social"}
ALLOWED_TARGET_TYPES = {
    "action_transition",
    "state_change",
    "participant_update",
    "spatial_consequence",
    "object_reveal",
    "visible_text_update",
}
ALLOWED_EVIDENCE_TYPES = {
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
TRIVIALITY_REJECTED = {
    "trivial_generic_motion",
    "trivial_static_state",
    "trivial_redundant_continuation",
    "trivial_low_information",
}
MINIMAL_REJECTION_REASONS = {
    "no_valid_target",
    "invalid_target",
    "insufficient_buildup",
    "underdetermined",
    "meaningful_but_noninferable",
    "trivial_target",
}
FRAGMENT_STARTS = (
    "knocking ",
    "sending ",
    "causing ",
    "making ",
    "leaving ",
    "revealing ",
    "showing ",
    "then ",
)
PRONOUN_STARTS = (
    "he ",
    "she ",
    "it ",
    "they ",
    "his ",
    "her ",
    "their ",
)

def _default_block_decision(reason: str, strategy: str) -> dict[str, Any]:
    return {
        "found_valid_target": False,
        "target_clip_id": None,
        "target_position_in_block": None,
        "target_type": None,
        "target_triviality": None,
        "target_validity_reason": reason,
        "selected_context_clip_ids": [],
        "selected_context_spans": [],
        "selected_window_size": 0,
        "predictability": "unpredictable",
        "expectedness": None,
        "evidence_sufficiency": 0.0,
        "reasoning_type": ["other"],
        "continuity_type": "discontinuous_unpredictable",
        "should_keep": False,
        "context_selection_strategy": strategy,
    }


def _is_missing_predictability(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"unknown", "missing", "n/a", "na"}
    return False


def _target_is_explicitly_invalid(qwen_output: dict[str, Any]) -> bool:
    target_type = str(qwen_output.get("target_type", "")).strip().lower()
    if target_type and target_type not in ALLOWED_TARGET_TYPES:
        return True
    triviality = str(qwen_output.get("target_triviality", "")).strip().lower()
    return triviality in TRIVIALITY_REJECTED


def _target_needs_immediate_antecedent(sentence: Any) -> bool:
    text = str(sentence or "").strip().lower()
    return text.startswith(FRAGMENT_STARTS) or text.startswith(PRONOUN_STARTS)


def _normalize_rejection_reason(result: dict[str, Any], qwen_output: dict[str, Any]) -> str:
    triviality = str(result.get("target_triviality", "")).strip().lower()
    if triviality == "meaningful_but_noninferable":
        return "meaningful_but_noninferable"
    if triviality in TRIVIALITY_REJECTED:
        return "trivial_target"
    predictability = result.get("predictability")
    if predictability == "underdetermined":
        return "underdetermined"
    if predictability == "unpredictable":
        return "insufficient_buildup"
    if not result.get("found_valid_target") or not result.get("target_clip_id") or result.get("target") is None:
        if (
            (qwen_output.get("target_clip_id") or qwen_output.get("target_position_in_block") is not None)
            and _target_is_explicitly_invalid(qwen_output)
        ):
            return "invalid_target"
        return "no_valid_target"
    if _target_is_explicitly_invalid(qwen_output):
        return "invalid_target"
    if not result.get("selected_context_clip_ids"):
        return "insufficient_buildup"
    return "insufficient_buildup"


def _reconcile_evidence_with_context(
    *,
    evidence_rows: list[dict[str, Any]],
    prefix_clips: list[dict[str, Any]],
    selected_context_ids: list[str],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Ensure evidence only references valid prior clips and is represented in context."""

    prefix_lookup = {row["clip_id"]: row for row in prefix_clips}
    valid_evidence: list[dict[str, Any]] = []
    merged_ids = list(selected_context_ids)
    seen_ids = set(merged_ids)
    for evidence in evidence_rows:
        clip_id = evidence.get("clip_id")
        if clip_id not in prefix_lookup:
            continue
        span = str(evidence.get("span") or "").strip()
        source_text = str(prefix_lookup[clip_id].get("audio_description") or "")
        if not span or span not in source_text:
            continue
        evidence_type = str(evidence.get("evidence_type") or "other").strip()
        if evidence_type not in ALLOWED_EVIDENCE_TYPES:
            evidence_type = "other"
        valid_evidence.append(
            {
                "clip_id": clip_id,
                "span": span,
                "evidence_type": evidence_type,
            }
        )
        if clip_id not in seen_ids:
            merged_ids.append(clip_id)
            seen_ids.add(clip_id)
    merged_context = [row for row in prefix_clips if row["clip_id"] in set(merged_ids)]
    ordered_ids = [row["clip_id"] for row in merged_context]
    return valid_evidence, ordered_ids, merged_context


def route_target_level_decision(result: dict[str, Any]) -> str:
    """Route a materialized 02b row into selected, challenge, or rejected."""

    if not target_level_passes_selection(result):
        if result.get("target_triviality") == "meaningful_but_noninferable" or result.get("predictability") == "underdetermined":
            return "challenge"
        return "rejected"
    predictability = result.get("predictability")
    if predictability == "underdetermined":
        return "challenge"
    if predictability == "unpredictable":
        return "rejected"
    return "selected"


def classify_search_block(
    block: dict[str, Any],
    *,
    qwen_client: QwenClient | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Ask Qwen to choose one valid target and its prior buildup from a search block."""

    if qwen_client is None or not qwen_client.is_enabled():
        return _materialize_search_block_result(
            block,
            _default_block_decision("qwen client unavailable", "no_qwen_available"),
        )

    prompt, video_paths = make_block_selection_video_prompt(block)
    try:
        if video_paths:
            qwen_output = qwen_client.generate_json_with_videos(
                "select_block_target_context",
                prompt,
                video_paths=video_paths,
                temperature=0.1,
            ).parsed
        else:
            qwen_output = qwen_client.generate_json("select_block_target_context", prompt, temperature=0.1).parsed
    except Exception as exc:
        qwen_output = _default_block_decision(
            f"qwen_parse_or_generation_error: {type(exc).__name__}",
            "qwen_block_search_error",
        )
    return _materialize_search_block_result(block, qwen_output, config=config)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match is None:
            return default
        try:
            return float(match.group(0))
        except ValueError:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0", ""}:
            return False
    return bool(value)


def _materialize_search_block_result(
    block: dict[str, Any],
    qwen_output: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clips = list(block.get("clips", []))
    clip_lookup = {row["clip_id"]: row for row in clips}
    clip_ids = [row["clip_id"] for row in clips]
    target_clip_id = qwen_output.get("target_clip_id")
    raw_target_position = qwen_output.get("target_position_in_block")
    target_position = int(raw_target_position) if raw_target_position not in {None, ""} else None

    if target_clip_id is None and target_position is not None and 1 <= target_position <= len(clips):
        target_clip_id = clips[target_position - 1]["clip_id"]
    if target_clip_id in clip_lookup:
        target_position = clip_ids.index(target_clip_id) + 1

    found_valid_target = _safe_bool(qwen_output.get("found_valid_target"), False)
    target = clip_lookup.get(target_clip_id) if target_clip_id is not None else None
    if target is None:
        found_valid_target = False
        target_position = None
        target_clip_id = None

    allowable_context_ids: set[str] = set()
    prefix_clips: list[dict[str, Any]] = []
    if target_position is not None:
        prefix_clips = clips[: target_position - 1]
        allowable_context_ids = {row["clip_id"] for row in prefix_clips}
    immediate_prior_clip_id = prefix_clips[-1]["clip_id"] if prefix_clips else None
    selected_ids = [clip_id for clip_id in qwen_output.get("selected_context_clip_ids", []) if clip_id in allowable_context_ids]
    raw_evidence = list(qwen_output.get("selected_context_spans", []))
    evidence, selected_ids, selected_context = _reconcile_evidence_with_context(
        evidence_rows=raw_evidence,
        prefix_clips=prefix_clips,
        selected_context_ids=selected_ids,
    )
    result = {
        "id": f"{block['id']}::{target_clip_id or 'no_target'}",
        "movie": block["movie"],
        "block_start_index": block["block_start_index"],
        "block_end_index": block["block_end_index"],
        "max_window_clips": block.get("max_window_clips", len(clips)),
        "full_block_clip_ids": clip_ids,
        "clips": clips,
        "target": target,
        "target_clip_id": target_clip_id,
        "target_sentence": target.get("audio_description") if target is not None else None,
        "target_type": qwen_output.get("target_type"),
        "target_triviality": qwen_output.get("target_triviality"),
        "target_position_in_block": target_position,
        "context": selected_context,
        "context_sentences": [row.get("audio_description") for row in selected_context],
        "full_prior_context_clip_ids": [row["clip_id"] for row in prefix_clips],
        "immediate_prior_clip_id": immediate_prior_clip_id,
        "selected_context_clip_ids": selected_ids,
        "selected_window_size": len(selected_ids),
        "context_selection_strategy": qwen_output.get("context_selection_strategy", "qwen_block_search"),
        "target_validity_reason": qwen_output.get("target_validity_reason"),
        "predictability": qwen_output.get("predictability", "unpredictable"),
        "expectedness": None,
        "evidence": evidence,
        "evidence_clip_ids": [
            clip_id
            for clip_id in qwen_output.get("evidence_clip_ids", [])
            if clip_id in allowable_context_ids
        ] or [row["clip_id"] for row in evidence],
        "evidence_sufficiency": _safe_float(qwen_output.get("evidence_sufficiency"), 0.0),
        "evidence_sufficiency_threshold": (config or {}).get("evidence_sufficiency_threshold", 0.0),
        "reasoning_type": normalize_reasoning_types(qwen_output.get("reasoning_type", ["other"])),
        "continuity_type": normalize_continuity_type(
            qwen_output.get("continuity_type"),
            qwen_output.get("reasoning_type", ["other"]),
        ),
        "should_keep": False,
        "rejection_reason": None,
        "found_valid_target": found_valid_target,
        "qwen_selection_output": qwen_output,
        "qwen_continuity_output": qwen_output,
    }
    if target is not None:
        features = build_continuity_features(prefix_clips, target)
        result.update(features)
    else:
        result.update(
            {
                "timestamp_gap": None,
                "semantic_similarity_last": 0.0,
                "semantic_similarity_mean": 0.0,
                "context_entities": [],
                "target_entities": [],
                "entity_overlap": 0.0,
                "action_overlap": 0.0,
                "location_overlap": 0.0,
                "continuity_scores": {},
            }
        )
    status = route_target_level_decision(result)
    result["should_keep"] = status == "selected"
    result["rejection_reason"] = None if status == "selected" else _normalize_rejection_reason(result, qwen_output)
    return result


def target_level_passes_selection(result: dict[str, Any]) -> bool:
    """Return whether a block-level Qwen decision is structurally valid."""

    return bool(
        result.get("found_valid_target") is True
        and result.get("target") is not None
        and bool(result.get("target_clip_id"))
        and result.get("target_type") in ALLOWED_TARGET_TYPES
        and result.get("target_triviality") == "nontrivial"
        and len(result.get("selected_context_clip_ids", [])) > 0
        and len(result.get("evidence", [])) > 0
        and result.get("evidence_sufficiency", 0.0) >= result.get("evidence_sufficiency_threshold", 0.0)
        and (
            not _target_needs_immediate_antecedent(result.get("target_sentence"))
            or result.get("immediate_prior_clip_id") in set(result.get("selected_context_clip_ids", []))
        )
        and all(
            clip["clip_id"] in set(result.get("full_prior_context_clip_ids", []))
            for clip in result.get("context", [])
        )
        and not _target_is_explicitly_invalid(result.get("qwen_selection_output", {}))
        and result.get("predictability") != "unpredictable"
    )


def select_valid_contexts_fixed_candidate_windows(
    windows: list[dict[str, Any]],
    *,
    qwen_client: QwenClient | None,
    config: dict[str, Any],
    progress_callback: callable | None = None,
) -> dict[str, Any]:
    """Legacy fixed-candidate selection kept for compatibility."""

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    challenge: list[dict[str, Any]] = []
    stats = Counter()
    for index, candidate in enumerate(windows, start=1):
        target = candidate["target"]
        features = build_continuity_features(candidate["context"], target)
        classified = candidate | features
        if qwen_client is None or not qwen_client.is_enabled():
            output = {
                "continuity_type": "discontinuous_unpredictable",
                "predictability": "unpredictable",
                "evidence": [],
                "evidence_sufficiency": 0.0,
                "reasoning_type": ["other"],
                "should_keep": False,
                "rejection_reason": "qwen client unavailable",
            }
        else:
            output = qwen_client.generate_json(
                "classify_continuity",
                make_continuity_prompt(classified),
                temperature=0.1,
            ).parsed
        row = classified | {
            "selected_window_size": len(candidate["context"]),
            "selected_context_clip_ids": [clip["clip_id"] for clip in candidate["context"]],
            "context_selection_strategy": "fixed_candidate_window",
            "rejection_reason": output.get("rejection_reason"),
            "qwen_continuity_output": output,
        }
        if progress_callback is not None:
            progress_callback("candidate_scored", candidate_index=index, target_clip_id=target["clip_id"])
        if (
            output.get("should_keep") is True
            and output.get("predictability") == "predictable"
            and output.get("continuity_type") in ALLOWED_CONTINUITY
            and float(output.get("evidence_sufficiency", 0.0)) >= config["evidence_sufficiency_threshold"]
        ):
            selected.append(row)
        elif output.get("predictability") == "underdetermined" or output.get("continuity_type") == "scene_cut_semantically_linked":
            challenge.append(row)
        else:
            rejected.append(row)
    stats["selected_targets"] = len(selected)
    stats["rejected_rows"] = len(rejected)
    stats["challenge_targets"] = len(challenge)
    return {"selected": selected, "rejected": rejected, "challenge": challenge, "stats": dict(stats)}


def iterate_target_level_context_decisions(
    windows: list[dict[str, Any]],
    *,
    qwen_client: QwenClient | None,
    config: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    """Yield one block-search decision at a time, honoring the target advancement rule."""

    blocks_by_movie: dict[str, dict[int, dict[str, Any]]] = {}
    for block in windows:
        blocks_by_movie.setdefault(block["movie"], {})[int(block["block_start_index"])] = block

    qwen_calls = 0
    max_qwen_calls = config.get("max_qwen_calls")
    for movie in sorted(blocks_by_movie):
        movie_blocks = blocks_by_movie[movie]
        if not movie_blocks:
            continue
        start_index = min(movie_blocks)
        while start_index in movie_blocks:
            block = movie_blocks[start_index]
            use_qwen = qwen_client if max_qwen_calls is None or qwen_calls < max_qwen_calls else None
            result = classify_search_block(block, qwen_client=use_qwen, config=config)
            if use_qwen is not None and result["context_selection_strategy"] == "qwen_block_search":
                qwen_calls += 1
            result["qwen_call_index"] = qwen_calls
            yield result

            if result.get("found_valid_target") and result.get("target_position_in_block") is not None:
                start_index = block["block_start_index"] + int(result["target_position_in_block"])
            else:
                start_index += 1
