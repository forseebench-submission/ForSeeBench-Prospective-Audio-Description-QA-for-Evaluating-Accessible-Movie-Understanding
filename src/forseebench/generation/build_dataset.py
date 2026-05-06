"""Assemble candidate benchmark examples and dataset splits."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from forseebench.generation.extract_actions import extract_target_action
from forseebench.generation.generate_questions import generate_question
from forseebench.generation.identify_context import identify_evidence
from forseebench.generation.normalization import (
    normalize_bool,
    normalize_candidate_example,
    normalize_continuity_type,
    normalize_evidence_rows,
    normalize_float,
    normalize_reasoning_types,
    normalize_validation_payload,
)
from forseebench.generation.validate_examples import validate_candidate
from forseebench.qwen.qwen_client import QwenClient
from forseebench.utils.schema import TARGET_TYPE_VALUES


def _build_selection_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_triviality": row.get("target_triviality"),
        "target_validity_reason": row.get("target_validity_reason"),
        "selected_window_size": row.get("selected_window_size", row.get("window_size", len(row.get("context", [])))),
    }


def _build_validation_metadata(row: dict[str, Any], distractor_quality: float = 0.0) -> dict[str, Any]:
    return {
        "oracle_pass": None,
        "grounding_pass": None,
        "distractor_quality": distractor_quality,
    }


def build_action_extraction_records(
    windows: list[dict[str, Any]],
    *,
    qwen_client: QwenClient,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Extract target actions for each temporal window."""

    records: list[dict[str, Any]] = []
    for window in windows[:limit]:
        target_action = extract_target_action(window, qwen_client)
        row = window | {
            "expectedness": target_action.get("expectedness"),
            "target_action": target_action,
        }
        if target_action.get("expectedness_warning") is not None:
            row["expectedness_warning"] = target_action["expectedness_warning"]
        records.append(row)
    return records


def build_candidate_examples(
    extraction_records: list[dict[str, Any]],
    *,
    qwen_client: QwenClient,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert extracted windows into candidate multiple-choice examples."""

    examples: list[dict[str, Any]] = []
    for row in extraction_records[:limit]:
        evidence_block = row.get("qwen_selection_output") or row.get("qwen_continuity_output")
        if evidence_block is None:
            evidence_block = identify_evidence(row, row["target_action"], qwen_client)
        evidence_block = {
            **evidence_block,
            "predictability": row.get("predictability", evidence_block.get("predictability", "unpredictable")),
            "continuity_type": row.get("continuity_type", evidence_block.get("continuity_type")),
            "evidence": row.get("evidence", evidence_block.get("evidence", evidence_block.get("selected_context_spans", []))),
            "evidence_sufficiency": row.get("evidence_sufficiency", evidence_block.get("evidence_sufficiency", 0.0)),
            "reasoning_type": row.get("reasoning_type", evidence_block.get("reasoning_type", ["other"])),
            "should_keep": row.get("should_keep", evidence_block.get("should_keep", False)),
        }
        evidence_rows = normalize_evidence_rows(
            evidence_block.get("evidence", evidence_block.get("selected_context_spans", []))
        )
        reasoning_types = normalize_reasoning_types(evidence_block.get("reasoning_type", ["other"]))
        continuity_type = normalize_continuity_type(evidence_block.get("continuity_type"), reasoning_types)
        question_block = generate_question(
            row,
            row["target_action"],
            evidence_rows,
            qwen_client,
        )
        example = {
            "id": row["id"],
            "movie": row["movie"],
            "target_clip_id": row["target_clip_id"],
            "target_sentence": row["target_action"].get("target_sentence", row["target_action"]["raw_description"]),
            "context_clip_ids": [clip["clip_id"] for clip in row["context"]],
            "context_sentences": [clip.get("audio_description") for clip in row["context"]],
            "question_type": question_block["question_type"],
            "question": question_block["question"],
            "options": question_block["options"],
            "answer_idx": question_block["answer_idx"],
            "answer_text": question_block["answer_text"],
            "predictability": evidence_block.get("predictability", "unpredictable"),
            "expectedness": row.get("expectedness"),
            "target_type": row["target_action"].get("target_type", row.get("target_type")),
            "evidence_clip_ids": row.get("evidence_clip_ids")
            or [item["clip_id"] for item in evidence_block.get("selected_context_spans", [])],
            "selection_metadata": _build_selection_metadata(row),
            "validation_metadata": _build_validation_metadata(row),
            "distractor_metadata": question_block["distractor_metadata"],
            "split": None,
            "context": row["context"],
            "target": row["target"],
            "evidence": evidence_rows,
            "reasoning_type": reasoning_types,
            "continuity_type": continuity_type,
            "continuity_scores": row.get("continuity_scores", {}),
            "semantic_similarity_last": row.get("semantic_similarity_last", 0.0),
            "semantic_similarity_mean": row.get("semantic_similarity_mean", 0.0),
            "timestamp_gap": row.get("timestamp_gap"),
            "entity_overlap": row.get("entity_overlap", 0.0),
            "action_overlap": row.get("action_overlap", 0.0),
            "location_overlap": row.get("location_overlap", 0.0),
            "selected_window_size": row.get("selected_window_size", row.get("window_size", len(row["context"]))),
            "context_selection_strategy": row.get("context_selection_strategy", "qwen_block_search"),
            "full_prior_context_clip_ids": row.get("full_block_clip_ids", [clip["clip_id"] for clip in row["context"]]),
            "selected_context_clip_ids": row.get("selected_context_clip_ids", [clip["clip_id"] for clip in row["context"]]),
            "rejection_reason": row.get("rejection_reason"),
            "qwen_selection_output": evidence_block,
            "qwen_continuity_output": row.get("qwen_continuity_output", evidence_block),
            "quality": {
                "qwen_confidence": 0.0,
                "evidence_sufficiency": normalize_float(evidence_block.get("evidence_sufficiency"), 0.0),
                "distractor_quality": 0.0,
                "should_keep": normalize_bool(
                    evidence_block.get("should_keep_for_main_benchmark", evidence_block.get("should_keep", False))
                ),
            },
        }
        examples.append(normalize_candidate_example(example))
    return examples


def candidate_passes_main_gate(example: dict[str, Any], *, quality_pass: bool, expectedness_pass: bool) -> bool:
    """Return whether a candidate belongs in the main benchmark."""

    selection_metadata = example.get("selection_metadata", {})
    return bool(
        example.get("target_type") in TARGET_TYPE_VALUES
        and selection_metadata.get("target_triviality") == "nontrivial"
        and example.get("predictability") == "predictable"
        and len(example.get("evidence", [])) > 0
        and len(example.get("evidence_clip_ids", [])) > 0
        and not example.get("validation", {}).get("failure_reasons", [])
        and quality_pass
        and expectedness_pass
        and example.get("quality", {}).get("should_keep") is True
    )


def candidate_routes_to_challenge(example: dict[str, Any]) -> bool:
    """Return whether a non-kept example is useful for challenge/audit splits."""

    return bool(
        example.get("predictability") in {"underdetermined", "unpredictable"}
        or example.get("continuity_type") == "scene_cut_semantically_linked"
        or example.get("selection_metadata", {}).get("target_triviality") == "meaningful_but_noninferable"
    )


def validate_and_filter_examples(
    candidates: list[dict[str, Any]],
    *,
    qwen_client: QwenClient | None,
    thresholds: dict[str, float],
    max_kept: int | None = None,
) -> dict[str, Any]:
    """Validate and split candidate examples into kept, challenge, and rejected."""

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    challenge: list[dict[str, Any]] = []
    stats = Counter()

    for example in candidates:
        normalize_candidate_example(example)
        validation = normalize_validation_payload(
            validate_candidate(example, qwen_client),
            existing_quality=example.get("quality", {}),
        )
        example["quality"].update(
            {
                "qwen_confidence": normalize_float(validation.get("qwen_confidence"), 0.0),
                "evidence_sufficiency": normalize_float(validation.get("evidence_sufficiency"), 0.0),
                "distractor_quality": normalize_float(validation.get("distractor_quality"), 0.0),
                "should_keep": normalize_bool(validation.get("should_keep"), False),
            }
        )
        example["validation_metadata"] = {
            "oracle_pass": normalize_bool(validation.get("should_keep"), False) and "multi_answer" not in validation.get("failure_reasons", []),
            "grounding_pass": "context does not support" not in " ".join(validation.get("failure_reasons", [])),
            "distractor_quality": normalize_float(validation.get("distractor_quality"), 0.0),
        }
        example["validation"] = validation
        quality_pass = (
            example["quality"]["qwen_confidence"] >= thresholds["qwen_confidence"]
            and example["quality"]["evidence_sufficiency"] >= thresholds["evidence_sufficiency"]
            and example["quality"]["distractor_quality"] >= thresholds["distractor_quality"]
        )
        expectedness_filter_enabled = thresholds.get("filter_by_expectedness", False)
        expectedness_pass = True
        if expectedness_filter_enabled:
            expectedness = example.get("expectedness")
            expectedness_pass = (
                expectedness is not None
                and thresholds["expectedness_main_min"] <= float(expectedness) <= thresholds["expectedness_main_max"]
            )
        main_benchmark_pass = candidate_passes_main_gate(
            example,
            quality_pass=quality_pass,
            expectedness_pass=expectedness_pass,
        )
        if main_benchmark_pass:
            kept.append(example)
            stats["kept"] += 1
        elif candidate_routes_to_challenge(example):
            challenge.append(example)
            stats["challenge"] += 1
        else:
            rejected.append(example)
            stats["rejected"] += 1
    overflow: list[dict[str, Any]] = []
    if max_kept is not None and len(kept) > max_kept:
        overflow = kept[max_kept:]
        kept = kept[:max_kept]
        for example in overflow:
            example.setdefault("validation", {})
            example["validation"]["failure_reasons"] = list(example["validation"].get("failure_reasons", [])) + [
                "overflow beyond pilot max_examples"
            ]
        rejected.extend(overflow)
        stats["pilot_overflow"] = len(overflow)
        stats["rejected"] += len(overflow)
        stats["kept"] = len(kept)
    return {
        "kept": kept,
        "challenge": challenge,
        "rejected": rejected,
        "stats": dict(stats),
    }


def split_examples_by_movie(
    examples: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    """Split examples by movie without leakage."""

    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("split ratios must sum to 1.0")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[example["movie"]].append(example)

    movie_sizes = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    total = len(examples)
    targets = {
        "train": total * train_ratio,
        "val": total * val_ratio,
        "test": total * test_ratio,
    }
    assignments = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for movie, movie_examples in movie_sizes:
        split = min(targets, key=lambda name: (counts[name] / targets[name] if targets[name] else 1.0, counts[name]))
        assignments[split].extend(movie_examples)
        counts[split] += len(movie_examples)
    return assignments


def make_public_row(example: dict[str, Any], split_name: str) -> dict[str, Any]:
    return {
        "id": example["id"],
        "movie": example["movie"],
        "target_clip_id": example["target_clip_id"],
        "target_sentence": example["target_sentence"],
        "context_clip_ids": list(example["context_clip_ids"]),
        "context_sentences": list(example["context_sentences"]),
        "question_type": example["question_type"],
        "question": example["question"],
        "options": list(example["options"]),
        "predictability": example["predictability"],
        "expectedness": example["expectedness"],
        "target_type": example["target_type"],
        "evidence_clip_ids": list(example["evidence_clip_ids"]),
        "split": f"{split_name}_public",
    }


def make_private_row(example: dict[str, Any], split_name: str) -> dict[str, Any]:
    row = make_public_row(example, split_name)
    row.update(
        {
            "answer_idx": example["answer_idx"],
            "answer_text": example["answer_text"],
            "answer_source": {
                "source": "target_sentence",
                "target_clip_id": example["target_clip_id"],
                "target_sentence": example["target_sentence"],
            },
            "split": f"{split_name}_private",
        }
    )
    return row
