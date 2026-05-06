"""Validation helpers for candidate examples."""

from __future__ import annotations

from typing import Any

from forseebench.generation.normalization import normalize_candidate_example
from forseebench.qwen.prompts import make_validate_prompt
from forseebench.qwen.qwen_client import QwenClient
from forseebench.utils.schema import validate_example


def validate_candidate(example: dict[str, Any], qwen_client: QwenClient | None) -> dict[str, Any]:
    """Run schema checks and optional Qwen validation."""

    normalize_candidate_example(example)
    schema_errors = validate_example(example)
    if schema_errors:
        return {
            "should_keep": False,
            "qwen_confidence": 0.0,
            "evidence_sufficiency": 0.0,
            "distractor_quality": 0.0,
            "failure_reasons": schema_errors,
            "recommended_fix": "Fix schema violations before validation.",
        }
    if qwen_client is None or not qwen_client.is_enabled():
        quality = example["quality"]
        return {
            "should_keep": quality.get("should_keep", False),
            "qwen_confidence": float(quality.get("qwen_confidence", 0.0)),
            "evidence_sufficiency": float(quality.get("evidence_sufficiency", 0.0)),
            "distractor_quality": float(quality.get("distractor_quality", 0.0)),
            "failure_reasons": [],
            "recommended_fix": "",
        }
    prompt = make_validate_prompt(example)
    return qwen_client.generate_json("validate_example", prompt, temperature=0.1).parsed
