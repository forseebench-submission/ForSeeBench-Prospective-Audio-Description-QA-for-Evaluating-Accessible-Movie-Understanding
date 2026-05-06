"""Run Qwen-assisted target-update extraction."""

from __future__ import annotations

from typing import Any

from forseebench.qwen.prompts import make_extract_action_prompt, make_score_expectedness_prompt
from forseebench.qwen.qwen_client import QwenClient


def _normalize_expectedness(value: Any) -> float | None:
    """Return a nullable numeric expectedness score from Qwen output."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def extract_target_action(window: dict[str, Any], qwen_client: QwenClient) -> dict[str, Any]:
    """Extract a structured target update from the target clip."""

    prompt = make_extract_action_prompt(window["context"], window["target"])
    result = qwen_client.generate_json("extract_target_action", prompt, temperature=0.1)
    payload = {
        "normalized_target": result.parsed.get(
            "normalized_target",
            window["target"]["audio_description"],
        )
    }
    payload["raw_description"] = window["target"]["audio_description"]
    payload["target_sentence"] = window["target"]["audio_description"]
    payload["target_type"] = window.get("target_type", "action_transition")
    expectedness_prompt = make_score_expectedness_prompt(window["context"], payload)
    expectedness_result = qwen_client.generate_json("score_expectedness", expectedness_prompt, temperature=0.1)
    expectedness = _normalize_expectedness(expectedness_result.parsed.get("expectedness"))
    warning = None
    if expectedness is None:
        warning = "invalid_expectedness_output"
    else:
        expectedness = max(0.0, min(1.0, expectedness))
    payload["expectedness"] = expectedness
    if warning is not None:
        payload["expectedness_warning"] = warning
    return payload
