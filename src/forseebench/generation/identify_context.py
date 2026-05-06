"""Identify evidence and predictability."""

from __future__ import annotations

from typing import Any

from forseebench.qwen.prompts import make_identify_evidence_prompt
from forseebench.qwen.qwen_client import QwenClient


def identify_evidence(window: dict[str, Any], target_action: dict[str, Any], qwen_client: QwenClient) -> dict[str, Any]:
    """Identify supporting evidence, predictability, and continuity labels."""

    prompt = make_identify_evidence_prompt(window["context"], target_action)
    return qwen_client.generate_json("identify_evidence", prompt, temperature=0.1).parsed
