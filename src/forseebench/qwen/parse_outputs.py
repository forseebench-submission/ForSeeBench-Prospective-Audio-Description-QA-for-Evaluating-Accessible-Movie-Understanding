"""Robust JSON extraction for Qwen outputs."""

from __future__ import annotations

import json
from json import JSONDecodeError
import re
from typing import Any


def extract_json_candidates(text: str) -> list[str]:
    """Extract likely JSON object substrings from free-form model text."""

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates: list[str] = []
    if cleaned.startswith("{") and cleaned.endswith("}"):
        candidates.append(cleaned)

    start = -1
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(cleaned):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(cleaned[start : index + 1])
                start = -1

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=len, reverse=True):
        stripped = candidate.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            deduped.append(stripped)
    return deduped


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first valid top-level JSON object from text."""

    if not text or not text.strip():
        raise ValueError("empty model output")
    errors: list[str] = []
    for candidate in extract_json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except JSONDecodeError as exc:
            errors.append(f"{exc.__class__.__name__}: {exc}")
            continue
        if not isinstance(parsed, dict):
            errors.append(f"top-level JSON was {type(parsed).__name__}")
            continue
        return parsed
    if errors:
        raise ValueError("failed to parse model JSON: " + "; ".join(errors))
    raise ValueError("no JSON object found in model output")
