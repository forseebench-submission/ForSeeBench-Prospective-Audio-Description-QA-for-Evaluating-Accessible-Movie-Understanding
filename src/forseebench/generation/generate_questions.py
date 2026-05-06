"""Generate multiple-choice question variants."""

from __future__ import annotations

from typing import Any

from forseebench.generation.normalization import normalize_question_type
from forseebench.qwen.prompts import make_question_prompt
from forseebench.qwen.qwen_client import QwenClient
from forseebench.utils.schema import DISTRACTOR_TYPES


def _normalize_answer_idx(value: Any, options: list[Any], parsed: dict[str, Any]) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 0 <= value < 4 else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            idx = int(text)
            return idx if 0 <= idx < 4 else None
        if len(text) == 1 and text.upper() in {"A", "B", "C", "D"}:
            return ord(text.upper()) - ord("A")
        for idx, option in enumerate(options):
            if text == str(option).strip():
                return idx
    answer = parsed.get("answer")
    if answer is not None and answer is not value:
        return _normalize_answer_idx(answer, options, {"answer": None})
    answer_text = parsed.get("answer_text")
    if answer_text is not None and answer_text is not value:
        return _normalize_answer_idx(answer_text, options, {"answer_text": None})
    return None


def _normalize_question_block(parsed: dict[str, Any], target_action: dict[str, Any]) -> dict[str, Any]:
    question_type = parsed.get("question_type")
    options = parsed.get("options")
    distractor_metadata = parsed.get("distractor_metadata")
    answer_idx = parsed.get("answer_idx")

    if isinstance(parsed.get("choices"), dict):
        choice_map = parsed["choices"]
        ordered_keys = sorted(choice_map.keys())
        options = [choice_map[key] for key in ordered_keys]
        answer_key = parsed.get("answer", "A")
        answer_idx = max(0, ordered_keys.index(answer_key)) if answer_key in ordered_keys else 0
        if isinstance(distractor_metadata, dict):
            distractor_metadata = [distractor_metadata.get(key, "plausible_unsupported") for key in ordered_keys]

    if not isinstance(options, list) or len(options) != 4:
        raise ValueError("generate_question must return exactly four options")
    answer_idx = _normalize_answer_idx(answer_idx, options, parsed)
    if answer_idx is None:
        raise ValueError("generate_question must return answer_idx in [0, 3]")
    if not isinstance(distractor_metadata, list) or len(distractor_metadata) != 4:
        distractor_metadata = [
            "correct" if idx == answer_idx else "plausible_unsupported"
            for idx in range(4)
        ]
    else:
        distractor_metadata = [
            label if label in DISTRACTOR_TYPES else "plausible_unsupported"
            for label in distractor_metadata
        ]
        distractor_metadata = [
            "correct" if idx == answer_idx else ("plausible_unsupported" if label == "correct" else label)
            for idx, label in enumerate(distractor_metadata)
        ]
    question_type = normalize_question_type(question_type, target_action.get("target_type"))
    return {
        "question_type": question_type,
        "question": parsed["question"],
        "options": options,
        "answer_idx": answer_idx,
        "answer_text": options[answer_idx],
        "distractor_metadata": distractor_metadata,
    }


def generate_question(
    window: dict[str, Any],
    target_action: dict[str, Any],
    evidence: list[dict[str, Any]],
    qwen_client: QwenClient,
) -> dict[str, Any]:
    """Generate a multiple-choice question and distractors."""

    prompt = make_question_prompt(window["context"], target_action, evidence)
    parsed = qwen_client.generate_json("generate_question", prompt, temperature=0.4).parsed
    return _normalize_question_block(parsed, target_action)
