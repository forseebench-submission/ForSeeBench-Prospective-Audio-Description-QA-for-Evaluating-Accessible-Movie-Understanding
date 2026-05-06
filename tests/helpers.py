from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FakeCallResult:
    prompt_name: str
    raw_text: str
    parsed: dict[str, Any]


class FakeQwenClient:
    """Small test-only stand-in for deterministic pipeline unit tests."""

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.calls_made = 0

    def is_enabled(self) -> bool:
        return True

    def generate_json(self, prompt_name: str, prompt: str, *, temperature: float | None = None) -> FakeCallResult:
        self.calls_made += 1
        if prompt_name not in self.responses:
            raise KeyError(f"no fake response configured for {prompt_name}")
        payload = self.responses[prompt_name]
        if isinstance(payload, list):
            if not payload:
                raise KeyError(f"no remaining fake responses configured for {prompt_name}")
            payload = payload.pop(0)
        return FakeCallResult(prompt_name=prompt_name, raw_text=str(payload), parsed=payload)
