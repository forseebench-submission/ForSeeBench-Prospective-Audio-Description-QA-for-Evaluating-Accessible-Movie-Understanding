"""Text normalization helpers for LSMDC-like captions."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """Normalize free-form caption text into a stable single-line string."""

    if text is None:
        return ""
    cleaned = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\n", " ").replace("\r", " ").strip()
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned
