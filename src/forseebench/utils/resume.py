"""Helpers for resumable pipeline scripts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


def file_has_content(path: str | Path) -> bool:
    """Return whether a file exists and is non-empty."""

    candidate = Path(path)
    return candidate.exists() and candidate.stat().st_size > 0


def load_existing_ids(path: str | Path, *, id_field: str = "id") -> set[str]:
    """Load existing row ids from a JSONL file."""

    candidate = Path(path)
    if not candidate.exists():
        return set()
    ids: set[str] = set()
    with candidate.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            value = payload.get(id_field)
            if isinstance(value, str):
                ids.add(value)
    return ids


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append JSONL rows and return how many were written."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def reset_file(path: str | Path) -> None:
    """Create or truncate a file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")


def write_progress(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a structured progress JSON payload."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def progress_payload(**kwargs: Any) -> dict[str, Any]:
    """Attach a standard UTC update timestamp to a progress payload."""

    return kwargs | {"updated_at": datetime.now(timezone.utc).isoformat()}
