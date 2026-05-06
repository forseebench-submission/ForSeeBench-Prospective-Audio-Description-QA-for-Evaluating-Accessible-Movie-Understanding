"""JSONL writing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Any


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write JSONL records and return the number of rows written."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
