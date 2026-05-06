"""Flexible LSMDC-style clip loader."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from forseebench.parsing.normalize_text import normalize_text


@dataclass(slots=True)
class ClipRecord:
    """A normalized clip-level record."""

    movie: str
    clip_id: str
    timestamp_start: str | None
    timestamp_end: str | None
    audio_description: str
    sequence_index: int = 0
    video_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_column(raw: dict[str, Any], *candidates: str, default: str | None = None) -> str | None:
    lower_map = {key.lower(): key for key in raw}
    for candidate in candidates:
        key = lower_map.get(candidate.lower())
        if key is not None:
            value = raw.get(key)
            return str(value).strip() if value not in {None, ""} else default
    return default


def _make_clip_id(raw: dict[str, Any], movie: str, sequence_index: int) -> str:
    """Return an explicit clip id or synthesize one for MAD-style rows."""

    clip_id = _find_column(raw, "clip_id", "clip", "id", "segment_id")
    if clip_id:
        return clip_id
    start = _find_column(raw, "timestamp_start", "start", "start_time")
    end = _find_column(raw, "timestamp_end", "end", "end_time")
    if start and end:
        safe_start = start.replace(".", "p")
        safe_end = end.replace(".", "p")
        return f"{movie}__{safe_start}_{safe_end}"
    return f"{movie}__{sequence_index:05d}"


def _normalize_row(raw: dict[str, Any], source_path: Path, sequence_index: int) -> ClipRecord | None:
    movie = _find_column(raw, "movie", "movie_id", "video_id", "imdb")
    description = _find_column(
        raw,
        "audio_description",
        "description",
        "sentence",
        "caption",
        "text",
        default="",
    )
    if not movie or not description:
        return None
    clip_id = _make_clip_id(raw, movie, sequence_index)
    return ClipRecord(
        movie=movie,
        clip_id=clip_id,
        timestamp_start=_find_column(raw, "timestamp_start", "start", "start_time"),
        timestamp_end=_find_column(raw, "timestamp_end", "end", "end_time"),
        audio_description=normalize_text(description),
        sequence_index=sequence_index,
        video_path=_find_column(raw, "video_path", "video"),
    )


def _iter_delimited_rows(path: Path, delimiter: str) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for raw in reader:
            yield raw


def _iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield payload


def load_lsmdc_records(input_path: str | Path) -> list[ClipRecord]:
    """Load clip rows from a file or directory and sort by movie and timestamp."""

    base = Path(input_path)
    files: list[Path]
    if base.is_dir():
        files = sorted(
            path for path in base.rglob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".jsonl"}
        )
    else:
        files = [base]

    records: list[ClipRecord] = []
    for path in files:
        if path.suffix.lower() == ".csv":
            iterator = _iter_delimited_rows(path, ",")
        elif path.suffix.lower() == ".tsv":
            iterator = _iter_delimited_rows(path, "\t")
        elif path.suffix.lower() == ".jsonl":
            iterator = _iter_jsonl_rows(path)
        else:
            continue
        for sequence_index, raw in enumerate(iterator):
            record = _normalize_row(raw, path, sequence_index)
            if record is not None:
                records.append(record)

    def sort_key(record: ClipRecord) -> tuple[str, float, int, str]:
        start = _safe_float(record.timestamp_start)
        return (record.movie, start, record.sequence_index, record.clip_id)

    return sorted(records, key=sort_key)


def _safe_float(value: str | None) -> float:
    if value is None:
        return float("inf")
    try:
        return float(value)
    except ValueError:
        return float("inf")


def group_by_movie(records: Iterable[ClipRecord]) -> dict[str, list[ClipRecord]]:
    """Group records by movie while preserving order."""

    grouped: dict[str, list[ClipRecord]] = {}
    for record in records:
        grouped.setdefault(record.movie, []).append(record)
    return grouped
