"""Helpers for matching MAD clip rows to local video files."""

from __future__ import annotations

from pathlib import Path


def format_mad_time(seconds: float) -> str:
    """Format seconds as the MAD clip filename timestamp HH.MM.SS.mmm."""

    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}.{minutes:02d}.{secs:02d}.{millis:03d}"


def find_mad_video_path(
    *,
    video_root: str | Path,
    movie: str,
    timestamp_start: str | None,
    timestamp_end: str | None,
) -> str | None:
    """Return the expected MAD clip video path if it exists."""

    if timestamp_start is None or timestamp_end is None:
        return None
    try:
        start = format_mad_time(float(timestamp_start))
        end = format_mad_time(float(timestamp_end))
    except ValueError:
        return None
    path = Path(video_root) / movie / f"{movie}_{start}-{end}.avi"
    return str(path) if path.exists() else None
