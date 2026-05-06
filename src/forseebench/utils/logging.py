"""Small logging helpers for scripts."""

from __future__ import annotations


def log(message: str) -> None:
    """Print a visible ForSeeBench-prefixed log line."""

    print(f"[ForSeeBench] {message}", flush=True)
