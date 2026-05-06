"""Build ordered max-lookahead search blocks from clip rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from forseebench.io.load_lsmdc import ClipRecord, group_by_movie


@dataclass(slots=True)
class SearchBlock:
    """One ordered search region that Qwen may inspect."""

    id: str
    movie: str
    block_start_index: int
    block_end_index: int
    clips: list[ClipRecord]
    max_window_clips: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "movie": self.movie,
            "block_start_index": self.block_start_index,
            "block_end_index": self.block_end_index,
            "clips": [asdict(row) for row in self.clips],
            "max_window_clips": self.max_window_clips,
        }


def build_temporal_windows(
    records: Iterable[ClipRecord],
    *,
    window_size: int = 3,
    max_windows: int | None = None,
    min_context_chars: int = 20,
) -> list[SearchBlock]:
    """Backward-compatible wrapper for building search blocks."""

    return build_search_blocks(
        records,
        max_window_clips=window_size,
        max_blocks=max_windows,
        min_block_chars=min_context_chars,
    )


def build_search_blocks(
    records: Iterable[ClipRecord],
    *,
    max_window_clips: int = 3,
    max_blocks: int | None = None,
    min_block_chars: int = 20,
) -> list[SearchBlock]:
    """Build sliding max-lookahead blocks per movie."""

    if max_window_clips < 1:
        raise ValueError("max_window_clips must be >= 1")

    grouped = group_by_movie(records)
    blocks: list[SearchBlock] = []
    for movie, movie_rows in grouped.items():
        for start_idx in range(len(movie_rows)):
            clips = movie_rows[start_idx : start_idx + max_window_clips]
            if not clips:
                continue
            block_text = " ".join(row.audio_description for row in clips).strip()
            if len(block_text) < min_block_chars:
                continue
            block = SearchBlock(
                id=f"lsmdc::{movie}::block::{start_idx:05d}_{start_idx + len(clips) - 1:05d}::w{max_window_clips}",
                movie=movie,
                block_start_index=start_idx,
                block_end_index=start_idx + len(clips) - 1,
                clips=clips,
                max_window_clips=max_window_clips,
            )
            blocks.append(block)
            if max_blocks is not None and len(blocks) >= max_blocks:
                return blocks
    return blocks
