from __future__ import annotations

from pathlib import Path

from forseebench.io.load_lsmdc import load_lsmdc_records
from forseebench.parsing.build_temporal_windows import build_search_blocks


def test_build_search_blocks_respects_movie_boundaries() -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample_lsmdc.csv"
    records = load_lsmdc_records(fixture)
    blocks = build_search_blocks(records, max_window_clips=3)

    assert len(blocks) == 8
    assert blocks[0].movie == "MovieA"
    assert blocks[0].block_start_index == 0
    assert blocks[0].block_end_index == 2
    assert [row.clip_id for row in blocks[0].clips] == ["a1", "a2", "a3"]

    movie_b_blocks = [block for block in blocks if block.movie == "MovieB"]
    assert movie_b_blocks[0].block_start_index == 0
    assert [row.clip_id for row in movie_b_blocks[0].clips] == ["b1", "b2", "b3"]
