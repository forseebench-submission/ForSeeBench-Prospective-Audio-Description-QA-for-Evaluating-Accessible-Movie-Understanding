from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_analysis_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_expectedness.py"
    spec = importlib.util.spec_from_file_location("analyze_expectedness", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_expectedness_reports_expected_buckets() -> None:
    module = _load_analysis_module()
    rows = [
        {"id": "a", "expectedness": 0.9, "reasoning_type": ["motion_trajectory"]},
        {"id": "b", "expectedness": 0.6, "reasoning_type": ["motion_trajectory", "object_affordance"]},
        {"id": "c", "expectedness": 0.2, "reasoning_type": ["other"]},
        {"id": "d", "expectedness": None, "reasoning_type": ["other"]},
    ]

    summary = module.summarize_expectedness(rows)

    assert summary["count"] == 4
    assert summary["mean_expectedness"] == (0.9 + 0.6 + 0.2) / 3
    assert summary["median_expectedness"] == 0.6
    assert summary["min_expectedness"] == 0.2
    assert summary["max_expectedness"] == 0.9
    assert summary["above_0_8"] == 1
    assert summary["between_0_4_and_0_8"] == 1
    assert summary["below_0_4"] == 1
    assert summary["null_count"] == 1
    assert summary["by_reasoning_type"]["motion_trajectory"]["count"] == 2
    assert summary["by_reasoning_type"]["motion_trajectory"]["mean"] == 0.75
    assert summary["by_reasoning_type"]["other"]["null"] == 1
