from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from forseebench.io.write_jsonl import read_jsonl
from forseebench.utils.schema import validate_release_public_example, validate_release_with_answers_example


ROOT = Path(__file__).resolve().parents[1]


def test_sample_with_answers_and_public_rows_validate() -> None:
    answer_rows = read_jsonl(ROOT / "hf_dataset/sample_data/sample_with_answers.jsonl")
    public_rows = read_jsonl(ROOT / "hf_dataset/sample_data/sample_public.jsonl")

    assert len(answer_rows) == 2
    assert len(public_rows) == 2
    assert all(validate_release_with_answers_example(row) == [] for row in answer_rows)
    assert all(validate_release_public_example(row) == [] for row in public_rows)


def test_full_qna_release_rows_validate() -> None:
    answer_rows = read_jsonl(ROOT / "hf_dataset/data/qna_with_answers.jsonl")
    public_rows = read_jsonl(ROOT / "hf_dataset/data/qna_test.jsonl")

    assert len(answer_rows) == 787
    assert len(public_rows) == 787
    assert all(validate_release_with_answers_example(row) == [] for row in answer_rows)
    assert all(validate_release_public_example(row) == [] for row in public_rows)


def test_validate_dataset_cli_default_full_qna() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_dataset.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "PASS" in result.stdout
