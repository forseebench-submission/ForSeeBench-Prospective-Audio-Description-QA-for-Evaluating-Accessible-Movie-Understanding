from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validate_dataset_public_sample_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_dataset.py",
            "--input",
            "hf_dataset/sample_data/sample_public.jsonl",
            "--schema",
            "public",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "PASS" in result.stdout
    assert "public schema" in result.stdout


def test_validate_dataset_with_answers_sample_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_dataset.py",
            "--input",
            "hf_dataset/sample_data/sample_with_answers.jsonl",
            "--schema",
            "with_answers",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "PASS" in result.stdout
    assert "with_answers schema" in result.stdout


def test_validate_dataset_missing_file_error() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_dataset.py", "--input", "hf_dataset/sample_data/does_not_exist.jsonl"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "file does not exist" in result.stdout
