from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validate_dataset_public_qna_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_dataset.py",
            "--input",
            "hf_dataset/data/qna_test.jsonl",
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


def test_validate_dataset_with_answers_qna_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_dataset.py",
            "--input",
            "hf_dataset/data/qna_with_answers.jsonl",
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


def test_validate_dataset_full_qna_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_dataset.py",
            "--input",
            "hf_dataset/data/qna_test.jsonl",
            "--schema",
            "public",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "787 rows" in result.stdout


def test_validate_dataset_missing_file_error() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_dataset.py", "--input", "hf_dataset/data/does_not_exist.jsonl"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "file does not exist" in result.stdout
