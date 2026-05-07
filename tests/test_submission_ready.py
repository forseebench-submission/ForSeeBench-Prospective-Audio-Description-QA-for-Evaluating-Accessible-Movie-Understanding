from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_ready_checker_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_submission_ready.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Root README" in result.stdout
    assert "Hugging Face release files" in result.stdout
    assert "Raw media scan" in result.stdout
    assert "HF card split framing" in result.stdout
