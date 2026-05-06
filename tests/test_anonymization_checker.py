from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_anonymization_checker_warns_only() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_anonymization.py", "README.md", "hf_dataset", "scripts"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "anonymization" in result.stdout
