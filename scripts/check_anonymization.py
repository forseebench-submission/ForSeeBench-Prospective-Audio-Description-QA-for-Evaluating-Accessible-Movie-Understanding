#!/usr/bin/env python3
"""Warn about obvious anonymization risks in release-facing files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (
    "README.md",
    "docs",
    "scripts",
    "src",
    "tests",
    "configs",
    "hf_dataset",
    "paper",
    "agent",
)
SKIP_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}
TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".sty",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class Pattern:
    label: str
    regex: re.Pattern[str]


RISK_PATTERNS = (
    Pattern("private path", re.compile(r"(/jumbo/|/thayerfs/|/Users/|/home/[A-Za-z0-9_.-]+)", re.IGNORECASE)),
    Pattern("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    Pattern("institution/lab string", re.compile(r"\b(jinlab|dartmouth|thayer|tufts|stanford|mit|berkeley|cmu|cornell)\b", re.IGNORECASE)),
    Pattern("api key wording", re.compile(r"\b(api[_-]?key|secret[_-]?key|access[_-]?token|hf_token|openai_api_key)\b", re.IGNORECASE)),
    Pattern("personal github/hf name", re.compile(r"\bgithub\.com/(?!forseebench-submission\b|huggingface\b)[A-Za-z0-9_.-]+", re.IGNORECASE)),
    Pattern("personal github/hf name", re.compile(r"\bhuggingface\.co/(?!api/|datasets/forseebench\b|forseebench\b)[A-Za-z0-9_.-]+", re.IGNORECASE)),
)


def is_scannable(path: Path) -> bool:
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return path.name in {"README", "LICENSE"}


def iter_files(paths: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = ROOT / raw
        if path.is_file() and is_scannable(path):
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(item for item in path.rglob("*") if item.is_file() and is_scannable(item)))
    return files


def line_matches(text: str, pattern: re.Pattern[str]) -> tuple[int, str] | None:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if "Pattern(" in line:
            continue
        if pattern.search(line):
            return line_number, line.strip()[:180]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=list(DEFAULT_PATHS), help="Files/directories to scan.")
    parser.add_argument("--max-findings", type=int, default=80, help="Maximum warnings to print.")
    args = parser.parse_args()

    findings: list[str] = []
    for path in iter_files(tuple(args.paths)):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(ROOT)
        for pattern in RISK_PATTERNS:
            match = line_matches(text, pattern.regex)
            if match is not None:
                line_number, excerpt = match
                findings.append(f"{rel}:{line_number}: {pattern.label}: {excerpt}")

    if findings:
        print(f"PARTIAL anonymization: {len(findings)} potential issue(s) found.")
        for finding in findings[: args.max_findings]:
            print(f"  - {finding}")
        if len(findings) > args.max_findings:
            print(f"  - ... {len(findings) - args.max_findings} more")
        print("Warnings only: review and scrub/exclude these before creating the anonymous mirror.")
        return 0

    print("PASS anonymization: no obvious release-facing identity/path patterns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
