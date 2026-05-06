#!/usr/bin/env python3
"""Prepare a clean anonymous GitHub release directory without uploading."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT.parent / "forseebench_anonymous_release"

RAW_MEDIA_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".flac"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar"}
LARGE_FILE_BYTES = 25 * 1024 * 1024
EXCLUDED_NAMES = {
    ".git",
    ".env",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "temp",
    "tmp",
}
EXCLUDED_SUFFIXES = RAW_MEDIA_EXTENSIONS | ARCHIVE_EXTENSIONS | {".pyc", ".pyo", ".log", ".aux", ".blg", ".bbl", ".fls", ".fdb_latexmk", ".out", ".pdf"}
TEXT_SUFFIXES = {".bib", ".cfg", ".csv", ".json", ".jsonl", ".md", ".py", ".sh", ".tex", ".txt", ".yaml", ".yml"}
PRIVATE_TEXT_PATTERNS = (
    "/jumbo/",
    "/thayerfs/",
    "/Users/",
    "openai_api_key",
    "api_key:",
    "access_token",
    "secret_key",
)
GUARD_SCRIPT_FILES = {
    "scripts/check_anonymization.py",
    "scripts/prepare_anonymous_release.py",
    "scripts/upload_hf_dataset.py",
}

ALLOWED_TOP_LEVEL_FILES = (
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "LICENSE",
    "LICENSE.md",
    "CITATION.cff",
)
ALLOWED_DIRS = (
    "configs",
    "src",
    "scripts",
    "tests",
)
ALLOWED_HF_FILES = (
    "hf_dataset/README.md",
    "hf_dataset/schema.md",
    "hf_dataset/croissant_rai_notes.md",
    "hf_dataset/croissant_metadata_draft.json",
)
ALLOWED_HF_DIRS = (
    "hf_dataset/data",
    "hf_dataset/sample_data",
)
ALLOWED_AGENT_FILES = (
    "agent/submission_readiness_audit.md",
    "agent/neurips_ed_compliance_checklist.md",
    "agent/source_dataset_citation_audit.md",
    "agent/final_submission_readiness_report.md",
)


@dataclass
class CopySummary:
    copied: list[str]
    skipped: list[str]


def has_excluded_part(path: Path) -> bool:
    return any(part in EXCLUDED_NAMES for part in path.parts)


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {"README", "LICENSE"}


def contains_private_text(path: Path) -> str | None:
    rel = path.relative_to(ROOT)
    if str(rel) in GUARD_SCRIPT_FILES:
        return None
    if not is_text_file(path):
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "non-utf8 text-like file"
    for line in text.splitlines():
        lowered = line.lower()
        if "todo(author)" in lowered or "private path" in lowered or "raw source" in lowered:
            continue
        for pattern in PRIVATE_TEXT_PATTERNS:
            if pattern.lower() in lowered:
                return f"matched private/secret pattern {pattern!r}"
    return None


def should_copy_file(src: Path, rel: Path) -> tuple[bool, str]:
    if has_excluded_part(rel):
        return False, "excluded directory/name"
    if src.suffix.lower() in EXCLUDED_SUFFIXES:
        return False, f"excluded suffix {src.suffix}"
    if src.name.startswith(".") and src.name != ".gitignore":
        return False, "hidden file"
    try:
        size = src.stat().st_size
    except OSError as exc:
        return False, f"cannot stat file: {exc}"
    if size > LARGE_FILE_BYTES:
        return False, "over 25MB"
    private_reason = contains_private_text(src)
    if private_reason:
        return False, private_reason
    return True, ""


def copy_file(src: Path, rel: Path, output: Path, summary: CopySummary) -> None:
    ok, reason = should_copy_file(src, rel)
    if not ok:
        summary.skipped.append(f"{rel}: {reason}")
        return
    dest = output / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    summary.copied.append(str(rel))


def copy_tree(src_dir: Path, rel_dir: Path, output: Path, summary: CopySummary) -> None:
    if not src_dir.exists():
        summary.skipped.append(f"{rel_dir}: missing")
        return
    for src in sorted(src_dir.rglob("*")):
        if src.is_file():
            rel = src.relative_to(ROOT)
            copy_file(src, rel, output, summary)


def prepare(output: Path, include_agent: bool, force: bool) -> CopySummary:
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} already exists; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    summary = CopySummary(copied=[], skipped=[])

    for raw in ALLOWED_TOP_LEVEL_FILES:
        src = ROOT / raw
        if src.exists():
            copy_file(src, Path(raw), output, summary)
        else:
            summary.skipped.append(f"{raw}: missing optional file")

    for raw in ALLOWED_DIRS:
        copy_tree(ROOT / raw, Path(raw), output, summary)

    for raw in ALLOWED_HF_FILES:
        src = ROOT / raw
        if src.exists():
            copy_file(src, Path(raw), output, summary)
        else:
            summary.skipped.append(f"{raw}: missing")

    for raw in ALLOWED_HF_DIRS:
        copy_tree(ROOT / raw, Path(raw), output, summary)

    if include_agent:
        for raw in ALLOWED_AGENT_FILES:
            src = ROOT / raw
            if src.exists():
                copy_file(src, Path(raw), output, summary)
            else:
                summary.skipped.append(f"{raw}: missing")
    else:
        summary.skipped.append("agent/*.md: excluded by default; pass --include-agent to copy whitelisted audit reports")

    return summary


def run_reminder(command: list[str]) -> int:
    print("+ " + " ".join(command))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Clean release directory. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--force", action="store_true", help="Replace output directory if it already exists.")
    parser.add_argument("--include-agent", action="store_true", help="Include whitelisted agent audit/report markdown files.")
    parser.add_argument("--run-checks", action="store_true", help="Run anonymization, readiness, and pytest before preparing.")
    args = parser.parse_args()

    if args.run_checks:
        failures = 0
        for command in (
            [sys.executable, "scripts/check_anonymization.py"],
            [sys.executable, "scripts/check_submission_ready.py"],
            [sys.executable, "-m", "pytest"],
        ):
            failures += int(run_reminder(command) != 0)
        if failures:
            print(f"Refusing to prepare release because {failures} check command(s) failed.", file=sys.stderr)
            return 1
    else:
        print("Reminder before release:")
        print("  python scripts/check_anonymization.py")
        print("  python scripts/check_submission_ready.py")
        print("  pytest")

    output = Path(args.output).resolve()
    try:
        summary = prepare(output, include_agent=args.include_agent, force=args.force)
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Prepared anonymous release directory: {output}")
    print(f"Copied files: {len(summary.copied)}")
    for rel in summary.copied[:80]:
        print(f"  COPY {rel}")
    if len(summary.copied) > 80:
        print(f"  ... {len(summary.copied) - 80} more copied")
    print(f"Skipped files: {len(summary.skipped)}")
    for rel in summary.skipped[:120]:
        print(f"  SKIP {rel}")
    if len(summary.skipped) > 120:
        print(f"  ... {len(summary.skipped) - 120} more skipped")
    print("No upload or git push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
