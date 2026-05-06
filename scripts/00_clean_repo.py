#!/usr/bin/env python3
"""Safely classify and optionally archive files irrelevant to ForSeeBench."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "archive_irrelevant_to_forseebench"
REPORT_PATH = ROOT / "REPO_CLEANUP_REPORT.md"
DO_NOT_TOUCH_PREFIXES = {
    ".git",
    "data/raw",
    "data/interim",
    "data/processed",
    "annotations",
    "configs",
}
ALWAYS_KEEP = {
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "environment.yml",
}


@dataclass(slots=True)
class Decision:
    label: str
    reason: str


def classify(path: Path) -> Decision:
    relative = path.relative_to(ROOT).as_posix()
    if relative == ".git" or relative.startswith(".git/") or "/.git/" in relative:
        return Decision("DO_NOT_TOUCH_RAW_DATA", "Version-control metadata must not be moved.")
    if relative.startswith(".pytest_cache/") or "/__pycache__/" in relative or path.suffix == ".pyc":
        return Decision("ARCHIVE_TEMP", "Generated cache or bytecode.")
    if any(relative == prefix or relative.startswith(prefix + "/") for prefix in DO_NOT_TOUCH_PREFIXES):
        return Decision("DO_NOT_TOUCH_RAW_DATA", "Protected project data/config path.")
    if relative in ALWAYS_KEEP:
        return Decision("KEEP_CORE", "Standard project root file.")
    if relative.startswith(("src/forseebench/", "scripts/", "docs/", "agents/", "tests/")):
        return Decision("KEEP_CORE", "Part of the new ForSeeBench pilot pipeline.")
    if relative.startswith("research_agents/forseebench_md_files/"):
        return Decision("KEEP_CORE", "Source specification files for ForSeeBench.")
    if relative.startswith(("research_agents/artifacts/", "node_modules/", "__pycache__/")):
        return Decision("ARCHIVE_TEMP", "Generated artifact or cache.")
    if relative.startswith(("experiments/", "literature/", "to_human/", "69e0f43d6a24b90f566e526b/")):
        return Decision("ARCHIVE_IRRELEVANT", "Not part of the pilot benchmark construction pipeline.")
    if relative.startswith("research_agents/"):
        return Decision("KEEP_MAYBE", "Legacy prototype code may still contain reusable ideas.")
    if relative.startswith("sprint-board/"):
        return Decision("KEEP_MAYBE", "Project management records.")
    return Decision("KEEP_MAYBE", "Unclassified; keep until reviewed manually.")


def scan_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ARCHIVE_DIR in path.parents:
            continue
        files.append(path)
    return sorted(files)


def move_to_archive(path: Path) -> Path:
    relative = path.relative_to(ROOT)
    destination = ARCHIVE_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    return destination


def build_report(rows: list[tuple[Path, Decision, Path | None]]) -> str:
    kept_core = [(path, decision) for path, decision, _ in rows if decision.label == "KEEP_CORE"]
    kept_maybe = [(path, decision) for path, decision, _ in rows if decision.label == "KEEP_MAYBE"]
    archived = [(path, decision, target) for path, decision, target in rows if decision.label.startswith("ARCHIVE_")]
    ambiguous = [(path, decision) for path, decision, _ in rows if decision.label == "KEEP_MAYBE"]
    lines = [
        "# Repo Cleanup Report",
        "",
        "## Summary",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Total files scanned: {len(rows)}",
        f"- Files kept: {len(kept_core) + len(kept_maybe)}",
        f"- Files archived: {len(archived)}",
        f"- Ambiguous files: {len(ambiguous)}",
        "",
        "## Kept Core Files",
        "",
        "| File | Reason |",
        "|---|---|",
    ]
    lines.extend(f"| `{path.relative_to(ROOT).as_posix()}` | {decision.reason} |" for path, decision in kept_core[:50])
    lines.extend([
        "",
        "## Kept Maybe Files",
        "",
        "| File | Reason |",
        "|---|---|",
    ])
    lines.extend(f"| `{path.relative_to(ROOT).as_posix()}` | {decision.reason} |" for path, decision in kept_maybe[:50])
    lines.extend([
        "",
        "## Archived Files",
        "",
        "| Original Path | Archive Path | Reason |",
        "|---|---|---|",
    ])
    for path, decision, target in archived[:100]:
        target_rel = target.relative_to(ROOT).as_posix() if target else f"archive_irrelevant_to_forseebench/{path.relative_to(ROOT).as_posix()}"
        lines.append(f"| `{path.relative_to(ROOT).as_posix()}` | `{target_rel}` | {decision.reason} |")
    lines.extend([
        "",
        "## Ambiguous Files",
        "",
        "| File | Why Ambiguous | Recommendation |",
        "|---|---|---|",
    ])
    for path, decision in ambiguous[:50]:
        lines.append(f"| `{path.relative_to(ROOT).as_posix()}` | {decision.reason} | Keep for now. |")
    lines.extend([
        "",
        "## Do Not Touch",
        "",
        "- `data/raw/`",
        "- `annotations/`",
        "- `.git/`",
        "",
        "## Recommended Final Repository Structure",
        "",
        "```text",
        "ForSeeBench/",
        "  agents/",
        "  configs/",
        "  data/",
        "  docs/",
        "  scripts/",
        "  src/",
        "  tests/",
        "```",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Move archivable files into the archive directory.")
    parser.add_argument("--dry_run", action="store_true", help="Preview actions without moving files.")
    args = parser.parse_args()

    apply = args.apply
    rows: list[tuple[Path, Decision, Path | None]] = []
    for path in scan_files():
        decision = classify(path)
        archive_target: Path | None = None
        if apply and decision.label.startswith("ARCHIVE_"):
            archive_target = move_to_archive(path)
        rows.append((path, decision, archive_target))

    REPORT_PATH.write_text(build_report(rows), encoding="utf-8")
    for path, decision, target in rows:
        action = "MOVE" if target else "KEEP"
        if decision.label.startswith("ARCHIVE_"):
            action = "MOVE" if apply else "PROPOSE_MOVE"
        print(f"{action:12} {decision.label:24} {path.relative_to(ROOT).as_posix()}")
    print(f"\nWrote cleanup report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
