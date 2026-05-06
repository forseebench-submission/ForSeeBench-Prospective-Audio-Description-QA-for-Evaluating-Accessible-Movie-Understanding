#!/usr/bin/env python3
"""Lightweight anonymous GitHub/Hugging Face submission-readiness checker."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".flac"}
LARGE_FILE_BYTES = 25 * 1024 * 1024
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules"}


@dataclass(frozen=True)
class Check:
    label: str
    paths: tuple[str, ...]
    required: bool = True
    note: str = ""


CHECKS = (
    Check("Root README", ("README.md",)),
    Check(
        "Reviewer docs",
        (
            "docs/reviewer_quickstart.md",
            "docs/reproducibility.md",
            "docs/dataset_construction.md",
            "docs/evaluation_protocol.md",
            "docs/benchmark_card.md",
            "docs/anonymization.md",
            "docs/source_data_redistribution_audit.md",
            "docs/source_dataset_citations.md",
            "docs/huggingface_release_plan.md",
            "docs/license_decision.md",
            "docs/release_commands.md",
        ),
    ),
    Check(
        "Hugging Face release draft",
        (
            "hf_dataset/README.md",
            "hf_dataset/schema.md",
            "hf_dataset/croissant_rai_notes.md",
            "hf_dataset/croissant_metadata_draft.json",
        ),
    ),
    Check(
        "Full Q/A release files",
        (
            "hf_dataset/data/qna_test.jsonl",
            "hf_dataset/data/qna_with_answers.jsonl",
        ),
    ),
    Check(
        "Sample data",
        (
            "hf_dataset/sample_data/sample_with_answers.jsonl",
            "hf_dataset/sample_data/sample_public.jsonl",
            "hf_dataset/sample_data/sample_predictions.jsonl",
        ),
    ),
    Check(
        "Reviewer scripts",
        (
            "scripts/check_submission_ready.py",
            "scripts/check_anonymization.py",
            "scripts/validate_dataset.py",
            "scripts/evaluate_mcq.py",
            "scripts/prepare_anonymous_release.py",
            "scripts/upload_hf_dataset.py",
        ),
    ),
    Check(
        "Core package",
        (
            "src/forseebench/utils/schema.py",
            "src/forseebench/evaluation/metrics.py",
            "src/forseebench/io/write_jsonl.py",
        ),
    ),
    Check(
        "Tests",
        (
            "tests/test_submission_ready.py",
            "tests/test_sample_data_loads.py",
            "tests/test_anonymization_checker.py",
            "tests/test_dataset_validation.py",
            "tests/test_metric_sanity.py",
        ),
    ),
    Check(
        "Paper draft",
        (
            "paper/69f1681b76d89ed4c70c745c/neurips_2026.tex",
            "paper/69f1681b76d89ed4c70c745c/checklist.tex",
        ),
        required=False,
        note="Optional in clean GitHub code artifact; paper may be submitted separately.",
    ),
    Check(
        "Audit files",
        (
            "agent/submission_readiness_audit.md",
            "agent/neurips_ed_compliance_checklist.md",
            "agent/source_dataset_citation_audit.md",
        ),
        required=False,
        note="Optional in clean GitHub code artifact; release docs carry the reviewer-facing policy.",
    ),
    Check("License", ("LICENSE", "LICENSE.md"), required=False, note="TODO(author): final license is unresolved."),
    Check(
        "Dependency file",
        ("requirements.txt", "pyproject.toml", "environment.yml", "environment.yaml"),
        required=False,
        note="TODO(author): full dependency pinning is unresolved.",
    ),
)


def status_for(paths: tuple[str, ...]) -> tuple[str, list[str], list[str]]:
    existing = [path for path in paths if (ROOT / path).exists()]
    missing = [path for path in paths if not (ROOT / path).exists()]
    if len(existing) == len(paths):
        return "PASS", existing, missing
    if existing:
        return "PARTIAL", existing, missing
    return "MISSING", existing, missing


def iter_repo_files() -> list[Path]:
    files: list[Path] = []
    for item in ROOT.rglob("*"):
        if not item.is_file():
            continue
        if any(part in SKIP_DIRS for part in item.relative_to(ROOT).parts):
            continue
        files.append(item)
    return files


def check_text_contains(path: str, snippets: tuple[str, ...], label: str) -> int:
    file_path = ROOT / path
    if not file_path.exists():
        print(f"MISSING {label}: {path} is absent.")
        return 1
    text = file_path.read_text(encoding="utf-8").lower()
    if all(snippet.lower() in text for snippet in snippets):
        print(f"PASS {label}: required wording found.")
        return 0
    print(f"PARTIAL {label}: required wording not found in {path}.")
    return 1


def check_urls() -> int:
    readme = ROOT / "README.md"
    if not readme.exists():
        print("MISSING Hosting URLs: README.md absent.")
        return 1
    text = readme.read_text(encoding="utf-8")
    failures = 0
    if "https://github.com/forseebench-submission/" in text:
        print("PASS GitHub URL: anonymous GitHub target present.")
    else:
        print("MISSING GitHub URL: anonymous GitHub target not found in README.")
        failures += 1
    if "forseebench-submission/forseebench" in text or "forseebench/forseebench" in text:
        print("PASS HF Dataset URL: planned HF Dataset target present.")
    else:
        print("MISSING HF Dataset URL: planned HF Dataset target not found in README.")
        failures += 1
    return failures


def check_raw_and_large_files() -> int:
    failures = 0
    raw_files: list[str] = []
    large_files: list[str] = []
    for path in iter_repo_files():
        rel = path.relative_to(ROOT)
        if path.suffix.lower() in RAW_EXTENSIONS:
            raw_files.append(str(rel))
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > LARGE_FILE_BYTES:
            large_files.append(f"{rel} ({size / (1024 * 1024):.1f}MB)")
    if raw_files:
        print("PARTIAL Raw media scan: disallowed raw media-like files found.")
        for item in raw_files[:20]:
            print(f"  - {item}")
        failures += 1
    else:
        print("PASS Raw media scan: no obvious raw media files found.")
    if large_files:
        print("PARTIAL Large file scan: files over 25MB found.")
        for item in large_files[:20]:
            print(f"  - {item}")
        print("  note: exclude large generated/source artifacts unless explicitly approved for review release.")
    else:
        print("PASS Large file scan: no files over 25MB found outside skipped cache dirs.")
    return failures


def check_hf_card() -> int:
    failures = 0
    card = ROOT / "hf_dataset/README.md"
    if not card.exists():
        print("MISSING HF card content: hf_dataset/README.md absent.")
        return 1
    text = card.read_text(encoding="utf-8").lower()
    if "mad" in text and "mad-eval" in text:
        print("PASS HF card source citation: MAD/MAD-eval mentioned.")
    else:
        print("PARTIAL HF card source citation: MAD/MAD-eval not clearly mentioned.")
        failures += 1
    if "raw movie videos" in text and "not redistribute" in text:
        print("PASS HF card raw-asset exclusion: raw assets are excluded.")
    else:
        print("PARTIAL HF card raw-asset exclusion: missing explicit raw-source exclusion language.")
        failures += 1
    split_terms = ("train_public", "train_private", "val_public", "val_private", "test_public", "test_private")
    unjustified_train_val_test = "train/validation/test" in text and "not presented as train/validation/test" not in text
    if any(term in text for term in split_terms) or unjustified_train_val_test:
        print("PARTIAL HF card split framing: train/validation/test release language remains without justification.")
        failures += 1
    else:
        print("PASS HF card split framing: no misleading train/validation/test release framing found.")
    if "evaluation benchmark" in text and "not a training corpus" in text:
        print("PASS HF card benchmark framing: evaluation-only framing found.")
    else:
        print("PARTIAL HF card benchmark framing: evaluation-only framing missing or weak.")
        failures += 1
    if "qna_test.jsonl" in text and "qna_with_answers.jsonl" in text:
        print("PASS HF card Q/A file names: primary Q/A files are documented.")
    else:
        print("PARTIAL HF card Q/A file names: qna_test/qna_with_answers are not clearly documented.")
        failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if required checks are missing or partial.")
    args = parser.parse_args()

    failures = 0
    for check in CHECKS:
        status, existing, missing = status_for(check.paths)
        if check.required and status != "PASS":
            failures += 1
        detail = f"{check.label}: {len(existing)}/{len(check.paths)} present"
        print(f"{status} {detail}")
        if missing:
            print("  missing: " + ", ".join(missing))
        if check.note:
            print(f"  note: {check.note}")

    failures += check_text_contains(
        "docs/source_data_redistribution_audit.md",
        ("raw videos are not redistributed", "must not be uploaded"),
        "Raw source exclusion",
    )
    failures += check_text_contains(
        "docs/source_dataset_citations.md",
        ("mad", "qwen", "narrad", "autoad"),
        "Source citation coverage",
    )
    failures += check_urls()
    failures += check_raw_and_large_files()
    failures += check_hf_card()

    if failures:
        print(f"PARTIAL submission readiness: {failures} required check(s) are incomplete.")
    else:
        print("PASS submission readiness file check.")
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
