#!/usr/bin/env python3
"""Upload hf_dataset/ contents to a Hugging Face Dataset repo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RAW_MEDIA_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".flac"}
SECRET_NAMES = {".env", ".env.local", ".envrc"}
SECRET_SUFFIXES = {".pem", ".key"}
LARGE_FILE_BYTES = 25 * 1024 * 1024


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def iter_upload_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file())


def validate_upload_files(folder: Path, allow_large: bool) -> tuple[list[Path], list[str]]:
    files = iter_upload_files(folder)
    errors: list[str] = []
    for path in files:
        rel = path.relative_to(folder)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in rel.parts):
            errors.append(f"{rel}: cache/git file is not allowed")
        if path.name in SECRET_NAMES or path.suffix.lower() in SECRET_SUFFIXES:
            errors.append(f"{rel}: secret-like file is not allowed")
        if path.suffix.lower() in RAW_MEDIA_EXTENSIONS:
            errors.append(f"{rel}: raw media extension is not allowed")
        try:
            size = path.stat().st_size
        except OSError as exc:
            errors.append(f"{rel}: cannot stat file: {exc}")
            continue
        if size > LARGE_FILE_BYTES and not allow_large:
            errors.append(f"{rel}: file is {size / (1024 * 1024):.1f}MB; pass --allow-large only after release review")
    return files, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="forseebench/forseebench", help="Hugging Face dataset repo id.")
    parser.add_argument("--folder", default="hf_dataset", help="Folder to upload. Default: hf_dataset")
    parser.add_argument("--private", type=parse_bool, default=True, help="Create repo as private: true/false. Default: true")
    parser.add_argument("--dry-run", action="store_true", help="List files and do not upload.")
    parser.add_argument("--allow-large", action="store_true", help="Allow files over 25MB after manual review.")
    args = parser.parse_args()

    folder = (ROOT / args.folder).resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"ERROR: upload folder does not exist or is not a directory: {folder}", file=sys.stderr)
        return 2

    files, errors = validate_upload_files(folder, allow_large=args.allow_large)
    if errors:
        print("Refusing upload because unsafe files were detected:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Repository: {args.repo_id}")
    print(f"Folder: {folder}")
    print(f"Private: {args.private}")
    print(f"Files to upload: {len(files)}")
    for path in files:
        rel = path.relative_to(folder)
        print(f"  {rel}")

    dataset_url = f"https://huggingface.co/datasets/{args.repo_id}"
    if args.dry_run:
        print("Dry run only: no Hugging Face upload was performed.")
        print(f"Dataset URL after upload would be: {dataset_url}")
        return 0

    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub is required for upload. Install with `python -m pip install huggingface_hub`.", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Upload ForSeeBench dataset artifact",
    )
    print(f"Uploaded dataset: {dataset_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
