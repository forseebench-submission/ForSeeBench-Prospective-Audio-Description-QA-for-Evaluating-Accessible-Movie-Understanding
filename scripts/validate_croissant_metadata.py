#!/usr/bin/env python3
"""Sanity-check the ForSeeBench Croissant + RAI metadata file.

This is a lightweight repository checker for the NeurIPS E&D hosting
requirements. It does not replace the official Croissant validator, but it
checks the fields and local file references that commonly break submissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "hf_dataset/croissant.json"
REQUIRED_DATASET_FIELDS = {
    "@context",
    "@type",
    "name",
    "description",
    "url",
    "license",
    "conformsTo",
    "distribution",
    "recordSet",
}
REQUIRED_RAI_FIELDS = {
    "rai:dataLimitations",
    "rai:dataBiases",
    "rai:personalSensitiveInformation",
    "rai:dataUseCases",
    "rai:dataSocialImpact",
    "rai:hasSyntheticData",
}
REQUIRED_FILE_FIELDS = {
    "@id",
    "@type",
    "name",
    "contentUrl",
    "encodingFormat",
    "sha256",
    "contentSize",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected top-level JSON object")
    return payload


def local_path_for_content_url(content_url: str) -> Path | None:
    parsed = urlparse(content_url)
    if parsed.scheme in {"http", "https"}:
        marker = "/resolve/main/"
        if marker not in parsed.path:
            return None
        rel = parsed.path.split(marker, 1)[1]
        return ROOT / "hf_dataset" / rel
    return ROOT / "hf_dataset" / content_url


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_fields(record_set: Any) -> list[dict[str, Any]]:
    record_sets = record_set if isinstance(record_set, list) else [record_set]
    fields: list[dict[str, Any]] = []
    for record in record_sets:
        if isinstance(record, dict):
            raw_fields = record.get("field", [])
            if isinstance(raw_fields, list):
                fields.extend(field for field in raw_fields if isinstance(field, dict))
    return fields


def validate(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    missing = sorted(REQUIRED_DATASET_FIELDS - set(payload))
    if missing:
        errors.append(f"missing required dataset field(s): {', '.join(missing)}")
    if payload.get("@type") != "sc:Dataset":
        errors.append('@type must be "sc:Dataset"')

    rai_missing = sorted(REQUIRED_RAI_FIELDS - set(payload))
    if rai_missing:
        errors.append(f"missing required RAI field(s): {', '.join(rai_missing)}")
    if "rai:hasSyntheticData" in payload and not isinstance(payload["rai:hasSyntheticData"], bool):
        errors.append("rai:hasSyntheticData must be a boolean")
    if "prov:wasDerivedFrom" not in payload:
        errors.append("missing provenance field: prov:wasDerivedFrom")
    if "prov:wasGeneratedBy" not in payload:
        errors.append("missing provenance field: prov:wasGeneratedBy")

    distribution = payload.get("distribution", [])
    if not isinstance(distribution, list) or not distribution:
        errors.append("distribution must be a non-empty list")
        distribution = []

    file_ids: set[str] = set()
    for index, file_object in enumerate(distribution, start=1):
        if not isinstance(file_object, dict):
            errors.append(f"distribution[{index}] must be an object")
            continue
        missing_file_fields = sorted(REQUIRED_FILE_FIELDS - set(file_object))
        if missing_file_fields:
            errors.append(f"distribution[{index}] missing field(s): {', '.join(missing_file_fields)}")
        file_id = file_object.get("@id")
        if isinstance(file_id, str):
            file_ids.add(file_id)
        if file_object.get("@type") != "cr:FileObject":
            errors.append(f"distribution[{index}] @type must be cr:FileObject")

        content_url = file_object.get("contentUrl")
        if isinstance(content_url, str):
            local_path = local_path_for_content_url(content_url)
            if local_path is None:
                errors.append(f"distribution[{index}] contentUrl is not mappable to a local file: {content_url}")
            elif not local_path.exists():
                errors.append(f"distribution[{index}] local file does not exist: {local_path.relative_to(ROOT)}")
            else:
                expected_hash = file_object.get("sha256")
                expected_size = file_object.get("contentSize")
                actual_hash = sha256(local_path)
                actual_size = local_path.stat().st_size
                if expected_hash != actual_hash:
                    errors.append(
                        f"distribution[{index}] sha256 mismatch for {local_path.relative_to(ROOT)}: "
                        f"{expected_hash} != {actual_hash}"
                    )
                try:
                    expected_size_int = int(expected_size)
                except (TypeError, ValueError):
                    errors.append(
                        f"distribution[{index}] contentSize must be a byte-count string for "
                        f"{local_path.relative_to(ROOT)}: {expected_size}"
                    )
                    expected_size_int = None
                if expected_size_int is not None and expected_size_int != actual_size:
                    errors.append(
                        f"distribution[{index}] contentSize mismatch for {local_path.relative_to(ROOT)}: "
                        f"{expected_size} != {actual_size}"
                    )

    record_set = payload.get("recordSet")
    if not record_set:
        errors.append("recordSet must be present and non-empty")
    fields = iter_fields(record_set)
    if not fields:
        errors.append("recordSet must define at least one field")
    for field in fields:
        source = field.get("source")
        if not isinstance(source, dict):
            errors.append(f"{field.get('@id', '<unknown field>')}: missing source object")
            continue
        file_ref = source.get("fileObject")
        if isinstance(file_ref, dict):
            file_ref = file_ref.get("@id")
        if file_ref not in file_ids:
            errors.append(f"{field.get('@id', '<unknown field>')}: source fileObject {file_ref!r} is not in distribution")
        extract = source.get("extract")
        if not isinstance(extract, dict) or "jsonPath" not in extract:
            errors.append(f"{field.get('@id', '<unknown field>')}: source.extract.jsonPath is required")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Croissant JSON file. Default: {DEFAULT_INPUT}")
    args = parser.parse_args()

    path = ROOT / args.input
    try:
        payload = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL {path}: {exc}")
        return 2

    errors = validate(payload)
    if errors:
        print(f"FAIL {path}: {len(errors)} issue(s)")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"PASS {path}: Croissant core, minimal RAI, provenance, and local file references look consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
