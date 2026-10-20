#!/usr/bin/env python3
"""Promote the frozen Inter-X val request to a method-neutral contract.

The original request was generated for CoShMDM selection and its manifest
deliberately scoped the artifact to that method.  The arrays are the official
544-sample Inter-X validation request shared by all methods.  This utility
validates the complete original provenance, copies those exact bytes, and
writes a method-neutral manifest without mutating completed CoShMDM evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCHEMA = "hhc_interx_validation_request_v1"
OUTPUT_SCHEMA = "hhc_interx_method_neutral_validation_request_v1"
SOURCE_PURPOSE = "CoShMDM checkpoint selection only; never a reported test row"
OUTPUT_PURPOSE = (
    "held-out official Inter-X checkpoint selection for all methods; "
    "never a reported test row"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_artifact(record: dict, label: str) -> Path:
    path = Path(record.get("path", "")).resolve()
    if artifact(path) != record:
        raise ValueError(f"{label} is absent or stale: {path}")
    return path


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "output/coshmdm_interx_val_requests/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/interx_official_val_requests",
    )
    args = parser.parse_args()

    source_manifest_path = args.source_manifest.resolve()
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    exact_contract = {
        "schema": SOURCE_SCHEMA,
        "status": "complete",
        "purpose": SOURCE_PURPOSE,
        "seed": 500123,
        "batch_size": 32,
        "samples": 544,
        "formal_hhc_validation_record_set_exact": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": source.get(key)}
        for key, expected in exact_contract.items()
        if source.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"source validation contract changed: {mismatches}")

    source_request = validate_artifact(source["request"], "source request")
    for key in (
        "split", "motions", "evaluator_options", "evaluator_weights",
        "formal_hhc_validation_selection", "implementation",
    ):
        validate_artifact(source[key], f"source {key}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_request = output_dir / "validation.npz"
    if output_request.is_file():
        if sha256_file(output_request) != source["request"]["sha256"]:
            raise ValueError("existing method-neutral request has different bytes")
    else:
        temporary = output_request.with_suffix(
            output_request.suffix + f".tmp.{os.getpid()}"
        )
        shutil.copyfile(source_request, temporary)
        os.replace(temporary, output_request)
    if artifact(output_request)["sha256"] != source["request"]["sha256"]:
        raise ValueError("method-neutral request copy is not byte-identical")

    manifest = {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "purpose": OUTPUT_PURPOSE,
        "seed": 500123,
        "batch_size": 32,
        "samples": 544,
        "request": artifact(output_request),
        "request_is_byte_identical_to_source": True,
        "test_set_used": False,
        "split": source["split"],
        "motions": source["motions"],
        "evaluator_options": source["evaluator_options"],
        "evaluator_weights": source["evaluator_weights"],
        "formal_hhc_validation_selection": source[
            "formal_hhc_validation_selection"
        ],
        "formal_hhc_validation_record_set_exact": True,
        "source_generation_manifest": artifact(source_manifest_path),
        "request_generation_implementation": source["implementation"],
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
