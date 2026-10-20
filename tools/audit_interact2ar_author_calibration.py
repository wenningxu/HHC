#!/usr/bin/env python3
"""Validate that an Interact2AR author-protocol calibration is still current."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def artifact_matches(record: dict) -> bool:
    if not isinstance(record, dict) or not {"path", "bytes", "sha256"} <= set(record):
        return False
    path = Path(record["path"])
    return (
        path.is_file()
        and path.stat().st_size == record["bytes"]
        and sha256_file(path) == record["sha256"]
    )


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path,
        default=Path("output/interact2ar_author_release_calibration/summary.json"),
    )
    parser.add_argument(
        "--current-audit", type=Path,
        default=Path("output/interact2ar_paper_repo_data_audit.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/interact2ar_author_release_calibration/audit.json"),
    )
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    current_audit_path = args.current_audit.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    current_audit = json.loads(current_audit_path.read_text(encoding="utf-8"))
    copied_audit_record = summary.get("audit", {})
    copied_audit_path = Path(copied_audit_record.get("path", ""))
    copied_audit = (
        json.loads(copied_audit_path.read_text(encoding="utf-8"))
        if copied_audit_path.is_file() else None
    )
    checks = {
        "calibration_schema_and_status": (
            summary.get("schema") == "interact2ar_author_release_calibration_v1"
            and summary.get("status") == "complete"
        ),
        "author_protocol_is_disclosed_gt_first_frame": (
            summary.get("scope")
            == "untouched_public_README_protocol_with_GT_first_frame"
            and summary.get("must_not_mix_with_unconditional_fair_table") is True
            and summary.get("first_frame_conditioning")
            == "random ground-truth frame at shared batch index 0..4"
        ),
        "released_and_paper_replications_are_not_conflated": (
            summary.get("released_replications") == 5
            and summary.get("paper_replications") == 20
        ),
        "paper_mean_diagnostic_gate_passes": summary.get("near_paper_mean") is True,
        "copied_prerequisite_audit_artifact_is_unchanged": artifact_matches(
            copied_audit_record
        ),
        "author_stdout_artifact_is_unchanged": artifact_matches(
            summary.get("stdout", {})
        ),
        "current_paper_repo_data_audit_passes": (
            current_audit.get("schema")
            == "interact2ar_paper_repo_data_audit_v1"
            and str(current_audit.get("status", "")).startswith("pass")
        ),
        "calibration_audit_equals_current_paper_repo_data_audit": (
            copied_audit == current_audit
        ),
    }
    payload = {
        "schema": "interact2ar_author_release_calibration_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "interpretation": (
            "The untouched released five-replication GT-first-frame run is only "
            "an author-protocol calibration. Passing this audit authorizes the "
            "separate twenty-replication unconditional joint-evaluator supplement; "
            "it does not authorize mixing the two tables."
        ),
        "summary": artifact(summary_path),
        "current_paper_repository_data_audit": artifact(current_audit_path),
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
