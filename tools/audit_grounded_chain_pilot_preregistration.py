#!/usr/bin/env python3
"""Freeze the grounded-chain pilot contract before its five outputs exist."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--pilot-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    preregistration = json.loads(
        args.preregistration.read_text(encoding="utf-8")
    )
    gate = json.loads(args.pilot_gate.read_text(encoding="utf-8"))
    optimizer = manifest.get("optimizer", {})
    checks = {
        "pilot_is_incomplete_at_preregistration": (
            manifest.get("complete") is False
            and int(manifest.get("completed_samples", -1)) < 5
        ),
        "five_frozen_samples": (
            manifest.get("requested_samples") == 5
            and len(manifest.get("selection", {}).get("selected_names", [])) == 5
            and manifest.get("selection", {}).get("selection_seed") == 20260828
        ),
        "manual_positive_timing_only": manifest.get("selection", {}).get(
            "positive_timing_only"
        ) is True,
        "ground_contract_matches_mmm": optimizer.get(
            "ground_alignment_policy"
        ) == "mmm_per_person_joint_min_restore_raw",
        "fixed_root_grounded_chain_policy": optimizer.get(
            "optimization_variable_policy"
        ) == "grounded_chain_fixed_root",
        "paper_schedule": (
            optimizer.get("iterations") == 100
            and optimizer.get("learning_rate") == 0.02
            and optimizer.get("fine_stage_start") == 50
        ),
        "metric_aligned_foot_loss": optimizer.get(
            "foot_contact_mask_policy"
        ) == "initial_or_current_grounded_horizontal_mean_speed",
        "active_contact_mean": optimizer.get(
            "contact_loss_reduction_policy"
        ) == "active_contact_mean",
        "preregistration_status": preregistration.get("status")
        == "preregistered_before_candidate_execution",
        "five_pair_gate_active": (
            gate.get("status") == "active"
            and gate.get("expected_sample_count") == 5
        ),
    }
    current_source_checks = {}
    for raw_path, expected_hash in manifest.get("implementation", {}).items():
        path = Path(raw_path)
        current_source_checks[str(path)] = (
            path.is_file() and sha256_file(path) == expected_hash
        )
    checks["all_run_sources_still_match"] = all(current_source_checks.values())
    passed = all(checks.values())
    payload = {
        "schema": "hhc_grounded_chain_pilot_preregistration_audit_v1",
        "status": "pass" if passed else "fail",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "source_hash_checks": current_source_checks,
        "completed_samples_at_audit": manifest.get("completed_samples"),
        "run_contract_sha256": manifest.get("contract_sha256"),
        "selected_names": manifest.get("selection", {}).get("selected_names"),
        "manifest": artifact(args.manifest),
        "preregistration": artifact(args.preregistration),
        "pilot_gate": artifact(args.pilot_gate),
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
