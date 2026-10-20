#!/usr/bin/env python3
"""Audit the preregistered contact-reduction stabilization as one factor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def without_reduction_policy(optimizer: dict) -> dict:
    result = dict(optimizer)
    result.pop("contact_loss_reduction_policy", None)
    return result


def build_checks(reference, candidate, preregistration, activation):
    immutable_top_level = (
        "schema", "plan_sha256", "plan_records_sha256",
        "contact_manifest_sha256", "source_motion_sha256", "selection",
        "requested_samples",
    )
    return {
        "conditional_experiment_was_preregistered": (
            preregistration.get("status") == "conditionally_preregistered"
            and preregistration.get(
                "created_before_frozen_initial_pilot_completion"
            ) is True
            and preregistration["single_factor_change"]["factor"]
            == "contact_loss_reduction_policy"
        ),
        "activation_condition_was_met": (
            activation.get("full_augmentation_authorized") is False
        ),
        "reference_is_complete": (
            reference.get("complete") is True
            and reference.get("completed_samples")
            == reference.get("requested_samples") == 25
        ),
        "candidate_is_complete": (
            candidate.get("complete") is True
            and candidate.get("completed_samples")
            == candidate.get("requested_samples") == 25
        ),
        "all_inputs_and_selection_identical": all(
            reference.get(key) == candidate.get(key)
            for key in immutable_top_level
        ),
        "all_optimizer_settings_except_reduction_identical": (
            without_reduction_policy(reference["optimizer"])
            == without_reduction_policy(candidate["optimizer"])
        ),
        "both_use_frozen_initial_foot_masks": (
            reference["optimizer"].get("foot_contact_mask_policy")
            == candidate["optimizer"].get("foot_contact_mask_policy")
            == "frozen_initial"
        ),
        "reference_is_paper_literal_raw_sum": (
            reference["optimizer"].get(
                "contact_loss_reduction_policy", "paper_literal_raw_sum"
            ) == "paper_literal_raw_sum"
        ),
        "candidate_is_active_contact_mean": (
            candidate["optimizer"].get("contact_loss_reduction_policy")
            == "active_contact_mean"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--activation-assessment", type=Path, required=True)
    parser.add_argument(
        "--preregistration", type=Path,
        default=ROOT / "configs/interx_shape_augmentation/"
        "normalized_contact_reduction_pilot_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    preregistration = json.loads(
        args.preregistration.read_text(encoding="utf-8")
    )
    activation = json.loads(
        args.activation_assessment.read_text(encoding="utf-8")
    )
    checks = build_checks(
        reference, candidate, preregistration, activation
    )
    passed = all(checks.values())
    payload = {
        "schema": "hhc_interx_contact_reduction_single_factor_audit_v1",
        "status": "pass" if passed else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "declared_difference": {
            "field": "optimizer.contact_loss_reduction_policy",
            "reference": "paper_literal_raw_sum",
            "candidate": "active_contact_mean",
        },
        "artifacts": {
            "reference_completion": artifact(args.reference),
            "candidate_completion": artifact(args.candidate),
            "activation_assessment": artifact(args.activation_assessment),
            "preregistration": artifact(args.preregistration),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
