#!/usr/bin/env python3
"""Audit the preregistered metric-aligned foot-loss pilot contract.

Frame batching is accepted only as an execution-only transformation when the
frozen multi-motion value/gradient gate selected the exact candidate size.
The sole scientific factor remains the foot-sliding loss contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOOT_POLICY = "initial_or_current_grounded_horizontal_mean_speed"


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


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def optimizer_without_authorized_changes(optimizer: dict) -> dict:
    result = dict(optimizer)
    result.pop("foot_contact_mask_policy", None)
    result.pop("volumetric_smpl_frame_batch_size", None)
    return result


def expected_candidate_sources() -> dict[str, str]:
    relative_paths = (
        "tools/run_interx_shape_augmentation.py",
        "optimize/optimize_full_body.py",
        "optimization/losses.py",
        "optimization/contact_loss_reduction.py",
        "optimization/foot_contact_constraints.py",
        "optimization/metric_aligned_foot_sliding.py",
        "optimization/volumetric_smpl_backend.py",
        "optimization/volumetric_smpl_batching.py",
    )
    return {
        str((ROOT / relative).resolve()): sha256_file(ROOT / relative)
        for relative in relative_paths
    }


def build_checks(reference, candidate, preregistration, activation, batching):
    immutable_top_level = (
        "schema",
        "plan_sha256",
        "plan_records_sha256",
        "contact_manifest_sha256",
        "source_motion_sha256",
        "selection",
        "requested_samples",
    )
    reference_optimizer = reference.get("optimizer", {})
    candidate_optimizer = candidate.get("optimizer", {})
    foot_decision = activation.get("metric_decisions", {}).get(
        "mean_foot_sliding_m_per_frame", {}
    )
    selected_batching = batching.get("selected_candidate", {})
    expected_sources = expected_candidate_sources()
    observed_sources = candidate.get("implementation", {})
    helper = ROOT / preregistration["candidate_implementation"]["helper"]
    test = ROOT / preregistration["candidate_implementation"]["test"]
    return {
        "conditional_experiment_was_preregistered": (
            preregistration.get("status") == "conditionally_preregistered"
            and preregistration.get(
                "created_before_contactmean_pilot_completion"
            ) is True
            and preregistration.get("single_factor_change", {}).get("factor")
            == "foot_sliding_loss_contract"
        ),
        "foot_specific_activation_condition_was_met": (
            activation.get("full_augmentation_authorized") is False
            and foot_decision.get("pass") is False
            and foot_decision.get("observed", {}).get(
                "paired_delta_after_minus_before_mean", 0.0
            ) > 0.0
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
        "all_objective_settings_except_foot_policy_identical": (
            optimizer_without_authorized_changes(reference_optimizer)
            == optimizer_without_authorized_changes(candidate_optimizer)
        ),
        "reference_uses_frozen_released_predicate": (
            reference_optimizer.get("foot_contact_mask_policy")
            == "frozen_initial"
        ),
        "candidate_uses_preregistered_metric_aligned_policy": (
            candidate_optimizer.get("foot_contact_mask_policy") == FOOT_POLICY
        ),
        "both_use_active_contact_mean": (
            reference_optimizer.get("contact_loss_reduction_policy")
            == candidate_optimizer.get("contact_loss_reduction_policy")
            == "active_contact_mean"
        ),
        "reference_is_serial_and_candidate_is_selected_batch8": (
            int(reference_optimizer.get(
                "volumetric_smpl_frame_batch_size", 1
            )) == 1
            and candidate_optimizer.get(
                "volumetric_smpl_frame_batch_size"
            ) == 8
        ),
        "frame_batching_gate_authorizes_exact_execution_change": (
            batching.get("status") == "pass"
            and batching.get("production_edit_authorized") is True
            and selected_batching.get("frame_batch_size") == 8
            and preregistration.get("frozen_unchanged_settings", {}).get(
                "frame_batch_size"
            ) == 8
        ),
        "candidate_run_binds_current_promoted_sources": (
            all(observed_sources.get(path) == digest
                for path, digest in expected_sources.items())
        ),
        "preregistered_helper_and_test_are_unchanged": (
            sha256_file(helper)
            == preregistration["candidate_implementation"]["helper_sha256"]
            and sha256_file(test)
            == preregistration["candidate_implementation"]["test_sha256"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--activation-assessment", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=ROOT / "configs/interx_shape_augmentation/"
        "metric_aligned_foot_sliding_pilot_v1.json",
    )
    parser.add_argument(
        "--frame-batching-assessment",
        type=Path,
        default=ROOT / "output/volumetric_smpl_flat_hand_serial_sampling_"
        "frame_batching_assessment.json",
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
    batching = json.loads(
        args.frame_batching_assessment.read_text(encoding="utf-8")
    )
    checks = build_checks(
        reference, candidate, preregistration, activation, batching
    )
    passed = all(checks.values())
    payload = {
        "schema": "hhc_interx_metric_aligned_foot_single_factor_audit_v1",
        "status": "pass" if passed else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "declared_scientific_difference": {
            "field": "optimizer.foot_contact_mask_policy",
            "reference": "frozen_initial",
            "candidate": FOOT_POLICY,
        },
        "execution_only_difference": {
            "field": "optimizer.volumetric_smpl_frame_batch_size",
            "reference": 1,
            "candidate": 8,
            "authorization": artifact(args.frame_batching_assessment),
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
