#!/usr/bin/env python3
"""Audit the preregistered metric-aligned foot-loss candidate."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/interx_shape_augmentation/metric_aligned_foot_sliding_pilot_v1.json"
OUTPUT = ROOT / "output/interx_shape_aug_metric_aligned_foot_preregistration_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    motivation = config["motivation_snapshot"]
    implementation = config["candidate_implementation"]
    gate_binding = config["authorization_gate"]
    snapshot_path = resolve(motivation["path"])
    helper_path = resolve(implementation["helper"])
    test_path = resolve(implementation["test"])
    gate_path = resolve(gate_binding["path"])
    batching_path = ROOT / (
        "output/volumetric_smpl_flat_hand_serial_sampling_"
        "frame_batching_assessment.json"
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    batching = json.loads(batching_path.read_text(encoding="utf-8"))
    records = snapshot.get("records", [])
    deltas = [row.get("delta_after_minus_before") for row in records]
    checks = {
        "schema_and_status_match": (
            config.get("schema")
            == "hhc_interx_shape_augmentation_metric_aligned_foot_sliding_pilot_v1"
            and config.get("status") == "conditionally_preregistered"
            and config.get("created_before_contactmean_pilot_completion") is True
        ),
        "activation_is_foot_sliding_only": (
            "fails the unchanged preregistered mean-foot-sliding criterion"
            in config.get("activation_condition", "")
        ),
        "snapshot_hash_matches": (
            sha256_file(snapshot_path) == motivation.get("sha256")
        ),
        "snapshot_is_partial12_and_all_worse": (
            snapshot.get("sample_count")
            == motivation.get("completed_samples_at_preregistration") == 12
            and len(records) == 12
            and len(set(row.get("sample_key") for row in records)) == 12
            and all(isinstance(delta, (int, float)) and delta > 0 for delta in deltas)
            and sum(delta > 0 for delta in deltas)
            == motivation.get("samples_with_worse_foot_sliding") == 12
        ),
        "snapshot_mean_matches": (
            snapshot.get("mean_delta_after_minus_before")
            == motivation.get("mean_delta_after_minus_random_m_per_frame")
        ),
        "helper_hash_matches": (
            sha256_file(helper_path) == implementation.get("helper_sha256")
        ),
        "test_hash_matches_and_pass_declared": (
            sha256_file(test_path) == implementation.get("test_sha256")
            and implementation.get("test_status_before_activation") == "pass"
        ),
        "single_factor_contract_matches": (
            config.get("single_factor_change", {}).get("factor")
            == "foot_sliding_loss_contract"
            and config["single_factor_change"].get("candidate")
            == "initial_or_current_grounded_horizontal_mean_speed"
        ),
        "gate_hash_and_thresholds_match": (
            sha256_file(gate_path) == gate_binding.get("sha256")
            and gate.get("status") == "active"
            and gate.get("expected_sample_count")
            == gate_binding.get("expected_sample_count") == 25
            and gate.get("criteria", {}).get(
                "mean_foot_sliding_m_per_frame", {}
            ).get("maximum_mean_delta") == 0.0
        ),
        "frame_batching_execution_gate_passed": (
            batching.get("status") == "pass"
            and batching.get("production_edit_authorized") is True
            and batching.get("selected_candidate", {}).get("frame_batch_size")
            == config.get("frozen_unchanged_settings", {}).get(
                "frame_batch_size"
            ) == 8
        ),
        "weights_remain_unchanged": (
            config.get("frozen_unchanged_settings", {}).get("loss_weights")
            == {
                "attract": 1.0,
                "pene": 5.0,
                "reg": 1.0,
                "smooth": 0.1,
                "skate": 100.0,
                "ground": 20.0,
            }
        ),
    }
    passed = all(checks.values())
    payload = {
        "schema": "hhc_metric_aligned_foot_preregistration_audit_v1",
        "status": "pass" if passed else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "artifacts": {
            "config": artifact(CONFIG),
            "motivation_snapshot": artifact(snapshot_path),
            "helper": artifact(helper_path),
            "test": artifact(test_path),
            "gate": artifact(gate_path),
            "frame_batching_assessment": artifact(batching_path),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
