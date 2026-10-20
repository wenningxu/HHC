#!/usr/bin/env python3
"""Fail-closed readiness audit for the GPU1 work after frozen-foot pilot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / (
    "output/interx_shape_aug_pilot_optimized_flathand_"
    "frozenfoot_query10475_seed20260828"
)
RANDOM = ROOT / "output/interx_shape_aug_pilot_random_seed20260828"
NORMALIZED_PREREG = ROOT / (
    "configs/interx_shape_augmentation/"
    "normalized_contact_reduction_pilot_v1.json"
)
MGT_STATE = ROOT / "output/hhc_global_masked_loss_control_training_state.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    manifest_path = FROZEN / "_completion_manifest.json"
    contract_path = FROZEN / "_run_contract.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    prereg = json.loads(NORMALIZED_PREREG.read_text(encoding="utf-8"))
    mgt_state = json.loads(MGT_STATE.read_text(encoding="utf-8"))

    source_hashes = manifest["implementation"]
    source_checks = {
        raw_path: (
            Path(raw_path).is_file()
            and sha256_file(Path(raw_path)) == expected_hash
        )
        for raw_path, expected_hash in source_hashes.items()
    }
    prereg_hash_checks = {}
    for label in ("motivation_snapshot", "authorization_gate"):
        record = prereg[label]
        path = ROOT / record["path"]
        prereg_hash_checks[label] = (
            path.is_file() and sha256_file(path) == record["sha256"]
        )

    frame_inputs = [
        RANDOM / "G024T003A019R024__shapeaug_s00000123_p1.npz",
        RANDOM / "G024T003A019R024__shapeaug_s00000123_p2.npz",
        RANDOM / "G043T005A022R010__shapeaug_s00000123_p1.npz",
        RANDOM / "G043T005A022R010__shapeaug_s00000123_p2.npz",
    ]
    required_tools = [
        ROOT / "tools/finalize_frozen_foot_pilot.sh",
        ROOT / "tools/run_volumetric_smpl_frame_batching_gate.sh",
        ROOT / "optimization/volumetric_smpl_batching.py",
        ROOT / "tools/benchmark_volumetric_smpl_frame_batching.py",
        ROOT / "tools/assess_volumetric_smpl_frame_batching.py",
        ROOT / "optimization/contact_loss_reduction.py",
        ROOT / "tools/audit_contact_reduction_single_factor.py",
        ROOT / "tools/finalize_normalized_contact_reduction_pilot.sh",
        ROOT / "tools/run_hhc_global_masked_loss_control.sh",
    ]
    checks = {
        "frozen_contract_matches_completion_manifest": (
            manifest["contract_sha256"] == contract["contract_sha256"]
            and canonical_sha256({
                key: value for key, value in contract.items()
                if key not in ("created_at_utc", "contract_sha256")
            }) == manifest["contract_sha256"]
        ),
        "frozen_hash_bound_sources_unchanged_while_running": all(
            source_checks.values()
        ),
        "frozen_selection_is_exact_25": (
            manifest["requested_samples"] == 25
            and len(manifest["selection"]["selected_names"]) == 25
        ),
        "normalized_preregistration_hashes_are_frozen": all(
            prereg_hash_checks.values()
        ),
        "frame_batching_real_inputs_exist": all(
            path.is_file() for path in frame_inputs
        ),
        "all_handoff_tools_exist": all(path.is_file() for path in required_tools),
        "mgt_control_is_not_prematurely_started": (
            mgt_state.get("status") == "new"
            and mgt_state.get("expected_final_step") == 50000
        ),
    }
    passed = all(checks.values())
    payload = {
        "schema": "hhc_post_frozen_gpu1_handoff_readiness_v1",
        "status": "ready_after_frozen_completion" if passed else "fail",
        "checks": checks,
        "frozen_progress": {
            "complete": manifest["complete"],
            "completed_samples": manifest["completed_samples"],
            "requested_samples": manifest["requested_samples"],
        },
        "source_checks": source_checks,
        "preregistration_hash_checks": prereg_hash_checks,
        "frame_inputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in frame_inputs
        ],
        "required_tools": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in required_tools
        ],
        "execution_order": [
            "finish and finalize frozen-foot pilot",
            "run real frame-batching equivalence/performance gate",
            "if frozen physical gate failed, implement and run preregistered active-contact-mean pilot",
            "launch online-W&B MGT global masked-token-mean control",
        ],
    }
    output = ROOT / "output/post_frozen_gpu1_handoff_readiness.json"
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
