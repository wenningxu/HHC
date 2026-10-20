#!/usr/bin/env python3
"""Fail closed if the shape-augmentation queue cannot retain all artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
from pathlib import Path


GIB = 1024 ** 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_apparent_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def project_training_bytes_per_arm(
    training_bytes: int,
    checkpoint_sizes: list[int],
    validation_checkpoints: int,
    rolling_checkpoints: int,
    validation_checkpoint_bytes: int | None = None,
) -> dict[str, int | str]:
    """Project mixed validation/recovery storage from measured artifacts."""
    if not checkpoint_sizes:
        raise ValueError("reference training run contains no checkpoint")
    measured_full_checkpoint_bytes = max(checkpoint_sizes)
    measured_validation_checkpoint_bytes = (
        measured_full_checkpoint_bytes
        if validation_checkpoint_bytes is None
        else int(validation_checkpoint_bytes)
    )
    if measured_validation_checkpoint_bytes <= 0:
        raise ValueError("validation checkpoint size must be positive")
    observed_checkpoint_bytes = sum(checkpoint_sizes)
    noncheckpoint_training_bytes = max(0, training_bytes - observed_checkpoint_bytes)
    planned_checkpoint_count = validation_checkpoints + rolling_checkpoints
    count_based_bytes = (
        measured_validation_checkpoint_bytes * validation_checkpoints
        + measured_full_checkpoint_bytes * rolling_checkpoints
        + noncheckpoint_training_bytes
    )
    # With no explicit validation snapshot, retain the legacy conservative
    # lower bound of the complete reference tree.  With an explicit measured
    # weights-only snapshot, the reference tree intentionally has a different
    # storage contract and must not override the mixed-format projection.
    projected_bytes = (
        max(training_bytes, count_based_bytes)
        if validation_checkpoint_bytes is None else count_based_bytes
    )
    return {
        "measured_full_checkpoint_bytes": measured_full_checkpoint_bytes,
        "measured_validation_checkpoint_bytes": (
            measured_validation_checkpoint_bytes
        ),
        "validation_checkpoint_storage": (
            "full_training_state"
            if validation_checkpoint_bytes is None else "weights_only"
        ),
        "observed_checkpoint_bytes": observed_checkpoint_bytes,
        "noncheckpoint_training_bytes": noncheckpoint_training_bytes,
        "planned_checkpoint_count": planned_checkpoint_count,
        "count_based_bytes": count_based_bytes,
        "projected_bytes": projected_bytes,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filesystem-root", type=Path, default=Path("."))
    parser.add_argument("--reference-label", type=Path, required=True)
    parser.add_argument("--reference-training-run", type=Path, required=True)
    parser.add_argument(
        "--reference-validation-checkpoint", type=Path,
        help=(
            "Measured weights-only validation checkpoint for the same model "
            "architecture. If omitted, validation candidates are conservatively "
            "projected as full training-state checkpoints."
        ),
    )
    parser.add_argument(
        "--reference-validation-manifest", type=Path,
        help="Hash-bound export manifest for the measured weights-only checkpoint.",
    )
    parser.add_argument("--arms", type=int, default=2)
    parser.add_argument("--labels-per-arm", type=int, default=2)
    parser.add_argument("--validation-checkpoints-per-arm", type=int, default=30)
    parser.add_argument("--rolling-full-checkpoints-per-arm", type=int, default=2)
    parser.add_argument("--asset-and-optimizer-reserve-gib", type=float, default=40)
    parser.add_argument("--physical-output-reserve-gib", type=float, default=30)
    parser.add_argument("--free-space-after-queue-gib", type=float, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_label = args.reference_label.resolve()
    reference_training = args.reference_training_run.resolve()
    if not reference_label.is_dir() or not reference_training.is_dir():
        raise FileNotFoundError(
            f"missing reference label/training run: {reference_label}, "
            f"{reference_training}"
        )
    if not (reference_label / "summary.json").is_file():
        raise ValueError("reference label is not a complete 20-rep result")
    summary = json.loads(
        (reference_label / "summary.json").read_text(encoding="utf-8")
    )
    if summary.get("replications") != 20:
        raise ValueError("reference label does not contain 20 replications")
    if (
        args.arms <= 0
        or args.labels_per_arm <= 0
        or args.validation_checkpoints_per_arm <= 0
        or args.rolling_full_checkpoints_per_arm <= 0
    ):
        raise ValueError("arms, labels and checkpoint counts must be positive")

    label_bytes = tree_apparent_bytes(reference_label)
    training_bytes = tree_apparent_bytes(reference_training)
    checkpoint_sizes = [
        item.stat().st_size
        for item in reference_training.rglob("*.ckpt")
        if item.is_file()
    ]
    validation_checkpoint = (
        args.reference_validation_checkpoint.resolve()
        if args.reference_validation_checkpoint is not None else None
    )
    if validation_checkpoint is not None and not validation_checkpoint.is_file():
        raise FileNotFoundError(validation_checkpoint)
    validation_manifest = (
        args.reference_validation_manifest.resolve()
        if args.reference_validation_manifest is not None else None
    )
    if (validation_checkpoint is None) != (validation_manifest is None):
        raise ValueError(
            "weights-only validation checkpoint and manifest must be supplied together"
        )
    validation_manifest_payload = None
    if validation_manifest is not None:
        if not validation_manifest.is_file():
            raise FileNotFoundError(validation_manifest)
        validation_manifest_payload = json.loads(
            validation_manifest.read_text(encoding="utf-8")
        )
        output_record = validation_manifest_payload.get("output", {})
        if not (
            validation_manifest_payload.get("schema")
            == "hhc_lightning_weights_only_capacity_reference_v1"
            and validation_manifest_payload.get("status") == "pass"
            and Path(output_record.get("path", "")).resolve()
            == validation_checkpoint
            and output_record.get("bytes") == validation_checkpoint.stat().st_size
            and output_record.get("sha256") == sha256_file(validation_checkpoint)
            and output_record.get("optimizer_states_present") is False
            and output_record.get("state_dict_tensors", 0) > 0
        ):
            raise ValueError("weights-only validation reference manifest is invalid")
    projection = project_training_bytes_per_arm(
        training_bytes,
        checkpoint_sizes,
        args.validation_checkpoints_per_arm,
        args.rolling_full_checkpoints_per_arm,
        (
            validation_checkpoint.stat().st_size
            if validation_checkpoint is not None else None
        ),
    )
    planned_label_bytes = label_bytes * args.arms * args.labels_per_arm
    planned_training_bytes = projection["projected_bytes"] * args.arms
    fixed_reserve_bytes = int(
        (args.asset_and_optimizer_reserve_gib
         + args.physical_output_reserve_gib) * GIB
    )
    post_queue_reserve_bytes = int(args.free_space_after_queue_gib * GIB)
    estimated_new_bytes = (
        planned_label_bytes + planned_training_bytes + fixed_reserve_bytes
    )
    required_free_bytes = estimated_new_bytes + post_queue_reserve_bytes
    usage = shutil.disk_usage(args.filesystem_root.resolve())
    checks = {
        "reference_label_complete": summary.get("replications") == 20,
        "enough_free_space_with_reserve": usage.free >= required_free_bytes,
    }
    payload = {
        "schema": "hhc_shape_augmentation_disk_capacity_audit_v1",
        "status": "pass" if all(checks.values()) else "insufficient_capacity",
        "checks": checks,
        "policy": (
            "conservative apparent-byte projection; stop without deleting any "
            "existing artifact when capacity is insufficient"
        ),
        "filesystem": {
            "root": str(args.filesystem_root.resolve()),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_gib": usage.free / GIB,
        },
        "projection": {
            "arms": args.arms,
            "labels_per_arm": args.labels_per_arm,
            "reference_label_bytes": label_bytes,
            "reference_training_run_bytes": training_bytes,
            "reference_checkpoint_count": len(checkpoint_sizes),
            "reference_checkpoint_median_bytes": int(statistics.median(checkpoint_sizes)),
            "measured_full_checkpoint_bytes": projection["measured_full_checkpoint_bytes"],
            "measured_validation_checkpoint_bytes": projection[
                "measured_validation_checkpoint_bytes"
            ],
            "validation_checkpoint_storage": projection[
                "validation_checkpoint_storage"
            ],
            "validation_checkpoints_per_arm": args.validation_checkpoints_per_arm,
            "rolling_full_checkpoints_per_arm": args.rolling_full_checkpoints_per_arm,
            "planned_checkpoint_count_per_arm": projection["planned_checkpoint_count"],
            "noncheckpoint_training_bytes": projection["noncheckpoint_training_bytes"],
            "count_based_training_bytes_per_arm": projection["count_based_bytes"],
            "projected_training_bytes_per_arm": projection["projected_bytes"],
            "planned_label_bytes": planned_label_bytes,
            "planned_training_bytes": planned_training_bytes,
            "asset_and_optimizer_reserve_gib": args.asset_and_optimizer_reserve_gib,
            "physical_output_reserve_gib": args.physical_output_reserve_gib,
            "estimated_new_bytes": estimated_new_bytes,
            "estimated_new_gib": estimated_new_bytes / GIB,
            "required_free_bytes": required_free_bytes,
            "required_free_gib": required_free_bytes / GIB,
            "free_space_after_queue_gib": args.free_space_after_queue_gib,
            "projected_free_after_queue_gib": (
                usage.free - estimated_new_bytes
            ) / GIB,
        },
        "references": {
            "label": {
                "path": str(reference_label),
                "summary_sha256": sha256_file(reference_label / "summary.json"),
            },
            "training_run": {"path": str(reference_training)},
            "validation_checkpoint": (
                {
                    "path": str(validation_checkpoint),
                    "bytes": validation_checkpoint.stat().st_size,
                    "sha256": sha256_file(validation_checkpoint),
                }
                if validation_checkpoint is not None else None
            ),
            "validation_checkpoint_manifest": (
                {
                    "path": str(validation_manifest),
                    "bytes": validation_manifest.stat().st_size,
                    "sha256": sha256_file(validation_manifest),
                }
                if validation_manifest is not None else None
            ),
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
