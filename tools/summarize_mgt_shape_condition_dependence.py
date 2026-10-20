#!/usr/bin/env python3
"""Pair train/validation HHC-M shape-dependence audits."""

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


def load(path: Path, expected_role: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "hhc_mgt_shape_condition_dependence_audit_v1":
        raise ValueError(f"wrong shape-dependence schema: {path}")
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete shape-dependence audit: {path}")
    if payload.get("split_role") != expected_role:
        raise ValueError(f"expected {expected_role} audit: {path}")
    return payload


def condition_effect(payload: dict, condition: str) -> dict:
    aggregate = payload["metrics"]["aggregate"]
    requested = aggregate["requested"]
    candidate = aggregate[condition]
    return {
        "requested_accuracy_percent": float(requested["accuracy_percent"]),
        "candidate_accuracy_percent": float(candidate["accuracy_percent"]),
        "requested_accuracy_advantage_percent_points": (
            float(requested["accuracy_percent"])
            - float(candidate["accuracy_percent"])
        ),
        "requested_cross_entropy": float(requested["cross_entropy"]),
        "candidate_cross_entropy": float(candidate["cross_entropy"]),
        "requested_cross_entropy_advantage": (
            float(candidate["cross_entropy"])
            - float(requested["cross_entropy"])
        ),
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
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--minimum-train-advantage-pp", type=float, default=5.0
    )
    parser.add_argument(
        "--minimum-train-minus-validation-pp", type=float, default=3.0
    )
    args = parser.parse_args()

    training = load(args.training, "training")
    validation = load(args.validation, "validation")
    for key in ("checkpoint", "seed"):
        if training.get(key) != validation.get(key):
            raise ValueError(f"train/validation pairing mismatch: {key}")
    effects = {}
    for condition in (
        "permuted_beta_and_gender", "zero_beta_requested_gender"
    ):
        train_effect = condition_effect(training, condition)
        validation_effect = condition_effect(validation, condition)
        train_minus_validation = (
            train_effect["requested_accuracy_advantage_percent_points"]
            - validation_effect["requested_accuracy_advantage_percent_points"]
        )
        effects[condition] = {
            "training": train_effect,
            "validation": validation_effect,
            "train_minus_validation_requested_accuracy_advantage_pp": (
                train_minus_validation
            ),
        }
    permuted = effects["permuted_beta_and_gender"]
    shortcut_supported = (
        permuted["training"][
            "requested_accuracy_advantage_percent_points"
        ] >= args.minimum_train_advantage_pp
        and permuted[
            "train_minus_validation_requested_accuracy_advantage_pp"
        ] >= args.minimum_train_minus_validation_pp
    )
    payload = {
        "schema": "hhc_mgt_shape_condition_dependence_pair_v1",
        "status": "pass",
        "decision": (
            "train_only_shape_identity_shortcut_supported"
            if shortcut_supported
            else "train_only_shape_identity_shortcut_not_established"
        ),
        "thresholds_preregistered_before_execution": {
            "minimum_train_requested_shape_advantage_percent_points": (
                args.minimum_train_advantage_pp
            ),
            "minimum_train_minus_validation_advantage_percent_points": (
                args.minimum_train_minus_validation_pp
            ),
        },
        "effects": effects,
        "checkpoint": training["checkpoint"],
        "inputs": {
            "training": {
                "path": str(args.training.resolve()),
                "sha256": sha256_file(args.training),
            },
            "validation": {
                "path": str(args.validation.resolve()),
                "sha256": sha256_file(args.validation),
            },
        },
        "interpretation": (
            "This paired token diagnostic tests shape-identity reliance only. "
            "It motivates but does not substitute for the matched decoded-motion "
            "shape-augmentation FID experiment."
        ),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
