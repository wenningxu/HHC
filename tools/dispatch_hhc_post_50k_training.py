#!/usr/bin/env python3
"""Freeze the evidence-driven action after the 50k masked-loss control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ASSESSMENT_SCHEMA = "hhc_global_masked_loss_control_assessment_v1"
SHAPE_SCHEMA = "hhc_mgt_shape_condition_dependence_pair_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path, schema: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise ValueError(f"wrong schema for {path}: {payload.get('schema')}")
    return payload


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def decide(assessment: dict, shape_pair: dict | None) -> dict:
    decision = assessment.get("decision")
    common = {
        "schema": "hhc_post_50k_training_dispatch_v1",
        "source_decision": decision,
        "shape_training_budget": 300000,
        "global_control_training_budget": 300000,
    }
    if decision == "retain_formal_contract_and_continue_shape_training":
        return {
            **common,
            "status": "ready",
            "action": "continue_shape_augmentation",
            "continue_global_control": False,
            "shape_training_loss_contract": "paper_masked_only",
            "shape_reference_family": "formal_paper_masked_only",
            "reason": "formal validation FID already passed the preregistered core threshold",
        }
    if decision == "promote_global_masked_loss_to_full_retrain_before_shape":
        control = assessment.get("control") or {}
        if control.get("hypothesis_supported") is not True:
            raise ValueError("promotion decision lacks supported control evidence")
        return {
            **common,
            "status": "ready",
            "action": "continue_global_control_to_300k_then_shape",
            "continue_global_control": True,
            # The combined shape arm is deliberately deferred until the full
            # control confirms that the early advantage survives 300k.  The
            # matched shape-only arms retain the paper-stated objective so the
            # data augmentation effect remains identifiable.
            "shape_training_loss_contract": "paper_masked_only",
            "shape_reference_family": "formal_paper_masked_only",
            "reason": "50k FID passed both preregistered promotion margins",
        }
    if decision != "global_loss_not_sufficient_investigate_mask_conditioning_next":
        raise ValueError(f"assessment is not dispatchable: {decision}")
    if shape_pair is None:
        raise ValueError("insufficient global-loss result requires shape-dependence evidence")
    shape_decision = shape_pair.get("decision")
    if shape_pair.get("status") != "pass":
        raise ValueError("shape-dependence pair is not complete/pass")
    if shape_decision == "train_only_shape_identity_shortcut_supported":
        return {
            **common,
            "status": "ready",
            "action": "continue_shape_augmentation",
            "continue_global_control": False,
            "shape_training_loss_contract": "paper_masked_only",
            "shape_reference_family": "formal_paper_masked_only",
            "shape_source_decision": shape_decision,
            "mask_conditioning_status": "shape_identity_shortcut_supported",
            "diagnostic_followup_required": False,
            "reason": "shape-ID shortcut passed both preregistered train/validation thresholds",
        }
    if shape_decision != "train_only_shape_identity_shortcut_not_established":
        raise ValueError(f"unknown shape-dependence decision: {shape_decision}")
    return {
        **common,
        "status": "ready",
        "action": "continue_shape_augmentation",
        "continue_global_control": False,
        "shape_training_loss_contract": "paper_masked_only",
        "shape_reference_family": "formal_paper_masked_only",
        "shape_source_decision": shape_decision,
        "mask_conditioning_status": "unresolved_after_two_negative_diagnostics",
        "diagnostic_followup_required": True,
        "reason": (
            "neither diagnostic explains the weak FID; preserve that unresolved "
            "finding while continuing the user-required matched shape-augmentation arms"
        ),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--shape-pair", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assessment = load(args.assessment, ASSESSMENT_SCHEMA)
    if assessment.get("status") != "complete":
        raise ValueError(
            f"masked-loss assessment is not complete: {assessment.get('status')}"
        )
    shape_pair = None
    if args.shape_pair is not None:
        shape_pair = load(args.shape_pair, SHAPE_SCHEMA)
    payload = decide(assessment, shape_pair)
    payload["evidence"] = {"masked_loss_assessment": artifact(args.assessment)}
    if args.shape_pair is not None:
        payload["evidence"]["shape_dependence_pair"] = artifact(args.shape_pair)
    payload["implementation"] = artifact(Path(__file__))
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
