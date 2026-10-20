#!/usr/bin/env python3
"""Quantify contact/regularizer scale balance in a shape-augmentation run.

This is a diagnostic only.  It does not change the preregistered optimizer or
reinterpret physical-quality gates.  In particular, it makes the sequence-
length and contact-pair factors of the current raw-sum reduction explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


LOSS_WEIGHTS = {
    "attract": 1.0,
    "pene": 5.0,
    "reg": 1.0,
    "smooth": 0.1,
    "acc": 0.1,
    "skate": 100.0,
    "ground": 20.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_summary(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {"count": 0, "minimum": None, "median": None,
                "mean": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def weighted_contributions(row, weights):
    return {
        key: float(row[key]) * float(weights[key])
        for key in LOSS_WEIGHTS
    }


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else None


def audit(run_dir: Path, require_complete: bool = False):
    manifest_path = run_dir / "_completion_manifest.json"
    contract_path = run_dir / "_run_contract.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    reduction_policy = contract["optimizer"].get(
        "contact_loss_reduction_policy", "paper_literal_raw_sum"
    )
    if reduction_policy not in (
        "paper_literal_raw_sum", "active_contact_mean"
    ):
        raise ValueError(
            f"unknown contact-loss reduction policy: {reduction_policy!r}"
        )
    requested = int(manifest["requested_samples"])
    completed = int(manifest["completed_samples"])
    complete = bool(manifest["complete"])
    if require_complete and not complete:
        raise ValueError(f"run is incomplete: {completed}/{requested}")

    records = []
    for path in sorted(run_dir.glob("G*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        diagnostics = payload["diagnostics"]
        history = diagnostics["loss_history"]
        fine_start = int(diagnostics["fine_stage_start"])
        if len(history) != int(diagnostics["iterations"]):
            raise ValueError(f"incomplete loss history: {path}")
        if history[fine_start - 1]["iteration"] != fine_start - 1:
            raise ValueError(f"unexpected coarse-stage boundary: {path}")
        if history[fine_start]["iteration"] != fine_start:
            raise ValueError(f"unexpected fine-stage boundary: {path}")
        weights = dict(LOSS_WEIGHTS)
        weights.update(diagnostics["loss_weights"])
        coarse = weighted_contributions(history[fine_start - 1], weights)
        fine = weighted_contributions(history[fine_start], weights)
        active_frames = int(payload["timing_positive_frames_crop"])
        contact_pairs = int(payload["structured_contact_rows"])
        anchor_pairs = int(diagnostics["frozen_contact_region_pairs"])
        anchor_vertices = int(diagnostics["frozen_contact_vertex_pairs"])
        reduction_denominator = active_frames * max(contact_pairs, 1)
        coarse_mean_contact = None
        if contact_pairs:
            coarse_mean_contact = coarse["attract"]
            if reduction_policy == "paper_literal_raw_sum":
                coarse_mean_contact /= reduction_denominator
        records.append({
            "source_name": payload["source_name"],
            "active_contact_frames": active_frames,
            "semantic_contact_pairs": contact_pairs,
            "fine_anchor_region_pairs": anchor_pairs,
            "fine_anchor_vertex_pairs": anchor_vertices,
            "raw_sum_over_active_frame_pair_mean_scale": (
                reduction_denominator if contact_pairs else None
            ),
            "coarse_last_weighted_contributions": coarse,
            "fine_first_weighted_contributions": fine,
            "coarse_contact_to_skate_ratio": (
                safe_ratio(coarse["attract"], coarse["skate"])
                if contact_pairs else None
            ),
            "coarse_active_mean_contact_to_skate_ratio": (
                safe_ratio(coarse_mean_contact, coarse["skate"])
                if contact_pairs else None
            ),
            "fine_contact_to_skate_ratio": (
                safe_ratio(fine["attract"], fine["skate"])
                if anchor_pairs else None
            ),
            "fine_stage_total_loss_jump_ratio": safe_ratio(
                history[fine_start]["total"],
                history[fine_start - 1]["total"],
            ),
        })

    def values(key):
        return [row[key] for row in records if row[key] is not None]

    output = {
        "schema": "hhc_shape_augmentation_loss_balance_audit_v1",
        "status": "pass",
        "scope": "diagnostic only; no optimizer or physical gate is changed",
        "run": {
            "directory": str(run_dir.resolve()),
            "contract_sha256": sha256_file(contract_path),
            "manifest_sha256": sha256_file(manifest_path),
            "complete": complete,
            "completed_samples": completed,
            "requested_samples": requested,
            "foot_contact_mask_policy": contract["optimizer"][
                "foot_contact_mask_policy"
            ],
            "contact_loss_reduction_policy": reduction_policy,
        },
        "reduction_contract": {
            "observed_contact_reduction": reduction_policy,
            "paper_literal_raw_sum": (
                "raw sum over active frames and semantic pairs; the fine "
                "anchor stage additionally sums matched anchor vertices"
            ),
            "comparison_reduction": (
                "mean over active frames and semantic pairs; this isolates "
                "reduction scale and is not an old-code equivalence claim"
            ),
            "raw_sum_over_mean_scale_formula": (
                "active_contact_frames * semantic_contact_pairs"
            ),
            "weights": LOSS_WEIGHTS,
        },
        "aggregate": {
            "raw_sum_over_active_frame_pair_mean_scale": finite_summary(
                values("raw_sum_over_active_frame_pair_mean_scale")
            ),
            "coarse_contact_to_skate_ratio": finite_summary(
                values("coarse_contact_to_skate_ratio")
            ),
            "coarse_active_mean_contact_to_skate_ratio": finite_summary(
                values("coarse_active_mean_contact_to_skate_ratio")
            ),
            "fine_contact_to_skate_ratio": finite_summary(
                values("fine_contact_to_skate_ratio")
            ),
            "fine_stage_total_loss_jump_ratio": finite_summary(
                values("fine_stage_total_loss_jump_ratio")
            ),
        },
        "records": records,
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit(args.run_dir, require_complete=args.require_complete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
