#!/usr/bin/env python3
"""Gate fast shape optimization with conservative per-sample fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {
        'path': str(path), 'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


SPEED_CORE_KEYS = (
    'iterations',
    'learning_rate',
    'fine_stage_start',
    'seed',
    'seed_policy',
    'foot_contact_mask_policy',
    'contact_loss_reduction_policy',
    'optimization_variable_policy',
    'temporal_mask',
    'mode',
)


def speed_work_contract(completion: dict) -> dict:
    """Return backend-independent work that must match for timing claims."""
    optimizer = completion.get('optimizer', {})
    return {
        'optimizer': {key: optimizer.get(key) for key in SPEED_CORE_KEYS},
        'requested_samples': completion.get('requested_samples'),
        'completed_samples': completion.get('completed_samples'),
        'completed_names': completion.get('completed_names'),
        'plan_records_sha256': completion.get('plan_records_sha256'),
        'contact_manifest_sha256': completion.get('contact_manifest_sha256'),
        'source_motion_sha256': completion.get('source_motion_sha256'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument(
        "--nearest-completion", type=Path,
        help=(
            "Optional same-sample nearest-vertex completion. When supplied, "
            "the VolumetricSMPL candidate must be strictly faster."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skate-tolerance", type=float, default=2e-6)
    parser.add_argument("--minimum-accepted", type=int, default=5)
    parser.add_argument("--maximum-two-gpu-hours", type=float, default=24.0)
    args = parser.parse_args()

    assessment = json.loads(args.assessment.read_text(encoding="utf-8"))
    completion = json.loads(args.completion.read_text(encoding="utf-8"))
    nearest = (
        json.loads(args.nearest_completion.read_text(encoding='utf-8'))
        if args.nearest_completion is not None else None
    )
    rows = assessment.get("rows", [])
    if assessment.get("samples") != 25 or len(rows) != 25:
        raise ValueError("promotion pilot must contain exactly 25 samples")
    if completion.get("complete") is not True or completion.get("completed_samples") != 25:
        raise ValueError("optimizer completion is incomplete")

    accepted = []
    selected_deltas = {key: [] for key in ("mismatch", "pene", "skate")}
    for row in rows:
        baseline = row["conditions"]["random_shape_mismatch"]
        candidate = row["conditions"]["foot_preserving_projection"]
        keep = (
            candidate["mismatch"] < baseline["mismatch"] - 1e-8
            and candidate["pene"] <= baseline["pene"] + 1e-8
            and abs(candidate["skate"] - baseline["skate"])
            <= args.skate_tolerance
        )
        if keep:
            accepted.append(row["augmented_name"])
        for metric in selected_deltas:
            selected = candidate[metric] if keep else baseline[metric]
            selected_deltas[metric].append(float(selected - baseline[metric]))

    runtime = completion["runtime_summary"]
    projected_single = float(runtime["projected_full_5868_optimizer_hours_single_gpu"])
    candidate_seconds = float(runtime['optimizer_wall_seconds_mean'])
    candidate_backend = completion.get('optimizer', {}).get(
        'signed_distance_backend', 'unknown'
    )
    nearest_seconds = None
    speed_contract_matches = None
    candidate_work_contract = speed_work_contract(completion)
    nearest_work_contract = None
    if nearest is not None:
        if (
            nearest.get('complete') is not True
            or nearest.get('completed_samples') != 25
            or nearest.get('completed_names') != completion.get('completed_names')
        ):
            raise ValueError(
                'nearest and VolumetricSMPL pilots are not the same 25 samples'
            )
        nearest_seconds = float(
            nearest['runtime_summary']['optimizer_wall_seconds_mean']
        )
        nearest_work_contract = speed_work_contract(nearest)
        speed_contract_matches = (
            candidate_work_contract == nearest_work_contract
        )
    summary = {
        "accepted": len(accepted),
        "total": len(rows),
        "acceptance_rate": len(accepted) / len(rows),
        "accepted_names": accepted,
        "selected_mismatch_delta_mean": sum(selected_deltas["mismatch"]) / len(rows),
        "selected_pene_delta_mean": sum(selected_deltas["pene"]) / len(rows),
        "selected_skate_delta_mean": sum(selected_deltas["skate"]) / len(rows),
        "selected_skate_abs_delta_max": max(map(abs, selected_deltas["skate"])),
        "projected_single_gpu_hours": projected_single,
        "projected_two_gpu_hours": projected_single / 2.0,
    }
    checks = {
        "enough_accepted_samples": len(accepted) >= args.minimum_accepted,
        "selected_mismatch_improves": summary["selected_mismatch_delta_mean"] < 0,
        "selected_penetration_does_not_degrade": summary["selected_pene_delta_mean"] <= 0,
        "selected_skate_within_tolerance": (
            summary["selected_skate_abs_delta_max"] <= args.skate_tolerance
        ),
        "two_gpu_projection_within_one_day": (
            summary["projected_two_gpu_hours"] <= args.maximum_two_gpu_hours
        ),
    }
    if nearest_seconds is not None:
        checks['same_optimizer_work_contract_for_speed'] = bool(
            speed_contract_matches
        )
        checks['volumetric_strictly_faster_than_same_sample_nearest'] = (
            candidate_backend == 'volumetric_smpl'
            and bool(speed_contract_matches)
            and candidate_seconds < nearest_seconds
        )
    passed = all(checks.values())
    payload = {
        "schema": "hhc_fast_selective_shape_pilot_gate_v2",
        "status": "pass" if passed else "fail",
        "selection_contract": (
            "use constant-delta foot-preserving projection only when mismatch "
            "strictly improves, penetration does not increase, and absolute "
            "foot-skate delta is within tolerance; otherwise retain random-shape input"
        ),
        "thresholds": {
            "samples": 25,
            "minimum_accepted": args.minimum_accepted,
            "skate_tolerance": args.skate_tolerance,
            "maximum_two_gpu_hours": args.maximum_two_gpu_hours,
        },
        "checks": checks,
        "summary": summary,
        "runtime_comparison": {
            "candidate_backend": candidate_backend,
            "candidate_mean_seconds": candidate_seconds,
            "nearest_mean_seconds": nearest_seconds,
            "same_optimizer_work_contract": speed_contract_matches,
            "candidate_work_contract": candidate_work_contract,
            "nearest_work_contract": nearest_work_contract,
            "speedup_over_nearest": (
                nearest_seconds / candidate_seconds
                if nearest_seconds is not None and speed_contract_matches
                else None
            ),
        },
        "artifacts": {
            "gate_implementation": artifact(Path(__file__)),
            "assessment": artifact(args.assessment),
            "volumetric_completion": artifact(args.completion),
            "nearest_completion": (
                artifact(args.nearest_completion)
                if args.nearest_completion is not None else None
            ),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
