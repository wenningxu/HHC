#!/usr/bin/env python
"""Fail closed on the 25-sample VolumetricSMPL shape-augmentation pilot."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nearest-completion", type=Path, required=True)
    parser.add_argument("--volumetric-completion", type=Path, required=True)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--query-gate", type=Path, required=True)
    parser.add_argument("--frame-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nearest = json.loads(args.nearest_completion.read_text())
    volume = json.loads(args.volumetric_completion.read_text())
    assessment = json.loads(args.assessment.read_text())
    query = json.loads(args.query_gate.read_text())
    frame = json.loads(args.frame_gate.read_text())
    raw = assessment["aggregate"]["original_mmm_raw_output"]
    policies = volume["optimizer"]
    nearest_runtime = float(nearest["runtime_summary"]["optimizer_wall_seconds_mean"])
    volume_runtime = float(volume["runtime_summary"]["optimizer_wall_seconds_mean"])
    projected_hours = 5868 * volume_runtime / 3600
    selected_frame = frame.get("selected_candidate") or {}
    selected_query = query.get("query_chunk_authorized")
    checks = {
        "same_25_selected_samples": (
            nearest.get("completed_samples") == 25
            and volume.get("completed_samples") == 25
            and nearest.get("completed_names") == volume.get("completed_names")
        ),
        "current_query_gate_passed": (
            query.get("status") == "pass"
            and isinstance(selected_query, int)
            and selected_query >= 1024
            and policies.get("query_chunk") == selected_query
        ),
        "current_frame_gate_passed": (
            frame.get("status") == "pass"
            and frame.get("production_edit_authorized") is True
            and policies.get("volumetric_smpl_frame_batch_size")
            == selected_frame.get("frame_batch_size")
        ),
        "only_authorized_optimizer_contract": (
            policies.get("signed_distance_backend") == "volumetric_smpl"
            and policies.get("contact_loss_reduction_policy") == "original_mmm"
            and policies.get("foot_contact_mask_policy") == "released_dynamic"
            and policies.get("optimization_variable_policy") == "released_all_per_frame"
            and policies.get("iterations") == 200
            and policies.get("fine_stage_start") == 100
            and policies.get("learning_rate") == 0.01
            and policies.get("loss_weight_overrides") == {}
            and policies.get("volumetric_smpl_penetration_vertex_count") is None
            and policies.get("volumetric_smpl_body_encoding_samples") is None
            and policies.get("volumetric_smpl_penetration_frame_count") is None
        ),
        "mismatch_mean_improves": raw["mismatch"]["mean_delta_vs_random"] < 0,
        "foot_skate_does_not_increase": raw["skate"]["mean_delta_vs_random"] <= 1e-6,
        "runtime_faster_than_nearest": volume_runtime < nearest_runtime,
        "projected_single_gpu_runtime_within_one_day": projected_hours <= 24.0,
    }
    passed = all(checks.values())
    result = {
        "schema": "hhc_volumetric_shape_pilot25_gate_v1",
        "status": "pass" if passed else "fail",
        "checks": checks,
        "metrics": raw,
        "runtime": {
            "nearest_mean_seconds": nearest_runtime,
            "volumetric_mean_seconds": volume_runtime,
            "speedup": nearest_runtime / volume_runtime,
            "projected_full_5868_hours_single_gpu": projected_hours,
        },
        "artifacts": {
            "nearest_completion": artifact(args.nearest_completion),
            "volumetric_completion": artifact(args.volumetric_completion),
            "assessment": artifact(args.assessment),
            "query_gate": artifact(args.query_gate),
            "frame_gate": artifact(args.frame_gate),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
