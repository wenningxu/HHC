#!/usr/bin/env python3
"""Authorize full shape augmentation only after a fast nearest-vertex pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


SCHEMA = "hhc_interx_shape_augmentation_runtime_v1"
OUTPUT_SCHEMA = "hhc_interx_shape_augmentation_runtime_gate_v1"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-pilot-samples", type=int, default=10)
    parser.add_argument("--target-seconds-per-sample", type=float, default=10.0)
    parser.add_argument("--maximum-projected-hours", type=float, default=24.0)
    args = parser.parse_args()

    analysis_path = args.analysis.resolve()
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    run = analysis.get("run", {})
    runtime = analysis.get("runtime") or {}
    elapsed = runtime.get("elapsed_seconds", {})
    projection = runtime.get("naive_serial_projection", {})
    bootstrap = runtime.get("bootstrap_mean_projection_95", {})
    completed = int(run.get("completed_samples", -1))
    mean_seconds = float(elapsed.get("mean", math.nan))
    median_seconds = float(elapsed.get("median", math.nan))
    projected_seconds = float(projection.get("seconds", math.nan))
    target_samples = int(projection.get("target_samples", -1))
    maximum_seconds = float(args.maximum_projected_hours * 3600.0)
    maximum_mean_seconds = (
        maximum_seconds / target_samples if target_samples > 0 else math.nan
    )
    checks = {
        "analysis_schema_matches": analysis.get("schema") == SCHEMA,
        "pilot_is_complete": analysis.get("status") == "complete",
        "optimized_mode": run.get("mode") == "optimized",
        "released_nearest_vertex_backend": (
            run.get("signed_distance_backend") == "nearest_vertex"
        ),
        "minimum_pilot_size_met": completed >= args.minimum_pilot_samples,
        "runtime_values_finite_positive": all(
            math.isfinite(value) and value > 0
            for value in (mean_seconds, median_seconds, projected_seconds)
        ),
        "projection_matches_mean_times_full_count": (
            target_samples > 0
            and math.isclose(
                projected_seconds,
                mean_seconds * target_samples,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
        ),
        "full_serial_projection_within_limit": (
            math.isfinite(projected_seconds)
            and projected_seconds <= maximum_seconds
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    payload = {
        "schema": OUTPUT_SCHEMA,
        "status": status,
        "full_augmentation_authorized": status == "pass",
        "checks": checks,
        "policy": {
            "backend": "nearest_vertex",
            "target_seconds_per_sample": args.target_seconds_per_sample,
            "maximum_projected_hours": args.maximum_projected_hours,
            "minimum_pilot_samples": args.minimum_pilot_samples,
            "decision_metric": "pilot arithmetic-mean runtime times full plan count",
            "bootstrap_interval_role": "reported uncertainty; not a second hidden gate",
        },
        "observed": {
            "pilot_samples": completed,
            "target_samples": target_samples,
            "mean_seconds_per_sample": mean_seconds,
            "median_seconds_per_sample": median_seconds,
            "target_seconds_per_sample": args.target_seconds_per_sample,
            "maximum_mean_seconds_for_full_count": maximum_mean_seconds,
            "projected_seconds": projected_seconds,
            "projected_hours": projected_seconds / 3600.0,
            "bootstrap_projected_gpu_days_95": (
                [bootstrap.get("gpu_days_lower"), bootstrap.get("gpu_days_upper")]
                if bootstrap else None
            ),
        },
        "analysis": artifact(analysis_path),
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
