#!/usr/bin/env python3
"""Fail closed unless exact frame batching passes multiple real-motion gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hhc_volumetric_smpl_frame_batch_benchmark_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


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


def assess(
    records: list[tuple[Path, dict]],
    *,
    minimum_motions: int,
    maximum_loss_relative_difference: float,
    maximum_gradient_relative_difference: float,
    minimum_speedup: float,
    maximum_peak_allocated_gib: float,
    expected_query_chunk: int,
) -> tuple[dict, bool]:
    if len(records) < minimum_motions:
        raise ValueError(
            f"need at least {minimum_motions} benchmark files, got {len(records)}"
        )
    production_path = ROOT / "optimization/volumetric_smpl_backend.py"
    experimental_path = ROOT / "optimization/volumetric_smpl_batching.py"
    benchmark_path = ROOT / "tools/benchmark_volumetric_smpl_frame_batching.py"
    expected_implementation = {
        "production": sha256_file(production_path),
        "experimental": sha256_file(experimental_path),
        "benchmark": sha256_file(benchmark_path),
    }
    input_identities = set()
    implementation_hashes = set()
    per_motion = []
    common_sizes = None
    for path, payload in records:
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"wrong benchmark schema: {path}")
        if payload.get("status") != "benchmark_complete_not_promoted":
            raise ValueError(f"benchmark was already promoted or incomplete: {path}")
        if not str(payload.get("device", "")).startswith("cuda"):
            raise ValueError(f"real promotion benchmark must use CUDA: {path}")
        inputs = payload.get("inputs", {})
        identity = (
            inputs.get("p1", {}).get("sha256"),
            inputs.get("p2", {}).get("sha256"),
        )
        if not all(isinstance(value, str) and len(value) == 64 for value in identity):
            raise ValueError(f"benchmark has invalid input hashes: {path}")
        if identity in input_identities:
            raise ValueError(f"duplicate real-motion benchmark input: {path}")
        input_identities.add(identity)

        implementation = payload.get("implementation", {})
        observed_implementation = (
            implementation.get("production", {}).get("sha256"),
            implementation.get("experimental", {}).get("sha256"),
            implementation.get("benchmark", {}).get("sha256"),
        )
        if observed_implementation != (
            expected_implementation["production"],
            expected_implementation["experimental"],
            expected_implementation["benchmark"],
        ):
            raise ValueError(f"benchmark source hashes are stale: {path}")
        implementation_hashes.add(observed_implementation)

        if not isinstance(payload.get("frames"), int) or payload["frames"] < 12:
            raise ValueError(f"benchmark must cover at least 12 real frames: {path}")
        if payload.get("query_chunk") != expected_query_chunk:
            raise ValueError(
                "benchmark query chunk differs from the promoted serial "
                f"optimizer contract {expected_query_chunk}: {path}"
            )

        serial = payload.get("serial", {})
        if not (
            finite(serial.get("loss"))
            and finite(serial.get("elapsed_seconds"))
            and float(serial["elapsed_seconds"]) > 0
            and isinstance(serial.get("peak_allocated_bytes"), int)
            and serial["peak_allocated_bytes"] >= 0
        ):
            raise ValueError(f"invalid serial measurement: {path}")
        candidates = {}
        for row in payload.get("experimental", []):
            size = int(row.get("frame_batch_size", 0))
            metrics_finite = all(finite(row.get(key)) for key in (
                "loss_relative_difference",
                "gradient_relative_to_serial_max",
                "speedup_over_serial",
                "elapsed_seconds",
            ))
            peak = row.get("peak_allocated_bytes")
            checks = {
                "all_metrics_finite": metrics_finite,
                "loss_equivalent": (
                    metrics_finite
                    and float(row["loss_relative_difference"])
                    <= maximum_loss_relative_difference
                ),
                "gradient_equivalent": (
                    metrics_finite
                    and float(row["gradient_relative_to_serial_max"])
                    <= maximum_gradient_relative_difference
                ),
                "speedup_sufficient": (
                    metrics_finite
                    and float(row["speedup_over_serial"]) >= minimum_speedup
                ),
                "peak_memory_within_limit": (
                    isinstance(peak, int)
                    and peak >= 0
                    and peak <= maximum_peak_allocated_gib * (1024 ** 3)
                ),
            }
            if size <= 1 or size in candidates:
                raise ValueError(f"invalid/duplicate frame batch size in {path}: {size}")
            candidates[size] = {
                "status": row.get("status", "complete"),
                "observed": {
                    "loss_relative_difference": row.get("loss_relative_difference"),
                    "gradient_relative_to_serial_max": row.get(
                        "gradient_relative_to_serial_max"
                    ),
                    "speedup_over_serial": row.get("speedup_over_serial"),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "peak_allocated_bytes": peak,
                },
                "checks": checks,
                "pass": all(checks.values()),
            }
        if not candidates:
            raise ValueError(f"benchmark has no experimental candidates: {path}")
        sizes = set(candidates)
        common_sizes = sizes if common_sizes is None else common_sizes & sizes
        per_motion.append({
            "benchmark": artifact(path),
            "input_identity": {"p1_sha256": identity[0], "p2_sha256": identity[1]},
            "frames": payload.get("frames"),
            "query_chunk": payload.get("query_chunk"),
            "serial": serial,
            "candidates": {str(key): candidates[key] for key in sorted(candidates)},
        })
    if len(implementation_hashes) != 1:
        raise ValueError("benchmark files do not use one immutable implementation")
    common_sizes = common_sizes or set()
    passing_sizes = []
    for size in sorted(common_sizes):
        rows = [motion["candidates"][str(size)] for motion in per_motion]
        if all(row["pass"] for row in rows):
            passing_sizes.append({
                "frame_batch_size": size,
                "minimum_speedup": min(
                    float(row["observed"]["speedup_over_serial"]) for row in rows
                ),
                "maximum_peak_allocated_bytes": max(
                    int(row["observed"]["peak_allocated_bytes"]) for row in rows
                ),
                "maximum_loss_relative_difference": max(
                    float(row["observed"]["loss_relative_difference"]) for row in rows
                ),
                "maximum_gradient_relative_difference": max(
                    float(row["observed"]["gradient_relative_to_serial_max"])
                    for row in rows
                ),
            })
    selected = None
    if passing_sizes:
        selected = max(
            passing_sizes,
            key=lambda row: (
                row["minimum_speedup"],
                -row["maximum_peak_allocated_bytes"],
                -row["frame_batch_size"],
            ),
        )
    passed = len(input_identities) >= minimum_motions and selected is not None
    return {
        "checks": {
            "minimum_distinct_real_motions": len(input_identities) >= minimum_motions,
            "single_immutable_implementation": len(implementation_hashes) == 1,
            "common_candidate_sizes_present": bool(common_sizes),
            "at_least_one_common_size_passes_every_motion": selected is not None,
        },
        "thresholds": {
            "minimum_motions": minimum_motions,
            "maximum_loss_relative_difference": maximum_loss_relative_difference,
            "maximum_gradient_relative_difference": maximum_gradient_relative_difference,
            "minimum_speedup": minimum_speedup,
            "maximum_peak_allocated_gib": maximum_peak_allocated_gib,
            "expected_query_chunk": expected_query_chunk,
        },
        "motions": per_motion,
        "passing_common_candidates": passing_sizes,
        "selected_candidate": selected,
        "production_edit_authorized": passed,
        "current_implementation": {
            "production": artifact(production_path),
            "experimental": artifact(experimental_path),
        },
    }, passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-motions", type=int, default=2)
    parser.add_argument("--maximum-loss-relative-difference", type=float, default=1e-5)
    parser.add_argument(
        "--maximum-gradient-relative-difference", type=float, default=1e-4
    )
    parser.add_argument("--minimum-speedup", type=float, default=1.25)
    parser.add_argument("--maximum-peak-allocated-gib", type=float, default=20.0)
    parser.add_argument(
        "--expected-query-chunk", type=int, default=10475,
        help="Promoted serial optimizer query chunk held fixed in the benchmark.",
    )
    args = parser.parse_args()
    if args.minimum_motions < 2:
        raise ValueError("promotion requires at least two distinct real motions")
    if min(
        args.maximum_loss_relative_difference,
        args.maximum_gradient_relative_difference,
        args.minimum_speedup,
        args.maximum_peak_allocated_gib,
        args.expected_query_chunk,
    ) <= 0:
        raise ValueError("all thresholds must be positive")
    records = [
        (path.resolve(), json.loads(path.read_text(encoding="utf-8")))
        for path in args.benchmark
    ]
    assessment, passed = assess(
        records,
        minimum_motions=args.minimum_motions,
        maximum_loss_relative_difference=args.maximum_loss_relative_difference,
        maximum_gradient_relative_difference=args.maximum_gradient_relative_difference,
        minimum_speedup=args.minimum_speedup,
        maximum_peak_allocated_gib=args.maximum_peak_allocated_gib,
        expected_query_chunk=args.expected_query_chunk,
    )
    payload = {
        "schema": "hhc_volumetric_smpl_frame_batching_assessment_v1",
        "status": "pass" if passed else "fail",
        **assessment,
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
