#!/usr/bin/env python3
"""Authorize a larger serial query chunk only after two real-motion tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-speedup", type=float, default=1.25)
    parser.add_argument("--maximum-loss-relative-difference", type=float, default=1e-5)
    parser.add_argument("--maximum-gradient-relative-difference", type=float, default=1e-4)
    parser.add_argument("--maximum-peak-allocated-gib", type=float, default=20.0)
    args = parser.parse_args()
    if len(args.benchmark) < 2:
        raise ValueError("at least two distinct real-motion benchmarks are required")

    production = ROOT / "optimization/volumetric_smpl_backend.py"
    benchmark_source = ROOT / "tools/benchmark_volumetric_smpl_query_chunk.py"
    expected_hashes = (sha256_file(production), sha256_file(benchmark_source))
    identities = set()
    common_chunks = None
    implementation_hashes = set()
    motions = []
    for path in args.benchmark:
        path = path.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "hhc_volumetric_smpl_query_chunk_benchmark_v1":
            raise ValueError(f"wrong schema: {path}")
        if payload.get("status") != "benchmark_complete_not_promoted":
            raise ValueError(f"incomplete/already promoted benchmark: {path}")
        if not str(payload.get("device", "")).startswith("cuda") or payload.get("frames", 0) < 12:
            raise ValueError(f"benchmark is not a >=12-frame CUDA run: {path}")
        identity = tuple(
            payload.get("inputs", {}).get(person, {}).get("sha256")
            for person in ("p1", "p2")
        )
        if identity in identities or not all(isinstance(value, str) and len(value) == 64 for value in identity):
            raise ValueError(f"duplicate or invalid input identity: {path}")
        identities.add(identity)
        observed_hashes = (
            payload.get("implementation", {}).get("production", {}).get("sha256"),
            payload.get("implementation", {}).get("benchmark", {}).get("sha256"),
        )
        if observed_hashes != expected_hashes:
            raise ValueError(f"stale benchmark implementation: {path}")
        implementation_hashes.add(observed_hashes)
        baseline = payload.get("baseline", {})
        if baseline.get("query_chunk") != 512 or not finite(baseline.get("loss")) or float(baseline["loss"]) <= 0:
            raise ValueError(f"baseline must be nonzero-loss query_chunk=512: {path}")
        candidates = {}
        for row in payload.get("candidates", []):
            chunk = int(row.get("query_chunk", 0))
            peak = row.get("peak_allocated_bytes")
            metrics_finite = all(finite(row.get(key)) for key in (
                "loss_relative_difference",
                "gradient_relative_to_baseline_max",
                "speedup_over_baseline",
            ))
            checks = {
                "metrics_finite": metrics_finite,
                "loss_equivalent": metrics_finite and float(row["loss_relative_difference"]) <= args.maximum_loss_relative_difference,
                "gradient_equivalent": metrics_finite and float(row["gradient_relative_to_baseline_max"]) <= args.maximum_gradient_relative_difference,
                "speedup_sufficient": metrics_finite and float(row["speedup_over_baseline"]) >= args.minimum_speedup,
                "memory_within_limit": isinstance(peak, int) and 0 <= peak <= args.maximum_peak_allocated_gib * 1024 ** 3,
            }
            if chunk <= 512 or chunk in candidates:
                raise ValueError(f"invalid/duplicate candidate chunk {chunk}: {path}")
            candidates[chunk] = {"observed": row, "checks": checks, "pass": all(checks.values())}
        if not candidates:
            raise ValueError(f"no candidates: {path}")
        common_chunks = set(candidates) if common_chunks is None else common_chunks & set(candidates)
        motions.append({
            "benchmark": artifact(path),
            "input_identity": {"p1_sha256": identity[0], "p2_sha256": identity[1]},
            "baseline": baseline,
            "candidates": {str(key): candidates[key] for key in sorted(candidates)},
        })
    if len(implementation_hashes) != 1:
        raise ValueError("benchmarks do not share an immutable implementation")
    passing = []
    for chunk in sorted(common_chunks or []):
        rows = [motion["candidates"][str(chunk)] for motion in motions]
        if all(row["pass"] for row in rows):
            passing.append({
                "query_chunk": chunk,
                "minimum_speedup": min(float(row["observed"]["speedup_over_baseline"]) for row in rows),
                "maximum_loss_relative_difference": max(float(row["observed"]["loss_relative_difference"]) for row in rows),
                "maximum_gradient_relative_difference": max(float(row["observed"]["gradient_relative_to_baseline_max"]) for row in rows),
                "maximum_peak_allocated_bytes": max(int(row["observed"]["peak_allocated_bytes"]) for row in rows),
            })
    selected = max(passing, key=lambda row: (row["minimum_speedup"], -row["maximum_peak_allocated_bytes"])) if passing else None
    passed = len(identities) >= 2 and selected is not None
    payload = {
        "schema": "hhc_volumetric_smpl_query_chunk_assessment_v1",
        "status": "pass" if passed else "fail",
        "checks": {
            "minimum_two_distinct_nonzero_loss_motions": len(identities) >= 2,
            "single_immutable_implementation": len(implementation_hashes) == 1,
            "common_candidate_passes_every_motion": selected is not None,
        },
        "thresholds": {
            "minimum_speedup": args.minimum_speedup,
            "maximum_loss_relative_difference": args.maximum_loss_relative_difference,
            "maximum_gradient_relative_difference": args.maximum_gradient_relative_difference,
            "maximum_peak_allocated_gib": args.maximum_peak_allocated_gib,
        },
        "motions": motions,
        "passing_common_candidates": passing,
        "selected_candidate": selected,
        "query_chunk_authorized": selected["query_chunk"] if selected else None,
        "implementation": {
            "production": artifact(production),
            "benchmark": artifact(benchmark_source),
            "assessment": artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
