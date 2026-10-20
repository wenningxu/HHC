#!/usr/bin/env python
"""Freeze serial-GPU runtime evidence for an Inter-X shape pilot.

The report is deliberately explicit about incomplete pilots and about the fact
that a positive-timing pilot is an unstratified extrapolation to the complete
augmentation plan.  It is a scheduling diagnostic, not an experiment-quality
gate and never authorizes the full asset run by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA = "hhc_interx_shape_augmentation_runtime_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def build_report(
    run_dir: Path,
    plan_path: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    run_dir = run_dir.resolve()
    plan_path = plan_path.resolve()
    manifest_path = run_dir / "_completion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    selected = manifest["selection"]["selected_names"]
    completed_names = manifest["completed_names"]
    if manifest["completed_samples"] != len(completed_names):
        raise ValueError("completion manifest count does not match completed_names")
    if len(completed_names) != len(set(completed_names)):
        raise ValueError("completion manifest contains duplicate names")
    if not set(completed_names) <= set(selected):
        raise ValueError("completed pilot output is outside the frozen selection")

    samples = []
    for name in completed_names:
        path = run_dir / f"{name}.json"
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        if sidecar["augmented_name"] != name:
            raise ValueError(f"{path}: augmented_name mismatch")
        diagnostics = sidecar["diagnostics"]
        elapsed = float(diagnostics["elapsed_seconds"])
        frames = int(sidecar["crop_stop"] - sidecar["crop_start"])
        timing_frames = int(sidecar["timing_positive_frames_crop"])
        if not np.isfinite(elapsed) or elapsed <= 0 or frames <= 0:
            raise ValueError(f"{path}: invalid runtime or frame count")
        samples.append({
            "name": name,
            "frames": frames,
            "timing_positive_frames": timing_frames,
            "timing_positive_fraction": timing_frames / frames,
            "elapsed_seconds": elapsed,
            "seconds_per_frame": elapsed / frames,
            "sidecar_sha256": sha256_file(path),
        })

    elapsed = np.asarray([row["elapsed_seconds"] for row in samples])
    frames = np.asarray([row["frames"] for row in samples], dtype=np.float64)
    timing = np.asarray(
        [row["timing_positive_frames"] for row in samples], dtype=np.float64,
    )
    full_count = int(plan["eligible_samples"])
    requested = int(manifest["requested_samples"])
    if requested != len(selected):
        raise ValueError("requested_samples does not match frozen selection")

    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if len(samples) == requested else "provisional",
        "interpretation": (
            "scheduling diagnostic only; this report neither measures physical "
            "quality nor authorizes the full asset run"
        ),
        "extrapolation_scope": (
            "unstratified positive-timing pilot extrapolated to all eligible "
            "plan records; use for a final scheduling decision only after the "
            "frozen pilot is complete"
        ),
        "run": {
            "directory": str(run_dir),
            "completion_manifest": str(manifest_path),
            "completion_manifest_sha256": sha256_file(manifest_path),
            "mode": manifest["optimizer"]["mode"],
            "signed_distance_backend": manifest["optimizer"].get(
                "signed_distance_backend", "legacy_unspecified"
            ),
            "query_chunk": manifest["optimizer"]["query_chunk"],
            "iterations": manifest["optimizer"]["iterations"],
            "requested_samples": requested,
            "completed_samples": len(samples),
            "remaining_samples": requested - len(samples),
        },
        "plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
            "eligible_samples": full_count,
            "positive_timing_samples": int(plan["positive_timing_samples"]),
        },
        "samples": samples,
    }
    if not samples:
        report["runtime"] = None
        return report

    runtime = {
        "elapsed_seconds": {
            "sum": float(elapsed.sum()),
            "mean": float(elapsed.mean()),
            "median": percentile(elapsed, 50),
            "minimum": float(elapsed.min()),
            "maximum": float(elapsed.max()),
            "p25": percentile(elapsed, 25),
            "p75": percentile(elapsed, 75),
        },
        "correlation": {
            "elapsed_vs_frames": correlation(elapsed, frames),
            "elapsed_vs_timing_positive_frames": correlation(elapsed, timing),
        },
        "naive_serial_projection": {
            "target_samples": full_count,
            "seconds": float(elapsed.mean() * full_count),
            "gpu_days": float(elapsed.mean() * full_count / 86400.0),
        },
    }
    if len(samples) >= 2 and bootstrap_replicates > 0:
        rng = np.random.default_rng(seed)
        means = rng.choice(
            elapsed,
            size=(bootstrap_replicates, len(elapsed)),
            replace=True,
        ).mean(axis=1)
        days = means * full_count / 86400.0
        runtime["bootstrap_mean_projection_95"] = {
            "seed": seed,
            "replicates": bootstrap_replicates,
            "gpu_days_lower": percentile(days, 2.5),
            "gpu_days_upper": percentile(days, 97.5),
        }
    report["runtime"] = runtime
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        args.run_dir,
        args.plan,
        args.bootstrap_replicates,
        args.seed,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
