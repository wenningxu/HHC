#!/usr/bin/env python3
"""Measure exact serial VolumetricSMPL query-chunk alternatives."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimization.volumetric_smpl_backend import VolumetricSMPLBackend
from tools.benchmark_volumetric_smpl_frame_batching import run_once, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--baseline-query-chunk", type=int, default=512)
    parser.add_argument(
        "--candidate-query-chunks", type=int, nargs="+", default=(1024, 2048)
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chunks = [args.baseline_query_chunk, *args.candidate_query_chunks]
    if args.frames < 12 or any(chunk < 1 for chunk in chunks):
        raise ValueError("use at least 12 frames and positive query chunks")
    if len(set(chunks)) != len(chunks):
        raise ValueError("query chunks must be unique")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("scheduling benchmark must run on CUDA")
    genders = set()
    available_frames = []
    for path in (args.p1, args.p2):
        with np.load(path, allow_pickle=False) as archive:
            genders.add(str(archive["gender"].item()))
            available_frames.append(int(archive["poses"].shape[0]))
    frames = min(args.frames, *available_frames)
    if frames < 12:
        raise ValueError("inputs do not contain 12 frames")
    backend = VolumetricSMPLBackend(
        model_root="body_models",
        device=device,
        genders=tuple(sorted(genders)),
        max_query_points_per_call=args.baseline_query_chunk,
    )

    runs = []
    for chunk in chunks:
        backend.max_query_points_per_call = int(chunk)
        result = run_once(
            backend, args.p1, args.p2, frames, device, frame_batch_size=1
        )
        result["query_chunk"] = int(chunk)
        runs.append(result)
    baseline = runs[0]
    baseline_gradients = baseline.pop("gradients")
    comparisons = []
    for result in runs[1:]:
        gradients = result.pop("gradients")
        gradient_max_abs = max(
            float((gradients[key] - baseline_gradients[key]).abs().max())
            for key in baseline_gradients
        )
        gradient_reference_max_abs = max(
            float(baseline_gradients[key].abs().max())
            for key in baseline_gradients
        )
        loss_abs = abs(result["loss"] - baseline["loss"])
        comparisons.append({
            **result,
            "speedup_over_baseline": (
                baseline["elapsed_seconds"] / result["elapsed_seconds"]
            ),
            "loss_absolute_difference": loss_abs,
            "loss_relative_difference": loss_abs / max(abs(baseline["loss"]), 1e-12),
            "gradient_max_absolute_difference": gradient_max_abs,
            "gradient_relative_to_baseline_max": (
                gradient_max_abs / max(gradient_reference_max_abs, 1e-12)
            ),
        })

    source = ROOT / "optimization/volumetric_smpl_backend.py"
    payload = {
        "schema": "hhc_volumetric_smpl_query_chunk_benchmark_v1",
        "status": "benchmark_complete_not_promoted",
        "device": str(device),
        "frames": frames,
        "inputs": {
            "p1": {"path": str(args.p1.resolve()), "sha256": sha256_file(args.p1)},
            "p2": {"path": str(args.p2.resolve()), "sha256": sha256_file(args.p2)},
        },
        "baseline": baseline,
        "candidates": comparisons,
        "promotion_gate": {
            "maximum_loss_relative_difference": 1e-5,
            "maximum_gradient_relative_difference": 1e-4,
            "minimum_speedup": 1.25,
            "maximum_peak_allocated_gib": 20.0,
            "minimum_distinct_real_motions": 2,
        },
        "implementation": {
            "production": {
                "path": str(source),
                "sha256": sha256_file(source),
            },
            "benchmark": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
