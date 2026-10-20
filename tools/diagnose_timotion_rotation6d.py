#!/usr/bin/env python3
"""Diagnose unconstrained 6D rotations in TiMotion official Inter-X H5 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def diagnose(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        motions = np.asarray(handle["motions"][:, 0], dtype=np.float32)
        lengths = np.asarray(handle["lengths"], dtype=np.int64)
        names = np.asarray(handle["names"]).astype("U")

    if motions.ndim != 3 or motions.shape[-1] != 56 * 12:
        raise ValueError(f"unexpected official motion shape: {motions.shape}")
    if np.any(lengths <= 0) or np.any(lengths > motions.shape[1]):
        raise ValueError("lengths fall outside the stored frame dimension")

    # Joint slot 55 stores root translation rather than a rotation. The
    # evaluator representation has rotations only in slots 0..54.
    rotations_by_sample = motions.reshape(
        motions.shape[0], motions.shape[1], 56, 12
    )[..., :55, :]
    rotations_by_sample = np.concatenate(
        (rotations_by_sample[..., :6], rotations_by_sample[..., 6:]), axis=2
    )
    valid = np.arange(motions.shape[1])[None, :] < lengths[:, None]
    rotations = rotations_by_sample[valid]

    a1 = rotations[..., :3].astype(np.float64)
    a2 = rotations[..., 3:].astype(np.float64)
    norm1 = np.linalg.norm(a1, axis=-1)
    norm2 = np.linalg.norm(a2, axis=-1)
    safe1 = np.maximum(norm1, 1e-30)
    safe2 = np.maximum(norm2, 1e-30)
    b1 = a1 / safe1[..., None]
    a2_orthogonal = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    norm2_orthogonal = np.linalg.norm(a2_orthogonal, axis=-1)
    b2 = a2_orthogonal / np.maximum(norm2_orthogonal, 1e-30)[..., None]
    canonical = np.concatenate((b1, b2), axis=-1)
    residual = np.linalg.norm(rotations.astype(np.float64) - canonical, axis=-1)
    abs_cosine = np.abs(np.sum(a1 * a2, axis=-1) / (safe1 * safe2))
    absolute = np.abs(rotations)

    sample_max_component = np.zeros(motions.shape[0], dtype=np.float64)
    sample_max_residual = np.zeros(motions.shape[0], dtype=np.float64)
    sample_p99_residual = np.zeros(motions.shape[0], dtype=np.float64)
    for index, length in enumerate(lengths):
        current = rotations_by_sample[index, :length].astype(np.float64)
        current_a1 = current[..., :3]
        current_a2 = current[..., 3:]
        current_b1 = current_a1 / np.maximum(
            np.linalg.norm(current_a1, axis=-1, keepdims=True), 1e-30
        )
        current_a2_orthogonal = current_a2 - np.sum(
            current_b1 * current_a2, axis=-1, keepdims=True
        ) * current_b1
        current_b2 = current_a2_orthogonal / np.maximum(
            np.linalg.norm(current_a2_orthogonal, axis=-1, keepdims=True),
            1e-30,
        )
        current_canonical = np.concatenate((current_b1, current_b2), axis=-1)
        current_residual = np.linalg.norm(current - current_canonical, axis=-1)
        sample_max_component[index] = np.max(np.abs(current))
        sample_max_residual[index] = np.max(current_residual)
        sample_p99_residual[index] = np.percentile(current_residual, 99)

    worst_index = int(np.argmax(sample_p99_residual))
    return {
        "path": str(path.resolve()),
        "samples": int(motions.shape[0]),
        "valid_frames": int(np.sum(lengths)),
        "rotation6d_vectors": int(rotations.shape[0] * rotations.shape[1]),
        "first_axis_norm": percentiles(norm1),
        "second_axis_norm": percentiles(norm2),
        "absolute_axis_cosine": percentiles(abs_cosine),
        "canonicalization_l2_residual": percentiles(residual),
        "degenerate_first_axis_count": int(np.sum(norm1 < 1e-6)),
        "degenerate_orthogonal_second_axis_count": int(
            np.sum(norm2_orthogonal < 1e-6)
        ),
        "component_abs_gt_1_fraction": float(np.mean(absolute > 1.0)),
        "component_abs_gt_2_fraction": float(np.mean(absolute > 2.0)),
        "worst_sample_by_p99_residual": {
            "index": worst_index,
            "name": str(names[worst_index]),
            "length": int(lengths[worst_index]),
            "p99_residual": float(sample_p99_residual[worst_index]),
            "max_residual": float(sample_max_residual[worst_index]),
            "max_abs_component": float(sample_max_component[worst_index]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([diagnose(path) for path in args.h5], indent=2))


if __name__ == "__main__":
    main()
