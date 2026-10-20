#!/usr/bin/env python3
"""Prove that a TiMotion validation weights checkpoint matches its full state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch


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
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--weights-checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    full_path = args.full_checkpoint.resolve()
    weights_path = args.weights_checkpoint.resolve()
    metrics_path = args.metrics.resolve()
    full = torch.load(full_path, map_location="cpu", weights_only=False)
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    full_state = full.get("state_dict")
    weights_state = weights.get("state_dict")
    if not isinstance(full_state, dict) or not isinstance(weights_state, dict):
        raise ValueError("both checkpoints must contain a Lightning state_dict")
    if int(full.get("global_step", -1)) != args.expected_step:
        raise ValueError("full checkpoint global_step mismatch")
    if set(full_state) != set(weights_state):
        raise ValueError("full and weights-only state_dict key sets differ")

    floating_tensors = 0
    tensor_values = 0
    for key in sorted(full_state):
        left, right = full_state[key], weights_state[key]
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            if left != right:
                raise ValueError(f"non-tensor state differs: {key}")
            continue
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError(f"tensor metadata differs: {key}")
        if not torch.equal(left, right):
            raise ValueError(f"tensor values differ: {key}")
        tensor_values += left.numel()
        if left.is_floating_point() or left.is_complex():
            floating_tensors += 1
            if not bool(torch.isfinite(left).all()):
                raise FloatingPointError(f"non-finite model tensor: {key}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = metrics.get("main_generation_metadata", {})
    weights_artifact = artifact(weights_path)
    checks = {
        "full_step_matches": int(full.get("global_step", -1)) == args.expected_step,
        "metrics_complete_on_544_validation_samples": (
            metrics.get("status") == "complete"
            and metrics.get("samples") == 544
            and metrics.get("generation_request_hash_verified") is True
        ),
        "metrics_are_finite": all(
            isinstance(metrics.get(name), (int, float))
            and math.isfinite(float(metrics[name]))
            for name in ("fid", "diversity", "matching", "top1", "top2", "top3")
        ),
        "metrics_bind_weights_checkpoint_path": (
            Path(metadata.get("checkpoint", "")).resolve() == weights_path
        ),
        "metrics_bind_weights_checkpoint_sha256": (
            metadata.get("checkpoint_sha256") == weights_artifact["sha256"]
        ),
        "state_dict_keys_identical": set(full_state) == set(weights_state),
        "state_dict_values_bitwise_identical": True,
        "state_dict_finite": True,
    }
    if not all(checks.values()):
        raise ValueError(f"TiMotion validation checkpoint pair failed: {checks}")
    payload = {
        "schema": "hhc_timotion_validation_checkpoint_pair_audit_v1",
        "status": "pass",
        "expected_step": args.expected_step,
        "checks": checks,
        "state_dict": {
            "keys": len(full_state),
            "floating_tensors": floating_tensors,
            "tensor_values": tensor_values,
        },
        "validation_metrics": {
            name: metrics[name]
            for name in ("fid", "diversity", "matching", "top1", "top2", "top3")
        },
        "artifacts": {
            "full_checkpoint": artifact(full_path),
            "weights_checkpoint": weights_artifact,
            "metrics": artifact(metrics_path),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
