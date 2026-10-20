#!/usr/bin/env python
"""Compare old and repaired same-step MGT training states bit for bit."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recursive_equal(left, right) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and np.array_equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(recursive_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(recursive_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def differing_keys(left: dict, right: dict) -> list[str]:
    return [
        key for key in sorted(set(left) | set(right))
        if key not in left or key not in right
        or not recursive_equal(left[key], right[key])
    ]


def checkpoint_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    steps = [int(reference.get("global_step", -1)), int(candidate.get("global_step", -1))]
    if steps != [args.expected_step, args.expected_step]:
        raise ValueError(f"same-step check failed: expected {args.expected_step}, got {steps}")
    reference_model = reference.get("state_dict", {})
    candidate_model = candidate.get("state_dict", {})
    model_differences = differing_keys(reference_model, candidate_model)
    exact = {
        "model_state": not model_differences,
        "optimizer_state": recursive_equal(
            reference.get("optimizer_states"), candidate.get("optimizer_states")
        ),
        "lr_scheduler_state": recursive_equal(
            reference.get("lr_schedulers"), candidate.get("lr_schedulers")
        ),
    }
    status = "pass" if all(exact.values()) else "fail"
    payload = {
        "schema": "hhc_mgt_historical_checkpoint_pair_v1",
        "status": status,
        "expected_step": args.expected_step,
        "reference": checkpoint_record(args.reference),
        "candidate": checkpoint_record(args.candidate),
        "exact": exact,
        "model_tensor_count": len(reference_model),
        "differing_model_keys": model_differences,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
