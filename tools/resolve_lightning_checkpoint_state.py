#!/usr/bin/env python
"""Resolve whether one full Lightning checkpoint needs a strict resume."""

import argparse
import json
import os
from pathlib import Path

import torch


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve(checkpoint_path: Path, expected_final_step: int) -> dict:
    checkpoint_path = checkpoint_path.resolve()
    if expected_final_step <= 0:
        raise ValueError("expected final step must be positive")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    # mmap avoids materializing multi-gigabyte model/Adam tensors merely to
    # read the Lightning progress and structural resume state.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    global_step = int(checkpoint.get("global_step", -1))
    if global_step < 0:
        raise ValueError("checkpoint has no valid global_step")
    if global_step > expected_final_step:
        raise ValueError(
            f"checkpoint step {global_step} exceeds expected final step "
            f"{expected_final_step}"
        )
    required_nonempty = ("state_dict", "optimizer_states", "lr_schedulers", "loops")
    missing = [key for key in required_nonempty if not checkpoint.get(key)]
    if missing:
        raise ValueError(
            "checkpoint cannot provide a full stateful resume; missing "
            f"{missing}"
        )

    status = "complete" if global_step == expected_final_step else "resume"
    stat = checkpoint_path.stat()
    return {
        "schema": "hhc_lightning_checkpoint_state_resolution_v1",
        "status": status,
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "global_step": global_step,
        "epoch": int(checkpoint.get("epoch", -1)),
        "expected_final_step": expected_final_step,
        "training_required": status == "resume",
        "resume_checkpoint": str(checkpoint_path) if status == "resume" else None,
        "full_state_present": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-final-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = resolve(args.checkpoint, args.expected_final_step)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
