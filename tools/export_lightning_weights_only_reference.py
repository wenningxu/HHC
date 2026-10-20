#!/usr/bin/env python3
"""Export a measured Lightning weights-only checkpoint for capacity planning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


WEIGHTS_ONLY_KEYS = (
    "epoch", "global_step", "pytorch-lightning_version", "state_dict",
    "loops", "hparams_name", "hyper_parameters",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    source = args.checkpoint.resolve()
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        source, map_location="cpu", mmap=True, weights_only=False
    )
    missing = [key for key in WEIGHTS_ONLY_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(f"source Lightning checkpoint lacks keys: {missing}")
    if "optimizer_states" not in checkpoint:
        raise ValueError("source is not a measured full training-state checkpoint")
    weights_only = {key: checkpoint[key] for key in WEIGHTS_ONLY_KEYS}
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    torch.save(weights_only, temporary)
    os.replace(temporary, output)

    observed = torch.load(
        output, map_location="cpu", mmap=True, weights_only=False
    )
    if set(observed) != set(WEIGHTS_ONLY_KEYS):
        raise ValueError("exported weights-only Lightning key set differs")
    if set(observed["state_dict"]) != set(checkpoint["state_dict"]):
        raise ValueError("exported state_dict key set differs")
    payload = {
        "schema": "hhc_lightning_weights_only_capacity_reference_v1",
        "status": "pass",
        "source": {
            "path": str(source), "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "output": {
            "path": str(output), "bytes": output.stat().st_size,
            "sha256": sha256_file(output), "keys": list(WEIGHTS_ONLY_KEYS),
            "optimizer_states_present": False,
            "state_dict_tensors": len(observed["state_dict"]),
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(manifest, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
