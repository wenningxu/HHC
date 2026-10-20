#!/usr/bin/env python3
"""Audit the mask-only stabilization of TiMotion's relative-orientation loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TIMOTION = ROOT.parent / "TIMotion"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qnormalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)


def qbetween(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    cross = torch.cross(v0, v1, dim=-1)
    scalar = torch.sqrt(
        (v0.square()).sum(-1, keepdim=True)
        * (v1.square()).sum(-1, keepdim=True)
    ) + (v0 * v1).sum(-1, keepdim=True)
    return qnormalize(torch.cat([scalar, cross], dim=-1))


def orientations(joints: torch.Tensor, valid: torch.Tensor, stabilized: bool):
    across = joints[..., 2, :] - joints[..., 1, :]
    if stabilized:
        fallback = torch.zeros_like(across)
        fallback[..., 0] = 1
        across = torch.where(valid.unsqueeze(-1), across, fallback)
    across = across / torch.linalg.vector_norm(across, dim=-1, keepdim=True)
    y_axis = torch.zeros_like(across)
    y_axis[..., 1] = 1
    forward = torch.cross(y_axis, across, dim=-1)
    forward = forward / torch.linalg.vector_norm(
        forward, dim=-1, keepdim=True
    )
    return qbetween(forward[..., 0, :], forward[..., 1, :])


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
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/timotion_padding_ro_stabilization_audit.json",
    )
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(123)
    prediction = torch.randn((2, 5, 2, 22, 3), generator=generator)
    target = torch.randn((2, 5, 2, 22, 3), generator=generator)
    valid = torch.zeros((2, 5, 2), dtype=torch.bool)
    valid[:, :3] = True
    target[:, 3:] = 0.0

    old_prediction = orientations(prediction, valid, stabilized=False)
    old_target = orientations(target, valid, stabilized=False)
    new_prediction = orientations(prediction, valid, stabilized=True)
    new_target = orientations(target, valid, stabilized=True)
    valid_time = valid[..., 0]
    old_error = (old_prediction[..., [0, 2]] - old_target[..., [0, 2]]).square()
    new_error = (new_prediction[..., [0, 2]] - new_target[..., [0, 2]]).square()
    old_masked = (old_error.mean(-1) * valid_time).sum() / valid_time.sum()
    new_masked = (new_error.mean(-1) * valid_time).sum() / valid_time.sum()

    loss_source = TIMOTION / "models/losses.py"
    source_text = loss_source.read_text(encoding="utf-8")
    bad_h5 = (
        TIMOTION / "checkpoints/"
        "TIMotion-Transformer-LPA-InterX-smoke-bs32x8-e1/"
        "official_validation/step_000000001_main.h5"
    )
    bad_values = None
    if bad_h5.is_file():
        with h5py.File(bad_h5, "r") as source:
            bad_values = int((~np.isfinite(source["motions"][:])).sum())

    remote_audit = ROOT / "output/timotion_interx_smoke_wandb_history_audit.json"
    remote = json.loads(remote_audit.read_text()) if remote_audit.is_file() else {}
    checks = {
        "released_formula_reproduces_padding_nan": (
            not torch.isfinite(old_target[:, 3:]).all().item()
            and not math.isfinite(float(old_masked))
        ),
        "stabilized_formula_is_finite": (
            torch.isfinite(new_prediction).all().item()
            and torch.isfinite(new_target).all().item()
            and math.isfinite(float(new_masked))
        ),
        "valid_prediction_values_bitwise_unchanged": torch.equal(
            old_prediction[valid_time], new_prediction[valid_time]
        ),
        "valid_target_values_bitwise_unchanged": torch.equal(
            old_target[valid_time], new_target[valid_time]
        ),
        "source_substitution_is_mask_outside_only": all(marker in source_text for marker in (
            "valid = self.mask[..., 0].bool()",
            "torch.where(valid.unsqueeze(-1), across, fallback_across)",
            "torch.where(valid.unsqueeze(-1), across_gt, fallback_across)",
        )),
        "failed_smoke_generation_contains_nonfinite_values": (
            bad_values is not None and bad_values > 0
        ),
        "fixed_smoke_remote_wandb_gate_passed": remote.get("status") == "pass",
    }
    payload = {
        "schema": "timotion_padding_ro_stabilization_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "interpretation": (
            "The change substitutes a finite direction only at padded/masked "
            "frames before the released orientation normalization. Values on "
            "the loss support are bitwise unchanged."
        ),
        "synthetic_runtime": {
            "old_masked_loss_finite": math.isfinite(float(old_masked)),
            "new_masked_loss": float(new_masked),
            "old_nonfinite_target_quaternions": int(
                (~torch.isfinite(old_target)).sum().item()
            ),
            "new_nonfinite_target_quaternions": int(
                (~torch.isfinite(new_target)).sum().item()
            ),
        },
        "failed_smoke_nonfinite_motion_values": bad_values,
        "source": {
            "path": str(loss_source.resolve()),
            "sha256": sha256_file(loss_source),
        },
        "remote_wandb_audit": {
            "path": str(remote_audit.resolve()),
            "sha256": sha256_file(remote_audit) if remote_audit.is_file() else None,
            "run_id": remote.get("run_id"),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
