#!/usr/bin/env python3
"""Bind TiMotion's released warm-up boundary to the formal W&B trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import wandb


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def released_factor(epoch: int, warmup: int, horizon: int) -> float:
    factor = 0.5 * (1.0 + math.cos(math.pi * epoch / horizon))
    # This inclusive boundary is intentional: it reproduces models/utils.py.
    if epoch <= warmup:
        factor *= (epoch + 1.0) / warmup
    return factor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--entity", default="edmond_gla")
    parser.add_argument("--minimum-transition-epochs", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest.get("paper_protocol", {})
    batch = manifest.get("batch_emulation", {})
    observability = manifest.get("observability", {})
    artifacts = manifest.get("artifacts", {})
    source_record = artifacts.get("scheduler_and_model_utils", {})
    source_path = Path(source_record.get("path", "")).resolve()
    source = source_path.read_text(encoding="utf-8")
    source_sha256 = sha256_file(source_path)

    base_lr = float(protocol.get("learning_rate", 0.0))
    warmup = int(protocol.get("warmup_epochs", 0))
    horizon = int(protocol.get("scheduler_horizon_epochs", 0))
    updates_per_epoch = int(batch.get("optimizer_updates_per_epoch", 0))
    run_id = observability.get("wandb_run_id")
    project = observability.get("wandb_project")
    if not all((base_lr > 0, warmup > 0, horizon > 0, updates_per_epoch > 0)):
        raise ValueError("manifest has an incomplete scheduler contract")
    if observability.get("wandb_mode") != "online" or not run_id or not project:
        raise ValueError("manifest is not bound to an online W&B run")

    run_path = f"{args.entity}/{project}/{run_id}"
    run = wandb.Api(timeout=30).run(run_path)
    history = []
    for row in run.scan_history(
        keys=["_step", "train/learning_rate"], page_size=5000
    ):
        value = row.get("train/learning_rate")
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise FloatingPointError(f"non-finite learning rate: {value}")
        history.append({"wandb_step": int(row.get("_step", -1)), "value": numeric})

    transitions = []
    previous = None
    for ordinal, row in enumerate(history):
        if previous is None or not math.isclose(
            row["value"], previous, rel_tol=1e-13, abs_tol=0.0
        ):
            epoch = ordinal // updates_per_epoch
            expected = base_lr * released_factor(epoch, warmup, horizon)
            transitions.append({
                "epoch_zero_based": epoch,
                "optimizer_update_ordinal": ordinal,
                "wandb_step": row["wandb_step"],
                "observed_learning_rate": row["value"],
                "released_expected_learning_rate": expected,
                "absolute_error": abs(row["value"] - expected),
            })
            previous = row["value"]

    required = args.minimum_transition_epochs
    checked = transitions[:required]
    boundary = {row["epoch_zero_based"]: row for row in transitions}
    checks = {
        "source_hash_matches_launch_manifest": (
            source_sha256 == source_record.get("sha256")
        ),
        "released_inclusive_boundary_present": (
            "if epoch <= self.warmup:" in source
            and "(epoch+1) * 1.0 / self.warmup" in source
        ),
        "runtime_has_required_epoch_transitions": len(checked) >= required,
        "runtime_transition_ordinals_are_epoch_aligned": all(
            row["optimizer_update_ordinal"]
            == row["epoch_zero_based"] * updates_per_epoch
            for row in checked
        ),
        "runtime_matches_released_formula": all(
            row["absolute_error"] <= 1e-15 for row in checked
        ),
        "inclusive_boundary_overshoot_observed": (
            warmup in boundary
            and boundary[warmup]["observed_learning_rate"] > base_lr
        ),
        "following_epoch_returns_to_cosine": (
            warmup + 1 in boundary
            and boundary[warmup + 1]["observed_learning_rate"] < base_lr
        ),
    }
    payload = {
        "schema": "timotion_released_scheduler_boundary_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "run_path": run_path,
        "remote_state": str(run.state),
        "contract": {
            "base_learning_rate": base_lr,
            "warmup_parameter": warmup,
            "cosine_horizon_epochs": horizon,
            "optimizer_updates_per_epoch": updates_per_epoch,
            "released_boundary_semantics": (
                "epoch <= warmup produces a one-epoch 11/10 overshoot at "
                "zero-based epoch 10, then returns to the cosine branch"
            ),
            "port_policy": "preserve released scheduler behavior exactly",
        },
        "transitions": checked,
        "source": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": source_sha256,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256_at_audit": sha256_file(manifest_path),
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
