#!/usr/bin/env python3
"""Align validation MModality collapse with no-shape training telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import wandb


TRAIN_METRICS = (
    "train/loss",
    "train/accuracy_all",
    "train/accuracy_masked",
    "train/accuracy_visible",
    "train/gradient_l2_norm",
    "train/weight_l2_norm",
    "train/gradient_to_weight_l2_ratio",
)
VAL_METRICS = (
    "val/FID",
    "val/Matching",
    "val/R_precision_top1",
    "val/R_precision_top2",
    "val/R_precision_top3",
    "val/Diversity",
    "val/Multimodality_diagnostic",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def correlation(left: list[float], right: list[float]) -> float:
    value = float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])
    if not math.isfinite(value):
        raise ValueError("non-finite correlation")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.wandb_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("remote_state") != "finished":
        raise ValueError("W&B audit is not a completed passing run")
    for metric in VAL_METRICS:
        events = audit.get("metrics", {}).get(metric, {}).get("events", [])
        if len(events) != 30:
            raise ValueError(f"{metric}: expected 30 validation events, got {len(events)}")
    validation_wandb_steps = [
        int(row["wandb_step"])
        for row in audit["metrics"]["val/Multimodality_diagnostic"]["events"]
    ]
    for metric in VAL_METRICS:
        steps = [int(row["wandb_step"]) for row in audit["metrics"][metric]["events"]]
        if steps != validation_wandb_steps:
            raise ValueError(f"{metric}: validation event order/steps differ")

    run = wandb.Api(timeout=30).run(audit["run_path"])
    training_rows = []
    keys = ["_step", "train/optimizer_step", *TRAIN_METRICS]
    for row in run.scan_history(keys=keys, page_size=2000):
        if row.get("train/optimizer_step") is None:
            continue
        if any(row.get(metric) is None for metric in TRAIN_METRICS):
            continue
        values = {metric: float(row[metric]) for metric in TRAIN_METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("non-finite training telemetry")
        training_rows.append({
            "wandb_step": int(row["_step"]),
            "optimizer_step": int(row["train/optimizer_step"]),
            **values,
        })
    if len(training_rows) < 30:
        raise ValueError(f"insufficient complete training telemetry: {len(training_rows)}")

    aligned = []
    for index, wandb_step in enumerate(validation_wandb_steps):
        eligible = [row for row in training_rows if row["wandb_step"] <= wandb_step]
        if not eligible:
            raise ValueError(f"no training event precedes validation event {index}")
        training = max(eligible, key=lambda row: row["wandb_step"])
        expected_optimizer_step = (index + 1) * 10_000
        if not expected_optimizer_step - 50 <= training["optimizer_step"] <= expected_optimizer_step:
            raise ValueError(
                f"validation {index}: nearest training step {training['optimizer_step']} "
                f"does not match expected {expected_optimizer_step}"
            )
        validation = {
            metric: float(audit["metrics"][metric]["events"][index]["value"])
            for metric in VAL_METRICS
        }
        aligned.append({
            "validation_index": index,
            "expected_optimizer_step": expected_optimizer_step,
            "validation_wandb_step": wandb_step,
            "training": training,
            "validation": validation,
        })

    mm = [row["validation"]["val/Multimodality_diagnostic"] for row in aligned]
    correlations = {
        metric: correlation(mm, [row["training"][metric] for row in aligned])
        for metric in TRAIN_METRICS
    }
    correlations.update({
        metric: correlation(mm, [row["validation"][metric] for row in aligned])
        for metric in VAL_METRICS if metric != "val/Multimodality_diagnostic"
    })
    first5 = np.asarray(mm[:5], dtype=np.float64)
    last5 = np.asarray(mm[-5:], dtype=np.float64)
    payload = {
        "schema": "hhc_no_shape_training_collapse_trajectory_v1",
        "status": "complete",
        "scope": "completed no-shape 300k validation/W&B trajectory; correlation is diagnostic, not causal",
        "run_path": audit["run_path"],
        "validation_interval": 10000,
        "aligned_events": aligned,
        "mmodality": {
            "first": mm[0],
            "last": mm[-1],
            "minimum": min(mm),
            "maximum": max(mm),
            "first5_mean": float(first5.mean()),
            "last5_mean": float(last5.mean()),
            "last5_minus_first5": float(last5.mean() - first5.mean()),
        },
        "pearson_correlation_with_mmodality": correlations,
        "interpretation_policy": (
            "Use signs/magnitudes to prioritize mechanisms only. This single-run "
            "trajectory cannot establish that a telemetry variable causes collapse."
        ),
        "evidence": {"wandb_audit": artifact(args.wandb_audit)},
        "implementation": artifact(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": payload["status"],
        "mmodality": payload["mmodality"],
        "pearson_correlation_with_mmodality": correlations,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
