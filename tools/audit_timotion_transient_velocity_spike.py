#!/usr/bin/env python3
"""Freeze a post-hoc, read-only audit of a TiMotion velocity-loss excursion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import wandb


TRAIN_METRICS = (
    "train/loss",
    "train/loss_components/simple",
    "train/loss_components/VEL_A",
    "train/loss_components/VEL_B",
)
VAL_METRICS = (
    "val/loss",
    "val/loss_components/simple",
    "val/loss_components/VEL_A",
    "val/loss_components/VEL_B",
)
TELEMETRY_METRICS = (
    "train/gradient_l2_norm",
    "train/gradient_max_abs",
    "train/weight_l2_norm",
    "train/weight_max_abs",
    "train/nonfinite_gradients",
    "train/nonfinite_weights",
)


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


def finite_history(run, metric: str) -> list[dict]:
    # Fetch metrics separately. W&B scan_history can return no rows when sparse
    # Lightning validation and manual optimizer-step keys are requested in one
    # combined query, even though every individual history is present.
    records = []
    for row in run.scan_history(keys=["_step", metric], page_size=1000):
        value = row.get(metric)
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise FloatingPointError(f"{metric} contains non-finite {value}")
        wandb_step = row.get("_step")
        if wandb_step is None:
            raise ValueError(f"{metric} history row lacks W&B _step")
        records.append({"wandb_step": int(wandb_step), "value": numeric})
    return records


def component_snapshot_alignment(
    records_by_metric: dict[str, list[dict]],
    *,
    remote_state: str,
    maximum_running_suffix_skew: int,
) -> dict:
    """Validate a common event prefix while a live run is still appending.

    W&B exposes no transaction spanning several ``scan_history`` calls.  A
    running job can therefore append a few events between the first and last
    component query.  Exact count equality would misclassify that harmless
    suffix race as missing telemetry.  Event-step equality over the entire
    common prefix still detects a missing or shifted event in the frozen
    portion; only a bounded trailing skew is tolerated while the remote run is
    explicitly ``running``.
    """
    counts = {metric: len(records) for metric, records in records_by_metric.items()}
    minimum = min(counts.values(), default=0)
    maximum = max(counts.values(), default=0)
    suffix_skew = maximum - minimum
    prefixes = {
        metric: tuple(record["wandb_step"] for record in records[:minimum])
        for metric, records in records_by_metric.items()
    }
    reference = next(iter(prefixes.values()), ())
    common_prefix_aligned = bool(minimum) and all(
        prefix == reference for prefix in prefixes.values()
    )
    exact_counts = suffix_skew == 0
    suffix_skew_allowed = exact_counts or (
        remote_state == "running"
        and suffix_skew <= maximum_running_suffix_skew
    )
    return {
        "counts": counts,
        "common_prefix_events": minimum,
        "common_prefix_aligned": common_prefix_aligned,
        "suffix_skew_events": suffix_skew,
        "maximum_running_suffix_skew": maximum_running_suffix_skew,
        "suffix_skew_allowed": suffix_skew_allowed,
        "exact_counts": exact_counts,
        "remote_state": remote_state,
        "pass": common_prefix_aligned and suffix_skew_allowed,
    }


def epoch_summary(values: list[float], updates_per_epoch: int) -> list[dict]:
    epochs = []
    for offset in range(0, len(values) - updates_per_epoch + 1,
                        updates_per_epoch):
        window = values[offset:offset + updates_per_epoch]
        epochs.append({
            "completed_epoch": len(epochs) + 1,
            "count": len(window),
            "mean": statistics.fmean(window),
            "median": statistics.median(window),
            "maximum": max(window),
            "minimum": min(window),
        })
    return epochs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--entity", default="edmond_gla")
    parser.add_argument("--analysis-start-epoch", type=int, default=40)
    parser.add_argument("--baseline-epochs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observability = manifest.get("observability", {})
    run_id = observability.get("wandb_run_id")
    project = observability.get("wandb_project")
    updates_per_epoch = int(
        manifest.get("batch_emulation", {}).get(
            "optimizer_updates_per_epoch", 0
        )
    )
    if observability.get("wandb_mode") != "online" or not run_id or not project:
        raise ValueError("manifest is not bound to an online W&B run")
    if updates_per_epoch <= 0:
        raise ValueError("manifest lacks optimizer_updates_per_epoch")

    run_path = f"{args.entity}/{project}/{run_id}"
    run = wandb.Api(timeout=30).run(run_path)
    history_records = {
        metric: finite_history(run, metric)
        for metric in TRAIN_METRICS + VAL_METRICS + TELEMETRY_METRICS
    }
    histories = {
        metric: [record["value"] for record in records]
        for metric, records in history_records.items()
    }
    train_counts = {metric: len(histories[metric]) for metric in TRAIN_METRICS}
    val_counts = {metric: len(histories[metric]) for metric in VAL_METRICS}
    remote_state = str(run.state)
    train_alignment = component_snapshot_alignment(
        {metric: history_records[metric] for metric in TRAIN_METRICS},
        remote_state=remote_state,
        maximum_running_suffix_skew=updates_per_epoch,
    )
    validation_alignment = component_snapshot_alignment(
        {metric: history_records[metric] for metric in VAL_METRICS},
        remote_state=remote_state,
        maximum_running_suffix_skew=1,
    )
    common_train_count = min(train_counts.values(), default=0)
    common_val_count = min(val_counts.values(), default=0)
    completed_train_epochs = common_train_count // updates_per_epoch
    completed_epochs = min(completed_train_epochs, common_val_count)
    if completed_epochs < args.analysis_start_epoch:
        raise ValueError(
            f"only {completed_epochs} complete epochs; analysis starts at "
            f"{args.analysis_start_epoch}"
        )

    train_epochs = {
        metric: epoch_summary(
            histories[metric][:completed_epochs * updates_per_epoch],
            updates_per_epoch,
        )
        for metric in TRAIN_METRICS
    }
    val_epochs = {
        metric: histories[metric][:completed_epochs] for metric in VAL_METRICS
    }
    analysis_indices = range(args.analysis_start_epoch - 1, completed_epochs)
    peak_index = max(analysis_indices, key=lambda index: val_epochs["val/loss"][index])
    baseline_start = max(
        args.analysis_start_epoch - 1, peak_index - args.baseline_epochs
    )
    baseline_indices = list(range(baseline_start, peak_index))
    if not baseline_indices:
        raise ValueError("no pre-spike baseline epochs in analysis window")

    def baseline(metric: str, source: dict) -> float:
        values = source[metric]
        if values and isinstance(values[0], dict):
            return statistics.median(values[index]["mean"] for index in baseline_indices)
        return statistics.median(values[index] for index in baseline_indices)

    peak_train_loss = train_epochs["train/loss"][peak_index]["mean"]
    baseline_train_loss = baseline("train/loss", train_epochs)
    peak_velocity = sum(
        train_epochs[metric][peak_index]["mean"]
        for metric in TRAIN_METRICS if metric.endswith(("VEL_A", "VEL_B"))
    )
    baseline_velocity = sum(
        baseline(metric, train_epochs)
        for metric in TRAIN_METRICS if metric.endswith(("VEL_A", "VEL_B"))
    )
    total_excess = peak_train_loss - baseline_train_loss
    velocity_excess = peak_velocity - baseline_velocity
    velocity_excess_fraction = (
        velocity_excess / total_excess if total_excess > 0 else None
    )
    latest_index = completed_epochs - 1
    latest_val = val_epochs["val/loss"][latest_index]
    baseline_val = baseline("val/loss", val_epochs)
    window_start = max(0, peak_index - args.baseline_epochs)
    window_end = min(completed_epochs, peak_index + args.baseline_epochs + 1)

    checks = {
        "manifest_declares_online": observability.get("wandb_mode") == "online",
        "all_histories_present": all(histories.values()),
        "train_component_common_prefix_aligned": train_alignment[
            "common_prefix_aligned"
        ],
        "train_component_suffix_skew_allowed": train_alignment[
            "suffix_skew_allowed"
        ],
        "validation_component_common_prefix_aligned": validation_alignment[
            "common_prefix_aligned"
        ],
        "validation_component_suffix_skew_allowed": validation_alignment[
            "suffix_skew_allowed"
        ],
        "nonfinite_gradient_counts_zero": all(
            value == 0 for value in histories["train/nonfinite_gradients"]
        ),
        "nonfinite_weight_counts_zero": all(
            value == 0 for value in histories["train/nonfinite_weights"]
        ),
    }
    payload = {
        "schema": "timotion_transient_velocity_spike_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "interpretation_policy": (
            "post-hoc operational diagnosis only; not a checkpoint-selection "
            "criterion and not evidence for changing the released loss"
        ),
        "checks": checks,
        "run_path": run_path,
        "remote_state": remote_state,
        "live_snapshot_alignment": {
            "policy": (
                "all component W&B _step values must match over the common "
                "prefix; a running run may have only a bounded trailing skew "
                "caused by sequential API queries; a finished run requires "
                "exact counts"
            ),
            "training": train_alignment,
            "validation": validation_alignment,
        },
        "updates_per_epoch": updates_per_epoch,
        "completed_epochs_analyzed": completed_epochs,
        "analysis_start_epoch": args.analysis_start_epoch,
        "peak": {
            "completed_epoch": peak_index + 1,
            "train_loss_mean": peak_train_loss,
            "validation_loss": val_epochs["val/loss"][peak_index],
            "train_simple_mean": train_epochs[
                "train/loss_components/simple"
            ][peak_index]["mean"],
            "train_velocity_a_mean": train_epochs[
                "train/loss_components/VEL_A"
            ][peak_index]["mean"],
            "train_velocity_b_mean": train_epochs[
                "train/loss_components/VEL_B"
            ][peak_index]["mean"],
        },
        "baseline": {
            "completed_epochs": [index + 1 for index in baseline_indices],
            "train_loss_median_of_epoch_means": baseline_train_loss,
            "validation_loss_median": baseline_val,
            "train_velocity_sum_median_of_epoch_means": baseline_velocity,
        },
        "decomposition": {
            "train_total_excess": total_excess,
            "train_velocity_sum_excess": velocity_excess,
            "velocity_excess_fraction_of_total_excess": velocity_excess_fraction,
        },
        "latest_complete_epoch": {
            "completed_epoch": latest_index + 1,
            "train_loss_mean": train_epochs["train/loss"][latest_index]["mean"],
            "validation_loss": latest_val,
            "validation_loss_over_pre_spike_baseline": latest_val / baseline_val,
            "train_velocity_a_mean": train_epochs[
                "train/loss_components/VEL_A"
            ][latest_index]["mean"],
            "train_velocity_b_mean": train_epochs[
                "train/loss_components/VEL_B"
            ][latest_index]["mean"],
        },
        "telemetry_latest": {
            metric: histories[metric][-1] for metric in TELEMETRY_METRICS
        },
        "history_counts": {**train_counts, **val_counts, **{
            metric: len(histories[metric]) for metric in TELEMETRY_METRICS
        }},
        "epoch_window": {
            metric: rows[window_start:window_end]
            for metric, rows in train_epochs.items()
        },
        "validation_window": {
            metric: [
                {"completed_epoch": index + 1, "value": values[index]}
                for index in range(window_start, window_end)
            ]
            for metric, values in val_epochs.items()
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
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
