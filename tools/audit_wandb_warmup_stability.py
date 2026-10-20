#!/usr/bin/env python3
"""Audit an online W&B run after its configured learning-rate warm-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/online_wandb_training_policy.json"

METRICS = (
    "train/gradient_l2_norm",
    "train/gradient_max_abs",
    "train/weight_l2_norm",
    "train/weight_max_abs",
    "train/nonfinite_gradients",
    "train/nonfinite_weights",
)
SPARSE_TELEMETRY_METRICS = METRICS + (
    "train/samples_per_second",
    "train/gpu_peak_memory_gib",
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


def resolve_optimizer_step_axis(run) -> str:
    """Resolve optimizer progress without treating W&B event order as steps."""
    for candidate in ("train/optimizer_step", "trainer/global_step"):
        for row in run.scan_history(keys=["_step", candidate], page_size=1000):
            if row.get(candidate) is not None:
                return candidate
    raise ValueError(
        "online W&B run lacks an explicit optimizer-step axis; "
        "W&B _step is not an optimizer step"
    )


def finite_history(run, metric: str, step_axis: str) -> list[dict]:
    return finite_histories(run, (metric,), step_axis)[metric]


def finite_histories(
    run, metrics: tuple[str, ...] | list[str], step_axis: str
) -> dict[str, list[dict]]:
    """Read several optimizer-indexed streams in one W&B history scan."""
    histories = {metric: [] for metric in metrics}
    for row in run.scan_history(
        keys=["_step", step_axis, *metrics], page_size=1000
    ):
        present = [metric for metric in metrics if row.get(metric) is not None]
        if not present:
            continue
        step_value = row.get(step_axis)
        if step_value is None or not math.isfinite(float(step_value)):
            raise ValueError(
                f"{present} lack a finite explicit optimizer step on "
                f"axis {step_axis!r}"
            )
        optimizer_step = int(step_value)
        if float(step_value) != optimizer_step:
            raise ValueError(
                f"{present} have non-integral optimizer step {step_value!r}"
            )
        for metric in present:
            value = row[metric]
            numeric = float(value)
            if not math.isfinite(numeric):
                raise FloatingPointError(
                    f"{metric} contains non-finite value {value}"
                )
            histories[metric].append({
                "optimizer_step": optimizer_step,
                "wandb_step": int(row.get("_step", -1)),
                "value": numeric,
            })
    return histories


def finite_event_history(run, metric: str) -> list[dict]:
    """Read validation history using W&B step only as event ordering."""
    return finite_event_histories(run, (metric,))[metric]


def finite_event_histories(
    run, metrics: tuple[str, ...] | list[str]
) -> dict[str, list[dict]]:
    """Read several event-ordered validation streams in one W&B scan."""
    histories = {metric: [] for metric in metrics}
    for row in run.scan_history(keys=["_step", *metrics], page_size=1000):
        for metric in metrics:
            value = row.get(metric)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise FloatingPointError(
                    f"{metric} contains non-finite value {value}"
                )
            histories[metric].append({
                "optimizer_step": None,
                "wandb_step": int(row.get("_step", -1)),
                "value": numeric,
            })
    return histories


def relative_span(values: list[float]) -> float:
    if not values:
        return math.inf
    reference = abs(values[0])
    if reference == 0:
        return 0.0 if all(value == 0 for value in values) else math.inf
    return (max(values) - min(values)) / reference


def summarize(rows: list[dict]) -> dict:
    values = [row["value"] for row in rows]
    return {
        "count": len(rows),
        "first_optimizer_step": rows[0]["optimizer_step"] if rows else None,
        "last_optimizer_step": rows[-1]["optimizer_step"] if rows else None,
        "first_wandb_step": rows[0]["wandb_step"] if rows else None,
        "last_wandb_step": rows[-1]["wandb_step"] if rows else None,
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "minimum": min(values, default=None),
        "maximum": max(values, default=None),
        "relative_span_from_first": relative_span(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--entity", default="edmond_gla")
    parser.add_argument(
        "--metric-profile",
        default=None,
        help=(
            "Frozen W&B metric profile for this launch. When omitted, use "
            "the profile recorded in the training manifest."
        ),
    )
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--max-weight-relative-span", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observability = manifest.get("observability", {})
    batch = manifest.get("batch_emulation", {})
    resolved_arguments = manifest.get("resolved_arguments", {})
    run_id = observability.get("wandb_run_id")
    project = observability.get("wandb_project")
    manifest_profile_name = observability.get("metric_profile")
    profile_name = args.metric_profile or manifest_profile_name
    if observability.get("wandb_mode") != "online" or not run_id or not project:
        raise ValueError("manifest is not bound to an online W&B run")
    interval = int(observability.get("telemetry_interval_optimizer_steps", 0))
    updates_per_epoch = int(batch.get("optimizer_updates_per_epoch", 0))
    configured_warmup_steps = int(resolved_arguments.get("warm_up_iter", 0))
    if interval <= 0:
        raise ValueError("manifest lacks a positive telemetry interval")
    if configured_warmup_steps > 0:
        warmup_contract = "configured_optimizer_steps"
        warmup_optimizer_steps = configured_warmup_steps
        if updates_per_epoch <= 0:
            updates_per_epoch = int(
                manifest.get("loader_contract", {})
                .get("train", {})
                .get("drop_last_batches", 0)
            )
    else:
        if updates_per_epoch <= 0:
            raise ValueError(
                "manifest lacks configured warm-up steps and optimizer updates "
                "per epoch"
            )
        warmup_contract = "requested_epochs_times_optimizer_updates"
        warmup_optimizer_steps = args.warmup_epochs * updates_per_epoch

    run_path = f"{args.entity}/{project}/{run_id}"
    run = wandb.Api(timeout=30).run(run_path)
    optimizer_step_axis = resolve_optimizer_step_axis(run)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    profile = policy.get("metric_profiles", {}).get(profile_name)
    if profile is None:
        raise ValueError(f"unknown online W&B metric profile: {profile_name!r}")
    required_training = profile["required_training_metrics"]
    # W&B scan_history(keys=...) returns only events containing every requested
    # key.  TiMotion intentionally logs dense per-update loss/LR separately
    # from the sparse 50-update telemetry event, so preserve those two actual
    # co-write groups rather than requesting an impossible all-key event.
    sparse_training = tuple(
        metric for metric in required_training
        if metric in SPARSE_TELEMETRY_METRICS
    )
    dense_training = tuple(
        metric for metric in required_training
        if metric not in SPARSE_TELEMETRY_METRICS
    )
    required_training_histories = {}
    for metric_group in (sparse_training, dense_training):
        if metric_group:
            required_training_histories.update(
                finite_histories(run, metric_group, optimizer_step_axis)
            )
    histories = {
        metric: required_training_histories[metric] for metric in METRICS
    }
    required_validation = profile["required_validation_metrics"]
    semantic_validation = tuple(
        metric for metric in required_validation if metric != "val/loss"
    )
    validation_histories = finite_event_histories(run, semantic_validation)
    if "val/loss" in required_validation:
        validation_histories.update(
            finite_event_histories(run, ("val/loss",))
        )
    val_rows = []
    for row in run.scan_history(keys=["_step", "epoch", "val/loss"], page_size=1000):
        if row.get("epoch") is None or row.get("val/loss") is None:
            continue
        value = float(row["val/loss"])
        if not math.isfinite(value):
            raise FloatingPointError(f"val/loss contains non-finite value {value}")
        val_rows.append({
            "epoch_zero_based": int(row["epoch"]),
            "completed_epoch": int(row["epoch"]) + 1,
            "wandb_step": int(row.get("_step", -1)),
            "value": value,
        })

    # Sparse telemetry at 50, 100, ... belongs to the warm-up window only when
    # its optimizer step does not exceed the final warm-up update.  For the
    # released TiMotion 710-update warm-up this is 50..700 (14 events), not an
    # incorrectly rounded-up event at step 750.
    expected_events = warmup_optimizer_steps // interval
    if expected_events < 1:
        raise ValueError(
            "warm-up ends before the first configured telemetry event"
        )
    completed_epochs = max(
        (row["completed_epoch"] for row in val_rows), default=0
    )
    completed_optimizer_step = max(
        (
            row["optimizer_step"]
            for rows in histories.values()
            for row in rows
        ),
        default=0,
    )
    # This artifact is specifically a warm-up audit.  A refresh performed
    # later in the same run must not silently widen its stability window to
    # the entire training trajectory.  Preserve the first expected telemetry
    # events as the immutable warm-up window and report the full trajectory in
    # a separate field for operational monitoring.
    warmup_histories = {
        metric: rows[:expected_events] for metric, rows in histories.items()
    }
    weight_l2 = [
        row["value"]
        for row in warmup_histories["train/weight_l2_norm"]
    ]
    weight_max = [
        row["value"]
        for row in warmup_histories["train/weight_max_abs"]
    ]
    checks = {
        "manifest_declares_online": observability.get("wandb_mode") == "online",
        "warmup_completed": (
            completed_optimizer_step >= warmup_optimizer_steps
            if warmup_contract == "configured_optimizer_steps"
            else completed_epochs >= args.warmup_epochs
        ),
        "telemetry_count_covers_warmup": all(
            len(rows) >= expected_events for rows in histories.values()
        ),
        "all_required_histories_present_and_finite": all(histories.values()),
        "all_profile_training_histories_present_and_finite": all(
            required_training_histories.values()
        ),
        "gradient_norms_positive": all(
            row["value"] > 0
            for metric in ("train/gradient_l2_norm", "train/gradient_max_abs")
            for row in warmup_histories[metric]
        ),
        "nonfinite_counts_zero": all(
            row["value"] == 0
            for metric in ("train/nonfinite_gradients", "train/nonfinite_weights")
            for row in histories[metric]
        ),
        "weight_l2_relative_span_bounded": (
            relative_span(weight_l2) <= args.max_weight_relative_span
        ),
        "weight_max_relative_span_bounded": (
            relative_span(weight_max) <= args.max_weight_relative_span
        ),
    }
    payload = {
        "schema": "hhc_wandb_warmup_stability_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "run_path": run_path,
        "manifest_metric_profile": manifest_profile_name,
        "effective_metric_profile": profile_name,
        "metric_profile_override": args.metric_profile,
        "remote_state": str(run.state),
        "warmup_epochs": args.warmup_epochs,
        "warmup_contract": warmup_contract,
        "warmup_optimizer_steps": warmup_optimizer_steps,
        "completed_epochs": completed_epochs,
        "completed_optimizer_step": completed_optimizer_step,
        "optimizer_updates_per_epoch": updates_per_epoch,
        "telemetry_interval_optimizer_steps": interval,
        "optimizer_step_axis": optimizer_step_axis,
        "minimum_expected_telemetry_events": expected_events,
        "max_weight_relative_span": args.max_weight_relative_span,
        "metric_window": "first telemetry events through warm-up only",
        "metrics": {
            metric: summarize(rows)
            for metric, rows in warmup_histories.items()
        },
        "full_run_metrics": {
            metric: summarize(rows) for metric, rows in histories.items()
        },
        "full_required_training_metrics": {
            metric: summarize(rows)
            for metric, rows in required_training_histories.items()
        },
        "training_metric_presence": {
            metric: bool(rows)
            for metric, rows in required_training_histories.items()
        },
        "full_validation_metrics": {
            metric: summarize(rows)
            for metric, rows in validation_histories.items()
        },
        "validation_metric_presence": {
            metric: bool(rows)
            for metric, rows in validation_histories.items()
        },
        "validation_event_window": (
            "all finite events observed so far; W&B internal step is event "
            "ordering only and is not interpreted as optimizer progress"
        ),
        "validation_loss": val_rows,
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "policy": {"path": str(POLICY), "sha256": sha256_file(POLICY)},
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
