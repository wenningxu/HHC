#!/usr/bin/env python3
"""Audit MGT's optimizer-step warm-up from its bound online W&B run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import wandb

try:
    from tools.audit_wandb_online_metric_history import (
        metric_history,
        resolve_optimizer_step_axis,
    )
except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
    from audit_wandb_online_metric_history import (
        metric_history,
        resolve_optimizer_step_axis,
    )


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/online_wandb_training_policy.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_span(values: list[float]) -> float | None:
    if not values:
        return None
    reference = abs(values[0])
    if reference == 0:
        return 0.0 if all(value == 0 for value in values) else None
    return (max(values) - min(values)) / reference


def summarize_metric(rows: list[dict]) -> dict:
    """Summarize one metric without changing the warm-up audit window."""
    metric_values = [row["value"] for row in rows]
    return {
        "count": len(rows),
        "first_optimizer_step": rows[0]["optimizer_step"] if rows else None,
        "last_optimizer_step": rows[-1]["optimizer_step"] if rows else None,
        "first_wandb_step": rows[0]["wandb_step"] if rows else None,
        "last_wandb_step": rows[-1]["wandb_step"] if rows else None,
        "first": metric_values[0] if metric_values else None,
        "last": metric_values[-1] if metric_values else None,
        "minimum": min(metric_values, default=None),
        "maximum": max(metric_values, default=None),
        "relative_span_from_first": relative_span(metric_values),
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--entity", default="edmond_gla")
    parser.add_argument("--max-weight-l2-relative-span", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observability = manifest.get("observability", {})
    resolved = manifest.get("resolved_arguments", {})
    run_id = observability.get("wandb_run_id")
    project = observability.get("wandb_project")
    profile_name = observability.get("metric_profile")
    interval = int(observability.get("telemetry_interval_optimizer_steps", 0))
    warmup_steps = int(resolved.get("warm_up_iter", 0))
    base_lr = float(resolved.get("lr", 0.0))
    if observability.get("wandb_mode") != "online" or not run_id or not project:
        raise ValueError("manifest is not bound to an online W&B run")
    if profile_name != "masked_token_classification":
        raise ValueError(f"unexpected MGT metric profile: {profile_name!r}")
    if interval <= 0 or warmup_steps <= 0 or warmup_steps % interval:
        raise ValueError("MGT warm-up must be a positive multiple of telemetry interval")

    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    profile = policy["metric_profiles"][profile_name]
    required = profile["required_training_metrics"]
    required_validation = profile["required_validation_metrics"]
    run_path = f"{args.entity}/{project}/{run_id}"
    run = wandb.Api(timeout=30).run(run_path)
    axis = resolve_optimizer_step_axis(run)
    histories = {
        metric: metric_history(run, metric, axis)
        for metric in required
    }
    validation_histories = {
        metric: metric_history(
            run, metric, require_optimizer_axis=False,
        )
        for metric in required_validation
    }
    expected_steps = list(range(interval, warmup_steps + 1, interval))
    warmup = {
        metric: [row for row in rows if row["optimizer_step"] <= warmup_steps]
        for metric, rows in histories.items()
    }
    values = {
        metric: [row["value"] for row in rows]
        for metric, rows in warmup.items()
    }
    observed_steps = {
        metric: [row["optimizer_step"] for row in rows]
        for metric, rows in warmup.items()
    }
    weight_l2_span = relative_span(values["train/weight_l2_norm"])
    weight_max_span = relative_span(values["train/weight_max_abs"])
    learning_rates = values["train/learning_rate"]
    losses = values["train/loss"]
    checks = {
        "manifest_declares_online": observability.get("wandb_mode") == "online",
        "optimizer_axis_is_explicit": axis == "train/optimizer_step",
        "every_required_metric_has_exact_warmup_events": all(
            steps == expected_steps for steps in observed_steps.values()
        ),
        "all_values_finite": all(
            math.isfinite(value)
            for metric_values in values.values()
            for value in metric_values
        ),
        "gradient_norms_positive": all(
            value > 0
            for metric in ("train/gradient_l2_norm", "train/gradient_max_abs")
            for value in values[metric]
        ),
        "nonfinite_counts_zero": all(
            value == 0
            for metric in ("train/nonfinite_gradients", "train/nonfinite_weights")
            for value in values[metric]
        ),
        "warmup_learning_rate_positive_and_monotone": (
            len(learning_rates) == len(expected_steps)
            and all(value > 0 for value in learning_rates)
            and all(a <= b for a, b in zip(learning_rates, learning_rates[1:]))
        ),
        "warmup_reaches_base_learning_rate": (
            bool(learning_rates)
            and math.isclose(learning_rates[-1], base_lr, rel_tol=1e-5, abs_tol=1e-12)
        ),
        "loss_lower_at_warmup_end": bool(losses) and losses[-1] < losses[0],
        "weight_l2_relative_span_bounded": (
            weight_l2_span is not None
            and weight_l2_span <= args.max_weight_l2_relative_span
        ),
    }
    summaries = {
        metric: summarize_metric(rows) for metric, rows in warmup.items()
    }
    # Keep all pass/fail stability checks bound to the immutable warm-up
    # window.  The full-run summaries are operational evidence that gradient,
    # weight and non-finite streams continued throughout a live training run.
    full_run_summaries = {
        metric: summarize_metric(rows) for metric, rows in histories.items()
    }
    # Validation metrics do not share a reliable optimizer-step axis across
    # Lightning callbacks.  Keep W&B's internal step strictly as event order
    # and report every finite event observed so far without making a live run
    # pretend that final validation coverage has already passed.
    full_validation_summaries = {
        metric: summarize_metric(rows)
        for metric, rows in validation_histories.items()
    }
    payload = {
        "schema": "hhc_mgt_wandb_optimizer_warmup_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "run_path": run_path,
        "remote_state": str(run.state),
        "optimizer_step_axis": axis,
        "telemetry_interval_optimizer_steps": interval,
        "warmup_optimizer_steps": warmup_steps,
        "expected_telemetry_events": len(expected_steps),
        "expected_optimizer_steps": expected_steps,
        "base_learning_rate": base_lr,
        "max_weight_l2_relative_span": args.max_weight_l2_relative_span,
        "observed_weight_l2_relative_span": weight_l2_span,
        "observed_weight_max_abs_relative_span": weight_max_span,
        "weight_max_abs_policy": (
            "reported_without_a_hard_drift_threshold; global max-abs is a "
            "single-coordinate diagnostic, while non-finite counts and global "
            "weight L2 provide the fail-closed health checks"
        ),
        "metric_window": "exact optimizer steps through warm-up only",
        "metrics": summaries,
        "full_run_metrics": full_run_summaries,
        "full_validation_metrics": full_validation_summaries,
        "validation_metric_presence": {
            metric: bool(rows)
            for metric, rows in validation_histories.items()
        },
        "validation_event_window": (
            "all finite events observed so far; W&B internal step is event "
            "ordering only and is not interpreted as optimizer progress"
        ),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
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
