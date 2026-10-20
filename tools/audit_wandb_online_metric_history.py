#!/usr/bin/env python3
"""Fail closed unless a completed online W&B run contains required metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import wandb


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/online_wandb_training_policy.json"
PROFILE_RUN_BINDINGS = {
    "diffusion_regression_20260831_launch": {
        "project": "HHC",
        "run_id": "kdz5s3ra",
    },
}


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


def history_rows_for_keys(run, keys, snapshot=None):
    if snapshot is None:
        return run.scan_history(keys=keys, page_size=1000)
    # Match W&B's requested-key intersection, including missing/null values.
    return (row for row in snapshot if all(row.get(key) is not None for key in keys))


def resolve_optimizer_step_axis(run, snapshot=None) -> str:
    """Find a real optimizer-progress key without querying absent W&B keys.

    W&B's public API returns an empty scan when a requested key does not exist,
    even if the other requested keys do.  Probe the supported axes separately
    before scanning metric histories.
    """
    for candidate in ("train/optimizer_step", "trainer/global_step"):
        for row in history_rows_for_keys(run, ["_step", candidate], snapshot):
            if row.get(candidate) is not None:
                return candidate
    raise ValueError(
        "online W&B run lacks an explicit optimizer-step axis; "
        "W&B _step is not an optimizer step"
    )


def metric_history(
    run, metric: str, step_axis: str | None = None,
    require_optimizer_axis: bool = True,
    snapshot=None,
) -> list[dict]:
    rows = []
    if require_optimizer_axis and step_axis is None:
        step_axis = resolve_optimizer_step_axis(run, snapshot)
    if require_optimizer_axis and step_axis not in (
        "train/optimizer_step", "trainer/global_step",
    ):
        raise ValueError(f"unsupported optimizer-step axis: {step_axis!r}")
    history_keys = (
        ["_step", step_axis, metric]
        if require_optimizer_axis else ["_step", metric]
    )
    for row in history_rows_for_keys(run, history_keys, snapshot):
        value = row.get(metric)
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise FloatingPointError(
                f"online W&B metric {metric} contains non-finite value {value}"
            )
        if require_optimizer_axis and row.get(step_axis) is None:
            raise ValueError(
                f"online W&B metric {metric} lacks an explicit optimizer-step "
                "axis; W&B _step is not an optimizer step"
            )
        optimizer_step = None
        if require_optimizer_axis:
            step_value = row[step_axis]
            optimizer_step = int(step_value)
            if (not math.isfinite(float(step_value))
                    or float(step_value) != optimizer_step):
                raise ValueError(
                    f"online W&B metric {metric} has invalid optimizer step "
                    f"{step_value!r}"
                )
        rows.append({
            "optimizer_step": optimizer_step,
            "wandb_step": int(row.get("_step", -1)),
            "step_source": (
                step_axis if require_optimizer_axis
                else "wandb_event_order_only"
            ),
            "value": numeric,
        })
    return rows


def history_coverage_checks(
    summaries: dict, profile: dict, expected_final_step: int,
) -> dict:
    """Require every mandatory stream to cover the completed training run.

    Checking only one training metric is insufficient: a continuously logged
    loss could otherwise hide gradient/weight telemetry that stopped near the
    beginning of training.  All required training and validation streams must
    reach the declared final step.  Non-finite counters are additionally
    required to remain zero for the entire remote history.
    """
    required_training = profile["required_training_metrics"]
    required_validation = profile["required_validation_metrics"]

    def reaches_final(metric: str) -> bool:
        summary = summaries.get(metric, {})
        last_step = summary.get("last_step")
        return last_step is not None and last_step >= expected_final_step

    nonfinite_metrics = (
        "train/nonfinite_gradients",
        "train/nonfinite_weights",
    )
    final_training_wandb_step = max(
        (
            summaries.get(metric, {}).get("last_wandb_step", -1)
            for metric in required_training
        ),
        default=-1,
    )
    return {
        "all_training_metrics_reach_expected_step": all(
            reaches_final(metric) for metric in required_training
        ),
        # Validation frameworks may attach a microbatch counter to epoch-level
        # metrics.  Do not compare that counter with optimizer updates.  W&B's
        # internal step is used only as event ordering here: every required
        # validation stream must have an event at or after the final training
        # event, whose optimizer coverage is established above.
        "all_validation_metrics_follow_final_training_event": (
            final_training_wandb_step >= 0
            and all(
                summaries.get(metric, {}).get("last_wandb_step", -1)
                >= final_training_wandb_step
                for metric in required_validation
            )
        ),
        "nonfinite_counts_zero": all(
            metric in summaries
            and summaries[metric].get("minimum") == 0.0
            and summaries[metric].get("maximum") == 0.0
            for metric in nonfinite_metrics
        ),
    }


def align_validation_events(histories, validation_metrics):
    """Attach inferred training axes to sparse validation events."""
    training = {}
    for metric, rows in histories.items():
        if metric in validation_metrics:
            continue
        for row in rows:
            training.setdefault(row["wandb_step"], set()).add(row["optimizer_step"])
    aligned = {}
    for metric in validation_metrics:
        events = []
        for row in histories.get(metric, []):
            candidates = [step for wb, steps in training.items() if wb <= row["wandb_step"] for step in steps]
            if not candidates:
                raise ValueError(f"validation event has no preceding training event: {row['wandb_step']}")
            latest_wb = max(wb for wb in training if wb <= row["wandb_step"])
            steps = training[latest_wb]
            if len(steps) != 1:
                raise ValueError(f"conflicting training optimizer axes at W&B step {latest_wb}: {sorted(steps)}")
            events.append({**row, "preceding_training_optimizer_step": next(iter(steps)),
                           "preceding_training_wandb_step": latest_wb,
                           "alignment": "inferred_from_latest_training_metric_event"})
        aligned[metric] = events
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--entity", default="edmond_gla")
    parser.add_argument("--metric-profile", default="diffusion_regression")
    parser.add_argument("--expected-final-step", type=int, required=True)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--retry-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    observability = manifest.get("observability", {})
    run_id = observability.get("wandb_run_id")
    project = observability.get("wandb_project")
    if observability.get("wandb_mode") != "online" or not run_id or not project:
        raise ValueError("training manifest is not bound to an online W&B run")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    profile = policy.get("metric_profiles", {}).get(args.metric_profile)
    if not profile:
        raise ValueError(f"unknown metric profile: {args.metric_profile}")
    required = (
        profile["required_training_metrics"]
        + profile["required_validation_metrics"]
    )
    run_path = f"{args.entity}/{project}/{run_id}"
    state = None
    histories = {}
    optimizer_step_axis = None
    error = None
    for attempt in range(1, args.attempts + 1):
        try:
            run = wandb.Api(timeout=30).run(run_path)
            state = str(run.state)
            snapshot = list(run.scan_history(page_size=1000))
            optimizer_step_axis = resolve_optimizer_step_axis(run, snapshot)
            histories = {}
            for metric in profile["required_training_metrics"]:
                histories[metric] = metric_history(
                    run, metric, optimizer_step_axis,
                    require_optimizer_axis=True,
                    snapshot=snapshot,
                )
            for metric in profile["required_validation_metrics"]:
                histories[metric] = metric_history(
                    run, metric, require_optimizer_axis=False, snapshot=snapshot,
                )
            if state == "finished" and all(histories.values()):
                break
            error = (
                f"attempt {attempt}: state={state}, missing="
                f"{[key for key, values in histories.items() if not values]}"
            )
        except Exception as caught:  # W&B transport errors are retried.
            error = f"attempt {attempt}: {type(caught).__name__}: {caught}"
        if attempt < args.attempts:
            time.sleep(args.retry_seconds)

    summaries = {}
    for metric, rows in histories.items():
        summaries[metric] = {
            "count": len(rows),
            "first_step": rows[0]["optimizer_step"] if rows else None,
            "last_step": rows[-1]["optimizer_step"] if rows else None,
            "first_wandb_step": rows[0]["wandb_step"] if rows else None,
            "last_wandb_step": rows[-1]["wandb_step"] if rows else None,
            "step_sources": sorted({
                row["step_source"] for row in rows
            }),
            "minimum": min((row["value"] for row in rows), default=None),
            "maximum": max((row["value"] for row in rows), default=None),
        }
        # Validation is sparse (normally one event per checkpoint), so retain
        # its actual trajectory for matched experiment comparisons.  Training
        # streams remain compact because they can contain thousands of events.
        if metric in profile["required_validation_metrics"]:
            summaries[metric]["events"] = [
                {
                    "wandb_step": row["wandb_step"],
                    "value": row["value"],
                }
                for row in rows
            ]
    aligned = align_validation_events(
        histories, profile["required_validation_metrics"]
    ) if histories else {}
    for metric, events in aligned.items():
        summaries[metric]["events"] = [
            {key: event[key] for key in (
                "wandb_step", "value", "preceding_training_optimizer_step",
                "preceding_training_wandb_step", "alignment"
            )} for event in events
        ]
    checks = {
        "manifest_declares_online": observability.get("wandb_mode") == "online",
        "remote_run_finished": state == "finished",
        "all_required_metrics_present_and_finite": all(histories.values()),
        "profile_run_binding_matches": (
            args.metric_profile not in PROFILE_RUN_BINDINGS
            or (
                project
                == PROFILE_RUN_BINDINGS[args.metric_profile]["project"]
                and run_id
                == PROFILE_RUN_BINDINGS[args.metric_profile]["run_id"]
            )
        ),
    }
    checks.update(history_coverage_checks(
        summaries, profile, args.expected_final_step
    ))
    payload = {
        "schema": "hhc_wandb_online_metric_history_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "run_path": run_path,
        "run_id": run_id,
        "project": project,
        "entity": args.entity,
        "remote_state": state,
        "metric_profile": args.metric_profile,
        "expected_final_step": args.expected_final_step,
        "optimizer_step_axis": optimizer_step_axis,
        "coverage_axis": (
            "training coverage uses explicit optimizer progress from "
            "train/optimizer_step or trainer/global_step; validation coverage "
            "requires a finite event at/after the final training event; W&B "
            "_step is event ordering only and is never optimizer progress"
        ),
        "metrics": summaries,
        "last_retry_error": error,
        "credential_disclosed": False,
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
        },
        "policy": {
            "path": str(POLICY.resolve()),
            "sha256": sha256_file(POLICY),
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
