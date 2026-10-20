#!/usr/bin/env python3
"""Read live W&B transaction progress locally without network or wait loops."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


DEFAULT_PROGRESS_KEYS = (
    "train/optimizer_step", "trainer/global_step", "completed_epoch", "epoch",
)


def summarize_training_metric_progress(metrics: dict) -> dict:
    """Summarize metric-attached progress without using validation events."""
    training_metric_progress = {
        key: summary["latest_progress"]
        for key, summary in metrics.items()
        if key.startswith("train/") and summary["latest_progress"] is not None
    }
    optimizer_axis = (
        "train/optimizer_step"
        if any(
            "train/optimizer_step" in progress
            for progress in training_metric_progress.values()
        )
        else "trainer/global_step"
    )
    training_steps = {
        key: int(progress[optimizer_axis])
        for key, progress in training_metric_progress.items()
        if isinstance(progress.get(optimizer_axis), (int, float))
        and math.isfinite(float(progress[optimizer_axis]))
    }
    return {
        "optimizer_axis": optimizer_axis,
        "per_metric_latest_step": training_steps,
        "minimum_covered_step": min(training_steps.values(), default=None),
        "maximum_observed_step": max(training_steps.values(), default=None),
        "interpretation": (
            "metric-attached training progress; latest_progress above is "
            "only the final event snapshot and may reflect validation"
        ),
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
    parser.add_argument("--wandb-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--progress-keys", nargs="+", default=list(DEFAULT_PROGRESS_KEYS)
    )
    parser.add_argument("--metrics", nargs="+", required=True)
    args = parser.parse_args()

    wandb_file = args.wandb_file.resolve()
    store = DataStore()
    store.open_for_scan(str(wandb_file))
    latest_progress = {key: None for key in args.progress_keys}
    metrics = {
        key: {
            "count": 0, "finite_count": 0, "nonfinite_count": 0,
            "latest": None, "minimum": None, "maximum": None,
            "latest_progress": None,
        }
        for key in args.metrics
    }
    history_rows = 0
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if not record.HasField("history"):
            continue
        history_rows += 1
        row = {
            item.key or "/".join(item.nested_key): json.loads(item.value_json)
            for item in record.history.item
        }
        for key in args.progress_keys:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                latest_progress[key] = value
        progress_snapshot = {
            key: value for key, value in latest_progress.items()
            if value is not None
        }
        for key, summary in metrics.items():
            if key not in row or not isinstance(row[key], (int, float)):
                continue
            value = float(row[key])
            summary["count"] += 1
            if not math.isfinite(value):
                summary["nonfinite_count"] += 1
                continue
            summary["finite_count"] += 1
            summary["latest"] = value
            summary["minimum"] = (
                value if summary["minimum"] is None
                else min(summary["minimum"], value)
            )
            summary["maximum"] = (
                value if summary["maximum"] is None
                else max(summary["maximum"], value)
            )
            summary["latest_progress"] = progress_snapshot

    stat = wandb_file.stat()
    checks = {
        "history_rows_present": history_rows > 0,
        "at_least_one_progress_axis_present": any(
            value is not None for value in latest_progress.values()
        ),
        "all_requested_metrics_present": all(
            summary["count"] > 0 for summary in metrics.values()
        ),
        "all_observed_requested_metrics_finite": all(
            summary["nonfinite_count"] == 0 for summary in metrics.values()
        ),
    }
    # ``latest_progress`` is the most recent value seen in any event.  Some
    # Lightning validation events attach a microbatch-like
    # ``trainer/global_step`` even when direct training logger calls use that
    # same field for optimizer updates.  Never present the final event's value
    # as training progress.  Instead expose the progress snapshot attached to
    # each requested training metric and its coverage range.
    training_progress_summary = summarize_training_metric_progress(metrics)
    payload = {
        "schema": "hhc_local_wandb_live_progress_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "role": (
            "live local monitoring only; final compliance requires the "
            "completed remote online-history audit"
        ),
        "checks": checks,
        "wandb_file": {
            "path": str(wandb_file), "bytes_at_scan": stat.st_size,
            "mtime_ns_at_scan": stat.st_mtime_ns,
            "sha256_omitted_while_live": True,
        },
        "history_rows": history_rows,
        "latest_progress": latest_progress,
        "training_metric_progress": training_progress_summary,
        "metrics": metrics,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
