#!/usr/bin/env python3
"""Audit the empirical paper mask-ratio streams of a live HHC MGT run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


METRICS = {
    "root": "train/mask_ratio_root",
    "body": "train/mask_ratio_body",
    "hand": "train/mask_ratio_hand",
}
# E[cos(pi*tau/2)] for max/median/min order statistics of three iid U(0,1)
# variables, respectively.  The paper orders tau_root >= tau_body >= tau_hand.
EXPECTED_MEANS = {
    "root": 0.3617856643091686,
    "body": 0.6644388981710445,
    "hand": 0.8836347546225309,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
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


def read_rows(path: Path) -> list[dict]:
    store = DataStore()
    store.open_for_scan(str(path))
    rows = []
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if not record.HasField("history"):
            continue
        row = {
            item.key or "/".join(item.nested_key): json.loads(item.value_json)
            for item in record.history.item
        }
        if not all(metric in row for metric in METRICS.values()):
            continue
        step = row.get("train/optimizer_step")
        if not isinstance(step, (int, float)) or not math.isfinite(float(step)):
            raise ValueError("mask-ratio event lacks train/optimizer_step")
        integer_step = int(step)
        if float(step) != integer_step:
            raise ValueError("mask-ratio event has a non-integral optimizer step")
        values = {part: float(row[metric]) for part, metric in METRICS.items()}
        rows.append({"optimizer_step": integer_step, **values})
    return rows


def summarize(values: list[float]) -> dict:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--through-step", type=int, required=True)
    parser.add_argument("--telemetry-interval", type=int, default=50)
    parser.add_argument("--mean-tolerance", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    resolved = manifest.get("resolved_arguments", {})
    rows = [
        row for row in read_rows(args.wandb_file.resolve())
        if row["optimizer_step"] <= args.through_step
    ]
    if not rows:
        raise ValueError("no complete mask-ratio W&B events were found")
    steps = [row["optimizer_step"] for row in rows]
    expected_steps = list(range(
        args.telemetry_interval, args.through_step + 1,
        args.telemetry_interval,
    ))
    summaries = {
        part: summarize([row[part] for row in rows]) for part in METRICS
    }
    checks = {
        "manifest_uses_paper_kinematic_ratio_contract": (
            resolved.get("training_mask_contract")
            == "historical_pipeline_paper_kinematic_ratio"
        ),
        "optimizer_steps_are_exact_and_complete": steps == expected_steps,
        "all_values_are_finite_probabilities": all(
            math.isfinite(row[part]) and 0.0 <= row[part] <= 1.0
            for row in rows for part in METRICS
        ),
        "every_event_obeys_root_body_hand_mask_order": all(
            row["root"] <= row["body"] <= row["hand"] for row in rows
        ),
        "empirical_means_match_order_statistic_expectations": all(
            abs(summaries[part]["mean"] - EXPECTED_MEANS[part])
            <= args.mean_tolerance
            for part in METRICS
        ),
    }
    wandb_path = args.wandb_file.resolve()
    stat = wandb_path.stat()
    payload = {
        "schema": "hhc_paper_mask_ratio_runtime_history_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "role": "live execution evidence; final run repeats this audit through step 300000",
        "checks": checks,
        "through_optimizer_step": args.through_step,
        "telemetry_interval_optimizer_steps": args.telemetry_interval,
        "events": len(rows),
        "expected_means": EXPECTED_MEANS,
        "mean_tolerance": args.mean_tolerance,
        "observed": summaries,
        "wandb_file": {
            "path": str(wandb_path),
            "bytes_at_scan": stat.st_size,
            "mtime_ns_at_scan": stat.st_mtime_ns,
            "sha256_omitted_while_live": True,
        },
        "artifacts": {
            "training_manifest": artifact(args.manifest),
            "mask_implementation": artifact(
                Path(__file__).resolve().parents[1] / "utils/motion_masks.py"
            ),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
