#!/usr/bin/env python
"""Compare exact-step MGT metrics from two local W&B transaction logs."""

import argparse
import hashlib
import json
import os
from pathlib import Path


METRICS = (
    "Train/all",
    "Train/every_token",
    "Train/masked",
    "Train/no_masked",
)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_training_rows(path):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(str(path))
    last_step = None
    rows = {}
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
        if "trainer/global_step" in row:
            last_step = int(row["trainer/global_step"])
        if last_step is None or "Train/all" not in row:
            continue
        rows[last_step] = {
            "trainer/global_step": last_step,
            **{metric: row[metric] for metric in METRICS if metric in row},
        }
    return rows


def file_record(path, include_sha256):
    stat = path.stat()
    record = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        record["sha256"] = sha256_file(path)
    return record


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    step_group = parser.add_mutually_exclusive_group(required=True)
    step_group.add_argument("--steps", type=int, nargs="+")
    step_group.add_argument(
        "--all-common-steps",
        action="store_true",
        help="Compare every training row present in both logs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-abs-loss-diff",
        type=float,
        default=None,
        help="Optional hard gate applied to Train/all at every requested step.",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip only the candidate hash while its log is still being appended.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print a compact status while retaining the full JSON artifact.",
    )
    args = parser.parse_args()

    for path in (args.reference, args.candidate):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference_rows = scan_training_rows(args.reference)
    candidate_rows = scan_training_rows(args.candidate)
    if args.all_common_steps:
        requested_steps = sorted(set(reference_rows) & set(candidate_rows))
        comparison_scope = "all_candidate_training_rows_present_in_reference"
    else:
        requested_steps = sorted(set(args.steps))
        comparison_scope = "explicit_steps"

    comparisons = []
    missing = []
    for step in requested_steps:
        reference = reference_rows.get(step)
        candidate = candidate_rows.get(step)
        if reference is None or candidate is None:
            missing.append({
                "trainer/global_step": step,
                "reference_missing": reference is None,
                "candidate_missing": candidate is None,
            })
            continue
        metric_differences = {}
        for metric in METRICS:
            if metric not in reference or metric not in candidate:
                continue
            difference = float(candidate[metric]) - float(reference[metric])
            denominator = max(abs(float(reference[metric])), 1e-12)
            metric_differences[metric] = {
                "signed": difference,
                "absolute": abs(difference),
                "relative_to_reference": abs(difference) / denominator,
            }
        comparisons.append({
            "trainer/global_step": step,
            "reference": reference,
            "candidate": candidate,
            "difference": metric_differences,
        })

    threshold_failures = []
    if args.max_abs_loss_diff is not None:
        threshold_failures = [
            {
                "trainer/global_step": row["trainer/global_step"],
                "absolute_loss_difference": row["difference"]["Train/all"]["absolute"],
                "maximum": args.max_abs_loss_diff,
            }
            for row in comparisons
            if row["difference"]["Train/all"]["absolute"]
            > args.max_abs_loss_diff
        ]

    status = "incomplete" if missing else "comparison_complete"
    if args.max_abs_loss_diff is not None and not missing:
        status = "fail" if threshold_failures else "pass"
    payload = {
        "schema": "hhc_mgt_wandb_training_trajectory_comparison_v1",
        "status": status,
        "reference": file_record(args.reference, True),
        "candidate": file_record(args.candidate, not args.skip_sha256),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "requested_steps": requested_steps,
        "comparison_scope": comparison_scope,
        "compared_step_count": len(comparisons),
        "latest_available_steps": {
            "reference": max(reference_rows) if reference_rows else None,
            "candidate": max(candidate_rows) if candidate_rows else None,
        },
        "comparisons": comparisons,
        "missing": missing,
        "maximum_absolute_loss_difference": args.max_abs_loss_diff,
        "threshold_failures": threshold_failures,
        "interpretation": (
            "Training metrics are diagnostic trajectory evidence only; formal "
            "model selection remains based on the agreed full-space ordinary "
            "text-to-motion Inter-X evaluator."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    printable = payload
    if args.summary_only:
        printable = {
            key: payload[key] for key in (
                "status", "comparison_scope", "compared_step_count",
                "latest_available_steps", "maximum_absolute_loss_difference",
                "missing", "threshold_failures",
            )
        }
    print(json.dumps(printable, indent=2))
    if status == "fail":
        raise SystemExit(2)
    if status == "incomplete":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
