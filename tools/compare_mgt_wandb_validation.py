#!/usr/bin/env python
"""Compare exact-step ordinary MGT validation metrics in local W&B logs."""

import argparse
import hashlib
import json
import os
from pathlib import Path


METRICS = (
    "val/FID",
    "val/Diversity",
    "val/TOP1",
    "val/TOP2",
    "val/TOP3",
    "val/Matching",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_validation_rows(path: Path) -> dict[int, dict]:
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
        if last_step is None or "val/FID" not in row:
            continue
        rows[last_step] = {
            "trainer/global_step": last_step,
            **{metric: row[metric] for metric in METRICS},
        }
    return rows


def file_record(path: Path, include_sha256: bool) -> dict:
    stat = path.stat()
    result = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        result["sha256"] = sha256_file(path)
    return result


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument(
        "--checkpoint-step-offset",
        type=int,
        default=0,
        help=(
            "Add this value only when reporting the conceptual checkpoint "
            "milestone for each W&B trainer/global_step. Metric lookup and "
            "comparison always remain on the unmodified W&B history step."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-abs-diff", type=float)
    parser.add_argument(
        "--skip-candidate-sha256",
        action="store_true",
        help="Use while the candidate transaction log is still open.",
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
    requested = sorted(set(args.steps))
    reference_rows = scan_validation_rows(args.reference)
    candidate_rows = scan_validation_rows(args.candidate)
    comparisons = []
    missing = []
    failures = []
    for step in requested:
        reference = reference_rows.get(step)
        candidate = candidate_rows.get(step)
        if reference is None or candidate is None:
            missing.append({
                "trainer/global_step": step,
                "reference_missing": reference is None,
                "candidate_missing": candidate is None,
            })
            continue
        differences = {}
        for metric in METRICS:
            difference = float(candidate[metric]) - float(reference[metric])
            differences[metric] = {
                "signed": difference,
                "absolute": abs(difference),
            }
            if args.max_abs_diff is not None and abs(difference) > args.max_abs_diff:
                failures.append({
                    "trainer/global_step": step,
                    "metric": metric,
                    "absolute_difference": abs(difference),
                    "maximum": args.max_abs_diff,
                })
        comparisons.append({
            "trainer/global_step": step,
            "conceptual_checkpoint_step": step + args.checkpoint_step_offset,
            "reference": reference,
            "candidate": candidate,
            "difference": differences,
        })

    if missing:
        status = "incomplete"
    elif args.max_abs_diff is None:
        status = "comparison_complete"
    else:
        status = "fail" if failures else "pass"
    payload = {
        "schema": "hhc_mgt_wandb_validation_comparison_v1",
        "status": status,
        "reference": file_record(args.reference, True),
        "candidate": file_record(args.candidate, not args.skip_candidate_sha256),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "requested_steps": requested,
        "step_contract": {
            "comparison_axis": "exact trainer/global_step in both W&B logs",
            "checkpoint_step_offset": args.checkpoint_step_offset,
            "conceptual_checkpoint_steps": [
                step + args.checkpoint_step_offset for step in requested
            ],
            "offset_affects_metric_lookup": False,
        },
        "available_steps": {
            "reference": sorted(reference_rows),
            "candidate": sorted(candidate_rows),
        },
        "comparisons": comparisons,
        "missing": missing,
        "maximum_absolute_difference": args.max_abs_diff,
        "threshold_failures": failures,
        "metric_contract": "ordinary full-space Inter-X validation only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    printable = payload
    if args.summary_only:
        printable = {
            key: payload[key] for key in (
                "status", "requested_steps", "available_steps",
                "maximum_absolute_difference", "missing",
                "threshold_failures",
            )
        }
    print(json.dumps(printable, indent=2))
    if status == "fail":
        raise SystemExit(2)
    if status == "incomplete":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
