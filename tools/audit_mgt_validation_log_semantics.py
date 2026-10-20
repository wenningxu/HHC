#!/usr/bin/env python3
"""Prove which historical/current W&B validation fields are pointwise comparable."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


CURRENT_METRICS = (
    "val/FID",
    "val/Diversity",
    "val/TOP1",
    "val/TOP2",
    "val/TOP3",
    "val/Matching",
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--historical-source-recovery", type=Path, required=True)
    parser.add_argument("--candidate-source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    recovery = json.loads(
        args.historical_source_recovery.read_text(encoding="utf-8")
    )
    candidate_manifest = json.loads(
        args.candidate_source_manifest.read_text(encoding="utf-8")
    )
    historical_train_record = recovery.get("snapshots", {}).get(
        "train/train_t2i_trans.py", {}
    )
    candidate_train_record = candidate_manifest.get("sources", {}).get(
        "train/train_t2i_trans.py", {}
    )
    historical_train = Path(historical_train_record.get("path", ""))
    candidate_train = Path(candidate_train_record.get("path", ""))
    if not historical_train.is_file() or not candidate_train.is_file():
        raise FileNotFoundError("one or both frozen training sources are absent")
    historical_text = historical_train.read_text(encoding="utf-8")
    candidate_text = candidate_train.read_text(encoding="utf-8")

    historical_expected = {
        "val/FID": "self.log('val/FID', fid)",
        "val/Diversity": "self.log('val/Diversity', self.best_div)",
        "val/TOP1": "self.log('val/TOP1', self.best_top1)",
        "val/TOP2": "self.log('val/TOP2', self.best_top2)",
        "val/TOP3": "self.log('val/TOP3', self.best_top3)",
        "val/Matching": "self.log('val/Matching', self.best_matching)",
    }
    candidate_expected = {
        metric: (
            f"self.log('{metric}', current_metrics['{key}'])"
        )
        for metric, key in {
            "val/FID": "FID",
            "val/Diversity": "Diversity",
            "val/TOP1": "R_precision_top1",
            "val/TOP2": "R_precision_top2",
            "val/TOP3": "R_precision_top3",
            "val/Matching": "Matching",
        }.items()
    }
    checks = {
        "comparison_complete": comparison.get("status") == "comparison_complete",
        "comparison_schema_matches": (
            comparison.get("schema") == "hhc_mgt_wandb_validation_comparison_v1"
        ),
        "historical_source_recovery_passes": recovery.get("status") == "pass",
        "historical_snapshot_hash_matches": (
            historical_train_record.get("sha256") == sha256_file(historical_train)
        ),
        "candidate_source_snapshot_passes": candidate_manifest.get("status") == "pass",
        "candidate_snapshot_hash_matches": (
            candidate_train_record.get("sha256") == sha256_file(candidate_train)
        ),
        "historical_logging_statements_match": all(
            statement in historical_text for statement in historical_expected.values()
        ),
        "candidate_logging_statements_match": all(
            statement in candidate_text for statement in candidate_expected.values()
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    payload = {
        "schema": "hhc_mgt_validation_log_semantics_v1",
        "status": status,
        "checks": checks,
        "pointwise_comparable_metrics": ["val/FID"],
        "not_pointwise_comparable_metrics": [
            metric for metric in CURRENT_METRICS if metric != "val/FID"
        ],
        "semantics": {
            "historical": {
                "val/FID": "current checkpoint value",
                "val/Diversity": "best-so-far closeness to real diversity",
                "val/TOP1": "best-so-far maximum",
                "val/TOP2": "best-so-far maximum",
                "val/TOP3": "best-so-far maximum",
                "val/Matching": "best-so-far minimum",
            },
            "candidate": {metric: "current checkpoint value" for metric in CURRENT_METRICS},
        },
        "interpretation": (
            "Only FID may be compared checkpoint-by-checkpoint in the supplied W&B "
            "histories. Historical Diversity/R-Precision/Matching rows are running "
            "envelopes, not the metrics of that checkpoint. Final comparisons require "
            "regeneration with the frozen official validation/test protocol."
        ),
        "artifacts": {
            "comparison": artifact(args.comparison),
            "historical_source_recovery": artifact(args.historical_source_recovery),
            "historical_train_source": artifact(historical_train),
            "candidate_source_manifest": artifact(args.candidate_source_manifest),
            "candidate_train_source": artifact(candidate_train),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
