#!/usr/bin/env python
"""Aggregate the 20-replication Interact2AR joint-evaluator supplement."""

import argparse
import glob
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np


PATTERN = re.compile(r"replication_(\d{2})\.json$")
WRAPPERS = ("JointsFull", "JointsBody", "JointsHands")
METRICS = ("fid", "matching", "top1", "top2", "top3", "diversity", "multimodality")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_artifact(record, label, verified_artifacts):
    path = Path(record.get("path", "")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    stat = path.stat()
    if stat.st_size != record.get("bytes"):
        raise ValueError(f"{label} changed after scoring")
    verification_key = (
        str(path), stat.st_dev, stat.st_ino, stat.st_size,
        stat.st_mtime_ns, stat.st_ctime_ns, record.get("sha256"),
    )
    if verification_key in verified_artifacts:
        return
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} changed after scoring")
    verified_artifacts.add(verification_key)


def statistic(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "ci95": float(1.96 * array.std() / math.sqrt(len(array))),
        "values": array.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-methods", nargs="+", required=True)
    parser.add_argument(
        "--author-calibration-audit", type=Path,
        default=Path("output/interact2ar_author_release_calibration/audit.json"),
    )
    parser.add_argument(
        "--scorer-parity-audit", type=Path,
        default=Path("output/interact2ar_joint_scorer_parity_audit.json"),
    )
    args = parser.parse_args()
    verified_artifacts = set()

    prerequisite_audits = {}
    for label, path, schema in (
        (
            "author_calibration", args.author_calibration_audit,
            "interact2ar_author_release_calibration_audit_v1",
        ),
        (
            "scorer_parity", args.scorer_parity_audit,
            "interact2ar_joint_scorer_parity_audit_v1",
        ),
    ):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != schema or record.get("status") != "pass":
            raise ValueError(f"{label} prerequisite audit did not pass")
        prerequisite_audits[label] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    methods = {}
    shared_provenance = None
    for method in args.expected_methods:
        paths = sorted(glob.glob(str(args.input_root / method / "replication_*.json")))
        ids = []
        records = []
        for path in paths:
            match = PATTERN.search(path)
            if match:
                ids.append(int(match.group(1)))
                records.append(json.loads(Path(path).read_text(encoding="utf-8")))
        if ids != list(range(20)):
            raise ValueError(f"{method}: expected replications 0..19, got {ids}")
        for replication, record in enumerate(records):
            if (
                record.get("schema") != "interact2ar_joint_evaluator_replication_v1"
                or record.get("status") != "complete"
                or record.get("method") != method
                or record.get("replication") != replication
                or record.get("samples") != 1696
                or record.get("mm_samples") != 100
                or record.get("mm_repeats") != 30
                or set(record.get("metrics", {})) != set(WRAPPERS)
            ):
                raise ValueError(f"{method}/replication_{replication:02d}: invalid contract")
            for label in (
                "requests", "raw_ground_truth", "main_generations", "mm_generations",
                "prior_official_interx_score",
            ):
                require_artifact(
                    record[label], f"{method}/{replication}/{label}",
                    verified_artifacts,
                )
            provenance = record["evaluator_provenance"]
            if provenance.get("separate_from_main_table") is not True:
                raise ValueError("joint evaluator result is not marked supplementary")
            if provenance.get("paper_aligned_fair_replications") != 20:
                raise ValueError("joint evaluator result is not the 20-replication protocol")
            require_artifact(
                provenance["prerequisite_paper_repository_data_audit"],
                f"{method}/{replication}/paper-repository-data audit",
                verified_artifacts,
            )
            require_artifact(
                provenance["prerequisite_author_release_calibration_audit"],
                f"{method}/{replication}/author-release calibration audit",
                verified_artifacts,
            )
            if (
                provenance["prerequisite_author_release_calibration_audit"]
                != prerequisite_audits["author_calibration"]
            ):
                raise ValueError(
                    f"{method}/{replication}: scorer did not bind the current "
                    "author calibration audit"
                )
            if shared_provenance is None:
                shared_provenance = provenance
            elif provenance != shared_provenance:
                raise ValueError("evaluator provenance differs across methods/replications")

        methods[method] = {
            wrapper: {
                metric: statistic([record["metrics"][wrapper][metric] for record in records])
                for metric in METRICS
            }
            for wrapper in WRAPPERS
        }

    result = {
        "schema": "interact2ar_joint_evaluator_suite_summary_v1",
        "status": "complete",
        "scope": "supplementary_same_unconditional_frozen_generations",
        "not_main_interx_metric_space": True,
        "replications": 20,
        "samples_per_replication": 1696,
        "methods": methods,
        "evaluator_provenance": shared_provenance,
        "prerequisite_audits": prerequisite_audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)

    lines = [
        "# Interact2AR released joint-evaluator supplement",
        "",
        "These numbers use the same unconditional frozen generations as the Inter-X main table. "
        "They are a separate evaluator space and must not replace or be mixed numerically with "
        "the official Inter-X full-672D evaluator.",
        "",
    ]
    for wrapper in WRAPPERS:
        lines.extend([
            f"## {wrapper}", "",
            "| Method | FID ↓ | Matching ↓ | R@1 ↑ | R@2 ↑ | R@3 ↑ | Diversity ↑ | MModality ↑ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method in args.expected_methods:
            metric = methods[method][wrapper]
            value = lambda key: f"{metric[key]['mean']:.4f} ± {metric[key]['ci95']:.4f}"
            lines.append(
                f"| {method} | {value('fid')} | {value('matching')} | {value('top1')} | "
                f"{value('top2')} | {value('top3')} | {value('diversity')} | "
                f"{value('multimodality')} |"
            )
        lines.append("")
    markdown = args.output.with_suffix(".md")
    markdown_temporary = markdown.with_suffix(
        markdown.suffix + f".tmp.{os.getpid()}"
    )
    markdown_temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(markdown_temporary, markdown)
    print(json.dumps({"status": "complete", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
