#!/usr/bin/env python3
"""Prove that a scorer-source refresh leaves frozen VQ metrics unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


METRICS = (
    "fid",
    "diversity_gt",
    "diversity",
    "top1",
    "top2",
    "top3",
    "matching",
)
FROZEN_PROVENANCE = (
    "options",
    "checkpoint",
    "wrapper_source",
    "option_loader_source",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    for label, record in (("before", before), ("after", after)):
        require(
            record.get("schema") == "fair_interx_main_only_metrics_v1"
            and record.get("status") == "complete",
            f"{label} VQ score is not a complete official main-only result",
        )
        require(record.get("samples") == 1696, f"{label} sample count changed")
        require(
            record.get("generation_length_policy") == "requested_lengths",
            f"{label} VQ score did not use exact requested lengths",
        )
        require(
            record.get("main_generation_metadata", {}).get(
                "generation_length_policy"
            ) == "exact_request_length",
            f"{label} reconstruction file is not exact-request-length",
        )

    for key in (
        "requests_sha256",
        "main_generations_sha256",
        "samples",
        "generation_length_policy",
        "evaluator_runtime",
    ):
        require(before.get(key) == after.get(key), f"{key} changed")
    require(
        before.get("main_generation_metadata")
        == after.get("main_generation_metadata"),
        "VQ reconstruction metadata changed",
    )

    before_provenance = before.get("evaluator_provenance", {})
    after_provenance = after.get("evaluator_provenance", {})
    require(
        before_provenance.get("contract")
        == after_provenance.get("contract")
        == "official_interx_full_672d",
        "official evaluator contract changed",
    )
    for key in FROZEN_PROVENANCE:
        require(
            before_provenance.get(key) == after_provenance.get(key),
            f"frozen evaluator provenance changed: {key}",
        )
    before_scorer = before_provenance.get("scorer_source", {})
    after_scorer = after_provenance.get("scorer_source", {})
    require(
        Path(before_scorer.get("path", "")).name
        == Path(after_scorer.get("path", "")).name
        == "score_fair_interx_main.py",
        "scorer identity changed",
    )
    current_scorer = Path(after_scorer["path"])
    require(current_scorer.is_file(), "current scorer source is missing")
    require(
        artifact(current_scorer) == after_scorer,
        "after result does not bind the current scorer bytes",
    )
    require(
        before_scorer.get("sha256") != after_scorer.get("sha256"),
        "migration audit requires a real scorer-source change",
    )

    differences = {}
    for metric in METRICS:
        old = before.get(metric)
        new = after.get(metric)
        require(old == new, f"VQ reconstruction metric changed: {metric}")
        differences[metric] = {
            "before": old,
            "after": new,
            "absolute": 0.0,
        }
    require(
        before.get("multimodality") is None
        and after.get("multimodality") is None,
        "main-only scorer fabricated multimodality",
    )

    payload = {
        "schema": "hhc_vq_reconstruction_scorer_migration_audit_v1",
        "status": "pass",
        "reason": (
            "the current scorer reads effective generation lengths for "
            "baselines that do not exactly match request lengths; this frozen "
            "VQ reconstruction is exact-request-length, so every metric must "
            "remain bitwise-identical"
        ),
        "checks": {
            "same_frozen_requests": True,
            "same_frozen_reconstruction_hdf5": True,
            "same_official_evaluator": True,
            "same_cpu_runtime_contract": True,
            "exact_request_length": True,
            "all_metrics_exact": True,
            "current_scorer_source_bound": True,
        },
        "scorer_source": {
            "before": before_scorer,
            "after": after_scorer,
        },
        "metric_differences": differences,
        "artifacts": {
            "before": artifact(args.before),
            "after": artifact(args.after),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
