#!/usr/bin/env python3
"""Verify and compare the one-factor HHC sampler validation diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


EXPECTED_REQUEST_SHA256 = (
    "7ec7fae8004bb91425c68503dd50574b6fb3bd8a824d27417d940e43adbe8d67"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "1107064f965bb71a539704fb8623da31c5b4a7bca289d198159d153bdac48149"
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--repredict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    repredict = json.loads(args.repredict.read_text(encoding="utf-8"))
    baseline_meta = baseline.get("main_generation_metadata", {})
    repredict_meta = repredict.get("main_generation_metadata", {})

    invariant_fields = (
        "batch_size", "cascade_order", "cfg_scale", "checkpoint_sha256",
        "contact_guidance", "contact_guidance_scale", "contact_map_sha256",
        "contact_timing_file_sha256", "contact_timing_source",
        "generation_length_policy", "inference_conditioning_contract",
        "mgt_contract", "normalization_mean_sha256",
        "normalization_std_sha256", "requests_sha256", "sampling_steps",
        "sampling_steps_unit", "seed", "seed_strategy", "temperature",
        "training_conditioning_contract", "vq_checkpoint_sha256",
        "vq_contract",
    )
    mismatches = {
        field: [baseline_meta.get(field), repredict_meta.get(field)]
        for field in invariant_fields
        if baseline_meta.get(field) != repredict_meta.get(field)
    }
    checks = {
        "both_complete": (
            baseline.get("status") == "complete"
            and repredict.get("status") == "complete"
        ),
        "same_544_validation_samples": (
            baseline.get("samples") == repredict.get("samples") == 544
        ),
        "frozen_request": (
            baseline.get("requests_sha256")
            == repredict.get("requests_sha256")
            == EXPECTED_REQUEST_SHA256
        ),
        "selected_checkpoint": (
            baseline_meta.get("checkpoint_sha256")
            == repredict_meta.get("checkpoint_sha256")
            == EXPECTED_CHECKPOINT_SHA256
        ),
        "only_declared_metadata_factor_differs": not mismatches,
        "baseline_sampler": baseline_meta.get("sampler") == "sample_intra_cas",
        "diagnostic_sampler": (
            repredict_meta.get("sampler") == "sample_intra_cas_repredict"
            and repredict_meta.get("diagnostic_changed_factor")
            == "token_update_support_only"
            and repredict_meta.get("diagnostic_not_formal_result") is True
        ),
        "finite_fids": (
            math.isfinite(float(baseline.get("fid", math.nan)))
            and math.isfinite(float(repredict.get("fid", math.nan)))
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"invalid re-prediction diagnostic: checks={checks}, "
            f"invariant_mismatches={mismatches}"
        )
    baseline_fid = float(baseline["fid"])
    repredict_fid = float(repredict["fid"])
    payload = {
        "schema": "hhc_repredict_validation_diagnostic_v1",
        "status": "complete",
        "interpretation": (
            "Diagnostic only: all formal sampling settings are fixed and only "
            "the set of positions eligible for re-prediction changes."
        ),
        "checks": checks,
        "invariant_mismatches": mismatches,
        "baseline": {
            "sampler": baseline_meta["sampler"],
            "fid": baseline_fid,
            "matching": baseline.get("matching"),
            "top1": baseline.get("top1"),
            "top2": baseline.get("top2"),
            "top3": baseline.get("top3"),
            "diversity": baseline.get("diversity"),
        },
        "repredict": {
            "sampler": repredict_meta["sampler"],
            "fid": repredict_fid,
            "matching": repredict.get("matching"),
            "top1": repredict.get("top1"),
            "top2": repredict.get("top2"),
            "top3": repredict.get("top3"),
            "diversity": repredict.get("diversity"),
        },
        "fid_delta_repredict_minus_baseline": repredict_fid - baseline_fid,
        "fid_relative_change": repredict_fid / baseline_fid - 1.0,
        "evidence": {
            "baseline": artifact(args.baseline),
            "repredict": artifact(args.repredict),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
