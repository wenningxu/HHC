#!/usr/bin/env python
"""Gate a repaired MGT against a same-step common-50 reference result."""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path


METRICS = ("fid", "diversity", "top1", "top2", "top3", "matching")
REQUEST_SHA256 = "01d894c2e95183b2aaa510470819d64cd6c5862beb47b114320e91270d0100d7"
VQ_SHA256 = "fdf5afaa5d3513f933254fafc831f6046f7c38b5e649db08fb6be7e0ce1006bb"
CONTACT_MAP_SHA256 = "56109a68bcf5d568cfbfc86de1ba373a8c651a6a4fdbf5007325f60a23809527"
MEAN_SHA256 = "a54dcfd012e07742077050e3a7ee1b4a2918ba2825466ab698d0cf6f9e438446"
STD_SHA256 = "23a2cf52a19cbaab16334821f2a3183e915c671f59fac702b68a1256742bc376"
EVALUATOR_OPTIONS_SHA256 = "f4ddaa81157259ce6b795f456be98263e50cd788a7800c30aacbfc2361b2d6dd"
EVALUATOR_CHECKPOINT_SHA256 = "b1699f13276c044d8b7ae5f6063eb241b7ef96d7e92c878857abd5081e40701e"
STEP_PATTERN = re.compile(r"(?:model-step=|model-0*)(\d+)(?:-|$)")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def checkpoint_step(record: dict) -> int:
    checkpoint = record.get("main_generation_metadata", {}).get("checkpoint", "")
    # Historical Lightning encoded the step in the parent directory
    # (``model-step=...-val/FID=...ckpt``), while the refactor encodes it in
    # the filename (``model-000...-FID.ckpt``).
    match = STEP_PATTERN.search(checkpoint)
    if not match:
        raise ValueError(f"cannot parse checkpoint step from {checkpoint!r}")
    return int(match.group(1))


def validate_artifact(record: dict, expected_sha256: str | None, label: str) -> None:
    path = Path(record.get("path", ""))
    require(path.is_file(), f"{label}: artifact is missing: {path}")
    require(record.get("bytes") == path.stat().st_size, f"{label}: byte size changed")
    actual = sha256_file(path)
    require(actual == record.get("sha256"), f"{label}: recorded hash changed")
    if expected_sha256 is not None:
        require(actual == expected_sha256, f"{label}: unexpected frozen hash")


def validate_evaluator(record: dict, label: str) -> dict:
    provenance = record.get("evaluator_provenance", {})
    require(
        provenance.get("contract") == "official_interx_full_672d",
        f"{label}: wrong evaluator contract",
    )
    expected = {
        "options": EVALUATOR_OPTIONS_SHA256,
        "checkpoint": EVALUATOR_CHECKPOINT_SHA256,
        "wrapper_source": None,
        "option_loader_source": None,
        "scorer_source": None,
    }
    for role, expected_sha256 in expected.items():
        artifact = provenance.get(role)
        require(isinstance(artifact, dict), f"{label}: missing evaluator {role}")
        validate_artifact(artifact, expected_sha256, f"{label} evaluator {role}")
    return provenance


def validate_evaluator_runtime(record: dict, label: str) -> dict:
    runtime = record.get("evaluator_runtime", {})
    require(
        runtime.get("contract")
        == "pytorch_official_evaluator_single_device_v1"
        and runtime.get("requested_device") == "cpu"
        and runtime.get("device_type") == "cpu"
        and runtime.get("device_index") is None
        and runtime.get("device_name") == "cpu"
        and isinstance(runtime.get("torch_version"), str)
        and bool(runtime.get("torch_version"))
        and (
            runtime.get("torch_cuda_version") is None
            or isinstance(runtime.get("torch_cuda_version"), str)
        ),
        f"{label}: evaluator runtime is missing or not the frozen CPU contract",
    )
    return runtime


def validate_generation_artifact(record: dict, role: str, label: str) -> None:
    path = Path(record.get(f"{role}_generations", ""))
    require(path.is_file(), f"{label}: {role} generations are missing")
    require(
        sha256_file(path) == record.get(f"{role}_generations_sha256"),
        f"{label}: {role} generation hash changed",
    )


def validate_metadata(
    metadata: dict,
    expected_seed: int,
    label: str,
    allow_legacy_compute_names: bool = False,
) -> tuple[bool, bool]:
    expected = {
        "sampler": "sample_intra_cas",
        "sampler_semantics": "paper_high_confidence_tokens_fixed",
        "sampling_steps": 50,
        "sampling_steps_unit": "outer_maskgit_iterations",
        "temperature": 0.5,
        "cfg_scale": 3.5,
        "cascade_order": "root,body,hands",
        "maximum_active_channel_stages_per_outer_iteration": 3,
        "maximum_model_forward_calls_per_outer_iteration_with_cfg": 6,
        "resolved_channels_skip_forward_but_not_outer_iterations": True,
        "contact_guidance": True,
        "contact_guidance_scale": 1.0,
        "contact_timing_source": "request_oracle",
        "contact_timing_file_sha256": REQUEST_SHA256,
        "requests_sha256": REQUEST_SHA256,
        "seed_strategy": "base_seed + repeat_index",
        "seed": expected_seed,
        "generation_length_policy": "exact_request_length",
        "mgt_contract": "historical_mmm",
        "vq_contract": "precleanup_raw12288",
        "vq_checkpoint_sha256": VQ_SHA256,
        "contact_map_sha256": CONTACT_MAP_SHA256,
        "normalization_mean_sha256": MEAN_SHA256,
        "normalization_std_sha256": STD_SHA256,
    }
    legacy_missing_conditioning = False
    conditioning_expected = {
        "training_conditioning_contract": "historical_mmm_shape_dropout",
        "inference_conditioning_contract": "paper_text_cfg",
    }
    if allow_legacy_compute_names and not all(
        key in metadata for key in conditioning_expected
    ):
        # The frozen Aug-28 reference predates explicit separation, but its
        # source-replayed first batch proves this exact inference behavior.
        require(
            not any(key in metadata for key in conditioning_expected),
            f"{label}: partial legacy conditioning metadata",
        )
        legacy_missing_conditioning = True
    else:
        expected.update(conditioning_expected)
    legacy_compute_names = False
    if allow_legacy_compute_names and not all(
        key in metadata for key in (
            "maximum_active_channel_stages_per_outer_iteration",
            "maximum_model_forward_calls_per_outer_iteration_with_cfg",
            "resolved_channels_skip_forward_but_not_outer_iterations",
        )
    ):
        require(
            metadata.get("channel_updates_per_outer_step") == 3
            and metadata.get("model_forward_calls_per_outer_step_with_cfg") == 6,
            f"{label}: legacy compute metadata is missing or invalid",
        )
        for key in (
            "maximum_active_channel_stages_per_outer_iteration",
            "maximum_model_forward_calls_per_outer_iteration_with_cfg",
            "resolved_channels_skip_forward_but_not_outer_iterations",
        ):
            expected.pop(key)
        legacy_compute_names = True
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    require(not mismatches, f"{label}: generation contract mismatch: {mismatches}")
    # The frozen reference HDF5 predates this explicit attribute, but its exact
    # train-only contact-map hash above proves the same formula. New outputs
    # must either carry the canonical value or remain byte-compatible legacy.
    require(
        metadata.get("contact_map_contract") in (None, "paper_formula_train_only"),
        f"{label}: wrong contact-map contract",
    )
    return legacy_compute_names, legacy_missing_conditioning


def validate(
    record: dict,
    expected_step: int,
    label: str,
    allow_legacy_compute_names: bool = False,
) -> dict:
    metadata = record.get("main_generation_metadata", {})
    mm_metadata = record.get("mm_generation_metadata", {})
    require(record.get("schema") == "fair_interx_ordinary_metrics_v2", f"{label}: wrong metric schema")
    require(record.get("status") == "complete", f"{label}: metrics are incomplete")
    require(record.get("samples") == 1696, f"{label}: expected 1696 samples")
    require(record.get("mm_samples") == 100 and record.get("mm_repeats") == 30, f"{label}: wrong MM contract")
    require(not any("body_controlled" in key for key in record), f"{label}: forbidden body-controlled metric")
    require(record.get("requests_sha256") == REQUEST_SHA256, f"{label}: wrong requests")
    require(metadata.get("requests_sha256") == REQUEST_SHA256, f"{label}: metadata request mismatch")
    require(record.get("generation_request_hash_verified") is True, f"{label}: request hash was not verified")
    require(record.get("generation_length_policy") == "requested_lengths", f"{label}: wrong scored length policy")
    main_legacy_compute, main_legacy_conditioning = validate_metadata(
        metadata, 1234, f"{label} main", allow_legacy_compute_names
    )
    mm_legacy_compute, mm_legacy_conditioning = validate_metadata(
        mm_metadata, 101234, f"{label} MM", allow_legacy_compute_names
    )
    require(
        main_legacy_compute == mm_legacy_compute,
        f"{label}: main/MM compute-metadata generations differ",
    )
    require(
        main_legacy_conditioning == mm_legacy_conditioning,
        f"{label}: main/MM conditioning metadata differ",
    )
    validate_generation_artifact(record, "main", label)
    validate_generation_artifact(record, "mm", label)
    evaluator = validate_evaluator(record, label)
    evaluator_runtime = validate_evaluator_runtime(record, label)
    require(metadata.get("checkpoint_sha256"), f"{label}: missing checkpoint hash")
    step = checkpoint_step(record)
    require(step == expected_step, f"{label}: checkpoint step {step} != {expected_step}")
    for metric in METRICS:
        value = record.get(metric)
        require(isinstance(value, (int, float)) and math.isfinite(value), f"{label}: invalid {metric}")
    return {
        "checkpoint_step": step,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "evaluator_provenance": evaluator,
        "evaluator_runtime": evaluator_runtime,
        "legacy_missing_contact_map_contract": (
            "contact_map_contract" not in metadata
            or "contact_map_contract" not in mm_metadata
        ),
        "legacy_imprecise_compute_field_names": main_legacy_compute,
        "legacy_missing_explicit_conditioning_contract": (
            main_legacy_conditioning
        ),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--max-abs-fid-diff", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(0 <= args.max_abs_fid_diff <= 0.02, "FID tolerance must be in [0, 0.02]")

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    reference_checkpoint = validate(
        reference, args.expected_step, "reference",
        allow_legacy_compute_names=True,
    )
    candidate_checkpoint = validate(candidate, args.expected_step, "candidate")
    require(
        reference_checkpoint["evaluator_provenance"]
        == candidate_checkpoint["evaluator_provenance"],
        "reference and candidate evaluator provenance differ",
    )
    require(
        reference_checkpoint["evaluator_runtime"]
        == candidate_checkpoint["evaluator_runtime"],
        "reference and candidate evaluator runtimes differ",
    )
    differences = {
        metric: {
            "signed": float(candidate[metric]) - float(reference[metric]),
            "absolute": abs(float(candidate[metric]) - float(reference[metric])),
        }
        for metric in METRICS
    }
    fid_difference = differences["fid"]["absolute"]
    status = "pass" if fid_difference <= args.max_abs_fid_diff else "fail"
    payload = {
        "schema": "hhc_mgt_common50_bridge_v1",
        "status": status,
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256_file(args.reference),
            **reference_checkpoint,
            **{metric: reference[metric] for metric in METRICS},
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": sha256_file(args.candidate),
            **candidate_checkpoint,
            **{metric: candidate[metric] for metric in METRICS},
        },
        "differences": differences,
        "maximum_absolute_fid_difference": args.max_abs_fid_diff,
        "ordinary_metric_contract": "official Inter-X full-space",
        "evaluated_samples": 1696,
        "requests_sha256": REQUEST_SHA256,
        "sampler": "sample_intra_cas",
        "sampling_steps": 50,
        "temperature": 0.5,
        "cfg_scale": 3.5,
        "contact_guidance": True,
        "contact_guidance_scale": 1.0,
        "contact_timing_source": "request_oracle",
        "contact_timing_file_sha256": REQUEST_SHA256,
        "training_conditioning_contract": "historical_mmm_shape_dropout",
        "inference_conditioning_contract": "paper_text_cfg",
        "contact_map_contract": "paper_formula_train_only_by_frozen_hash",
        "contact_map_sha256": CONTACT_MAP_SHA256,
        "vq_contract": "precleanup_raw12288",
        "vq_checkpoint_sha256": VQ_SHA256,
        "normalization_mean_sha256": MEAN_SHA256,
        "normalization_std_sha256": STD_SHA256,
        "evaluator_provenance": reference_checkpoint["evaluator_provenance"],
        "evaluator_runtime": reference_checkpoint["evaluator_runtime"],
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
