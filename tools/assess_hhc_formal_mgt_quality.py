#!/usr/bin/env python3
"""Assess whether a current paper-contract HHC MGT is safe to build upon."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs/hhc_formal_mgt_quality_gate_v1.json"


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


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def artifact_record_valid(record: dict) -> bool:
    if not isinstance(record, dict):
        return False
    path = Path(record.get("path", ""))
    return (
        path.is_file()
        and path.stat().st_size == record.get("bytes")
        and sha256_file(path) == record.get("sha256")
    )


def metric_protocol_checks(metric: dict, policy: dict, role: str) -> dict:
    metadata = metric.get("main_generation_metadata", {})
    evaluator = metric.get("evaluator_provenance", {})
    sampling = policy["sampling_contract"]
    expected_mgt_contract = "current" if role == "candidate" else "historical_mmm"
    expected_training_contract = (
        "paper_text_cfg" if role == "candidate"
        else "historical_mmm_shape_dropout"
    )
    checks = {
        "metric_complete": (
            metric.get("schema") == "fair_interx_main_only_metrics_v1"
            and metric.get("status") == "complete"
        ),
        "sample_count": metric.get("samples") == policy["expected_samples"],
        "request_hash": (
            metric.get("requests_sha256")
            == policy["expected_requests_sha256"]
            and metric.get("generation_request_hash_verified") is True
        ),
        "vq_checkpoint": (
            metadata.get("vq_checkpoint_sha256")
            == policy["expected_vq_checkpoint_sha256"]
            and metadata.get("vq_contract") == "precleanup_raw12288"
        ),
        "normalization": (
            metadata.get("normalization_mean_sha256")
            == policy["expected_normalization_mean_sha256"]
            and metadata.get("normalization_std_sha256")
            == policy["expected_normalization_std_sha256"]
        ),
        "sampler": metadata.get("sampler") == sampling["sampler"],
        "sampling_steps": (
            metadata.get("sampling_steps") == sampling["sampling_steps"]
            and metadata.get("sampling_steps_unit")
            == sampling["sampling_steps_unit"]
        ),
        "temperature": metadata.get("temperature") == sampling["temperature"],
        "cfg_scale": metadata.get("cfg_scale") == sampling["cfg_scale"],
        "contact_guidance": (
            metadata.get("contact_guidance") is sampling["contact_guidance"]
            and metadata.get("contact_guidance_scale")
            == sampling["contact_guidance_scale"]
            and metadata.get("contact_timing_source")
            == sampling["contact_timing_source"]
            and metadata.get("contact_timing_file_sha256")
            == policy["expected_requests_sha256"]
        ),
        "mgt_contract": metadata.get("mgt_contract") == expected_mgt_contract,
        "training_conditioning_contract": (
            metadata.get("training_conditioning_contract")
            == expected_training_contract
        ),
        "inference_conditioning_contract": (
            metadata.get("inference_conditioning_contract") == "paper_text_cfg"
        ),
        "evaluator_contract": (
            evaluator.get("contract") == "official_interx_full_672d"
            and evaluator.get("checkpoint", {}).get("sha256")
            == policy["expected_evaluator_checkpoint_sha256"]
            and evaluator.get("options", {}).get("sha256")
            == policy["expected_evaluator_options_sha256"]
        ),
        "evaluator_artifacts_content_bound": all(
            artifact_record_valid(evaluator.get(name, {}))
            for name in (
                "checkpoint", "options", "wrapper_source",
                "option_loader_source", "scorer_source",
            )
        ),
        "cpu_evaluator_runtime": (
            metric.get("evaluator_runtime", {}).get("contract")
            == "pytorch_official_evaluator_single_device_v1"
            and metric.get("evaluator_runtime", {}).get("device_type") == "cpu"
        ),
    }
    return checks


def assess(
    candidate: dict,
    reference: dict,
    selection: dict,
    training_contract: dict,
    policy: dict,
) -> tuple[dict, bool]:
    candidate_protocol = metric_protocol_checks(candidate, policy, "candidate")
    reference_protocol = metric_protocol_checks(reference, policy, "reference")
    candidate_meta = candidate.get("main_generation_metadata", {})
    candidate_fid = candidate.get("fid")
    reference_fid = reference.get("fid")
    thresholds = policy["quality_thresholds"]
    delta = (
        float(candidate_fid) - float(reference_fid)
        if finite(candidate_fid) and finite(reference_fid) else math.nan
    )
    ratio = (
        float(candidate_fid) / float(reference_fid)
        if finite(candidate_fid) and finite(reference_fid)
        and float(reference_fid) > 0 else math.nan
    )
    quality_checks = {
        "reference_fid_finite": finite(reference_fid),
        "candidate_fid_finite": finite(candidate_fid),
        "reference_fid_at_or_below_limit": (
            finite(reference_fid)
            and float(reference_fid) <= thresholds["maximum_reference_fid"]
        ),
        "candidate_fid_at_or_below_limit": (
            finite(candidate_fid)
            and float(candidate_fid) <= thresholds["maximum_candidate_fid"]
        ),
        "candidate_minus_reference_at_or_below_limit": (
            finite(delta)
            and delta <= thresholds["maximum_candidate_minus_reference_fid"]
        ),
        "candidate_to_reference_ratio_at_or_below_limit": (
            finite(ratio)
            and ratio <= thresholds["maximum_candidate_to_reference_fid_ratio"]
        ),
    }
    formal_contract = training_contract.get("formal_implementation_contract", {})
    training_checks = {
        "selection_complete": selection.get("run_complete") is True,
        "selection_did_not_use_test_metrics": (
            selection.get("test_metrics_used_for_selection") is False
        ),
        "selected_checkpoint_matches_generation": (
            selection.get("selected_checkpoint_sha256")
            == candidate_meta.get("checkpoint_sha256")
        ),
        "training_manifest_matches_selection": (
            training_contract.get("active_training_manifest", {}).get("sha256")
            == selection.get("training_manifest_sha256")
        ),
        "paper_contract_audit_passed": (
            training_contract.get("schema")
            == "hhc_paper_training_contract_audit_v1"
            and training_contract.get("status") == "pass_with_disclosed_exceptions"
            and formal_contract.get("complete_paper_training_contract") is True
            and not training_contract.get("formal_implementation_mismatches")
        ),
        "paper_model_and_objective": (
            formal_contract.get("paper_architecture") is True
            and formal_contract.get("mask_schedule") == "paper_kinematic"
            and formal_contract.get("objective") == "paper_masked_only"
            and formal_contract.get("conditioning") == "paper_text_cfg"
        ),
    }
    passed = (
        all(candidate_protocol.values())
        and all(reference_protocol.values())
        and all(training_checks.values())
        and all(quality_checks.values())
    )
    return {
        "candidate_protocol_checks": candidate_protocol,
        "reference_protocol_checks": reference_protocol,
        "training_contract_checks": training_checks,
        "quality_checks": quality_checks,
        "observed": {
            "candidate_fid": candidate_fid,
            "reference_fid": reference_fid,
            "candidate_minus_reference_fid": delta,
            "candidate_to_reference_fid_ratio": ratio,
            "candidate_checkpoint_sha256": candidate_meta.get("checkpoint_sha256"),
            "selected_checkpoint_step": selection.get("selected_step"),
        },
        "thresholds": thresholds,
        "shape_mgt_training_authorized": passed,
    }, passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = load(args.policy)
    if (
        policy.get("schema") != "hhc_formal_mgt_quality_gate_policy_v1"
        or policy.get("status") != "active"
    ):
        raise ValueError("formal MGT quality policy is not active v1")
    candidate = load(args.candidate)
    reference = load(args.reference)
    selection = load(args.selection)
    training_contract = load(args.training_contract)
    assessment, passed = assess(
        candidate, reference, selection, training_contract, policy
    )
    payload = {
        "schema": "hhc_formal_mgt_quality_gate_v1",
        "status": "pass" if passed else "fail",
        **assessment,
        "evidence": {
            "candidate": artifact(args.candidate),
            "reference": artifact(args.reference),
            "selection": artifact(args.selection),
            "training_contract": artifact(args.training_contract),
            "policy": artifact(args.policy),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
