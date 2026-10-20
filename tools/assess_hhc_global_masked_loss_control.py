#!/usr/bin/env python3
"""Compare the 50k global-token masked-NLL control to formal HHC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


MATCHED_ARGUMENTS = (
    "batch_size",
    "block_size",
    "nb_code",
    "code_dim",
    "clip_dim",
    "width",
    "shape_condition_dim",
    "embed_dim_gpt",
    "n_head_gpt",
    "num_layers",
    "num_local_layer",
    "drop_out_rate",
    "ff_rate",
    "lr",
    "warm_up_iter",
    "weight_decay",
    "optimizer",
    "total_iter",
    "checkpoint_iter",
    "eval_iter",
    "eval_seed",
    "seed",
    "paper_architecture",
    "contact",
    "contact_conditioning",
    "shape",
    "training_mask_contract",
    "optimizer_contract",
    "conditioning_contract",
    "training_token_length_contract",
    "training_validation_contract",
    "sampler",
    "sampling_steps",
    "temperature",
    "cfg_scale",
    "contact_guidance_scale",
    "token_cache_contract",
    "vq_checkpoint",
    "cache_root",
    "codebook_dir",
    "historical_codebook_source",
    "historical_contact_source",
    "stats_dir",
    "stats_prefix",
    "train_motion_file",
    "train_split_file",
    "val_motion_file",
    "val_split_file",
    "all_motion_file",
    "all_split_file",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selection(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("run_complete"):
        raise ValueError(f"selection is not complete: {path}")
    if payload.get("test_metrics_used_for_selection") is not False:
        raise ValueError(f"selection is not validation-only: {path}")
    if payload.get("selection_metric") != "minimum fixed-protocol validation FID":
        raise ValueError(f"selection metric changed: {path}")
    expected_validation = {
        "sampler": "sample_intra_cas",
        "sampling_steps": 50,
        "temperature": 0.5,
        "cfg_scale": 3.5,
        "contact_guidance_scale": 1.0,
        "contact_timing_source": "validation_dataset_annotation_oracle",
        "contact_timing_is_test_data": False,
        "contact_as_transformer_training_input": False,
    }
    validation = payload.get("validation_inference_contract", {})
    mismatches = {
        key: {"expected": expected, "actual": validation.get(key)}
        for key, expected in expected_validation.items()
        if validation.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"selection validation contract changed: {mismatches}")
    fid = float(payload["selected_validation_fid"])
    if not math.isfinite(fid) or fid < 0:
        raise ValueError(f"invalid selected validation FID: {fid}")
    expected_final_step = int(payload.get("expected_final_step", -1))
    selected_step = int(payload.get("selected_step", -1))
    if not 0 < selected_step <= expected_final_step:
        raise ValueError(
            f"selected step {selected_step} is outside (0,{expected_final_step}]"
        )
    candidates = payload.get("retained_validation_candidates", [])
    if not candidates:
        raise ValueError(f"selection has no validation candidates: {path}")
    for candidate in candidates:
        candidate_step = int(candidate.get("step", -1))
        candidate_fid = float(candidate.get("validation_fid", math.nan))
        if (not 0 < candidate_step <= expected_final_step
                or not math.isfinite(candidate_fid) or candidate_fid < 0):
            raise ValueError(f"invalid validation candidate: {candidate}")
    return payload


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_training_manifest(selection: dict) -> tuple[Path, dict]:
    path = Path(selection["training_manifest"]).resolve()
    payload = load_json(path)
    if not isinstance(payload.get("resolved_arguments"), dict):
        raise ValueError(f"training manifest lacks resolved arguments: {path}")
    return path, payload


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_single_factor_contract(
    formal_args: dict, control_args: dict | None = None
) -> None:
    if formal_args.get("training_loss_contract") != "paper_masked_only":
        raise ValueError("formal reference does not use paper masked-only loss")
    if formal_args.get("training_token_length_contract") != "paper_50":
        raise ValueError("formal reference does not use the 50-token window")
    if control_args is None:
        return
    if (
        control_args.get("training_loss_contract")
        != "paper_masked_global_token_mean"
    ):
        raise ValueError("control does not use global masked-token-mean loss")
    mismatches = {
        key: {"formal": formal_args.get(key), "control": control_args.get(key)}
        for key in MATCHED_ARGUMENTS
        if formal_args.get(key) != control_args.get(key)
    }
    if mismatches:
        raise ValueError(
            "global masked-loss control is not single-factor matched: "
            f"{mismatches}"
        )


def select_audit_checkpoint(
    audit: dict,
    *,
    checkpoint_sha256: str | None = None,
    checkpoint_path: str | None = None,
) -> dict:
    if (checkpoint_sha256 is None) == (checkpoint_path is None):
        raise ValueError("select an audit checkpoint by exactly one identity")
    matches = [
        row for row in audit.get("checkpoints", [])
        if (
            row.get("sha256") == checkpoint_sha256
            if checkpoint_sha256 is not None
            else str(Path(row.get("path", "")).resolve())
            == str(Path(checkpoint_path).resolve())
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "mask-exposure audit must contain exactly one selected checkpoint: "
            f"sha256={checkpoint_sha256} path={checkpoint_path} "
            f"matches={len(matches)}"
        )
    return matches[0]


def aggregate_stage_metrics(checkpoint: dict) -> dict:
    """Aggregate the three stages used by the Root->Body->Hands sampler."""
    metrics = checkpoint["metrics"]
    stages = (
        ("all_parts_fully_masked", "root"),
        ("body_after_gt_root", "body"),
        ("hand_after_gt_root_body", "hand"),
    )
    rows = [metrics[condition][part] for condition, part in stages]
    tokens = sum(int(row["tokens"]) for row in rows)
    if tokens <= 0:
        raise ValueError("mask-exposure cascade stages contain no tokens")
    correct = sum(
        float(row["accuracy_percent"]) * int(row["tokens"]) / 100.0
        for row in rows
    )
    loss_sum = sum(
        float(row["cross_entropy"]) * int(row["tokens"])
        for row in rows
    )
    return {
        "tokens": tokens,
        "accuracy_percent": 100.0 * correct / tokens,
        "cross_entropy": loss_sum / tokens,
        "stages": {
            f"{condition}/{part}": metrics[condition][part]
            for condition, part in stages
        },
    }


def compare_mask_exposure(
    baseline_checkpoint_path: str,
    control_selection: dict,
    baseline_audit_path: Path,
    control_audit_path: Path,
    expected_split_role: str,
) -> dict:
    baseline_audit = load_json(baseline_audit_path)
    control_audit = load_json(control_audit_path)
    if baseline_audit.get("split_role") != expected_split_role:
        raise ValueError(f"baseline audit is not {expected_split_role}")
    if control_audit.get("split_role") != expected_split_role:
        raise ValueError(f"control audit is not {expected_split_role}")
    for key in ("split_file_sha256", "sample_names_sha256", "samples", "seed"):
        if baseline_audit.get(key) != control_audit.get(key):
            raise ValueError(
                f"mask-exposure {expected_split_role} pairing mismatch for {key}: "
                f"{baseline_audit.get(key)} != {control_audit.get(key)}"
            )
    baseline = aggregate_stage_metrics(select_audit_checkpoint(
        baseline_audit, checkpoint_path=baseline_checkpoint_path
    ))
    control = aggregate_stage_metrics(select_audit_checkpoint(
        control_audit,
        checkpoint_sha256=control_selection["selected_checkpoint_sha256"],
    ))
    return {
        "split_role": expected_split_role,
        "samples": int(baseline_audit["samples"]),
        "sample_names_sha256": baseline_audit["sample_names_sha256"],
        "baseline": baseline,
        "control": control,
        "accuracy_percent_delta_control_minus_baseline": (
            control["accuracy_percent"] - baseline["accuracy_percent"]
        ),
        "cross_entropy_delta_control_minus_baseline": (
            control["cross_entropy"] - baseline["cross_entropy"]
        ),
        "baseline_audit": {
            "path": str(baseline_audit_path.resolve()),
            "sha256": sha256_file(baseline_audit_path),
        },
        "control_audit": {
            "path": str(control_audit_path.resolve()),
            "sha256": sha256_file(control_audit_path),
        },
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-selection", type=Path, required=True)
    parser.add_argument("--control-selection", type=Path)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--comparison-step", type=int, default=50000)
    parser.add_argument("--minimum-absolute-improvement", type=float, default=0.1)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.15)
    parser.add_argument("--baseline-validation-mask-exposure", type=Path)
    parser.add_argument("--baseline-training-mask-exposure", type=Path)
    parser.add_argument("--control-validation-mask-exposure", type=Path)
    parser.add_argument("--control-training-mask-exposure", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    formal = load_selection(args.formal_selection)
    if int(formal["expected_final_step"]) != 300000:
        raise ValueError("formal selection is not the complete 300k run")
    formal_manifest_path, formal_manifest = load_training_manifest(formal)
    formal_args = formal_manifest["resolved_arguments"]
    validate_single_factor_contract(formal_args)
    final_best = float(formal["selected_validation_fid"])
    formal_candidates = [
        candidate for candidate in formal["retained_validation_candidates"]
        if int(candidate["step"]) <= args.comparison_step
    ]
    if not formal_candidates:
        raise ValueError("formal selection has no matched early candidates")
    formal_early = min(
        formal_candidates, key=lambda row: (float(row["validation_fid"]), int(row["step"]))
    )
    payload = {
        "schema": "hhc_global_masked_loss_control_assessment_v1",
        "status": "complete",
        "shape_training_ready": final_best <= args.threshold,
        "formal_selected_validation_fid": final_best,
        "core_fid_threshold": args.threshold,
        "comparison_step": args.comparison_step,
        "formal_early_best": formal_early,
        "formal_selection": {
            "path": str(args.formal_selection.resolve()),
            "sha256": sha256_file(args.formal_selection),
        },
        "formal_training_manifest": artifact(formal_manifest_path),
        "control": None,
    }
    if final_best <= args.threshold:
        payload["decision"] = "retain_formal_contract_and_continue_shape_training"
    elif args.control_selection is None:
        payload["status"] = "diagnostic_required"
        payload["decision"] = "run_global_masked_token_mean_50k_control"
    else:
        control = load_selection(args.control_selection)
        if int(control["expected_final_step"]) != args.comparison_step:
            raise ValueError("control selection did not end at the comparison step")
        control_manifest_path, control_manifest = load_training_manifest(control)
        control_args = control_manifest["resolved_arguments"]
        validate_single_factor_contract(formal_args, control_args)
        control_fid = float(control["selected_validation_fid"])
        formal_fid = float(formal_early["validation_fid"])
        absolute = formal_fid - control_fid
        relative = absolute / formal_fid
        supported = (
            absolute >= args.minimum_absolute_improvement
            and relative >= args.minimum_relative_improvement
        )
        payload["control"] = {
            "path": str(args.control_selection.resolve()),
            "sha256": sha256_file(args.control_selection),
            "training_manifest": artifact(control_manifest_path),
            "selected_step": int(control["selected_step"]),
            "selected_validation_fid": control_fid,
            "absolute_fid_improvement": absolute,
            "relative_fid_improvement": relative,
            "minimum_absolute_improvement": args.minimum_absolute_improvement,
            "minimum_relative_improvement": args.minimum_relative_improvement,
            "hypothesis_supported": supported,
        }
        audit_paths = (
            args.baseline_validation_mask_exposure,
            args.baseline_training_mask_exposure,
            args.control_validation_mask_exposure,
            args.control_training_mask_exposure,
        )
        if not all(audit_paths):
            raise ValueError(
                "final control assessment requires paired validation and training "
                "mask-exposure audits"
            )
        payload["denoising_generalization"] = {
            "validation": compare_mask_exposure(
                formal_early["path"], control,
                args.baseline_validation_mask_exposure,
                args.control_validation_mask_exposure,
                "validation",
            ),
            "training": compare_mask_exposure(
                formal_early["path"], control,
                args.baseline_training_mask_exposure,
                args.control_training_mask_exposure,
                "training",
            ),
            "interpretation": (
                "Paired diagnostic only. FID remains the promotion criterion; "
                "token accuracy cannot replace decoded-motion evaluation."
            ),
        }
        payload["shape_training_ready"] = False
        payload["decision"] = (
            "promote_global_masked_loss_to_full_retrain_before_shape"
            if supported else
            "global_loss_not_sufficient_investigate_mask_conditioning_next"
        )
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
