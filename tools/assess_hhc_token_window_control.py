#!/usr/bin/env python3
"""Assess the matched original-MMM 38-token-window control by val FID."""

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
    "training_loss_contract",
    "optimizer_contract",
    "conditioning_contract",
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


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_selection(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_complete") is not True:
        raise ValueError(f"selection is incomplete: {path}")
    if payload.get("test_metrics_used_for_selection") is not False:
        raise ValueError(f"selection used test metrics: {path}")
    if payload.get("selection_metric") != "minimum fixed-protocol validation FID":
        raise ValueError(f"wrong selection metric: {path}")
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
        if (
            not 0 < candidate_step <= expected_final_step
            or not math.isfinite(candidate_fid)
            or candidate_fid < 0
        ):
            raise ValueError(f"invalid validation candidate: {candidate}")
    return payload


def load_manifest(selection: dict) -> tuple[Path, dict]:
    path = Path(selection["training_manifest"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    return path, payload


def early_best(selection: dict, comparison_step: int) -> dict:
    candidates = [
        row for row in selection["retained_validation_candidates"]
        if int(row["step"]) <= comparison_step
    ]
    if not candidates:
        raise ValueError("formal selection has no matched early checkpoint")
    return min(
        candidates,
        key=lambda row: (float(row["validation_fid"]), int(row["step"])),
    )


def candidate_at_step(selection: dict, comparison_step: int) -> dict:
    candidates = [
        row for row in selection["retained_validation_candidates"]
        if int(row["step"]) == comparison_step
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"selection must contain exactly one validation candidate at "
            f"step {comparison_step}, got {len(candidates)}"
        )
    return candidates[0]


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
    parser.add_argument("--formal-selection", type=Path, required=True)
    parser.add_argument("--alignment-audit", type=Path, required=True)
    parser.add_argument("--control-selection", type=Path)
    parser.add_argument("--comparison-step", type=int, default=50000)
    parser.add_argument("--minimum-absolute-improvement", type=float, default=0.1)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    alignment = json.loads(args.alignment_audit.read_text(encoding="utf-8"))
    if (
        alignment.get("schema") != "hhc_mgt_token_window_alignment_audit_v1"
        or alignment.get("status") != "pass"
    ):
        raise ValueError("token-window alignment audit does not pass")
    formal = load_selection(args.formal_selection)
    if int(formal["expected_final_step"]) != 300000:
        raise ValueError("formal reference is not the complete 300k run")
    formal_manifest_path, formal_manifest = load_manifest(formal)
    formal_args = formal_manifest["resolved_arguments"]
    if formal_args["training_token_length_contract"] != "paper_50":
        raise ValueError("formal reference is not the audited 50-token run")
    formal_early = early_best(formal, args.comparison_step)
    formal_endpoint = candidate_at_step(formal, args.comparison_step)

    payload = {
        "schema": "hhc_original38_token_window_control_assessment_v1",
        "status": "diagnostic_required",
        "comparison_step": args.comparison_step,
        "formal_50_early_best": formal_early,
        "formal_50_matched_endpoint": formal_endpoint,
        "control_38": None,
        "decision": "run_original_mmm_38_token_window_50k_control",
        "thresholds": {
            "minimum_absolute_fid_improvement": args.minimum_absolute_improvement,
            "minimum_relative_fid_improvement": args.minimum_relative_improvement,
        },
        "artifacts": {
            "formal_selection": artifact(args.formal_selection),
            "formal_training_manifest": artifact(formal_manifest_path),
            "alignment_audit": artifact(args.alignment_audit),
            "implementation": artifact(Path(__file__)),
        },
    }
    if args.control_selection is not None:
        control = load_selection(args.control_selection)
        if int(control["expected_final_step"]) != args.comparison_step:
            raise ValueError("38-token control did not end at comparison step")
        control_manifest_path, control_manifest = load_manifest(control)
        control_args = control_manifest["resolved_arguments"]
        if control_args["training_token_length_contract"] != "original_mmm_38":
            raise ValueError("control does not use original MMM 38-token window")
        mismatches = {
            key: {"formal": formal_args.get(key), "control": control_args.get(key)}
            for key in MATCHED_ARGUMENTS
            if formal_args.get(key) != control_args.get(key)
        }
        if mismatches:
            raise ValueError(f"38-token control is not matched: {mismatches}")
        formal_fid = float(formal_early["validation_fid"])
        control_fid = float(control["selected_validation_fid"])
        absolute = formal_fid - control_fid
        relative = absolute / formal_fid
        supported = (
            absolute >= args.minimum_absolute_improvement
            and relative >= args.minimum_relative_improvement
        )
        control_endpoint = candidate_at_step(control, args.comparison_step)
        formal_endpoint_fid = float(formal_endpoint["validation_fid"])
        control_endpoint_fid = float(control_endpoint["validation_fid"])
        endpoint_absolute = formal_endpoint_fid - control_endpoint_fid
        endpoint_relative = endpoint_absolute / formal_endpoint_fid
        endpoint_supported = (
            endpoint_absolute >= args.minimum_absolute_improvement
            and endpoint_relative >= args.minimum_relative_improvement
        )
        payload.update({
            "status": "complete",
            "control_38": {
                "selected_step": int(control["selected_step"]),
                "selected_validation_fid": control_fid,
                "absolute_fid_improvement": absolute,
                "relative_fid_improvement": relative,
                "hypothesis_supported": supported,
                "selection_based_hypothesis_supported": supported,
                "matched_endpoint": {
                    "step": args.comparison_step,
                    "formal_validation_fid": formal_endpoint_fid,
                    "control_validation_fid": control_endpoint_fid,
                    "absolute_fid_improvement": endpoint_absolute,
                    "relative_fid_improvement": endpoint_relative,
                    "hypothesis_supported": endpoint_supported,
                },
            },
            "decision_basis": (
                "minimum validation FID over the same 10k-spaced candidate "
                "set through comparison_step; matched endpoint is reported "
                "separately and is not substituted for checkpoint selection"
            ),
            "decision": (
                "promote_original_mmm_38_window"
                if supported and endpoint_supported
                else (
                    "checkpoint_selection_supports_38_but_matched_endpoint_does_not"
                    if supported
                    else "retain_50_window_for_global_loss_diagnostic"
                )
            ),
        })
        payload["artifacts"].update({
            "control_selection": artifact(args.control_selection),
            "control_training_manifest": artifact(control_manifest_path),
        })

    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
