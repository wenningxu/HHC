#!/usr/bin/env python3
"""Select an HHC checkpoint solely on frozen official Inter-X validation FID."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATTERN = re.compile(r"model-(\d{9})-[0-9.]+\.ckpt$")
LATEST_PATTERN = re.compile(r"latest-(\d{9})\.ckpt$")
REQUEST_SCHEMA = "hhc_interx_method_neutral_validation_request_v1"
REQUEST_PURPOSE = (
    "held-out official Inter-X checkpoint selection for all methods; "
    "never a reported test row"
)
EXPECTED_CONTRACT = {
    "sampler": "sample_intra_cas",
    "sampling_steps": 50,
    "temperature": 0.5,
    "cfg_scale": 3.5,
    "contact_guidance_scale": 1.0,
    "contact": True,
    "contact_conditioning": False,
    "contact_timing_source": "validation_dataset_annotation_oracle",
    "contact_timing_is_test_data": False,
    "inference_conditioning_contract": "paper_text_cfg",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def validate_artifact(record: dict, label: str) -> Path:
    path = Path(record.get("path", "")).resolve()
    if not path.is_file() or artifact(path) != record:
        raise ValueError(f"{label}: artifact is absent or stale: {path}")
    return path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def checkpoint_step(path: Path) -> int:
    match = MODEL_PATTERN.match(path.name)
    if match is None:
        match = LATEST_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"invalid checkpoint name: {path}")
    filename_step = int(match.group(1))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_step = int(payload.get("global_step", -1))
    if state_step != filename_step:
        raise ValueError(
            f"checkpoint filename/state step mismatch: {filename_step} != {state_step}"
        )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint has no state_dict: {path}")
    if not all(
        bool(torch.isfinite(value).all())
        for value in state.values()
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())
    ):
        raise FloatingPointError(f"checkpoint contains non-finite weights: {path}")
    return state_step


def validate_inputs(args):
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = manifest.get("resolved_arguments", {})
    policy = manifest.get("checkpoint_policy", {})
    required = {
        "training_mask_contract": "historical_pipeline_paper_kinematic_ratio",
        "training_loss_contract": "historical_all_valid",
        "optimizer_contract": "historical_multistep",
        "conditioning_contract": "historical_mmm_shape_dropout",
        "training_token_length_contract": "historical_38",
        "training_validation_contract": "historical_test_side_effect",
        "save_all_validation_checkpoints": True,
        "total_iter": args.expected_final_step,
        "eval_iter": args.interval,
        "seed": 114514,
    }
    mismatches = {
        key: {"expected": expected, "actual": resolved.get(key)}
        for key, expected in required.items() if resolved.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"training contract mismatch: {mismatches}")
    if (
        policy.get("retain_all_validation_checkpoints") is not True
        or policy.get("test_metrics_used_for_selection") is not False
        or policy.get("historical_test_metric_controls_retention") is not False
    ):
        raise ValueError("training checkpoint retention still depends on test metrics")

    latest = run_dir / f"latest-{args.expected_final_step:09d}.ckpt"
    if checkpoint_step(latest) != args.expected_final_step:
        raise ValueError("training run is not complete")
    final_checkpoint_audit_path = args.final_checkpoint_audit.resolve()
    final_checkpoint_audit = json.loads(
        final_checkpoint_audit_path.read_text(encoding="utf-8")
    )
    latest_artifact = artifact(latest)
    if (
        final_checkpoint_audit.get("status") != "pass"
        or int(final_checkpoint_audit.get("global_step", -1))
        != args.expected_final_step
        or Path(final_checkpoint_audit.get("checkpoint", "")).resolve()
        != latest.resolve()
        or final_checkpoint_audit.get("checkpoint_sha256")
        != latest_artifact["sha256"]
        or final_checkpoint_audit.get("model_state", {}).get(
            "all_floating_tensors_finite"
        ) is not True
        or final_checkpoint_audit.get("optimizer_state", {}).get(
            "all_floating_tensors_finite"
        ) is not True
        or final_checkpoint_audit.get("has_loop_state") is not True
    ):
        raise ValueError("final 300k checkpoint audit is absent or invalid")

    wandb_history_audit_path = args.wandb_history_audit.resolve()
    wandb_history_audit = json.loads(
        wandb_history_audit_path.read_text(encoding="utf-8")
    )
    observability = manifest.get("observability", {})
    if (
        wandb_history_audit.get("schema")
        != "hhc_wandb_online_metric_history_audit_v1"
        or wandb_history_audit.get("status") != "pass"
        or wandb_history_audit.get("remote_state") != "finished"
        or wandb_history_audit.get("metric_profile")
        != "masked_token_classification"
        or int(wandb_history_audit.get("expected_final_step", -1))
        != args.expected_final_step
        or wandb_history_audit.get("run_id")
        != observability.get("wandb_run_id")
        or wandb_history_audit.get("project")
        != observability.get("wandb_project")
        or wandb_history_audit.get("manifest", {}).get("sha256")
        != sha256_file(manifest_path)
        or not wandb_history_audit.get("checks")
        or not all(wandb_history_audit["checks"].values())
    ):
        raise ValueError("complete online W&B history audit is absent or invalid")

    sampling_runtime_audit_path = args.sampling_runtime_audit.resolve()
    sampling_runtime_audit = json.loads(
        sampling_runtime_audit_path.read_text(encoding="utf-8")
    )
    sampling_contract = sampling_runtime_audit.get("contract", {})
    expected_sampling_contract = {
        "sampler": "sample_intra_cas",
        "model_architecture_contract": "historical_1024d_with_end_token",
        "sampling_steps": 50,
        "sampling_steps_unit": "outer_maskgit_iterations",
        "cascade_order": ["root", "body", "hands"],
        "maximum_active_channel_stages_per_outer_iteration": 3,
        "maximum_model_forward_calls_per_outer_iteration_with_cfg": 6,
        "maximum_common50_active_channel_stages": 150,
        "maximum_common50_model_forward_calls_with_cfg": 300,
        "resolved_channels_skip_forward_but_not_outer_iterations": True,
    }
    if (
        sampling_runtime_audit.get("schema") != "hhc_sampling_runtime_audit_v1"
        or sampling_runtime_audit.get("status") != "pass"
        or sampling_contract != expected_sampling_contract
        or not sampling_runtime_audit.get("checks")
        or not all(sampling_runtime_audit["checks"].values())
        or sampling_runtime_audit.get("source", {}).get("sha256")
        != sha256_file(ROOT / "models/two_person_transformer.py")
    ):
        raise ValueError("HHC common-50 sampler runtime audit is absent or stale")

    model_structure_audit_path = args.model_structure_audit.resolve()
    model_structure_audit = json.loads(
        model_structure_audit_path.read_text(encoding="utf-8")
    )
    structure_artifacts = model_structure_audit.get("artifacts", {})
    validated_structure_artifacts = {
        key: validate_artifact(
            structure_artifacts.get(key, {}), f"model_structure/{key}"
        )
        for key in (
            "checkpoint", "training_manifest", "sampling_runtime_audit",
            "model_source", "motion_source", "exporter", "implementation",
        )
    }
    if (
        model_structure_audit.get("schema")
        != "hhc_historical_model_checkpoint_structure_audit_v1"
        or model_structure_audit.get("status") != "pass"
        or int(model_structure_audit.get("expected_step", -1))
        != args.expected_final_step
        or not model_structure_audit.get("checks")
        or not all(model_structure_audit["checks"].values())
        or model_structure_audit.get("contract", {}).get("hidden_dimension")
        != 1024
        or model_structure_audit.get("contract", {}).get("attention_heads")
        != 16
        or structure_artifacts.get("checkpoint", {}).get("sha256")
        != latest_artifact["sha256"]
        or structure_artifacts.get("training_manifest", {}).get("sha256")
        != sha256_file(manifest_path)
        or structure_artifacts.get("sampling_runtime_audit", {}).get("sha256")
        != sha256_file(sampling_runtime_audit_path)
        or validated_structure_artifacts["model_source"]
        != (ROOT / "models/two_person_transformer.py").resolve()
        or validated_structure_artifacts["motion_source"]
        != (ROOT / "models/motion_transformer.py").resolve()
        or validated_structure_artifacts["exporter"]
        != (ROOT / "tools/export_hhc_official_interx.py").resolve()
        or validated_structure_artifacts["implementation"]
        != (ROOT / "tools/audit_hhc_historical_model_checkpoint_structure.py").resolve()
    ):
        raise ValueError("historical 1024-D checkpoint structure audit is invalid")

    candidates = sorted(run_dir.glob("model-*.ckpt"))
    indexed = {checkpoint_step(path): path for path in candidates}
    expected_steps = list(range(args.interval, args.expected_final_step + 1, args.interval))
    if sorted(indexed) != expected_steps:
        raise ValueError(
            f"validation checkpoint inventory mismatch: expected={expected_steps}, "
            f"actual={sorted(indexed)}"
        )

    request_manifest_path = args.request_manifest.resolve()
    request_manifest = json.loads(
        request_manifest_path.read_text(encoding="utf-8")
    )
    if (
        request_manifest.get("schema") != REQUEST_SCHEMA
        or request_manifest.get("status") != "complete"
        or request_manifest.get("samples") != 544
        or request_manifest.get("seed") != 500123
        or request_manifest.get("purpose") != REQUEST_PURPOSE
        or request_manifest.get("test_set_used") is not False
        or request_manifest.get("request_is_byte_identical_to_source") is not True
    ):
        raise ValueError("official validation request contract changed")
    neutral_artifacts = {
        key: validate_artifact(request_manifest.get(key, {}), f"request/{key}")
        for key in (
            "split", "motions", "evaluator_options", "evaluator_weights",
            "formal_hhc_validation_selection", "source_generation_manifest",
            "request_generation_implementation", "implementation",
        )
    }
    if neutral_artifacts["implementation"] != (
        ROOT / "tools/promote_interx_validation_request_contract.py"
    ).resolve():
        raise ValueError("method-neutral validation promoter changed")
    if neutral_artifacts["request_generation_implementation"] != (
        ROOT / "tools/prepare_interx_validation_requests.py"
    ).resolve():
        raise ValueError("validation request generator changed")
    request = Path(request_manifest["request"]["path"]).resolve()
    if artifact(request) != request_manifest["request"]:
        raise ValueError("official validation request hash changed")
    return (
        manifest_path, latest, indexed, request_manifest_path, request,
        final_checkpoint_audit_path, wandb_history_audit_path,
        sampling_runtime_audit_path,
        model_structure_audit_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--request-manifest", type=Path,
        default=ROOT / "output/interx_official_val_requests/manifest.json",
    )
    parser.add_argument(
        "--vq-checkpoint", type=Path,
        default=ROOT / "checkpoints/vq_precleanup_raw12288/net_175000.pth",
    )
    parser.add_argument(
        "--contact-map", type=Path,
        default=(ROOT / "checkpoints/vq_precleanup_raw12288_historical_mmm_cache/"
                 "code_contact_probs.npy"),
    )
    parser.add_argument(
        "--final-checkpoint-audit", type=Path,
        default=(ROOT / "output/hhc_historical_plus_paper_mask_ratio_"
                 "final_checkpoint_audit.json"),
    )
    parser.add_argument(
        "--wandb-history-audit", type=Path,
        default=(ROOT / "output/hhc_historical_plus_paper_mask_ratio_"
                 "wandb_history_audit.json"),
    )
    parser.add_argument(
        "--sampling-runtime-audit", type=Path,
        default=ROOT / "output/hhc_sampling_runtime_audit.json",
    )
    parser.add_argument(
        "--model-structure-audit", type=Path,
        default=(ROOT / "output/hhc_historical_plus_paper_mask_ratio_"
                 "model_structure_audit.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--expected-final-step", type=int, default=300000)
    parser.add_argument("--interval", type=int, default=10000)
    args = parser.parse_args()

    (training_manifest, latest, candidates, request_manifest, request,
     final_checkpoint_audit, wandb_history_audit,
     sampling_runtime_audit, model_structure_audit) = validate_inputs(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output.resolve()
    base_payload = {
        "schema": "hhc_official_interx_validation_selection_v1",
        "status": "running",
        "purpose": "held-out official Inter-X validation checkpoint selection only",
        "test_metrics_used_for_selection": False,
        "test_metrics_controlled_candidate_retention": False,
        "expected_final_step": args.expected_final_step,
        "candidate_interval": args.interval,
        "expected_candidates": len(candidates),
        "validation_inference_contract": EXPECTED_CONTRACT,
        "artifacts": {
            "training_manifest": artifact(training_manifest),
            "final_latest_checkpoint": artifact(latest),
            "final_checkpoint_audit": artifact(final_checkpoint_audit),
            "wandb_history_audit": artifact(wandb_history_audit),
            "sampling_runtime_audit": artifact(sampling_runtime_audit),
            "model_structure_audit": artifact(model_structure_audit),
            "request_manifest": artifact(request_manifest),
            "requests": artifact(request),
            "vq_checkpoint": artifact(args.vq_checkpoint),
            "contact_map": artifact(args.contact_map),
            "exporter": artifact(ROOT / "tools/export_hhc_official_interx.py"),
            "scorer": artifact(ROOT / "tools/score_fair_interx_main.py"),
            "selector": artifact(Path(__file__)),
        },
        "candidates": [],
    }
    records = []
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "schema", "purpose", "test_metrics_used_for_selection",
            "test_metrics_controlled_candidate_retention",
            "expected_final_step", "candidate_interval",
            "expected_candidates", "validation_inference_contract", "artifacts",
        )
        mismatches = {
            key: {"expected": base_payload.get(key), "actual": previous.get(key)}
            for key in immutable_keys
            if previous.get(key) != base_payload.get(key)
        }
        if mismatches:
            raise ValueError(
                "existing validation selection belongs to another immutable "
                f"contract: {mismatches}"
            )
        seen_steps = set()
        for record in previous.get("candidates", []):
            step = int(record.get("step", -1))
            if step in seen_steps or step not in candidates:
                raise ValueError(f"invalid resumed candidate step: {step}")
            seen_steps.add(step)
            checkpoint = validate_artifact(
                record.get("checkpoint", {}), f"step {step}/checkpoint"
            )
            if checkpoint != candidates[step].resolve():
                raise ValueError(f"step {step}: resumed checkpoint path changed")
            validate_artifact(
                record.get("generations", {}), f"step {step}/generations"
            )
            metrics = validate_artifact(
                record.get("metrics", {}), f"step {step}/metrics"
            )
            score = json.loads(metrics.read_text(encoding="utf-8"))
            expected_values = {
                "validation_fid": float(score["fid"]),
                "top1": float(score["top1"]), "top2": float(score["top2"]),
                "top3": float(score["top3"]),
                "matching": float(score["matching"]),
                "diversity": float(score["diversity"]),
            }
            value_mismatches = {
                key: (record.get(key), value)
                for key, value in expected_values.items()
                if record.get(key) != value
            }
            if value_mismatches:
                raise ValueError(
                    f"step {step}: resumed metric values changed: {value_mismatches}"
                )
            records.append(record)
        records.sort(key=lambda row: int(row["step"]))
        print({
            "resume_verified_candidates": len(records),
            "expected_candidates": len(candidates),
        }, flush=True)
    payload = {**base_payload, "candidates": records}
    atomic_json(output_path, payload)

    completed_steps = {int(record["step"]) for record in records}
    for step, checkpoint in candidates.items():
        if step in completed_steps:
            continue
        generation = output_dir / f"step_{step:09d}.h5"
        metrics = output_dir / f"step_{step:09d}.json"
        subprocess.run([
            sys.executable, "tools/export_hhc_official_interx.py",
            "--checkpoint", str(checkpoint.resolve()),
            "--vq-checkpoint", str(args.vq_checkpoint.resolve()),
            "--vq-contract", "precleanup_raw12288",
            "--mgt-contract", "historical_mmm",
            "--inference-conditioning-contract", "paper_text_cfg",
            "--requests", str(request), "--output", str(generation),
            "--device", args.device, "--batch-size", str(args.batch_size),
            "--repeats", "1", "--seed", "500123", "--resume",
            "--contact-map", str(args.contact_map.resolve()),
            "--contact-map-contract", "paper_formula_train_only",
            "--contact-guidance", "--contact-timing-source", "request_oracle",
            "--sampling-steps", "50", "--temperature", "0.5",
            "--cfg-scale", "3.5", "--contact-guidance-scale", "1.0",
        ], cwd=ROOT, check=True)
        subprocess.run([
            sys.executable, "tools/score_fair_interx_main.py",
            "--requests", str(request), "--main-generations", str(generation),
            "--output", str(metrics), "--device", args.device,
            "--batch-size", str(args.batch_size),
        ], cwd=ROOT, check=True)
        score = json.loads(metrics.read_text(encoding="utf-8"))
        fid = float(score["fid"])
        if not math.isfinite(fid) or fid < 0:
            raise ValueError(f"invalid validation FID at step {step}: {fid}")
        records.append({
            "step": step, "validation_fid": fid,
            "top1": float(score["top1"]), "top2": float(score["top2"]),
            "top3": float(score["top3"]), "matching": float(score["matching"]),
            "diversity": float(score["diversity"]),
            "checkpoint": artifact(checkpoint),
            "generations": artifact(generation), "metrics": artifact(metrics),
        })
        payload["candidates"] = records
        atomic_json(output_path, payload)
        print({"step": step, "validation_fid": fid,
               "completed": len(records), "total": len(candidates)}, flush=True)

    selected = min(records, key=lambda row: (row["validation_fid"], row["step"]))
    payload.update({
        "status": "complete", "evaluated_candidates": len(records),
        "selection_metric": "minimum held-out official Inter-X validation FID",
        "selected_step": selected["step"],
        "selected_validation_fid": selected["validation_fid"],
        "selected_checkpoint": selected["checkpoint"]["path"],
        "selected_checkpoint_sha256": selected["checkpoint"]["sha256"],
        "selected_checkpoint_bytes": selected["checkpoint"]["bytes"],
    })
    atomic_json(output_path, payload)
    print(json.dumps({key: payload[key] for key in (
        "status", "selected_step", "selected_validation_fid",
        "selected_checkpoint")}, indent=2))


if __name__ == "__main__":
    main()
