#!/usr/bin/env python3
"""Select no-shape MGT on val with common-50 and Interact2AR joints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"model-(\d{9})-[0-9]+(?:\.[0-9]+)?\.ckpt$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_run(args) -> tuple[Path, dict[int, Path]]:
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = manifest["resolved_arguments"]
    total_iter = int(resolved.get("total_iter", -1))
    if args.final_step <= 0 or args.final_step > total_iter:
        raise ValueError(
            f"final step {args.final_step} must be in [1, total_iter={total_iter}]"
        )
    # A diagnostic run stops before the configured training endpoint, whereas a
    # full-budget run has no diagnostic stop.  Treating both cases as
    # diagnostic_stop_iter == final_step incorrectly rejects a healthy 300k
    # continuation whose total_iter is itself 300k.
    expected_diagnostic_stop = args.final_step if args.final_step < total_iter else 0
    required = {
        "shape": False,
        "contact": False,
        "contact_conditioning": False,
        "training_mask_contract": "historical_pipeline_paper_kinematic_ratio",
        "training_loss_contract": args.expected_training_loss_contract,
        "training_validation_contract": "paper_isolated_common",
        "sampler": "sample_intra_cas",
        "sampling_steps": 50,
        "temperature": 0.5,
        "cfg_scale": 3.5,
        "diagnostic_stop_iter": expected_diagnostic_stop,
        "eval_iter": args.interval,
        "save_all_validation_checkpoints": True,
        "wandb_mode": "online",
    }
    mismatches = {key: {"expected": value, "actual": resolved.get(key)}
                  for key, value in required.items() if resolved.get(key) != value}
    if mismatches:
        raise ValueError(f"training contract mismatch: {mismatches}")
    if not (run_dir / f"latest-{args.final_step:09d}.ckpt").is_file():
        raise FileNotFoundError("final recovery checkpoint is absent")
    candidates = {}
    for path in run_dir.glob("model-*.ckpt"):
        match = PATTERN.fullmatch(path.name)
        if match:
            step = int(match.group(1))
            if step in candidates:
                raise ValueError(f"duplicate validation checkpoint at step {step}")
            candidates[step] = path.resolve()
    for path in getattr(args, "recovery_checkpoint", []):
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        step = int(checkpoint["global_step"])
        saved = vars(checkpoint["hyper_parameters"]["args"])
        if Path(saved["out_dir"]).resolve() != run_dir:
            raise ValueError("recovery checkpoint belongs to a different run")
        for key in (*required, "seed", "exp_name"):
            if saved.get(key) != resolved.get(key):
                raise ValueError(f"recovery checkpoint training contract differs: {key}")
        if step in candidates:
            raise ValueError(f"duplicate recovery/validation checkpoint at step {step}")
        candidates[step] = path.resolve()
        del checkpoint
    expected = list(range(args.interval, args.final_step + 1, args.interval))
    if sorted(candidates) != expected:
        raise ValueError(f"candidate steps {sorted(candidates)} != {expected}")
    return manifest_path, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--recovery-checkpoint", type=Path, action="append", default=[],
        help="Explicit missing-validation candidate; verify saved step and run, then replay normally.",
    )
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--vq-checkpoint", type=Path,
                        default=Path("checkpoints/vq_precleanup_raw12288/net_175000.pth"))
    parser.add_argument("--request-manifest", type=Path,
                        default=Path("output/interx_official_val_requests/manifest.json"))
    parser.add_argument(
        "--raw-ground-truth-manifest", type=Path,
        default=Path("output/interx_validation_raw_ground_truth_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--final-step", type=int, default=50_000)
    parser.add_argument("--interval", type=int, default=10_000)
    parser.add_argument(
        "--expected-training-loss-contract",
        choices=("historical_all_valid", "paper_masked_only"),
        default="historical_all_valid",
        help=(
            "Fail-closed expected single-factor training objective. The "
            "default preserves existing no-shape selections; masked-only is "
            "used only for the matched diagnostic arm."
        ),
    )
    args = parser.parse_args()

    manifest_path, candidates = validate_run(args)
    request_manifest_path = args.request_manifest.resolve()
    request_manifest = json.loads(request_manifest_path.read_text(encoding="utf-8"))
    if (request_manifest.get("schema")
            != "hhc_interx_method_neutral_validation_request_v1"
            or request_manifest.get("status") != "complete"
            or request_manifest.get("purpose")
            != "held-out official Inter-X checkpoint selection for all methods; never a reported test row"):
        raise ValueError("method-neutral held-out validation request is invalid")
    request = Path(request_manifest["request"]["path"]).resolve()
    if artifact(request) != request_manifest["request"]:
        raise ValueError("validation request hash is stale")
    raw_gt_manifest_path = args.raw_ground_truth_manifest.resolve()
    raw_gt_manifest = json.loads(raw_gt_manifest_path.read_text(encoding="utf-8"))
    if (raw_gt_manifest.get("schema")
            != "hhc_interx_validation_raw_ground_truth_v1"
            or raw_gt_manifest.get("status") != "complete"
            or raw_gt_manifest.get("request") != artifact(request)
            or raw_gt_manifest.get("request_manifest")
            != artifact(request_manifest_path)):
        raise ValueError("validation raw-GT replay is absent, stale, or mismatched")
    raw_ground_truth = Path(
        raw_gt_manifest["raw_ground_truth"]["path"]
    ).resolve()
    if artifact(raw_ground_truth) != raw_gt_manifest["raw_ground_truth"]:
        raise ValueError("validation raw-GT artifact is stale")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output.resolve()
    contract = {
        "split": "held-out official Inter-X validation",
        "test_metrics_used": False,
        "mgt_shape": False,
        "training_loss_contract": args.expected_training_loss_contract,
        "contact_guidance": False,
        "inference_conditioning_contract": "paper_text_cfg",
        "sampler": "sample_intra_cas",
        "sampling_steps": 50,
        "temperature": 0.5,
        "cfg_scale": 3.5,
        "seed": 500123,
        "selection_evaluator": "Interact2AR released joint JointsFull",
        "selection_metric": "JointsFull/FID",
        "candidate_final_step": args.final_step,
        "candidate_interval": args.interval,
        "supplementary_validation_metrics": ["JointsBody", "JointsHands"],
    }
    records = []
    for step, checkpoint in sorted(candidates.items()):
        generation = output_dir / f"step_{step:09d}.h5"
        metrics = output_dir / f"step_{step:09d}.json"
        subprocess.run([
            sys.executable, "tools/export_hhc_official_interx.py",
            "--checkpoint", str(checkpoint),
            "--vq-checkpoint", str(args.vq_checkpoint.resolve()),
            "--vq-contract", "precleanup_raw12288",
            "--mgt-contract", "historical_mmm", "--no-mgt-shape",
            "--inference-conditioning-contract", "paper_text_cfg",
            "--stats-dir", str(args.stats_dir.resolve()), "--stats-prefix", "Interx",
            "--requests", str(request), "--output", str(generation),
            "--device", args.device, "--batch-size", str(args.batch_size),
            "--repeats", "1", "--seed", "500123", "--resume",
            "--contact-map", str((args.cache_root / "code_contact_probs.npy").resolve()),
            "--contact-map-contract", "paper_formula_train_only",
            "--no-contact-guidance", "--contact-timing-source", "none",
            "--sampling-steps", "50", "--temperature", "0.5",
            "--cfg-scale", "3.5", "--contact-guidance-scale", "1.0",
        ], cwd=ROOT, check=True)
        subprocess.run([
            sys.executable, "tools/score_interact2ar_joint_validation.py",
            "--requests", str(request),
            "--raw-ground-truth", str(raw_ground_truth),
            "--generations", str(generation), "--output", str(metrics),
            "--device", args.device, "--batch-size", str(args.batch_size),
        ], cwd=ROOT, check=True)
        score = json.loads(metrics.read_text(encoding="utf-8"))
        joint_metrics = score["metrics"]
        fid = float(joint_metrics["JointsFull"]["fid"])
        if not math.isfinite(fid) or fid < 0:
            raise ValueError(f"invalid validation FID at step {step}: {fid}")
        records.append({
            "step": step, "validation_fid": fid,
            "selection_metric": "JointsFull/FID",
            "joint_metrics": joint_metrics,
            "checkpoint": artifact(checkpoint), "generations": artifact(generation),
            "metrics": artifact(metrics),
        })
        atomic_json(output_path, {
            "schema": "hhc_no_shape_joint_val_selection_v2", "status": "running",
            "contract": contract, "training_manifest": artifact(manifest_path),
            "request_manifest": artifact(request_manifest_path),
            "raw_ground_truth_manifest": artifact(raw_gt_manifest_path),
            "candidates": records,
        })

    selected = min(records, key=lambda row: (row["validation_fid"], row["step"]))
    payload = {
        "schema": "hhc_no_shape_joint_val_selection_v2", "status": "complete",
        "contract": contract, "training_manifest": artifact(manifest_path),
        "request_manifest": artifact(request_manifest_path),
        "raw_ground_truth_manifest": artifact(raw_gt_manifest_path),
        "candidates": records,
        "selected_step": selected["step"],
        "selected_validation_fid": selected["validation_fid"],
        "selected_validation_metric": "JointsFull/FID",
        "selected_checkpoint": selected["checkpoint"]["path"],
        "selected_checkpoint_sha256": selected["checkpoint"]["sha256"],
    }
    atomic_json(output_path, payload)
    print(json.dumps({key: payload[key] for key in (
        "status", "selected_step", "selected_validation_fid", "selected_checkpoint"
    )}, indent=2))


if __name__ == "__main__":
    main()
