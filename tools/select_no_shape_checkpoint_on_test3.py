#!/usr/bin/env python3
"""Incrementally select no-shape HHC checkpoints on three official test draws.

This is intentionally a user-directed, test-guided selection protocol.  It is
kept separate from the held-out selector so the resulting formal table cannot
be mislabeled as unbiased validation selection.
"""

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
MODEL_PATTERN = re.compile(r"model-(\d{9})-[0-9]+(?:\.[0-9]+)?\.ckpt$")
REPLICATIONS = (0, 1, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_training_manifest(run_dir: Path) -> tuple[Path, dict]:
    manifest_path = run_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    args = manifest["resolved_arguments"]
    required = {
        "shape": False,
        "contact": False,
        "contact_conditioning": False,
        "training_mask_contract": "historical_pipeline_paper_kinematic_ratio",
        "training_loss_contract": "historical_all_valid",
        "paper_architecture": False,
        "sampler": "sample_intra_cas",
        "sampling_steps": 50,
        "temperature": 0.5,
        "cfg_scale": 3.5,
        "total_iter": 300000,
        "wandb_mode": "online",
    }
    mismatches = {key: {"expected": expected, "actual": args.get(key)}
                  for key, expected in required.items()
                  if args.get(key) != expected}
    if mismatches:
        raise ValueError(f"training contract mismatch: {mismatches}")
    return manifest_path, manifest


def discover_checkpoints(run_dir: Path, expected_steps: list[int]) -> dict[int, Path]:
    checkpoints = {}
    expected = set(expected_steps)
    for path in run_dir.glob("model-*.ckpt"):
        match = MODEL_PATTERN.fullmatch(path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step in expected:
            if step in checkpoints:
                raise ValueError(f"duplicate model checkpoint at step {step}")
            checkpoints[step] = path.resolve()
    if sorted(checkpoints) != expected_steps:
        raise ValueError(
            f"checkpoint steps {sorted(checkpoints)} != {expected_steps}"
        )
    return checkpoints


def expected_score_provenance() -> dict:
    interact2ar = (ROOT.parent / "Interact2Ar").resolve(strict=True)
    mmm = Path("/home/wenning/projects/MMM").resolve(strict=True)
    return {
        "contract": "interact2ar_released_joint_evaluator_test_selection_v1",
        "repository": str(interact2ar),
        "repository_revision": subprocess.check_output(
            ["git", "-C", str(interact2ar), "rev-parse", "HEAD"], text=True
        ).strip(),
        "representation": "joints",
        "normalized": False,
        "parts": ["JointsFull", "JointsBody", "JointsHands"],
        "batch_size": 32,
        "diversity_pairs": 300,
        "metric_rng": "request_seed+1000*wrapper_index",
        "primary_selection_metric": "JointsFull/FID",
        "wrapper_source": artifact(
            interact2ar / "src/evaluation/evaluator_wrapper.py"
        ),
        "metrics_source": artifact(interact2ar / "src/evaluation/metrics.py"),
        "scorer_source": artifact(ROOT / "tools/score_interact2ar_joint_validation.py"),
        "body_model": artifact(mmm / "body_models/smplx/SMPLX_NEUTRAL.npz"),
        "checkpoints": {
            "JointsBody": artifact(
                interact2ar / "release-assets/checkpoints-evaluator/joints_body/model/finest.tar"
            ),
            "JointsFull": artifact(
                interact2ar / "release-assets/checkpoints-evaluator/joints_full/model/finest.tar"
            ),
            "JointsHands": artifact(
                interact2ar / "release-assets/checkpoints-evaluator/joints_hands/model/finest.tar"
            ),
        },
        "prerequisite_author_release_calibration_audit": artifact(
            ROOT / "output/interact2ar_author_release_calibration/audit.json"
        ),
        "prerequisite_paper_repository_data_audit": artifact(
            ROOT / "output/interact2ar_paper_repo_data_audit.json"
        ),
    }


def score_is_current(path: Path, request: Path, raw_gt: Path,
                     generation: Path, expected_provenance: dict) -> bool:
    if not path.is_file() or not generation.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    provenance = record.get("evaluator_provenance", {})
    return (
        record.get("status") == "complete"
        and record.get("evaluation_role") == "test_guided_checkpoint_selection"
        and record.get("requests") == artifact(request)
        and record.get("raw_ground_truth") == artifact(raw_gt)
        and record.get("generations") == artifact(generation)
        and all(provenance.get(key) == value
                for key, value in expected_provenance.items())
        and provenance.get("role")
        == "user-directed three-replication official-test checkpoint selection; "
           "test-guided and not an unbiased held-out selection"
    )


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    candidates = parser.add_mutually_exclusive_group(required=True)
    candidates.add_argument("--end-step", type=int)
    candidates.add_argument("--candidate-steps", type=int, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--requests-dir", type=Path,
                        default=Path("output/fair_interx_requests"))
    parser.add_argument("--raw-gt-dir", type=Path,
                        default=Path("output/interact2ar_joint_evaluator_ground_truth"))
    parser.add_argument("--stats-dir", type=Path, default=Path("dataset/interx"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("checkpoints/vq_precleanup_raw12288_historical_mmm_cache"))
    parser.add_argument("--vq-checkpoint", type=Path,
                        default=Path("checkpoints/vq_precleanup_raw12288/net_175000.pth"))
    parser.add_argument("--reuse-step30-formal", action="store_true")
    args = parser.parse_args()
    if args.candidate_steps is None:
        if args.end_step <= 0 or args.end_step > 300000 or args.end_step % 10000:
            raise ValueError("end-step must be a 10k multiple in [10k, 300k]")
        candidate_steps = list(range(10000, args.end_step + 1, 10000))
        selection_status = (
            "complete" if args.end_step == 300000 else "incremental_running_training"
        )
        candidate_contract = f"every 10k through {args.end_step}"
    else:
        candidate_steps = sorted(set(args.candidate_steps))
        if (
            candidate_steps != args.candidate_steps
            or any(step <= 0 or step > 300000 or step % 10000
                   for step in candidate_steps)
        ):
            raise ValueError(
                "candidate-steps must be unique, sorted 10k multiples in [10k, 300k]"
            )
        selection_status = "complete"
        candidate_contract = candidate_steps
    if args.batch_size != 32:
        raise ValueError("test-guided selection freezes batch size 32")

    run_dir = args.run_dir.resolve()
    manifest_path, _ = validate_training_manifest(run_dir)
    checkpoints = discover_checkpoints(run_dir, candidate_steps)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output.resolve()
    # Hash the released evaluator stack once per selector invocation.  Every
    # cached score must match this exact stack; the hashes are deliberately not
    # recomputed for every checkpoint/replication.
    expected_provenance = expected_score_provenance()
    records = []

    for step, checkpoint in sorted(checkpoints.items()):
        checkpoint_record = artifact(checkpoint)
        replications = []
        for replication in REPLICATIONS:
            request = (args.requests_dir / f"replication_{replication:02d}.npz").resolve()
            raw_gt = (args.raw_gt_dir / f"replication_{replication:02d}.h5").resolve()
            step_dir = output_dir / f"step_{step:09d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            generation = step_dir / f"replication_{replication:02d}_main.h5"
            if args.reuse_step30_formal and step == 30000:
                formal = ROOT / (
                    "output/fair_eval/hhc_no_mgt_shape_v4_pilot50k_bestval_common50/"
                    f"replication_{replication:02d}_main.h5"
                )
                if not formal.is_file():
                    raise FileNotFoundError(formal)
                generation = formal.resolve()
            score = step_dir / f"replication_{replication:02d}_joint.json"

            # Always enter the exporter's --resume path, including when the
            # HDF5 already exists.  That path validates checkpoint/request/VQ/
            # normalization hashes and every frozen sampling attribute before
            # returning early for a complete file.  Merely trusting the step
            # directory name could otherwise reuse a stale generation after a
            # checkpoint or protocol change.  A mismatch raises and preserves
            # the existing file rather than silently overwriting it.
            run([
                sys.executable, "tools/export_hhc_official_interx.py",
                "--checkpoint", str(checkpoint),
                "--vq-checkpoint", str(args.vq_checkpoint.resolve()),
                "--vq-contract", "precleanup_raw12288",
                "--mgt-contract", "historical_mmm", "--no-mgt-shape",
                "--inference-conditioning-contract", "paper_text_cfg",
                "--stats-dir", str(args.stats_dir.resolve()),
                "--stats-prefix", "Interx", "--requests", str(request),
                "--output", str(generation), "--device", args.device,
                "--batch-size", "32", "--repeats", "1",
                "--seed", str(1234 + replication), "--resume",
                "--contact-map", str((args.cache_root / "code_contact_probs.npy").resolve()),
                "--contact-map-contract", "paper_formula_train_only",
                "--no-contact-guidance", "--contact-timing-source", "none",
                "--sampling-steps", "50", "--temperature", "0.5",
                "--cfg-scale", "3.5", "--contact-guidance-scale", "1.0",
            ])
            if not score_is_current(
                    score, request, raw_gt, generation, expected_provenance):
                run([
                    sys.executable, "tools/score_interact2ar_joint_validation.py",
                    "--requests", str(request), "--raw-ground-truth", str(raw_gt),
                    "--generations", str(generation), "--output", str(score),
                    "--device", args.device, "--batch-size", "32",
                    "--evaluation-role", "test_guided_checkpoint_selection",
                ])
            scored = json.loads(score.read_text(encoding="utf-8"))
            fid = float(scored["metrics"]["JointsFull"]["fid"])
            if not math.isfinite(fid) or fid < 0:
                raise ValueError(f"step {step} replication {replication}: invalid FID")
            replications.append({
                "replication": replication, "fid": fid,
                "request": artifact(request), "raw_ground_truth": artifact(raw_gt),
                "generation": artifact(generation), "score": artifact(score),
            })
        mean_fid = sum(row["fid"] for row in replications) / len(replications)
        records.append({"step": step, "checkpoint": checkpoint_record,
                        "replications": replications, "mean_joints_full_fid": mean_fid})
        selected = min(records, key=lambda row: (row["mean_joints_full_fid"], row["step"]))
        atomic_json(output_path, {
            "schema": "hhc_no_shape_test3_checkpoint_selection_v1",
            "status": selection_status,
            "selection_is_test_guided": True,
            "unbiased_heldout_selection_claim_allowed": False,
            "contract": {
                "split": "official Inter-X test",
                "replications": list(REPLICATIONS),
                "metric": "released Interact2AR JointsFull FID mean over three fixed draws",
                "sampler": "sample_intra_cas/50/T0.5/CFG3.5",
                "checkpoint_candidates": candidate_contract,
            },
            "training_manifest": artifact(manifest_path),
            "evaluated_through_step": step,
            "requested_end_step": args.end_step,
            "requested_candidate_steps": candidate_steps,
            "candidates": records,
            "selected_step": selected["step"],
            "selected_mean_joints_full_fid": selected["mean_joints_full_fid"],
            "selected_checkpoint": selected["checkpoint"]["path"],
            "selected_checkpoint_sha256": selected["checkpoint"]["sha256"],
            "implementation": artifact(Path(__file__)),
        })
        print(json.dumps({"evaluated_step": step, "mean_fid": mean_fid,
                          "best_step": selected["step"],
                          "best_mean_fid": selected["mean_joints_full_fid"]}), flush=True)


if __name__ == "__main__":
    main()
