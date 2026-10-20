#!/usr/bin/env python3
"""Select a capped CoShMDM milestone on frozen Inter-X validation FID."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COSH_ROOT = Path("/mnt/8T-HDD/wenning/coshmdm-code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch

from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from tools.score_fair_interx_generations import (
    calculate_batched_retrieval,
    embed_motions,
)
from utils.eval_trans import (
    calculate_activation_statistics,
    calculate_frechet_distance,
)


MILESTONE = re.compile(r"milestone-(\d{9})\.ckpt$")
ACCEPTED_CAP_STATUSES = {
    "stopped_gracefully", "sigterm_sent_after_180s_grace",
    "trainer_exited_before_monitor_signal",
}
CAP_POLICY = "stop after stable 200k checkpoint, otherwise stop at hard deadline"


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


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def discover(directory: Path, maximum_step: int) -> list[tuple[int, Path]]:
    records = []
    for value in glob.glob(str(directory / "milestone-*.ckpt")):
        path = Path(value).resolve()
        match = MILESTONE.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        if 0 < step <= maximum_step and step % 20000 == 0:
            records.append((step, path))
    records.sort()
    if not records:
        raise RuntimeError(f"no completed milestones at or below {maximum_step}")
    return records


def validate_budget_contract(payload: dict) -> None:
    if (
        payload.get("schema") != "hhc_coshmdm_user_training_cap_v1"
        or payload.get("status") not in ACCEPTED_CAP_STATUSES
        or payload.get("preferred_terminal_optimizer_step") != 200000
        or payload.get("policy") != CAP_POLICY
    ):
        raise ValueError(
            "CoShMDM validation selection requires a finalized valid two-day "
            "training-cap manifest"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--budget-cap-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/coshmdm_interx_val_selection"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/coshmdm_interx_val_selection.json"))
    parser.add_argument("--maximum-step", type=int, default=200000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    requests = args.requests.resolve()
    request_manifest = args.request_manifest.resolve()
    budget_cap = args.budget_cap_manifest.resolve()
    for required in (requests, request_manifest, budget_cap):
        if not required.is_file():
            raise FileNotFoundError(required)
    request_contract = json.loads(request_manifest.read_text())
    budget_contract = json.loads(budget_cap.read_text())
    validate_budget_contract(budget_contract)
    formal_selection = ROOT / "output/mgt_validation_selection_manifest.json"
    if (
        request_contract.get("schema") != "hhc_interx_validation_request_v1"
        or request_contract.get("status") != "complete"
        or request_contract.get("seed") != 500123
        or request_contract.get("samples") != 544
        or request_contract.get("formal_hhc_validation_record_set_exact") is not True
        or request_contract.get("request", {}).get("sha256") != sha256_file(requests)
        or request_contract.get("formal_hhc_validation_selection", {}).get("sha256")
        != sha256_file(formal_selection)
    ):
        raise ValueError(
            "CoShMDM validation request is not exactly bound to the formal "
            "HHC 544-record validation selection contract"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen = np.load(requests, allow_pickle=False)
    names = frozen["names"].astype(str)
    lengths = frozen["lengths"].astype(np.int64)
    gt_embeddings = frozen["gt_embeddings"]
    text_embeddings = frozen["text_embeddings"]
    seed = int(frozen["seed"])
    if len(names) % args.batch_size:
        raise ValueError("validation request must contain complete evaluator batches")

    evaluator = EvaluatorModelWrapper(
        get_opt(
            str(ROOT / "checkpoints/hhi/Comp_v6_KLD01/opt.txt"),
            f"cuda:{args.gpu}",
        )
    )
    gt_mu, gt_cov = calculate_activation_statistics(gt_embeddings)
    candidates = discover(checkpoint_dir, args.maximum_step)
    exporter = COSH_ROOT / "tools/export_official_interx.py"
    exporter_hash = sha256_file(exporter)
    environment = os.environ.copy()
    environment["COSHMDM_STATS_PREFIX"] = "interx_global"

    result = {
        "schema": "hhc_coshmdm_capped_validation_selection_v1",
        "status": "running",
        "purpose": (
            "checkpoint selection on held-out Inter-X validation only; test "
            "metrics are not used"
        ),
        "selection_metric": "minimum official-evaluator validation FID",
        "test_metrics_used_for_selection": False,
        "maximum_user_allowed_step": args.maximum_step,
        "training_cap_status": budget_contract["status"],
        "training_cap_stop_reason": budget_contract.get("stop_reason"),
        "candidate_steps": [step for step, _ in candidates],
        "validation_seed": seed,
        "validation_samples": len(names),
        "formal_hhc_validation_record_set_exact": True,
        "formal_hhc_validation_selection_sha256": sha256_file(formal_selection),
        "batch_size": args.batch_size,
        "sampling_contract": "CoShMDM DDIM50 eta0 CFG3.5",
        "artifacts": {
            "requests": artifact(requests),
            "request_manifest": artifact(request_manifest),
            "formal_hhc_validation_selection": artifact(formal_selection),
            "budget_cap_manifest": artifact(budget_cap),
            "exporter": artifact(exporter),
            "evaluator_options": artifact(
                ROOT / "checkpoints/hhi/Comp_v6_KLD01/opt.txt"
            ),
            "evaluator_weights": artifact(
                ROOT / "checkpoints/hhi/text_mot_match/model/finest.tar"
            ),
            "implementation": artifact(Path(__file__)),
        },
        "candidates": [],
    }
    atomic_json(args.output.resolve(), result)

    for step, checkpoint in candidates:
        checkpoint_record = artifact(checkpoint)
        checkpoint_hash = checkpoint_record["sha256"]
        # The exporter is resumable.  Include the immutable checkpoint digest
        # in the cache key so a same-step checkpoint from another restarted
        # run can never inherit stale validation generations.
        generations = output_dir / (
            f"step_{step:09d}_{checkpoint_hash[:16]}_main.h5"
        )
        command = [
            sys.executable, "tools/export_official_interx.py",
            "--checkpoint", str(checkpoint),
            "--requests", str(requests),
            "--output", str(generations),
            "--gpu", str(args.gpu),
            "--batch-size", str(args.batch_size),
            "--repeats", "1", "--seed", str(seed), "--resume",
        ]
        print(json.dumps({"cwd": str(COSH_ROOT), "command": command}), flush=True)
        subprocess.run(command, cwd=COSH_ROOT, env=environment, check=True)
        if sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError(
                f"CoShMDM checkpoint changed during validation export: {checkpoint}"
            )

        with h5py.File(generations, "r") as source:
            if source["motions"].shape[:2] != (len(names), 1):
                raise ValueError(f"generation shape mismatch at step {step}")
            if not np.array_equal(source["names"][:].astype(str), names):
                raise ValueError(f"generation names mismatch at step {step}")
            if not np.array_equal(source["lengths"][:], lengths):
                raise ValueError(f"generation lengths mismatch at step {step}")
            generated = embed_motions(
                evaluator, source["motions"][:, 0], lengths, args.batch_size
            )
        pred_mu, pred_cov = calculate_activation_statistics(generated)
        fid = float(calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov))
        r_precision, matching = calculate_batched_retrieval(
            text_embeddings, generated, args.batch_size
        )
        record = {
            "step": step,
            "checkpoint": checkpoint_record,
            "checkpoint_source_stable_during_export": True,
            "generations": artifact(generations),
            "validation_fid": fid,
            "validation_r_precision": r_precision.tolist(),
            "validation_matching_score": float(matching),
        }
        result["candidates"].append(record)
        atomic_json(args.output.resolve(), result)

    selected = min(result["candidates"], key=lambda item: (
        item["validation_fid"], item["step"]
    ))
    if sha256_file(exporter) != exporter_hash:
        raise RuntimeError("CoShMDM exporter changed during validation selection")
    result.update({
        "status": "complete",
        "selected_step": selected["step"],
        "selected_validation_fid": selected["validation_fid"],
        "selected_checkpoint": selected["checkpoint"]["path"],
        "selected_checkpoint_sha256": selected["checkpoint"]["sha256"],
        "exporter_source_stable_during_selection": True,
    })
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
