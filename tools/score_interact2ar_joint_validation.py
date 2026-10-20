#!/usr/bin/env python3
"""Score one held-out validation generation with Interact2AR joint encoders."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.score_interact2ar_joint_evaluators as joint


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def nonnegative_fid(value: float, tolerance: float = 1e-6) -> float:
    value = float(value)
    if not math.isfinite(value) or value < -tolerance:
        raise FloatingPointError(f"invalid FID value {value}")
    return 0.0 if abs(value) <= tolerance else value


def score_embeddings(text, ground_truth, generated, batch_size, metric_seed):
    gt_mu, gt_cov = joint.calculate_activation_statistics(ground_truth)
    pred_mu, pred_cov = joint.calculate_activation_statistics(generated)
    matching, r_precision = joint.batched_retrieval(text, generated, batch_size)
    np.random.seed(metric_seed)
    diversity_gt = float(joint.calculate_diversity(ground_truth, 300))
    diversity = float(joint.calculate_diversity(generated, 300))
    return {
        "fid": nonnegative_fid(joint.calculate_frechet_distance(
            gt_mu, gt_cov, pred_mu, pred_cov
        )),
        "matching": float(matching),
        "top1": float(r_precision[0]),
        "top2": float(r_precision[1]),
        "top3": float(r_precision[2]),
        "diversity_gt": diversity_gt,
        "diversity": diversity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--raw-ground-truth", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--evaluation-role",
        choices=("heldout_validation", "test_guided_checkpoint_selection"),
        default="heldout_validation",
    )
    parser.add_argument(
        "--prerequisite-audit", type=Path,
        default=Path("output/interact2ar_paper_repo_data_audit.json"),
    )
    parser.add_argument(
        "--author-calibration-audit", type=Path,
        default=Path("output/interact2ar_author_release_calibration/audit.json"),
    )
    args = parser.parse_args()
    if args.batch_size != 32:
        raise ValueError("joint validation selection requires batch size 32")

    request_path = args.requests.resolve()
    gt_path = args.raw_ground_truth.resolve()
    generation_path = args.generations.resolve()
    output_path = args.output.resolve()
    request = np.load(request_path, allow_pickle=False)
    names = request["names"].astype(str)
    captions = request["captions"].astype(str)
    lengths = request["lengths"].astype(np.int64)
    if len(names) % args.batch_size:
        raise ValueError("validation set must contain complete retrieval batches")

    prerequisite = joint.load_prerequisite_audit(args.prerequisite_audit)
    calibration = joint.load_author_calibration_audit(
        args.author_calibration_audit, prerequisite
    )
    text_dir = joint.INTERACT2AR_ROOT / "data/texts_processed"
    vectorizer = joint.WordVectorizer(
        str(joint.INTERACT2AR_ROOT / "data/glove"), "hhi_vab"
    )
    os.chdir(joint.INTERACT2AR_ROOT)
    wrappers = {
        name: joint.EvaluatorModelWrapper(
            SimpleNamespace(), args.device, checkpoint,
            representation="joints", part=part, normalized=False,
        )
        for name, (part, checkpoint) in joint.WRAPPERS.items()
    }
    words, pos, sent_lens = joint.vectorize_captions(
        names, captions, text_dir, vectorizer
    )
    text_embeddings = {
        name: joint.embed_text(wrapper, words, pos, sent_lens, args.batch_size)
        for name, wrapper in wrappers.items()
    }

    with h5py.File(gt_path, "r") as gt_source, h5py.File(
        generation_path, "r"
    ) as generation_source:
        joint.validate_h5(gt_source, names, lengths, 1, "validation raw ground truth")
        joint.validate_h5(generation_source, names, lengths, 1, "validation generation")
        request_sha = joint.sha256_file(request_path)
        if gt_source.attrs.get("requests_sha256") != request_sha:
            raise ValueError("validation raw GT is not bound to the request")
        if generation_source.attrs.get("requests_sha256") != request_sha:
            raise ValueError("validation generation is not bound to the request")
        generation_lengths = joint.effective_lengths(generation_source, lengths)
        gt_embeddings = joint.embed_motion_all_wrappers(
            wrappers, gt_source["motions"], lengths, args.batch_size
        )
        generated_embeddings = joint.embed_motion_all_wrappers(
            wrappers, generation_source["motions"], generation_lengths,
            args.batch_size,
        )
        generation_metadata = joint.serializable_attributes(generation_source)

    metrics = {}
    for wrapper_index, wrapper_name in enumerate(wrappers):
        metrics[wrapper_name] = score_embeddings(
            text_embeddings[wrapper_name], gt_embeddings[wrapper_name],
            generated_embeddings[wrapper_name], args.batch_size,
            int(request["seed"]) + 1000 * wrapper_index,
        )

    role_contract = {
        "heldout_validation": (
            "interact2ar_released_joint_evaluator_validation_v1",
            "held-out checkpoint selection only; never a reported test row",
        ),
        "test_guided_checkpoint_selection": (
            "interact2ar_released_joint_evaluator_test_selection_v1",
            "user-directed three-replication official-test checkpoint selection; "
            "test-guided and not an unbiased held-out selection",
        ),
    }
    evaluator_contract, evaluator_role = role_contract[args.evaluation_role]
    provenance = {
        "contract": evaluator_contract,
        "role": evaluator_role,
        "repository": str(joint.INTERACT2AR_ROOT),
        "repository_revision": joint.repository_revision(),
        "representation": "joints",
        "normalized": False,
        "parts": list(joint.WRAPPERS),
        "batch_size": args.batch_size,
        "diversity_pairs": 300,
        "primary_selection_metric": "JointsFull/FID",
        "metric_rng": "request_seed+1000*wrapper_index",
        "wrapper_source": joint.artifact(
            joint.INTERACT2AR_ROOT / "src/evaluation/evaluator_wrapper.py"
        ),
        "metrics_source": joint.artifact(
            joint.INTERACT2AR_ROOT / "src/evaluation/metrics.py"
        ),
        "scorer_source": joint.artifact(Path(__file__)),
        "body_model": joint.artifact(
            joint.INTERACT2AR_ROOT / "data/body_models/smplx/SMPLX_NEUTRAL.npz"
        ),
        "checkpoints": {
            name: joint.artifact(joint.INTERACT2AR_ROOT / checkpoint)
            for name, (_, checkpoint) in joint.WRAPPERS.items()
        },
        "prerequisite_paper_repository_data_audit": prerequisite,
        "prerequisite_author_release_calibration_audit": calibration,
    }
    payload = {
        "schema": "hhc_interact2ar_joint_validation_score_v1",
        "status": "complete",
        "samples": len(names),
        "seed": int(request["seed"]),
        "evaluation_role": args.evaluation_role,
        "requests": joint.artifact(request_path),
        "raw_ground_truth": joint.artifact(gt_path),
        "generations": joint.artifact(generation_path),
        "generation_metadata": generation_metadata,
        "effective_length_policy": (
            "generated_lengths"
            if not np.array_equal(generation_lengths, lengths)
            else "requested_lengths"
        ),
        "metrics": metrics,
        "evaluator_provenance": provenance,
    }
    atomic_json(output_path, payload)
    print(json.dumps({"status": "complete", "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
