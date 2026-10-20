#!/usr/bin/env python
"""Score one main-generation HDF5 without claiming a multimodality result."""

import argparse
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from tools.score_fair_interx_generations import (
    EVALUATOR_CHECKPOINT,
    EVALUATOR_OPTIONS,
    EVALUATOR_OPTION_SOURCE,
    EVALUATOR_WRAPPER_SOURCE,
    artifact_record,
    calculate_batched_retrieval,
    embed_motions,
    embedding_lengths,
    evaluator_runtime_record,
    require_finite_motion_dataset,
    serializable_attributes,
    sha256_file,
    validate_generation_file,
)
from utils.eval_trans import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--main-generations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    request = np.load(args.requests, allow_pickle=False)
    names = request["names"].astype(str)
    lengths = request["lengths"].astype(np.int64)
    text_embeddings = request["text_embeddings"]
    gt_embeddings = request["gt_embeddings"]
    request_sha256 = sha256_file(args.requests)
    evaluator = EvaluatorModelWrapper(
        get_opt(str(EVALUATOR_OPTIONS), args.device)
    )

    with h5py.File(args.main_generations, "r") as source:
        metadata = serializable_attributes(source)
        validate_generation_file(source, names, lengths, 1, "main")
        require_finite_motion_dataset(source["motions"], "main")
        generated_lengths = embedding_lengths(source, lengths)
        generated = embed_motions(
            evaluator, source["motions"][:, 0], generated_lengths,
            args.batch_size,
        )
    if not np.isfinite(generated).all():
        raise FloatingPointError(
            "official evaluator produced non-finite motion embeddings"
        )
    recorded_request_hash = metadata.get("requests_sha256")
    if recorded_request_hash is None:
        recorded_request_path = metadata.get("requests")
        if (
            not recorded_request_path
            or not os.path.isfile(recorded_request_path)
            or sha256_file(recorded_request_path) != request_sha256
        ):
            raise ValueError(
                "main generation lacks a verifiable exact frozen-request binding"
            )
    elif recorded_request_hash != request_sha256:
        raise ValueError("main generation belongs to another frozen request")

    np.random.seed(int(request["seed"]))
    gt_mu, gt_cov = calculate_activation_statistics(gt_embeddings)
    pred_mu, pred_cov = calculate_activation_statistics(generated)
    retrieval, matching = calculate_batched_retrieval(
        text_embeddings, generated, args.batch_size
    )
    result = {
        "schema": "fair_interx_main_only_metrics_v1",
        "status": "complete",
        "fid": float(calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)),
        "diversity_gt": float(calculate_diversity(gt_embeddings, 300)),
        "diversity": float(calculate_diversity(generated, 300)),
        "top1": float(retrieval[0]),
        "top2": float(retrieval[1]),
        "top3": float(retrieval[2]),
        "matching": float(matching),
        "multimodality": None,
        "multimodality_status": (
            "not_computed_main_only; optimization of one repeat cannot be "
            "reported as a 30-repeat multimodality estimate"
        ),
        "samples": int(len(lengths)),
        "requests": os.path.abspath(args.requests),
        "requests_sha256": request_sha256,
        "main_generations": os.path.abspath(args.main_generations),
        "main_generations_sha256": sha256_file(args.main_generations),
        "generation_request_hash_verified": True,
        "generation_length_policy": (
            "generated_lengths"
            if not np.array_equal(generated_lengths, lengths)
            else "requested_lengths"
        ),
        "main_generation_metadata": metadata,
        "evaluator_provenance": {
            "contract": "official_interx_full_672d",
            "options": artifact_record(EVALUATOR_OPTIONS),
            "checkpoint": artifact_record(EVALUATOR_CHECKPOINT),
            "wrapper_source": artifact_record(EVALUATOR_WRAPPER_SOURCE),
            "option_loader_source": artifact_record(EVALUATOR_OPTION_SOURCE),
            "scorer_source": artifact_record(__file__),
        },
        "evaluator_runtime": evaluator_runtime_record(args.device),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = os.path.abspath(args.output) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
