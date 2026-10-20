#!/usr/bin/env python3
"""Decompose CoShMDM's official-evaluator gap in embedding space."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.evaluator_wrapper import EvaluatorModelWrapper  # noqa: E402
from options.get_eval_option import get_opt  # noqa: E402
from tools.score_fair_interx_generations import (  # noqa: E402
    EVALUATOR_CHECKPOINT,
    EVALUATOR_OPTIONS,
    EVALUATOR_OPTION_SOURCE,
    EVALUATOR_WRAPPER_SOURCE,
    artifact_record,
    embed_motions,
    evaluator_runtime_record,
    sha256_file,
    validate_generation_file,
)
from utils.eval_trans import (  # noqa: E402
    calculate_activation_statistics,
    calculate_frechet_distance,
)


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def embed_file(evaluator, path: Path, names, lengths, batch_size: int):
    with h5py.File(path, "r") as source:
        validate_generation_file(source, names, lengths, 1, path.name)
        return embed_motions(
            evaluator, source["motions"][:, 0], lengths, batch_size
        )


def distribution_summary(embeddings: np.ndarray) -> dict:
    mean, covariance = calculate_activation_statistics(embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "samples": int(len(embeddings)),
        "dimensions": int(embeddings.shape[1]),
        "finite": bool(np.isfinite(embeddings).all()),
        "embedding_norm_mean": float(np.mean(norms)),
        "embedding_norm_std": float(np.std(norms)),
        "centroid_l2_norm": float(np.linalg.norm(mean)),
        "covariance_trace": float(np.trace(covariance)),
    }


def fid_decomposition(left: np.ndarray, right: np.ndarray) -> dict:
    left_mean, left_covariance = calculate_activation_statistics(left)
    right_mean, right_covariance = calculate_activation_statistics(right)
    mean_term = float(np.sum(np.square(left_mean - right_mean)))
    total = float(calculate_frechet_distance(
        left_mean, left_covariance, right_mean, right_covariance
    ))
    covariance_term = total - mean_term
    return {
        "fid": total,
        "mean_term": mean_term,
        "covariance_term": covariance_term,
        "mean_term_fraction": mean_term / total if total else None,
        "centroid_l2_distance": float(np.sqrt(mean_term)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests", type=Path,
        default=ROOT / "output/fair_interx_requests/replication_00.npz",
    )
    parser.add_argument(
        "--model-generations", type=Path,
        default=(
            ROOT / "output/fair_eval/"
            "coshmdm_interx_retrained_2daycap_seed123/replication_00_main.h5"
        ),
    )
    parser.add_argument(
        "--representation-oracle", type=Path,
        default=ROOT / "output/coshmdm_representation_oracle/main.h5",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/coshmdm_embedding_gap_diagnosis.json",
    )
    args = parser.parse_args()

    request = np.load(args.requests, allow_pickle=False)
    names = request["names"].astype(str)
    lengths = request["lengths"].astype(np.int64)
    ground_truth = np.asarray(request["gt_embeddings"], dtype=np.float64)
    evaluator = EvaluatorModelWrapper(
        get_opt(str(EVALUATOR_OPTIONS), args.device)
    )
    model = embed_file(
        evaluator, args.model_generations, names, lengths, args.batch_size
    ).astype(np.float64)
    oracle = embed_file(
        evaluator, args.representation_oracle, names, lengths, args.batch_size
    ).astype(np.float64)
    if not all(np.isfinite(value).all() for value in (ground_truth, oracle, model)):
        raise FloatingPointError("non-finite official evaluator embeddings")

    gt_to_model = fid_decomposition(ground_truth, model)
    gt_to_oracle = fid_decomposition(ground_truth, oracle)
    oracle_to_model = fid_decomposition(oracle, model)
    payload = {
        "schema": "hhc_coshmdm_embedding_gap_diagnosis_v1",
        "status": "complete",
        "purpose": (
            "diagnostic-only decomposition in the same official Inter-X "
            "full-672D evaluator used by the fair table"
        ),
        "not_a_body_controlled_fid": True,
        "not_a_reported_comparison_row": True,
        "requests": artifact_record(args.requests),
        "model_generations": artifact_record(args.model_generations),
        "representation_oracle": artifact_record(args.representation_oracle),
        "distributions": {
            "official_ground_truth": distribution_summary(ground_truth),
            "representation_oracle": distribution_summary(oracle),
            "coshmdm_model": distribution_summary(model),
        },
        "gaps": {
            "official_ground_truth_to_coshmdm_model": gt_to_model,
            "official_ground_truth_to_representation_oracle": gt_to_oracle,
            "representation_oracle_to_coshmdm_model": oracle_to_model,
        },
        "derived": {
            "model_fid_minus_representation_oracle_fid": (
                gt_to_model["fid"] - gt_to_oracle["fid"]
            ),
            "representation_oracle_fraction_of_model_fid": (
                gt_to_oracle["fid"] / gt_to_model["fid"]
                if gt_to_model["fid"] else None
            ),
            "fid_is_not_additive_warning": True,
        },
        "interpretation": (
            "The GT-to-oracle gap measures information discarded by the "
            "CoShMDM representation/adapter. The oracle-to-model gap measures "
            "how far the learned generator is from that representation's own "
            "projected GT distribution. These FIDs are diagnostic and are not "
            "additive; the formal baseline remains GT-to-model."
        ),
        "evaluator_provenance": {
            "contract": "official_interx_full_672d",
            "options": artifact_record(EVALUATOR_OPTIONS),
            "checkpoint": artifact_record(EVALUATOR_CHECKPOINT),
            "wrapper_source": artifact_record(EVALUATOR_WRAPPER_SOURCE),
            "option_loader_source": artifact_record(EVALUATOR_OPTION_SOURCE),
        },
        "evaluator_runtime": evaluator_runtime_record(args.device),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
