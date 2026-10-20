#!/usr/bin/env python3
"""Bind the strong historical HHC pipeline to its one-factor paper-mask run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_REFERENCE = ROOT / (
    "output/t2i_trans/2026-08-29-16-44-04_"
    "mgt_known_good_historical_contract_v7_trainmode_runtime_loss_sampler_"
    "seed114514/training_manifest.json"
)
DEFAULT_CANDIDATE = ROOT / (
    "output/t2i_trans/2026-09-02-11-49-08_"
    "hhc_historical_plus_paper_mask_ratio_raw12288_300k_seed114514_v2/"
    "training_manifest.json"
)
NUMERICAL_TRAINING_KEYS = (
    "batch_size", "total_iter", "warm_up_iter", "lr", "weight_decay",
    "lr_scheduler", "gamma", "nb_code", "code_dim", "shape_condition_dim",
    "width", "block_size", "embed_dim_gpt", "n_head_gpt", "num_layers",
    "num_local_layer", "ff_rate", "drop_out_rate", "contact",
    "contact_conditioning", "shape", "paper_architecture", "pkeep", "seed",
    "token_cache_contract", "training_mask_contract",
    "training_loss_contract", "optimizer_contract", "conditioning_contract",
    "training_token_length_contract", "training_validation_contract",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--historical-audit", type=Path,
        default=ROOT / "output/historical_mgt_training_contract_audit.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/hhc_historical_plus_paper_mask_contract.json",
    )
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    historical_audit = json.loads(
        args.historical_audit.read_text(encoding="utf-8")
    )
    ref = reference["resolved_arguments"]
    cand = candidate["resolved_arguments"]
    differences = {
        key: {"historical": ref.get(key), "candidate": cand.get(key)}
        for key in NUMERICAL_TRAINING_KEYS if ref.get(key) != cand.get(key)
    }
    expected_difference = {
        "training_mask_contract": {
            "historical": "historical_flat_random_replace",
            "candidate": "historical_pipeline_paper_kinematic_ratio",
        }
    }
    ref_cache = reference["cache_manifest"]
    cand_cache = candidate["cache_manifest"]
    immutable_cache_fields = (
        "vq_checkpoint_sha256", "code_cache_sha256", "contact_cache_sha256",
        "train_split_sha256", "val_split_sha256",
        "normalization_mean_sha256", "normalization_std_sha256",
    )
    cache_matches = {
        key: ref_cache.get(key) == cand_cache.get(key)
        for key in immutable_cache_fields
    }
    checkpoint_policy = candidate.get("checkpoint_policy", {})
    checks = {
        "historical_reproduction_audit_passes": historical_audit.get("status")
        == "pass",
        "only_numerical_training_change_is_mask_contract": differences
        == expected_difference,
        "same_vq_cache_splits_and_stats": all(cache_matches.values()),
        "historical_model_topology_retained": (
            cand["embed_dim_gpt"] == 1024 and cand["n_head_gpt"] == 16
            and cand["num_layers"] == 9 and cand["num_local_layer"] == 2
            and cand["paper_architecture"] is False
        ),
        "historical_loss_optimizer_conditioning_retained": (
            cand["training_loss_contract"] == "historical_all_valid"
            and cand["optimizer_contract"] == "historical_multistep"
            and cand["conditioning_contract"]
            == "historical_mmm_shape_dropout"
        ),
        "all_candidates_retained_without_test_filtering": (
            checkpoint_policy.get("retain_all_validation_checkpoints") is True
            and checkpoint_policy.get("test_metrics_used_for_selection") is False
            and checkpoint_policy.get(
                "historical_test_metric_controls_retention"
            ) is False
        ),
        "online_wandb_required": candidate["observability"]["wandb_mode"]
        == "online",
        "paper_mask_function_wired": (
            "sample_historical_mmm_corruption_with_kinematic_ratios"
            in (ROOT / "train/train_t2i_trans.py").read_text(encoding="utf-8")
        ),
    }

    # Execution-level check of the paper ordering: tau_root > tau_body >
    # tau_hand and therefore gamma_root < gamma_body < gamma_hand.
    from utils.motion_masks import sample_kinematic_cascade_mask
    valid = torch.ones(512, 38, 3, dtype=torch.bool)
    mask, ratios, taus = sample_kinematic_cascade_mask(
        valid, generator=torch.Generator().manual_seed(5325)
    )
    counts = mask.sum(dim=1)
    checks["paper_tau_and_mask_count_order_holds"] = bool(
        torch.all(taus[:, 2] >= taus[:, 0])
        and torch.all(taus[:, 0] >= taus[:, 1])
        and torch.all(ratios[:, 2] <= ratios[:, 0])
        and torch.all(ratios[:, 0] <= ratios[:, 1])
        and torch.all(counts[:, 2] <= counts[:, 0])
        and torch.all(counts[:, 0] <= counts[:, 1])
    )
    status = "pass" if all(checks.values()) else "fail"
    payload = {
        "schema": "hhc_historical_plus_paper_mask_contract_v1",
        "status": status,
        "run_status": "training_in_progress",
        "interpretation": (
            "Performance-aligned HHC retraining: retain the exactly reproduced "
            "historical pipeline and change only the paper's asymmetric "
            "kinematic mask-ratio rule. The earlier multi-change 512D run is "
            "diagnostic and must not be called the final paper implementation."
        ),
        "paper_claims": {
            "source": artifact(
                ROOT / "HHC__Generating_Human_Human_Close_Interactions_with_Contacts.pdf"
            ),
            "section": "3.1, Training, lines 252-272",
            "mask_schedule": "gamma(tau)=cos(pi*tau/2)",
            "ordering": "tau_root > tau_body > tau_hand",
            "resulting_visibility": "root is unmasked more frequently than body/hands",
            "objective_ambiguity": (
                "prose says masked tokens, while Eq.3 sums forall positions; "
                "the historical all-valid loss is retained for the requested "
                "single-factor performance alignment"
            ),
        },
        "checks": checks,
        "numerical_training_differences": differences,
        "cache_identity": cache_matches,
        "artifacts": {
            "historical_manifest": artifact(args.reference),
            "candidate_manifest": artifact(args.candidate),
            "historical_reproduction_audit": artifact(args.historical_audit),
            "mask_implementation": artifact(ROOT / "utils/motion_masks.py"),
            "training_implementation": artifact(ROOT / "train/train_t2i_trans.py"),
            "focused_test": artifact(ROOT / "tests/test_historical_paper_mask_ratio.py"),
            "launcher": artifact(ROOT / "tools/run_hhc_historical_plus_paper_mask_ratio.sh"),
        },
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({
        "status": status, "checks": checks,
        "numerical_training_differences": differences,
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
