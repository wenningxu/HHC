#!/usr/bin/env python3
"""Ensure shape-augmentation MGT arms inherit the strong primary HHC setup."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/run_hhc_shape_aug_historical_paper_mask_arm.sh"
HYBRID_HELPER = ROOT / "utils/hybrid_shape_token_cache.py"
OUTPUT = ROOT / "output/hhc_shape_aug_training_launch_contract.json"


EXPECTED_TRAINING_OPTIONS = {
    "--nb-code": "12288",
    "--code-dim": "256",
    "--shape-condition-dim": "8",
    "--width": "512",
    "--embed-dim-gpt": "1024",
    "--n-head-gpt": "16",
    "--num-layers": "9",
    "--num-local-layer": "2",
    "--batch-size": "64",
    "--total-iter": "300000",
    "--warm-up-iter": "1000",
    "--lr": "0.0002",
    "--weight-decay": "0.000001",
    "--lr-scheduler": "150000",
    "--gamma": "0.05",
    "--training-mask-contract": "historical_pipeline_paper_kinematic_ratio",
    "--training-loss-contract": "historical_all_valid",
    "--optimizer-contract": "historical_multistep",
    "--conditioning-contract": "historical_mmm_shape_dropout",
    "--training-token-length-contract": "historical_38",
    "--training-validation-contract": "historical_test_side_effect",
    "--pkeep": "0.5",
    "--sampler": "sample_intra_cas",
    "--temperature": "0.0",
    "--sampling-steps": "10",
    "--cfg-scale": "3.5",
    "--contact-guidance-scale": "1.0",
    "--wandb-mode": "online",
    "--seed": "114514",
}
EXPECTED_TRAINING_SWITCHES = {
    "--no-contact",
    "--no-contact-conditioning",
    "--shape",
    "--no-paper-architecture",
    "--save-all-validation-checkpoints",
}


def training_option_map(source: str) -> tuple[dict[str, object], list[str]]:
    """Parse only the model-training command, rejecting duplicate flags."""
    block_start = source.index('if [[ "${training_status}" != complete ]]')
    command_start = source.index("train/train_t2i_trans.py", block_start)
    block_end = source.index("\nfi", command_start)
    command = source[command_start:block_end]
    command = re.sub(r"#.*", "", command).replace("\\\n", " ")
    tokens = shlex.split(command)
    options: dict[str, object] = {}
    duplicates: list[str] = []
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if token in options:
            duplicates.append(token)
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            options[token] = tokens[index + 1]
            index += 2
        else:
            options[token] = True
            index += 1
    return options, sorted(set(duplicates))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    training_options, duplicate_training_options = training_option_map(source)
    archived_launcher = (
        "Archived shape-training launcher:" in source
        and re.search(r"(?m)^exit 2$", source) is not None
    )
    required = (
        "--embed-dim-gpt 1024 --n-head-gpt 16 --num-layers 9 --num-local-layer 2",
        "--batch-size 64 --total-iter 300000 --warm-up-iter 1000",
        "--lr 0.0002 --weight-decay 0.000001 --lr-scheduler 150000 --gamma 0.05",
        "--no-contact --no-contact-conditioning --shape --no-paper-architecture",
        "--training-mask-contract historical_pipeline_paper_kinematic_ratio",
        "--training-loss-contract historical_all_valid",
        "--optimizer-contract historical_multistep",
        "--conditioning-contract historical_mmm_shape_dropout",
        "--training-token-length-contract historical_38",
        "--training-validation-contract historical_test_side_effect",
        "--save-all-validation-checkpoints --pkeep 0.5",
        "--sampler sample_intra_cas --temperature 0.0 --sampling-steps 10",
        "--wandb-mode online --seed 114514",
        "--sampling-steps 50 --temperature 0.5 --cfg-scale 3.5",
        (
            "--token-cache-contract "
            "hybrid_historical_original_plus_deterministic_augmented_v1"
        ),
        "--historical-codebook-source /home/wenning/projects/MMM/checkpoints/vq/codebook",
        "--historical-contact-source /home/wenning/projects/MMM/checkpoints/vq/contact_token_label",
        "tools/audit_hybrid_shape_cache_preflight.py",
        '--reference-validation-checkpoint "${weights_only_reference}"',
        '--reference-validation-manifest "${weights_only_reference_manifest}"',
        "--free-space-after-queue-gib 40",
        "tools/audit_lightning_checkpoint.py",
        "tools/audit_hhc_sampling_runtime.py",
        "tools/audit_hhc_historical_model_checkpoint_structure.py",
        "--expected-step 300000 --expected-lr 0.00001",
        "tools/audit_wandb_online_metric_history.py",
        "--metric-profile masked_token_classification --expected-final-step 300000",
        "tools/promote_interx_validation_request_contract.py",
        '--final-checkpoint-audit "${final_checkpoint_audit}"',
        '--wandb-history-audit "${wandb_history_audit}"',
    )
    checks = {
        "launcher_is_executable_not_archived": not archived_launcher,
        "strong_primary_numerical_contract_is_reused": all(
            item in source for item in required
        ),
        "both_matched_augmentation_arms_supported": (
            '"${arm}" != random' in source and '"${arm}" != optimized' in source
        ),
        "shape_stage_requires_main_results": (
            "timotion_transformer_lpa_interx_retrained_fair/summary.json" in source
            and "coshmdm_interx_retrained_2daycap_seed123/summary.json" in source
        ),
        "shape_stage_requires_pre_shape_review_results": all(
            item in source for item in (
                "output/current_interx_main_table.json",
                "output/current_interx_optimization_backend_contract.json",
                "output/optimization_historical_paper_mask_rep0/formal_manifest.json",
                "output/current_interx_mesh_physical/manifest.json",
                "output/current_interx_qualitative/manifest.json",
                (
                    "output/fair_eval/"
                    "hhc_historical_paper_mask_contact_timing_ablation_rep0/"
                    "summary.json"
                ),
                (
                    "output/fair_eval/"
                    "hhc_historical_paper_mask_sampling_pareto_no_timing_rep0/"
                    "summary.json"
                ),
            )
        ),
        "shape_stage_requires_interact2ar_author_and_fair_joint_protocols": all(
            item in source for item in (
                "output/interact2ar_author_release_calibration/summary.json",
                "output/interact2ar_author_release_calibration/audit.json",
                "output/interact2ar_joint_scorer_parity_audit.json",
                "output/interact2ar_joint_evaluator_suite/pre_shape_summary.json",
            )
        ),
        "launcher_has_no_wait_loop": "while true" not in source and "sleep " not in source,
        "gpu_must_be_free": "used_mib > 1024" in source,
        "per_arm_disk_gate_precedes_cache_and_training": (
            "tools/audit_shape_augmentation_disk_capacity.py" in source
            and "--arms 1 --labels-per-arm 2" in source
            and "--validation-checkpoints-per-arm 30" in source
            and '--reference-validation-checkpoint "${weights_only_reference}"'
            in source
            and '--reference-validation-manifest "${weights_only_reference_manifest}"'
            in source
            and source.index("tools/audit_shape_augmentation_disk_capacity.py")
            < source.index("train/train_t2i_trans.py --cache-only")
        ),
        "validation_candidates_are_weights_only_but_recovery_is_full_state": (
            "save_weights_only=True" in (
                ROOT / "train/train_t2i_trans.py"
            ).read_text(encoding="utf-8")
            and "latest-000300000.ckpt" in source
        ),
        "cache_only_is_not_a_training_wandb_run": (
            "--cache-only" in source and "--seed 114514 --wandb-mode disabled" in source
        ),
        "model_training_is_online_wandb": "--wandb-mode online --seed 114514" in source,
        "training_command_has_no_duplicate_options": not duplicate_training_options,
        "training_command_exactly_matches_primary_numerical_options": all(
            training_options.get(key) == value
            for key, value in EXPECTED_TRAINING_OPTIONS.items()
        ) and all(
            training_options.get(key) is True
            for key in EXPECTED_TRAINING_SWITCHES
        ),
        "training_validation_rng_contract_matches_primary": (
            "--training-validation-contract historical_test_side_effect" in source
            and "--sampler sample_intra_cas --temperature 0.0 --sampling-steps 10"
            in source
        ),
        "original_samples_reuse_byte_exact_historical_cache": all(
            item in source for item in (
                (
                    "--token-cache-contract "
                    "hybrid_historical_original_plus_deterministic_augmented_v1"
                ),
                (
                    "--historical-codebook-source "
                    "/home/wenning/projects/MMM/checkpoints/vq/codebook"
                ),
                (
                    "--historical-contact-source "
                    "/home/wenning/projects/MMM/checkpoints/vq/contact_token_label"
                ),
                "tools/audit_hybrid_shape_cache_preflight.py",
            )
        ),
        "selection_is_official_validation_only": (
            "select_hhc_checkpoint_on_official_interx_val.py" in source
        ),
        "each_arm_has_independent_final_state_and_wandb_audits": (
            'final_checkpoint_audit="output/${experiment}_final_checkpoint_audit.json"'
            in source
            and 'wandb_history_audit="output/${experiment}_wandb_history_audit.json"'
            in source
            and 'model_structure_audit="output/${experiment}_model_structure_audit.json"'
            in source
        ),
        "final_test_uses_user_fixed_common50": (
            "--sampling-steps 50 --temperature 0.5 --cfg-scale 3.5" in source
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    payload = {
        "schema": "hhc_shape_aug_training_launch_contract_v1",
        "status": status,
        "checks": checks,
        "parsed_training_command": {
            "archived_launcher": archived_launcher,
            "duplicate_options": duplicate_training_options,
            "expected_numerical_options": EXPECTED_TRAINING_OPTIONS,
            "expected_switches": sorted(EXPECTED_TRAINING_SWITCHES),
        },
        "interpretation": (
            "Both augmentation arms must byte-exactly reuse the historical MMM "
            "token/contact cache for every original train+val sample and tokenize "
            "only newly augmented names deterministically. This prevents cache/crop "
            "drift from being attributed to shape augmentation. The in-training "
            "10-step diagnostic preserves the primary run's historical RNG path; "
            "official checkpoint selection and final generation use the user-fixed "
            "50-step protocol."
        ),
        "launcher": {
            "path": str(LAUNCHER), "bytes": LAUNCHER.stat().st_size,
            "sha256": sha256_file(LAUNCHER),
        },
        "hybrid_helper": {
            "path": str(HYBRID_HELPER), "bytes": HYBRID_HELPER.stat().st_size,
            "sha256": sha256_file(HYBRID_HELPER),
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "bytes": Path(__file__).resolve().stat().st_size,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
