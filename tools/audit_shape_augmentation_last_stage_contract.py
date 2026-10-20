#!/usr/bin/env python3
"""Audit that shape augmentation is last, fast-gated, and nearest-vertex."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/run_interx_shape_augmentation_nearest_last_stage.sh"
RUNNER = ROOT / "tools/run_interx_shape_augmentation.py"
RUNTIME_GATE = ROOT / "tools/require_shape_augmentation_runtime_gate.py"
DISK_GATE = ROOT / "tools/audit_shape_augmentation_disk_capacity.py"
WEIGHTS_ONLY_EXPORTER = ROOT / "tools/export_lightning_weights_only_reference.py"
PRE_SHAPE_REVIEW = ROOT / "tools/run_current_interx_review_experiments_before_shape.sh"
LEGACY_QUEUE = ROOT / "tools/queue_interx_shape_augmentation_gpu1.sh"
LEGACY_CHAIN = ROOT / "tools/queue_hhc_timing_after_post_gpu1.sh"
LEGACY_POST_SHAPE = ROOT / "tools/queue_post_shape_review_experiments_gpu1.sh"
OUTPUT = ROOT / "output/interx_shape_augmentation_last_stage_contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    legacy_queue = LEGACY_QUEUE.read_text(encoding="utf-8")
    legacy_chain = LEGACY_CHAIN.read_text(encoding="utf-8")
    legacy_post_shape = LEGACY_POST_SHAPE.read_text(encoding="utf-8")
    ast.parse(RUNTIME_GATE.read_text(encoding="utf-8"), filename=str(RUNTIME_GATE))
    prerequisites = (
        "output/external_baseline_reproduction_audit.json",
        "hhc_historical_plus_paper_mask_ratio_contact_common50/summary.json",
        "hhc_historical_plus_paper_mask_ratio_no_contact_common50/summary.json",
        "timotion_transformer_lpa_interx_retrained_fair/summary.json",
        "coshmdm_interx_retrained_2daycap_seed123/summary.json",
        "intermask_official_exact_common50/summary.json",
        "interact2ar_official_fair_no_gt_first_frame/summary.json",
        "output/interact2ar_author_release_calibration/audit.json",
        "output/interact2ar_joint_scorer_parity_audit.json",
        "output/interact2ar_joint_evaluator_suite/pre_shape_summary.json",
    )
    review_prerequisites = (
        "output/current_interx_main_table.json",
        "output/current_interx_optimization_backend_contract.json",
        "output/optimization_historical_paper_mask_rep0/formal_manifest.json",
        "output/current_interx_mesh_physical/manifest.json",
        "output/current_interx_qualitative/manifest.json",
        (
            "output/fair_eval/"
            "hhc_historical_paper_mask_contact_timing_ablation_rep0/summary.json"
        ),
        (
            "output/fair_eval/"
            "hhc_historical_paper_mask_sampling_pareto_no_timing_rep0/summary.json"
        ),
    )
    checks = {
        "launcher_has_no_wait_or_sleep_loop": (
            "while true" not in launcher and "sleep " not in launcher
        ),
        "legacy_waiter_entrypoints_fail_closed": (
            "Archived shape queue" in legacy_queue
            and legacy_queue.index("exit 2") < legacy_queue.index("while true")
            and "Archived dependency chain" in legacy_chain
            and legacy_chain.index("exit 2") < legacy_chain.index("while ps")
            and "Archived duplicate post-shape queue" in legacy_post_shape
            and legacy_post_shape.index("exit 2")
            < legacy_post_shape.index("tools/require_interx_volumetric_smpl_contract.py")
        ),
        "all_main_interx_results_are_prerequisites": all(
            value in launcher for value in prerequisites
        ),
        "all_pre_shape_review_results_are_prerequisites": all(
            value in launcher for value in review_prerequisites
        ),
        "gpu_must_be_free": "memory.used" in launcher and "used_mib > 1024" in launcher,
        "nearest_vertex_is_explicit": (
            "--signed-distance-backend nearest_vertex" in launcher
        ),
        "new_output_namespace_avoids_old_volumetric_outputs": (
            "interx_shape_aug_full_nearest_vertex_seed20260828" in launcher
            and "interx_shape_aug_full_random_seed20260828" in launcher
            and "full_batch8_query10475" not in launcher
        ),
        "matched_25_sample_pilot_precedes_full": (
            launcher.index("--positive-timing-only --limit 25")
            < launcher.index('full="output/interx_shape_aug_full_nearest_vertex')
        ),
        "matched_full_random_control_is_generated_and_materialized": (
            'full_random="output/interx_shape_aug_full_random_seed20260828"' in launcher
            and '--output-dir "${full_random}" --mode random_shape_only' in launcher
            and 'random_overlay="dataset/interx_shape_aug_random_seed123_v3"' in launcher
            and '--output-root "${random_overlay}"' in launcher
            and launcher.index('full_random="output/interx_shape_aug_full_random_seed20260828"')
            < launcher.index('full="output/interx_shape_aug_full_nearest_vertex')
        ),
        "runtime_gate_precedes_full": (
            launcher.index("tools/require_shape_augmentation_runtime_gate.py")
            < launcher.index('full="output/interx_shape_aug_full_nearest_vertex')
        ),
        "disk_capacity_gate_precedes_full": (
            launcher.index("tools/audit_shape_augmentation_disk_capacity.py")
            < launcher.index('full="output/interx_shape_aug_full_nearest_vertex')
            and "--validation-checkpoints-per-arm 30" in launcher
            and "--rolling-full-checkpoints-per-arm 2" in launcher
            and "tools/export_lightning_weights_only_reference.py" in launcher
            and '--reference-validation-checkpoint "${weights_only_reference}"'
            in launcher
            and '--reference-validation-manifest "${weights_only_reference_manifest}"'
            in launcher
            and "--free-space-after-queue-gib 40" in launcher
            and launcher.index("tools/export_lightning_weights_only_reference.py")
            < launcher.index("tools/audit_shape_augmentation_disk_capacity.py")
        ),
        "runtime_limit_is_one_day": "--maximum-projected-hours 24" in launcher,
        "ten_second_target_is_recorded": "--target-seconds-per-sample 10" in launcher,
        "physical_quality_gate_precedes_full": (
            launcher.index("tools/assess_shape_augmentation_pilot.py")
            < launcher.index('full="output/interx_shape_aug_full_nearest_vertex')
        ),
        "two_gpu_deterministic_sharding_and_coverage_audit": all(
            value in launcher for value in (
                'gpu1="${2:-}"',
                'first_count=$((sample_count / 2))',
                '--offset "${first_count}" --limit "${second_count}"',
                'tools/aggregate_interx_shape_augmentation_shards.py',
                '--expected-shards 2',
            )
        ),
        "matched_shape_training_arms_run_after_materialization": (
            launcher.index(
                'tools/run_hhc_shape_aug_historical_paper_mask_arm.sh "${gpu0}" random'
            ) > launcher.index('--output-root "${overlay}"')
            and launcher.index(
                'tools/run_hhc_shape_aug_historical_paper_mask_arm.sh "${gpu1}" optimized'
            ) > launcher.index('--output-root "${overlay}"')
        ),
        "post_shape_joint_evaluator_and_final_bundle_are_mandatory": (
            'run_interact2ar_joint_evaluator_suite.sh "${gpu0}" post_shape'
            in launcher
            and 'finalize_current_interx_shape_augmentation_results.sh'
            in launcher
        ),
        "runner_default_is_nearest_vertex": (
            "default='nearest_vertex'" in runner
        ),
        "runner_records_backend_in_contract": (
            "'signed_distance_backend': args.signed_distance_backend" in runner
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    payload = {
        "schema": "hhc_interx_shape_augmentation_last_stage_contract_v1",
        "status": status,
        "checks": checks,
        "policy": {
            "ordering": (
                "after HHC, TiMotion, CoShMDM, InterMask and Interact2AR main "
                "Inter-X results, paired main table, paper optimization, "
                "mesh-physical table, qualitative comparison and contact-"
                "timing ablation and no-timing MModality sampling analysis"
            ),
            "optimizer_backend": "nearest_vertex",
            "target_seconds_per_sample": 10,
            "target_full_serial_hours": round(6238 * 10 / 3600, 6),
            "maximum_full_serial_hours": 24,
            "disk_projection_per_training_arm": {
                "validation_candidate_checkpoints": 30,
                "full_recovery_checkpoints": 2,
                "validation_storage": "measured weights-only Lightning checkpoint",
                "recovery_storage": "full optimizer/scheduler training state",
                "asset_and_optimizer_reserve_gib": 40,
                "physical_output_reserve_gib": 30,
                "final_free_space_reserve_gib": 40,
                "policy": "hash-bound measured mixed checkpoint sizes; no deletion",
            },
            "eligible_samples": 6238,
            "old_volumetric_partial_outputs": "retained_diagnostic_only_not_resumed",
        },
        "artifacts": {
            "launcher": artifact(LAUNCHER),
            "runner": artifact(RUNNER),
            "runtime_gate": artifact(RUNTIME_GATE),
            "disk_capacity_gate": artifact(DISK_GATE),
            "weights_only_reference_exporter": artifact(WEIGHTS_ONLY_EXPORTER),
            "pre_shape_review_entrypoint": artifact(PRE_SHAPE_REVIEW),
            "archived_legacy_queue": artifact(LEGACY_QUEUE),
            "archived_legacy_chain": artifact(LEGACY_CHAIN),
            "archived_legacy_post_shape": artifact(LEGACY_POST_SHAPE),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
