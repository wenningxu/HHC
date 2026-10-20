#!/usr/bin/env python3
"""Build one fail-closed bundle for the completed current Inter-X experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "hhc_historical_plus_paper_mask_ratio_no_contact_common50",
    "hhc_historical_plus_paper_mask_ratio_contact_common50",
    "intermask_official_exact_common50",
    "interact2ar_official_fair_no_gt_first_frame",
    "coshmdm_interx_retrained_2daycap_seed123",
    "timotion_transformer_lpa_interx_retrained_fair",
)


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
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_embedded_artifacts(value, label="root") -> int:
    """Re-hash every nested path/SHA artifact record in one result payload."""
    verified = 0
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(
            value.get("sha256"), str
        ):
            path = Path(value["path"]).resolve()
            if not path.is_file() or sha256_file(path) != value["sha256"]:
                raise ValueError(f"{label}: embedded artifact is absent or stale: {path}")
            if "bytes" in value and path.stat().st_size != int(value["bytes"]):
                raise ValueError(f"{label}: embedded artifact byte count changed: {path}")
            verified += 1
        for key, child in value.items():
            verified += validate_embedded_artifacts(child, f"{label}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            verified += validate_embedded_artifacts(child, f"{label}/{index}")
    return verified


def load(path: Path, schema: str, status=None) -> tuple[dict, dict, int]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise ValueError(f"{path}: expected schema {schema}")
    if status is not None and payload.get("status") != status:
        raise ValueError(f"{path}: expected status {status}")
    verified = validate_embedded_artifacts(payload, path.name)
    return payload, artifact(path), verified


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_write(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def finite(value, label):
    require(isinstance(value, (int, float)) and math.isfinite(float(value)), label)
    return float(value)


def validate_multimodality_report(report, official_main, label):
    """Validate and compact one current method's exhaustive-MM report."""
    official = report.get("official_10_pairs", {})
    exhaustive = report.get("exhaustive_all_pairs", {})
    require(report.get("replications") == 20,
            f"{label}: exhaustive MM does not contain 20 replications")
    require(report.get("all_pairs_per_prompt") == 435,
            f"{label}: exhaustive MM is not 30 choose 2")
    require(report.get("legacy_provenance_allowed") is False,
            f"{label}: exhaustive MM permits legacy provenance")
    require(official.get("values") == official_main.get("values"),
            f"{label}: exhaustive MM official values are stale")
    require(len(exhaustive.get("values", [])) == 20,
            f"{label}: exhaustive MM values are incomplete")
    for metric_name, metric in (
        ("official_10_pairs", official),
        ("exhaustive_all_435_pairs", exhaustive),
    ):
        finite(metric.get("mean"), f"{label}/{metric_name}/mean")
        finite(metric.get("ci95"), f"{label}/{metric_name}/ci95")
    require(
        [row.get("replication") for row in report.get("per_replication", [])]
        == list(range(20)),
        f"{label}: exhaustive MM replication IDs are incomplete",
    )
    return {
        "official_10_pairs": official,
        "exhaustive_all_435_pairs": exhaustive,
        "official_minus_all": report.get("official_minus_all"),
        "root_mean_square_pair_subsampling_error": finite(
            report.get("root_mean_square_pair_subsampling_error"),
            f"{label}/MM pair-subsampling RMSE",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("output/current_interx_review_bundle.json"))
    parser.add_argument("--markdown", type=Path,
                        default=Path("output/current_interx_review_bundle.md"))
    args = parser.parse_args()
    specs = {
        "main_table": (
            Path("output/current_interx_main_table.json"),
            "hhc_current_interx_main_table_v1", "complete",
        ),
        "shape_optimization_pilot": (
            Path(
                "output/"
                "interx_shape_aug_original_mmm_pilot25_seed20260828_gate.json"
            ),
            "hhc_original_mmm_shape_pilot_gate_v1",
            "diagnostic_not_production",
        ),
        "mesh_physical": (
            Path("output/current_interx_mesh_physical/manifest.json"),
            "hhc_final_interx_mesh_physical_main_table_v1", "complete",
        ),
        "mesh_physical_paired": (
            Path(
                "output/current_interx_mesh_physical/"
                "paired_vs_hhc_no_timing.json"
            ),
            "hhc_current_interx_mesh_paired_methods_v1", "complete",
        ),
        "hhc_oracle_physical_paired": (
            Path(
                "output/current_interx_mesh_physical/"
                "hhc_oracle_contact_vs_no_contact_paired.json"
            ),
            "hhc_paired_physical_comparison_v1", None,
        ),
        "qualitative": (
            Path("output/current_interx_qualitative/manifest.json"),
            "hhc_fair_interx_qualitative_render_v1", "complete",
        ),
        "contact_timing": (
            Path("output/fair_eval/hhc_historical_paper_mask_contact_timing_ablation_rep0/summary.json"),
            "hhc_contact_timing_ablation_summary_v1", None,
        ),
        "sampling_grid": (
            Path("output/fair_eval/hhc_historical_paper_mask_sampling_pareto_no_timing_rep0/grid_manifest.json"),
            "hhc_current_sampling_pareto_grid_v1", "complete",
        ),
        "sampling_pareto": (
            Path("output/fair_eval/hhc_historical_paper_mask_sampling_pareto_no_timing_rep0/summary.json"),
            "hhc_sampling_pareto_summary_v1", None,
        ),
        "shape_augmentation": (
            Path("output/current_interx_shape_augmentation_results.json"),
            "hhc_current_shape_augmentation_results_v1", "complete",
        ),
        "review_coverage": (
            Path("output/review_requirement_coverage.json"),
            "hhc_review_requirement_coverage_audit_v1", "pass",
        ),
        "hhc_selection": (
            Path("output/hhc_historical_plus_paper_mask_ratio_selection.json"),
            "hhc_official_interx_validation_selection_v1", "complete",
        ),
        "timotion_selection": (
            Path("output/timotion_interx_val_selection.json"),
            "hhc_timotion_interx_validation_selection_v1", "complete",
        ),
        "coshmdm_training_diagnosis": (
            Path("output/coshmdm_training_trajectory_diagnosis.json"),
            "hhc_coshmdm_training_trajectory_diagnosis_v1", "complete",
        ),
        "coshmdm_embedding_diagnosis": (
            Path("output/coshmdm_embedding_gap_diagnosis.json"),
            "hhc_coshmdm_embedding_gap_diagnosis_v1", "complete",
        ),
        "coshmdm_adapter_audit": (
            Path("output/coshmdm_representation_adapter_audit_live.json"),
            "hhc_coshmdm_representation_adapter_audit_v1", "pass",
        ),
        "interact2ar_paper_repo_data": (
            Path("output/interact2ar_paper_repo_data_audit.json"),
            "interact2ar_paper_repo_data_audit_v1", None,
        ),
        "interact2ar_author_calibration": (
            Path("output/interact2ar_author_release_calibration/audit.json"),
            "interact2ar_author_release_calibration_audit_v1", "pass",
        ),
        "interact2ar_scorer_parity": (
            Path("output/interact2ar_joint_scorer_parity_audit.json"),
            "interact2ar_joint_scorer_parity_audit_v1", "pass",
        ),
        "interact2ar_joint_evaluators": (
            Path("output/interact2ar_joint_evaluator_suite/summary.json"),
            "interact2ar_joint_evaluator_suite_summary_v1", "complete",
        ),
        "interact2ar_joint_pre_shape": (
            Path("output/interact2ar_joint_evaluator_suite/pre_shape_summary.json"),
            "interact2ar_joint_evaluator_suite_summary_v1", "complete",
        ),
        "intermask_native_protocol": (
            Path(
                "output/fair_eval/intermask_official_exact/"
                "native20_hdf5_protocol_audit.json"
            ),
            "intermask_existing_native20_hdf5_protocol_audit_v1", "pass",
        ),
        "coshmdm_training_contract": (
            Path("output/coshmdm_training_contract_audit_live.json"),
            "hhc_coshmdm_interx_training_contract_v1", "pass",
        ),
        "coshmdm_public_release_scope": (
            Path("output/coshmdm_public_release_scope_audit.json"),
            "hhc_coshmdm_public_release_scope_audit_v1", "pass",
        ),
        "timotion_training_contract": (
            Path("output/timotion_interx_training_contract_audit.json"),
            "hhc_timotion_interx_training_contract_v1", "pass",
        ),
        "timotion_generation_contract": (
            Path("output/timotion_generation_contract_audit_standard_formal.json"),
            "hhc_timotion_generation_contract_audit_v1", "pass",
        ),
        "timotion_wandb": (
            Path("output/timotion_interx_formal_wandb_history_audit.json"),
            "hhc_wandb_online_metric_history_audit_v1", "pass",
        ),
        "external_baseline_reproduction": (
            Path("output/external_baseline_reproduction_audit.json"),
            "hhc_external_baseline_reproduction_audit_v1", "pass",
        ),
    }
    payloads = {}
    artifacts = {}
    verified_artifacts = {}
    for name, (relative, schema, status) in specs.items():
        payloads[name], artifacts[name], verified_artifacts[name] = load(
            ROOT / relative, schema, status
        )
    cosh_run_path = ROOT / (
        "output/fair_eval/coshmdm_interx_retrained_2daycap_seed123/"
        "run_manifest.json"
    )
    cosh_run = json.loads(cosh_run_path.read_text(encoding="utf-8"))
    require(
        cosh_run.get("method") == "coshmdm"
        and cosh_run.get("implementation_source_stability_verified_at_completion")
        is True,
        "formal CoShMDM run did not freeze and verify its execution sources",
    )
    artifacts["coshmdm_run_manifest"] = artifact(cosh_run_path)
    verified_artifacts["coshmdm_run_manifest"] = validate_embedded_artifacts(
        cosh_run.get("provenance", {}), "coshmdm_run_manifest/provenance"
    )
    cosh_generation_path = ROOT / "output/coshmdm_generation_contract_audit.json"
    cosh_generation = json.loads(cosh_generation_path.read_text(encoding="utf-8"))
    require(
        cosh_generation.get("schema")
        == "hhc_coshmdm_generation_contract_audit_v1"
        and cosh_generation.get("status") == "pass"
        and cosh_run.get("provenance", {}).get("artifacts", {}).get(
            "generation_contract_audit", {}
        ).get("sha256") == sha256_file(cosh_generation_path),
        "CoShMDM generation audit is not the frozen formal-run artifact",
    )
    artifacts["coshmdm_generation_audit"] = artifact(cosh_generation_path)

    main = payloads["main_table"]
    require(main.get("protocol", {}).get("dataset") == "Inter-X", "wrong dataset")
    require(main["protocol"].get("replications") == 20, "main table is not 20 rep")
    require(main["protocol"].get("samples_per_replication") == 1696,
            "main table sample count changed")
    require(set(main.get("methods", {})) == set(METHODS), "main methods changed")
    multimodality = {}
    multimodality_artifacts = {}
    for label in METHODS:
        path = ROOT / "output/fair_eval" / label / "multimodality_audit_summary.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        official_main = main["methods"][label]["metrics"]["multimodality"]
        multimodality[label] = validate_multimodality_report(
            report, official_main, label
        )
        multimodality_artifacts[label] = artifact(path)

    shape_optimization = payloads["shape_optimization_pilot"]
    shape_aggregate = shape_optimization.get("aggregate", {})
    require(
        shape_optimization.get("samples") == 25
        and shape_optimization.get("gate_pass") is True
        and shape_optimization.get("accepted", 0) > 0
        and shape_optimization.get("contract", {}).get("optimizer_and_loss")
        == "unmodified original MMM"
        and shape_aggregate.get("timing_mismatch_selected_delta", 1.0) < 0
        and abs(shape_aggregate.get("skate_selected_delta", 1.0)) <= 1e-6,
        "train-only shape optimization pilot did not improve timed mismatch "
        "while preserving foot skating",
    )
    require(
        shape_aggregate.get("runtime", {}).get(
            "projected_full_5868_optimizer_hours_single_gpu", 25.0
        ) <= 24.0,
        "shape optimization pilot projects beyond the one-day single-GPU budget",
    )
    expected_original_loss_sha = (
        "29f67e9830c0a0c9dbfffc8361ce4ae91dc76fef691cca17cff39613d7151f7c"
    )
    require(
        shape_optimization.get("artifacts", {}).get(
            "original_mmm_loss", {}
        ).get("sha256") == expected_original_loss_sha,
        "shape pilot did not bind the restored original-MMM loss",
    )

    physical = payloads["mesh_physical"]
    require(tuple(physical.get("method_order", [])) == METHODS,
            "mesh physical method order changed")
    require(physical.get("samples_per_method") == 1696,
            "mesh physical sample count changed")
    physical_paired = payloads["mesh_physical_paired"]
    require(
        physical_paired.get("reference") == METHODS[0]
        and physical_paired.get("comparators") == list(METHODS[1:])
        and physical_paired.get("sample_count") == 1696
        and physical_paired.get("bootstrap_samples") == 10000
        and sum(
            len(rows)
            for rows in physical_paired.get("comparisons", {}).values()
        ) == 35,
        "cross-method paired mesh statistics are incomplete",
    )
    hhc_oracle_physical = payloads["hhc_oracle_physical_paired"]
    require(
        hhc_oracle_physical.get("sample_count") == 1696
        and hhc_oracle_physical.get("bootstrap_samples") == 10000
        and set(hhc_oracle_physical.get("comparisons", {}))
        == set(physical.get("metric_keys", [])),
        "HHC oracle/no-contact paired physical comparison is incomplete",
    )

    qualitative = payloads["qualitative"]
    require(tuple(qualitative.get("methods", [])) == METHODS,
            "qualitative methods changed")
    require(qualitative.get("same_request_repeat_camera_contract_verified") is True
            and qualitative.get("strict_request_hashes_verified") is True
            and len(qualitative.get("figures", [])) == 9,
            "qualitative comparison is incomplete")

    timing = payloads["contact_timing"]
    require(set(timing.get("rows", {})) == {
        "request_oracle", "none", "text_predicted", "full_valid_weak"
    }, "contact timing rows changed")
    for name, row in timing["rows"].items():
        for metric in ("fid", "top1", "matching", "contact_f1_10cm"):
            finite(row.get(metric), f"contact timing {name}/{metric} is non-finite")

    grid = payloads["sampling_grid"]
    require(grid.get("mgt_contract") == "historical_mmm"
            and grid.get("inference_conditioning_contract") == "paper_text_cfg"
            and grid.get("contact_timing_source") == "none"
            and grid.get("contact_guidance") is False,
            "sampling grid is not the current no-timing HHC")
    require(len(grid.get("outputs", {})) == 8, "sampling grid is incomplete")
    pareto = payloads["sampling_pareto"]
    require(len(pareto.get("rows", [])) == 8, "sampling Pareto table is incomplete")
    require(pareto.get("inputs", {}).get("manifest")
            == artifacts["sampling_grid"]["sha256"],
            "sampling Pareto summary is stale")

    shape = payloads["shape_augmentation"]
    require(set(shape.get("conditions", {})) == {"no_contact", "oracle_contact"},
            "shape timing conditions changed")
    for condition, rows in shape["conditions"].items():
        require(set(rows) == {"baseline", "random_shape_only", "optimized_shape"},
                f"shape arms changed for {condition}")

    coverage = payloads["review_coverage"]
    require(coverage.get("coverage_complete") is True,
            "review concern coverage is incomplete")
    hhc_selection = payloads["hhc_selection"]
    require(hhc_selection.get("test_metrics_used_for_selection") is False,
            "HHC checkpoint selection used test metrics")
    timotion_selection = payloads["timotion_selection"]
    require(timotion_selection.get("test_metrics_used_for_selection") is False
            and timotion_selection.get("validation_contract")
            == "standard_row_concatenated_v1",
            "TiMotion selection is not standard-6D held-out validation")
    cosh_training = payloads["coshmdm_training_diagnosis"]
    cosh_embedding = payloads["coshmdm_embedding_diagnosis"]
    require(
        cosh_training.get("checks", {}).get("all_logged_losses_finite") is True
        and cosh_training.get("checks", {}).get(
            "validation_fid_worsens_after_selected_checkpoint"
        ) is True,
        "CoShMDM training diagnosis does not establish the selected trajectory",
    )
    require(
        cosh_embedding.get("not_a_body_controlled_fid") is True
        and cosh_embedding.get("not_a_reported_comparison_row") is True,
        "CoShMDM representation diagnostic reporting boundary changed",
    )

    interact2ar_data = payloads["interact2ar_paper_repo_data"]
    require(
        str(interact2ar_data.get("status", "")).startswith("pass")
        and interact2ar_data.get("paper_contract", {}).get("replications") == 20,
        "Interact2AR paper/repository/data audit is not passing",
    )
    interact2ar_joint = payloads["interact2ar_joint_evaluators"]
    require(
        interact2ar_joint.get("scope")
        == "supplementary_same_unconditional_frozen_generations"
        and interact2ar_joint.get("not_main_interx_metric_space") is True
        and interact2ar_joint.get("replications") == 20
        and interact2ar_joint.get("samples_per_replication") == 1696,
        "Interact2AR joint-evaluator scope was mixed with the main metric space",
    )
    pre_shape_joint = payloads["interact2ar_joint_pre_shape"]
    pre_shape_methods = {"hhc", "intermask", "interact2ar", "coshmdm", "timotion"}
    require(
        set(pre_shape_joint.get("methods", {})) == pre_shape_methods
        and pre_shape_joint.get("replications") == 20
        and pre_shape_joint.get("not_main_interx_metric_space") is True,
        "pre-shape Interact2AR joint-evaluator gate is incomplete",
    )
    joint_methods = (
        "hhc", "intermask", "interact2ar", "coshmdm", "timotion",
        "hhc_shape_random", "hhc_shape_optimized",
    )
    joint_wrappers = ("JointsFull", "JointsBody", "JointsHands")
    require(
        set(interact2ar_joint.get("methods", {})) == set(joint_methods),
        "Interact2AR joint-evaluator method scope changed",
    )
    for method in joint_methods:
        require(
            set(interact2ar_joint["methods"][method]) == set(joint_wrappers),
            f"Interact2AR joint evaluators are incomplete for {method}",
        )
        for wrapper in joint_wrappers:
            for metric in ("fid", "matching", "top1", "top2", "top3",
                           "diversity", "multimodality"):
                record = interact2ar_joint["methods"][method][wrapper][metric]
                finite(record.get("mean"), f"{method}/{wrapper}/{metric}/mean")
                finite(record.get("ci95"), f"{method}/{wrapper}/{metric}/ci95")

    intermask_native = payloads["intermask_native_protocol"]
    external_baselines = payloads["external_baseline_reproduction"]
    require(
        all(external_baselines.get("checks", {}).values())
        and set(external_baselines.get("methods", {}))
        == {"intermask", "coshmdm", "timotion", "interact2ar"},
        "external baseline paper/repository/data/protocol aggregate audit failed",
    )
    require(
        intermask_native.get("classification")
        == "author-native-20-steps_exact-length_main20_mm20-extension_not-common50"
        and intermask_native.get("replications") == 20
        and intermask_native.get("samples_per_replication") == 1696
        and intermask_native.get("sampler_protocol", {}).get("time_steps") == 20,
        "InterMask author-native protocol calibration is incomplete",
    )
    cosh_contract = payloads["coshmdm_training_contract"]
    cosh_release_scope = payloads["coshmdm_public_release_scope"]
    require(
        cosh_contract.get("released_checkpoint_compatibility", {}).get(
            "documented_interx_checkpoint_available"
        ) is False
        and cosh_contract.get("dataset", {}).get("split_counts")
        == {"train": 9110, "val": 570, "test": 1708},
        "CoShMDM public-code port scope/data contract changed",
    )
    require(
        cosh_release_scope.get("classification")
        == "public_repository_runnable_diffusion_port_not_paper_architecture_reproduction"
        and cosh_release_scope.get("released_runnable_pipeline", {}).get(
            "shape_conditioning"
        ) is False
        and cosh_release_scope.get("released_runnable_pipeline", {}).get(
            "vq_latent_pipeline"
        ) is False
        and cosh_release_scope.get("released_runnable_pipeline", {}).get(
            "rl_sdf_policy"
        ) is False,
        "CoShMDM public repository was mislabeled as the paper architecture",
    )
    timotion_contract = payloads["timotion_training_contract"]
    timotion_generation = payloads["timotion_generation_contract"]
    timotion_wandb = payloads["timotion_wandb"]
    require(
        timotion_contract.get("reported_method")
        == "public TiMotion Transformer+LPA (not unreleased RWKV/Mamba)"
        and timotion_contract.get("dataset", {}).get("splits", {}).get(
            "train", {}
        ).get("sha256") == "47c5c9a09d7293e90e127fb49ba9dc9398afbff66c8f2b45ada3ef6c4453157a"
        and all(timotion_generation.get("checks", {}).values())
        and timotion_wandb.get("remote_state") == "finished"
        and all(timotion_wandb.get("checks", {}).values()),
        "TiMotion public Transformer+LPA reproduction contract is incomplete",
    )

    result = {
        "schema": "hhc_current_interx_review_bundle_v1",
        "status": "complete",
        "scope": "current Inter-X machine experiments",
        "protocol": main["protocol"],
        "main_methods": main["methods"],
        "shape_optimization_pilot": {
            "contract": shape_optimization["contract"],
            "samples": shape_optimization["samples"],
            "accepted": shape_optimization["accepted"],
            "aggregate": shape_optimization["aggregate"],
        },
        "mesh_physical": {
            "statistical_scope": physical["statistical_scope"],
            "methods": physical["methods"],
        },
        "hhc_oracle_contact_physical_effect": {
            "sample_count": hhc_oracle_physical["sample_count"],
            "bootstrap_samples": hhc_oracle_physical["bootstrap_samples"],
            "comparisons": hhc_oracle_physical["comparisons"],
        },
        "contact_timing": timing["rows"],
        "sampling_pareto": pareto["rows"],
        "multimodality": multimodality,
        "coshmdm_diagnosis": {
            "training": {
                "diagnosis": cosh_training["diagnosis"],
                "aggregate": cosh_training["aggregate"],
                "checks": cosh_training["checks"],
            },
            "representation": {
                "gaps": cosh_embedding["gaps"],
                "interpretation": cosh_embedding["interpretation"],
                "not_a_body_controlled_fid": True,
                "not_a_reported_comparison_row": True,
            },
            "adapter_contract": payloads["coshmdm_adapter_audit"][
                "adapter_contract"
            ],
            "generation_contract_checks": cosh_generation["checks"],
        },
        "interact2ar_joint_evaluator_supplement": {
            "scope": interact2ar_joint["scope"],
            "not_main_interx_metric_space": True,
            "methods": interact2ar_joint["methods"],
            "evaluator_provenance": interact2ar_joint["evaluator_provenance"],
        },
        "external_baseline_reproduction_scope": {
            "intermask": {
                "repository_protocol_calibration": intermask_native["classification"],
                "native_replications": 20,
                "main_table_protocol": "user-fixed common50 on frozen requests",
            },
            "coshmdm": {
                "scope": (
                    "public-repository runnable 262D diffusion Inter-X port "
                    "under user two-day cap; not the paper-described "
                    "VQ/shape/RL-SDF architecture"
                ),
                "native_interx_checkpoint_available": False,
                "repository_revision": cosh_contract["repository"]["revision"],
                "release_scope_classification": cosh_release_scope[
                    "classification"
                ],
            },
            "timotion": {
                "scope": timotion_contract["reported_method"],
                "paper_best_rwkv_claimed": False,
                "online_wandb_complete": True,
            },
            "interact2ar": {
                "repository_revision": interact2ar_data["repository"]["revision"],
                "author_gt_first_frame_separate": True,
                "fair_unconditional_and_joint_protocols_separate": True,
            },
            "aggregate_audit_policy": external_baselines["policy"],
        },
        "shape_augmentation": shape["conditions"],
        "qualitative_figures": qualitative["figures"],
        "disclosed_nonmachine_or_deferred_items": sorted(set(
            coverage.get("deliberately_out_of_current_scope", [])
        ) | {"figure_2_typo"}),
        "embedded_artifacts_rehashed": verified_artifacts,
        "artifacts": {
            **artifacts,
            "multimodality": multimodality_artifacts,
            "coshmdm_reproduction_diagnosis": artifact(
                ROOT / "output/coshmdm_interx_reproduction_diagnosis.md"
            ),
            "baseline_availability_audit": artifact(
                ROOT / "docs/baseline_availability_audit.md"
            ),
        },
        "implementation": artifact(Path(__file__)),
    }
    atomic_write(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")

    rows = []
    for label in METHODS:
        record = main["methods"][label]
        metric = record["metrics"]
        rows.append(
            f"| {record['display_name']} | {metric['fid']['mean']:.4f} ± "
            f"{metric['fid']['ci95']:.4f} | {metric['top1']['mean']:.4f} ± "
            f"{metric['top1']['ci95']:.4f} | {metric['matching']['mean']:.4f} ± "
            f"{metric['matching']['ci95']:.4f} |\n"
        )
    joint_rows = []
    for wrapper in joint_wrappers:
        for method in joint_methods:
            metric = interact2ar_joint["methods"][method][wrapper]
            joint_rows.append(
                f"| {wrapper} | {method} | {metric['fid']['mean']:.4f} ± "
                f"{metric['fid']['ci95']:.4f} | {metric['top1']['mean']:.4f} ± "
                f"{metric['top1']['ci95']:.4f} | {metric['matching']['mean']:.4f} ± "
                f"{metric['matching']['ci95']:.4f} |\n"
            )
    markdown = (
        "# Current Inter-X review bundle\n\n"
        "All semantic rows use 20 frozen paired replications × 1,696 samples. "
        "Optimization/mesh/qualitative sensitivity results are replication-0 "
        "unless their source artifact states otherwise.\n\n"
        "| Method | FID ↓ | R@1 ↑ | Matching ↓ |\n"
        "|---|---:|---:|---:|\n" + "".join(rows) + "\n"
        "External-method scope is enforced by machine-readable audits: InterMask "
        "has an independent author-native 20-step calibration; CoShMDM is a public-"
        "code Inter-X port under the two-day cap; TiMotion is the public Transformer+"
        "LPA port and is not labeled as the unreleased RWKV result; Interact2AR's "
        "GT-first-frame author protocol remains separate.\n\n"
        "## Interact2AR released joint-evaluator supplement\n\n"
        "This separate table re-scores the same unconditional frozen generations; "
        "its values are not comparable to the legacy Inter-X metric columns above.\n\n"
        "| Evaluator | Method | FID ↓ | R@1 ↑ | Matching ↓ |\n"
        "|---|---|---:|---:|---:|\n" + "".join(joint_rows) + "\n"
        "The machine-readable bundle binds current HHC/TiMotion validation "
        "selection, nearest-vertex optimization, physical metrics, contact-timing, "
        "MModality, qualitative figures and matched shape augmentation.\n"
    )
    atomic_write(args.markdown, markdown)
    print(json.dumps({
        "status": "complete", "output": str(args.output.resolve()),
        "markdown": str(args.markdown.resolve()),
        "embedded_artifacts_rehashed": sum(verified_artifacts.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
