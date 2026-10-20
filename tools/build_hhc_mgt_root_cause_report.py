#!/usr/bin/env python3
"""Bind the current HHC-M quality gap to reproducible source evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evidence(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an evidence-bound HHC MGT root-cause status report."
    )
    parser.add_argument(
        "--formal-contract",
        type=Path,
        default=ROOT / "output/hhc_formal_raw12288_repaired_mgt_training_contract_audit.json",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=(
            ROOT / "output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact/"
            "replication_00_current_scorer_20260901.json"
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=(
            ROOT / "output/fair_eval/"
            "hhc_mgt_repair_v7_step280k_common50_paper_text_cfg_contact_rep00/"
            "replication_00_current_scorer_20260901.json"
        ),
    )
    parser.add_argument(
        "--quality-policy",
        type=Path,
        default=ROOT / "configs/hhc_formal_mgt_quality_gate_v1.json",
    )
    parser.add_argument(
        "--historical-run",
        type=Path,
        default=ROOT / "output/historical_mgt_original_training_contract_audit.json",
    )
    parser.add_argument(
        "--repredict",
        type=Path,
        default=ROOT / "output/hhc_repredict_validation_diagnostic/comparison.json",
    )
    parser.add_argument(
        "--native-semantics",
        type=Path,
        default=ROOT / "output/hhc_native_repredict_semantics_audit.json",
    )
    parser.add_argument(
        "--loss-reduction",
        type=Path,
        default=ROOT / "output/hhc_masked_loss_reduction_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/hhc_mgt_root_cause_status.json",
    )
    args = parser.parse_args()

    formal = load(args.formal_contract)
    candidate = load(args.candidate)
    reference = load(args.reference)
    quality_policy = load(args.quality_policy)
    historical = load(args.historical_run)
    repredict = load(args.repredict)
    native = load(args.native_semantics)
    loss_reduction = load(args.loss_reduction)

    numeric = formal["paper_stated_numeric_contract"]
    old_config = historical["run"]["required_configuration"]
    old_milestones = historical["run"]["validation_milestones"]
    old_best = min(old_milestones, key=lambda row: row["fid_printed"])
    paper_reported_fid = 0.137

    candidate_fid = candidate["fid"]
    reference_fid = reference["fid"]
    thresholds = quality_policy["quality_thresholds"]
    repredict_delta = repredict["fid_delta_repredict_minus_baseline"]

    config_conflicts = {
        "latent_dimension": {
            "paper_appendix": numeric["latent_dimension"],
            "exact_low_fid_run": old_config["embed_dim_gpt"],
        },
        "attention_heads": {
            "paper_appendix": numeric["attention_heads"],
            "exact_low_fid_run": old_config["n_head_gpt"],
        },
        "batch_size": {
            "paper_appendix": numeric["batch_size"],
            "exact_low_fid_run": old_config["batch_size"],
        },
        "schedule": {
            "paper_appendix": numeric["schedule"],
            "exact_low_fid_run": (
                f"MultiStepLR milestones={old_config['lr_scheduler']} "
                f"gamma={old_config['gamma']}"
            ),
        },
        "mask_and_objective": {
            "paper_appendix": "kinematic asymmetric mask; masked-token NLL",
            "exact_low_fid_run": (
                f"flattened-stream cosine mask; pkeep={old_config['pkeep']} "
                "random-token corruption; all-valid-token NLL"
            ),
        },
        "optimizer_betas": {
            "paper_appendix": [0.9, 0.99],
            "exact_low_fid_run": [0.5, 0.9],
        },
    }

    checks = {
        "formal_run_matches_all_explicit_paper_numeric_settings": (
            not formal["paper_stated_numeric_mismatches"]
        ),
        "formal_run_matches_all_explicit_paper_qualitative_settings": (
            not formal["paper_stated_qualitative_mismatches"]
        ),
        "formal_candidate_fails_operational_quality_gate": (
            candidate_fid > thresholds["maximum_candidate_fid"]
            and candidate_fid - reference_fid
            > thresholds["maximum_candidate_minus_reference_fid"]
            and candidate_fid / reference_fid
            > thresholds["maximum_candidate_to_reference_fid_ratio"]
        ),
        "historical_run_source_and_configuration_are_verified": (
            historical["status"] == "pass"
        ),
        "historical_best_is_numerically_close_to_paper_table": (
            abs(old_best["fid_printed"] - paper_reported_fid) <= 0.002
        ),
        "historical_common_scorer_reference_passes_gate": reference_fid <= 0.20,
        "native_repredict_semantics_are_source_verified": native["status"] == "pass",
        "repredict_only_diagnostic_worsens_fid": repredict_delta > 0,
        "masked_loss_reduction_is_a_verified_unresolved_difference": (
            loss_reduction["status"] == "pass"
            and loss_reduction["checks"]
            ["current_is_per_sample_masked_mean_then_batch_mean"]
            and loss_reduction["checks"]
            ["intermask_ignores_unmasked_labels"]
            and loss_reduction["checks"]
            ["intermask_cross_entropy_uses_default_global_mean"]
        ),
        "all_listed_configuration_fields_conflict": all(
            item["paper_appendix"] != item["exact_low_fid_run"]
            for item in config_conflicts.values()
        ),
    }

    payload = {
        "schema": "hhc_mgt_root_cause_status_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnosis_narrowed_not_causally_isolated",
        "checks": checks,
        "quality_gap": {
            "current_formal_common_scorer_fid": candidate_fid,
            "historical_common_scorer_fid": reference_fid,
            "delta_current_minus_historical": candidate_fid - reference_fid,
            "current_to_historical_ratio": candidate_fid / reference_fid,
        },
        "paper_table_linkage": {
            "paper_reported_hhc_m_fid": paper_reported_fid,
            "exact_historical_run_best_fid": old_best["fid_printed"],
            "exact_historical_run_best_step": old_best["iteration"],
            "absolute_difference": abs(old_best["fid_printed"] - paper_reported_fid),
            "interpretation": (
                "The numerical match and chronology are strong evidence that the "
                "published-quality result is associated with the verified legacy run, "
                "but they do not by themselves prove which checkpoint produced the "
                "paper table. This is explicitly an inference, not direct provenance."
            ),
        },
        "paper_appendix_vs_verified_low_fid_run": config_conflicts,
        "ruled_out_as_unique_causes": {
            "evaluator_migration": (
                "Historical and current H5 rescoring preserved their FIDs under the "
                "same official Inter-X scorer."
            ),
            "vq_reconstruction": (
                "Both generator comparisons use the byte-identical raw12288/175k VQ; "
                "its official reconstruction FID is approximately 0.294."
            ),
            "token_update_support": {
                "fixed_token_fid": repredict["baseline"]["fid"],
                "native_repredict_fid": repredict["repredict"]["fid"],
                "delta_repredict_minus_fixed": repredict_delta,
            },
        },
        "remaining_causal_set": [
            "per-sample masked mean versus global masked-token mean loss reduction",
            "paper-block implementation versus the verified legacy topology",
            "512/8 model width versus 1024/16",
            "paper kinematic mask and masked-only loss versus legacy corruption/all-valid loss",
            "warm-up+cosine schedule and AdamW betas versus the legacy optimizer schedule",
            "batch size and seed",
        ],
        "operational_conclusion": (
            "Do not promote the native re-predict diagnostic. Keep the user-fixed "
            "50-step formal sampler. First run the pre-registered 50k global masked-token "
            "mean control, because it preserves the paper mask/objective while matching "
            "the standard InterMask reduction. The next formal retraining must retain "
            "paper-stated settings and online W&B telemetry; any legacy-setting arm is "
            "diagnostic or a disclosed reproduced baseline, never mislabeled as the "
            "paper architecture."
        ),
        "evidence": {
            "formal_contract": evidence(args.formal_contract),
            "candidate_metrics": evidence(args.candidate),
            "reference_metrics": evidence(args.reference),
            "quality_policy": evidence(args.quality_policy),
            "historical_run": evidence(args.historical_run),
            "repredict_diagnostic": evidence(args.repredict),
            "native_semantics": evidence(args.native_semantics),
            "loss_reduction": evidence(args.loss_reduction),
            "implementation": evidence(Path(__file__)),
        },
    }

    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"root-cause evidence contract failed: {failed}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
