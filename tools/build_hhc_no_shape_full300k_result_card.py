#!/usr/bin/env python3
"""Freeze the full-budget no-MGT-shape result without a causal overclaim."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / (
    "output/t2i_trans/"
    "2026-09-07-11-49-54_hhc_historical_paper_mask_no_mgt_shape_"
    "pilot50k_seed114514_v4"
)
LABEL = "hhc_no_mgt_shape_v4_full300k_besttest3_common50"
JOINT_ROOT = ROOT / "output/interact2ar_joint_evaluator_candidates" / LABEL
FAIR_ROOT = ROOT / "output/fair_eval" / LABEL
FILES = {
    "training_manifest": RUN / "training_manifest.json",
    "terminal_checkpoint_audit": ROOT / "output/hhc_no_shape_v4_full300k_final_checkpoint_audit.json",
    "wandb_history_audit": ROOT / "output/hhc_no_shape_v4_full300k_wandb_history_audit.json",
    "test3_selection": ROOT / "output/hhc_no_shape_v4_test3_shortlist_selection.json",
    "legacy_summary": FAIR_ROOT / "summary.json",
    "joint_summary": JOINT_ROOT / "summary.json",
    "paper_acceptance": JOINT_ROOT / "paper_acceptance.json",
    "paired_shape_comparison": JOINT_ROOT / "shape_conditioned_vs_no_shape_paired.json",
    "mm_repeat_collapse": JOINT_ROOT / "mm_repeat_collapse.json",
}
WRAPPERS = ("JointsFull", "JointsBody", "JointsHands")
METRICS = ("fid", "matching", "top1", "top2", "top3", "diversity", "multimodality")
LEGACY_METRICS = (
    "fid", "matching", "top1", "top2", "top3", "diversity",
    "diversity_gt", "multimodality",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def load(label: str) -> dict:
    return json.loads(FILES[label].read_text(encoding="utf-8"))


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def compact_joint_metrics(summary: dict) -> dict:
    require(summary.get("schema") == "interact2ar_joint_evaluator_suite_summary_v1",
            "unexpected joint-summary schema")
    require(summary.get("status") == "complete", "joint summary is incomplete")
    require(summary.get("replications") == 20, "joint summary is not 20 replications")
    require(summary.get("samples_per_replication") == 1696,
            "joint summary does not use 1696 samples")
    require(set(summary.get("methods", {})) == {LABEL}, "joint method set changed")
    output = {}
    for wrapper in WRAPPERS:
        source = summary["methods"][LABEL][wrapper]
        require(set(source) == set(METRICS), f"{wrapper}: metric set changed")
        output[wrapper] = {}
        for metric in METRICS:
            record = source[metric]
            values = record.get("values")
            require(isinstance(values, list) and len(values) == 20,
                    f"{wrapper}/{metric}: incomplete replications")
            require(all(isinstance(x, (int, float)) and math.isfinite(x)
                        for x in values), f"{wrapper}/{metric}: non-finite")
            output[wrapper][metric] = {
                "mean": float(record["mean"]), "ci95": float(record["ci95"])
            }
    return output


def compact_legacy_metrics(summary: dict) -> dict:
    require(summary.get("replications") == 20,
            "legacy summary is not 20 replications")
    require(summary.get("samples_per_replication") == 1696,
            "legacy summary does not use 1696 samples")
    metrics = summary.get("metrics", {})
    require(set(metrics) == set(LEGACY_METRICS), "legacy metric set changed")
    output = {}
    for metric in LEGACY_METRICS:
        record = metrics[metric]
        values = record.get("values")
        require(isinstance(values, list) and len(values) == 20,
                f"legacy/{metric}: incomplete replications")
        require(all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in values), f"legacy/{metric}: non-finite")
        output[metric] = {
            "mean": float(record["mean"]), "ci95": float(record["ci95"])
        }
    return output


def main() -> None:
    manifest = load("training_manifest")
    checkpoint = load("terminal_checkpoint_audit")
    wandb = load("wandb_history_audit")
    selection = load("test3_selection")
    legacy_summary = load("legacy_summary")
    summary = load("joint_summary")
    acceptance = load("paper_acceptance")
    paired = load("paired_shape_comparison")
    collapse = load("mm_repeat_collapse")

    args = manifest["resolved_arguments"]
    require(args.get("shape") is False, "MGT shape condition is not disabled")
    require(args.get("total_iter") == 300000 and args.get("diagnostic_stop_iter") == 0,
            "run is not the normal 300k endpoint")
    require(args.get("training_mask_contract") ==
            "historical_pipeline_paper_kinematic_ratio", "mask contract changed")
    require(args.get("training_loss_contract") == "historical_all_valid",
            "training loss contract changed")
    require(args.get("paper_architecture") is False, "historical topology changed")
    require(args.get("sampler") == "sample_intra_cas", "sampler changed")
    require(args.get("sampling_steps") == 50, "sampling steps changed")
    require(args.get("temperature") == 0.5 and args.get("cfg_scale") == 3.5,
            "sampling temperature/CFG changed")
    require(checkpoint.get("status") == "pass" and checkpoint.get("global_step") == 300000,
            "terminal checkpoint audit failed")
    require(wandb.get("status") == "pass" and wandb.get("remote_state") == "finished",
            "online W&B audit failed")
    require(selection.get("status") == "complete", "test3 selection is incomplete")
    require(selection.get("selection_is_test_guided") is True
            and selection.get("unbiased_heldout_selection_claim_allowed") is False,
            "test-guided selection disclosure is missing")
    expected_candidate_steps = [80000, 220000, 260000, 280000, 300000]
    require(selection.get("requested_candidate_steps") == expected_candidate_steps,
            "test3 shortlist changed")
    require(len(selection.get("candidates", [])) == 5,
            "not all five shortlisted checkpoints were selected")
    candidate_means = {}
    for row in selection["candidates"]:
        step = int(row["step"])
        replications = row.get("replications", [])
        require(len(replications) == 3, f"step {step}: selection is not three replications")
        require([int(item["replication"]) for item in replications] == [0, 1, 2],
                f"step {step}: selection replication IDs changed")
        values = [float(item["fid"]) for item in replications]
        require(all(math.isfinite(value) and value >= 0 for value in values),
                f"step {step}: invalid selection FID")
        recorded_mean = float(row["mean_joints_full_fid"])
        require(math.isclose(recorded_mean, sum(values) / 3, rel_tol=0, abs_tol=1e-12),
                f"step {step}: selection mean does not match its three replications")
        candidate_means[str(step)] = {
            "replication_fid": values,
            "mean_joints_full_fid": recorded_mean,
        }
    require(sorted(map(int, candidate_means)) == expected_candidate_steps,
            "selection candidate steps changed")
    selected_step = int(selection["selected_step"])
    require(selected_step in range(10000, 300001, 10000), "selected step is invalid")
    recomputed_best_step = min(
        candidate_means,
        key=lambda key: (candidate_means[key]["mean_joints_full_fid"], int(key)),
    )
    require(selected_step == int(recomputed_best_step),
            "selected step is not the minimum three-replication mean")
    require(math.isclose(
        float(selection["selected_mean_joints_full_fid"]),
        candidate_means[recomputed_best_step]["mean_joints_full_fid"],
        rel_tol=0, abs_tol=1e-12,
    ), "selected mean does not match the recomputed minimum")
    selected_path = Path(selection["selected_checkpoint"])
    require(sha256_file(selected_path) == selection["selected_checkpoint_sha256"],
            "selected checkpoint hash changed")
    require(acceptance.get("schema") == "hhc_vs_interact2ar_joint_paper_acceptance_v2"
            and acceptance.get("status") == "complete", "paper acceptance is incomplete")
    require(paired.get("schema") ==
            "hhc_interact2ar_joint_shape_condition_ablation_paired_v1",
            "paired comparison schema changed")
    require(paired.get("claim_scope") ==
            "full_training_budget_test_guided_selection_and_validation_rng_differs"
            and paired.get("causal_training_ablation_claim_allowed") is False,
            "paired comparison permits an invalid causal claim")
    require(collapse.get("schema") == "hhc_mm_repeat_collapse_diagnostic_v1"
            and collapse.get("status") == "complete", "MM collapse diagnostic incomplete")

    payload = {
        "schema": "hhc_no_mgt_shape_full300k_result_card_v1",
        "status": "complete",
        "claim_scope": "full_training_budget_noncausal_shape_condition_diagnostic",
        "causal_shape_condition_ablation_claim_allowed": False,
        "reason_not_causal": [
            "the retained shape-conditioned run used historical validation RNG side effects",
            "this no-shape run used RNG-isolated periodic validation",
            "the user-directed final checkpoint is selected on three official test replications",
        ],
        "training": {
            "optimizer_steps": 300000,
            "selected_step": selected_step,
            "mgt_shape_condition": False,
            "architecture": "historical 1024D/16-head 7-global+2-local+9-output topology",
            "vq": "raw Inter-X 12288-code checkpoint net_175000.pth",
            "mask": "paper asymmetric Root/Body/Hands cosine kinematic ratios",
            "loss": "historical all-valid-token NLL",
            "external_shape_dataset_used": False,
            "wandb": {"entity": wandb["entity"], "project": wandb["project"],
                      "run_id": wandb["run_id"], "state": wandb["remote_state"]},
        },
        "selection": {
            "split": "official Inter-X test replications 0, 1 and 2",
            "metric": "mean released Interact2AR JointsFull FID over three fixed test draws",
            "selection_is_test_guided": True,
            "unbiased_heldout_selection_claim_allowed": False,
            "candidate_steps": expected_candidate_steps,
            "selected_step": selected_step,
            "selected_test3_mean_fid": selection["selected_mean_joints_full_fid"],
            "candidate_test3_joints_full_fid": candidate_means,
            "checkpoint": artifact(selected_path),
        },
        "evaluation": {
            "split": "official Inter-X test", "replications": 20,
            "samples_per_replication": 1696,
            "multimodality": "100 prompts x 30 generations; 10 evaluator pairs",
            "sampler": "sample_intra_cas, 50 outer steps, T=0.5, text CFG=3.5",
            "fid_namespace_warning": (
                "legacy 672D and released Interact2AR joint evaluator metrics "
                "are separate namespaces and must not be compared directly"
            ),
            "legacy_672d_metrics": compact_legacy_metrics(legacy_summary),
            "joint_metrics": compact_joint_metrics(summary),
            "interact2ar_acceptance": {
                "numerical_all_joint_metrics_pass": acceptance[
                    "numerical_all_joint_metrics_pass"
                ],
                "paper_gt_first_frame_protocol_matched": acceptance[
                    "paper_gt_first_frame_protocol_matched"
                ],
                "fair_paper_superiority_claim_pass": acceptance[
                    "fair_paper_superiority_claim_pass"
                ],
                "checks": acceptance["checks"],
            },
        },
        "paired_shape_conditioned_comparison": paired["metrics"],
        "mm_repeat_collapse": collapse["methods"][LABEL],
        "evidence": {label: artifact(path) for label, path in FILES.items()},
        "implementation": artifact(Path(__file__)),
    }
    output = ROOT / "output/hhc_no_mgt_shape_full300k_result_card.json"
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": "complete", "output": str(output),
                      "selected_step": selected_step,
                      "JointsFull": payload["evaluation"]["joint_metrics"]["JointsFull"],
                      "numerical_all_interact2ar_metrics_pass": payload["evaluation"]
                      ["interact2ar_acceptance"]
                      ["numerical_all_joint_metrics_pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
