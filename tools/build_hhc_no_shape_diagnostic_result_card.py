#!/usr/bin/env python3
"""Freeze the completed no-MGT-shape Inter-X diagnostic without causal overclaim."""

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
LABEL = "hhc_no_mgt_shape_v4_pilot50k_bestval_common50"
JOINT_ROOT = ROOT / "output/interact2ar_joint_evaluator_candidates" / LABEL
FILES = {
    "training_manifest": RUN / "training_manifest.json",
    "terminal_checkpoint_audit": ROOT / "output/hhc_no_shape_v4_pilot50k_final_checkpoint_audit.json",
    "wandb_history_audit": ROOT / "output/hhc_no_shape_v4_pilot50k_wandb_history_audit.json",
    "validation_selection": ROOT / "output/hhc_no_shape_v4_pilot50k_validation_selection.json",
    "joint_summary": JOINT_ROOT / "summary.json",
    "paper_acceptance": JOINT_ROOT / "paper_acceptance.json",
    "paired_shape_comparison": JOINT_ROOT / "shape_conditioned_vs_no_shape_paired.json",
    "mm_repeat_collapse": JOINT_ROOT / "mm_repeat_collapse.json",
}
WRAPPERS = ("JointsFull", "JointsBody", "JointsHands")
METRICS = ("fid", "matching", "top1", "top2", "top3", "diversity", "multimodality")


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
    require(set(summary.get("methods", {})) == {LABEL},
            "joint summary method set changed")
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


def main() -> None:
    manifest = load("training_manifest")
    checkpoint = load("terminal_checkpoint_audit")
    wandb = load("wandb_history_audit")
    selection = load("validation_selection")
    summary = load("joint_summary")
    acceptance = load("paper_acceptance")
    paired = load("paired_shape_comparison")
    collapse = load("mm_repeat_collapse")

    args = manifest["resolved_arguments"]
    require(args.get("shape") is False, "MGT shape condition is not disabled")
    require(args.get("training_mask_contract") ==
            "historical_pipeline_paper_kinematic_ratio", "mask contract changed")
    require(args.get("training_loss_contract") == "historical_all_valid",
            "training loss contract changed")
    require(args.get("paper_architecture") is False, "historical topology changed")
    require(args.get("sampler") == "sample_intra_cas", "sampler changed")
    require(args.get("sampling_steps") == 50, "sampling steps changed")
    require(args.get("temperature") == 0.5 and args.get("cfg_scale") == 3.5,
            "sampling temperature/CFG changed")
    require(checkpoint.get("status") == "pass" and checkpoint.get("global_step") == 50000,
            "terminal checkpoint audit failed")
    require(wandb.get("status") == "pass" and wandb.get("remote_state") == "finished",
            "online W&B audit failed")
    require(selection.get("status") == "complete" and selection.get("selected_step") == 30000,
            "official validation selection changed")
    selected_path = Path(selection["selected_checkpoint"])
    require(sha256_file(selected_path) == selection["selected_checkpoint_sha256"],
            "selected checkpoint hash changed")
    require(acceptance.get("schema") == "hhc_vs_interact2ar_joint_paper_acceptance_v2"
            and acceptance.get("status") == "complete", "paper acceptance is incomplete")
    require(paired.get("schema") ==
            "hhc_interact2ar_joint_shape_condition_ablation_paired_v1",
            "paired comparison schema changed")
    require(paired.get("claim_scope") == "diagnostic_unmatched_training_budget"
            and paired.get("causal_training_ablation_claim_allowed") is False,
            "paired comparison permits an invalid causal claim")
    require(collapse.get("schema") == "hhc_mm_repeat_collapse_diagnostic_v1"
            and collapse.get("status") == "complete", "MM collapse diagnostic incomplete")

    payload = {
        "schema": "hhc_no_mgt_shape_diagnostic_result_card_v1",
        "status": "complete",
        "claim_scope": "targeted_diagnostic_unmatched_training_budget",
        "causal_shape_condition_ablation_claim_allowed": False,
        "reason_not_causal": [
            "shape-conditioned reference trained for 300k and selected at 260k; this pilot trained for 50k and selected at 30k",
            "the runs used different periodic-validation and validation-RNG side-effect contracts",
        ],
        "training": {
            "optimizer_steps": 50000,
            "selected_step": 30000,
            "mgt_shape_condition": False,
            "architecture": (
                "historical 1024D/16-head model: 7 global self-attention plus "
                "2 local cross-attention base blocks, followed by 9 output "
                "self-attention blocks"
            ),
            "vq": "raw Inter-X 12288-code checkpoint net_175000.pth",
            "mask": "paper asymmetric Root/Body/Hands cosine kinematic ratios",
            "loss": "historical all-valid-token NLL",
            "wandb": {
                "entity": wandb["entity"], "project": wandb["project"],
                "run_id": wandb["run_id"], "state": wandb["remote_state"],
            },
        },
        "selection": {
            "split": "frozen official Inter-X held-out validation",
            "metric": "released Interact2AR JointsFull FID",
            "selected_step": selection["selected_step"],
            "selected_validation_fid": selection["selected_validation_fid"],
            "checkpoint": artifact(selected_path),
        },
        "evaluation": {
            "split": "official Inter-X test",
            "replications": 20,
            "samples_per_replication": 1696,
            "multimodality": "100 prompts x 30 generations; 10 evaluator pairs",
            "sampler": (
                "sample_intra_cas Root->Body->Hands, 50 outer MaskGIT steps, "
                "fixed tokens never remasked, temperature 0.5, text CFG 3.5"
            ),
            "joint_metrics": compact_joint_metrics(summary),
            "numerical_all_interact2ar_metrics_pass": acceptance[
                "numerical_all_joint_metrics_pass"
            ],
            "paper_gt_first_frame_protocol_matched": acceptance[
                "paper_gt_first_frame_protocol_matched"
            ],
        },
        "paired_shape_conditioned_comparison": paired["metrics"],
        "mm_repeat_collapse": collapse["methods"][LABEL],
        "evidence": {label: artifact(path) for label, path in FILES.items()},
        "implementation": artifact(Path(__file__)),
    }
    output = ROOT / "output/hhc_no_mgt_shape_diagnostic_result_card.json"
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        "status": "complete", "output": str(output),
        "JointsFull": payload["evaluation"]["joint_metrics"]["JointsFull"],
        "numerical_all_interact2ar_metrics_pass": payload["evaluation"][
            "numerical_all_interact2ar_metrics_pass"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
