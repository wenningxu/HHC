#!/usr/bin/env python3
"""Compare the 50k masked-only no-shape pilot with its matched control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


METRICS = (
    "val/FID",
    "val/Matching",
    "val/R_precision_top1",
    "val/R_precision_top2",
    "val/R_precision_top3",
    "val/Diversity",
    "val/Multimodality_diagnostic",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validation_values(audit: dict, metric: str, final_step: int) -> dict[int, float]:
    events = audit["metrics"][metric].get("events", [])
    values = {}
    for row in events:
        step = row.get("preceding_training_optimizer_step")
        require(step is not None and step > 0 and step % 10000 == 0,
                f"{metric}: missing/unexpected training-step alignment; rerun history audit")
        require(step not in values, f"{metric}: duplicate validation at optimizer step {step}")
        values[step] = float(row["value"])
    return {step: value for step, value in values.items() if step <= final_step}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--control-wandb-audit", type=Path, required=True)
    parser.add_argument("--control-selection", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--pilot-wandb-audit", type=Path, required=True)
    parser.add_argument("--pilot-selection", type=Path, required=True)
    parser.add_argument("--source-equivalence-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cm, pm = load(args.control_manifest), load(args.pilot_manifest)
    ca, pa = load(args.control_wandb_audit), load(args.pilot_wandb_audit)
    cs, ps = load(args.control_selection), load(args.pilot_selection)
    source_audit = load(args.source_equivalence_audit)
    cr, pr = cm["resolved_arguments"], pm["resolved_arguments"]
    single_factor_keys = (
        "shape", "contact", "contact_conditioning", "paper_architecture",
        "training_mask_contract", "optimizer_contract", "conditioning_contract",
        "training_token_length_contract", "training_validation_contract", "sampler",
        "sampling_steps", "temperature", "cfg_scale", "batch_size", "seed",
    )
    mismatches = {
        key: {"control": cr.get(key), "pilot": pr.get(key)}
        for key in single_factor_keys if cr.get(key) != pr.get(key)
    }
    require(not mismatches, f"non-loss training factors differ: {mismatches}")
    require(cr.get("training_loss_contract") == "historical_all_valid",
            "control is not historical_all_valid")
    require(pr.get("training_loss_contract") == "paper_masked_only",
            "pilot is not paper_masked_only")
    require(int(cr["eval_iter"]) == int(pr["eval_iter"]) == 10_000,
            "validation interval changed")
    require(ca.get("status") == pa.get("status") == "pass", "W&B audit failed")
    require(cs.get("status") == ps.get("status") == "complete", "selection incomplete")
    require(ps["contract"].get("training_loss_contract") == "paper_masked_only",
            "pilot selection loss provenance is wrong")
    require(source_audit.get("status") == "pass",
            "source-equivalence audit did not pass")
    require(source_audit.get("behavioral_single_factor") is True,
            "source audit does not establish a behavioral single factor")

    count = 5
    trajectory = []
    values = {
        "control": {metric: validation_values(ca, metric, 50000) for metric in METRICS},
        "masked_only": {metric: validation_values(pa, metric, 50000) for metric in METRICS},
    }
    for index in range(count):
        step = (index + 1) * 10_000
        trajectory.append({
            "step": step,
            "control": {metric: values["control"][metric].get(step) for metric in METRICS},
            "masked_only": {metric: values["masked_only"][metric].get(step) for metric in METRICS},
            "delta_masked_only_minus_control": {
                metric: (values["masked_only"][metric][step] - values["control"][metric][step]
                         if step in values["masked_only"][metric] and step in values["control"][metric]
                         else None)
                for metric in METRICS
            },
        })

    selected_step = int(ps["selected_step"])
    baseline_selected_step = int(cs["selected_step"])
    require(baseline_selected_step in values["control"]["val/Multimodality_diagnostic"],
            "selected control checkpoint lacks MM validation")
    pilot_mm = values["masked_only"]["val/Multimodality_diagnostic"].get(selected_step)
    control_mm = values["control"]["val/Multimodality_diagnostic"][baseline_selected_step]
    pj = next(row for row in ps["candidates"] if row["step"] == selected_step)["joint_metrics"]["JointsFull"]
    cj = next(row for row in cs["candidates"] if row["step"] == baseline_selected_step)["joint_metrics"]["JointsFull"]
    quality_screen = {
        "fid_within_5_percent": float(pj["fid"]) <= float(cj["fid"]) * 1.05,
        "matching_within_0p05": float(pj["matching"]) <= float(cj["matching"]) + 0.05,
        "r1_within_0p01": float(pj["top1"]) >= float(cj["top1"]) - 0.01,
        "r2_within_0p01": float(pj["top2"]) >= float(cj["top2"]) - 0.01,
        "r3_within_0p01": float(pj["top3"]) >= float(cj["top3"]) - 0.01,
        "mm_not_lower": None if pilot_mm is None else pilot_mm >= control_mm,
    }
    promote = all(quality_screen.values())
    quality_pass = all(value for key, value in quality_screen.items() if key != "mm_not_lower")
    decision = (
        "continue_same_masked_only_run_to_300k" if promote
        else "needs_selected_checkpoint_mm_replay" if quality_pass and pilot_mm is None
        else "do_not_continue_masked_only_arm"
    )
    payload = {
        "schema": "hhc_no_shape_masked_only_pilot_comparison_v1",
        "status": "complete",
        "claim_scope": "single_seed_50k_validation_screen_not_test_reporting",
        "resume_caveat": "pilot resumed at 10k without saved RNG state; optimizer/step continuity is verified, exact minibatch/corruption RNG replay is not established",
        "single_factor": "training_loss_contract: historical_all_valid -> paper_masked_only",
        "trajectory": trajectory,
        "trajectory_alignment": "preceding explicit training optimizer step in synchronous validation; missing events remain null",
        "selected": {
            "control_step": baseline_selected_step,
            "masked_only_step": selected_step,
            "control_joints_full": cj,
            "masked_only_joints_full": pj,
            "control_mmodality_diagnostic": control_mm,
            "masked_only_mmodality_diagnostic": pilot_mm,
        },
        "promotion_screen": quality_screen,
        "decision": decision,
        "missing_mm_policy": "missing selected MM prevents promotion; a failed quality screen alone is sufficient to reject continuation",
        "formal_test_metrics_used_for_decision": False,
        "evidence": {
            name: artifact(path) for name, path in vars(args).items()
            if name != "output"
        },
        "implementation": artifact(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": "complete", "decision": payload["decision"],
                      "selected": payload["selected"]}, indent=2))


if __name__ == "__main__":
    main()
