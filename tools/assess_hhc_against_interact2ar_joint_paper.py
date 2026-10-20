#!/usr/bin/env python
"""Assess one HHC joint-evaluator summary against Interact2AR Table 2."""

import argparse
import json
import os
from pathlib import Path


METRICS = {
    "R1": ("top1", "higher"),
    "R2": ("top2", "higher"),
    "R3": ("top3", "higher"),
    "FID": ("fid", "lower"),
    "MM_Dist": ("matching", "lower"),
    "Diversity": ("diversity", "closer_to_ground_truth"),
    "MModality": ("multimodality", "higher"),
}


def compare(value, target, direction, ground_truth=None):
    if direction == "higher":
        return value > target, value - target
    if direction == "lower":
        return value < target, target - value
    if direction == "closer_to_ground_truth":
        candidate_error = abs(value - ground_truth)
        target_error = abs(target - ground_truth)
        return candidate_error < target_error, target_error - candidate_error
    raise ValueError(direction)


def generation_conditioning(summary):
    provenance = summary.get("evaluator_provenance") or {}
    return str(provenance.get("generation_conditioning", "unknown"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=Path("configs/interact2ar_joint_paper_acceptance_v2.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("joint evaluator summary is not complete")
    if args.method not in summary.get("methods", {}):
        raise KeyError(f"method not present in summary: {args.method}")

    checks = {}
    all_pass = True
    for wrapper, target_metrics in acceptance[
        "interact2ar_autoregressive_threshold"
    ].items():
        actual_metrics = summary["methods"][args.method][wrapper]
        gt_metrics = acceptance["ground_truth"][wrapper]
        wrapper_checks = {}
        for paper_name, (summary_name, direction) in METRICS.items():
            actual = float(actual_metrics[summary_name]["mean"])
            target = float(target_metrics[paper_name])
            gt = (
                float(gt_metrics[paper_name])
                if direction == "closer_to_ground_truth" else None
            )
            passed, margin = compare(actual, target, direction, gt)
            wrapper_checks[paper_name] = {
                "actual_mean": actual,
                "interact2ar_paper_mean": target,
                "direction": direction,
                "ground_truth_mean": gt,
                "margin_in_favour_of_hhc": margin,
                "pass": bool(passed),
            }
            all_pass &= bool(passed)
        checks[wrapper] = wrapper_checks

    conditioning = generation_conditioning(summary)
    first_frame_matched = (
        "first" in conditioning.lower()
        and "ground" in conditioning.lower()
        and "unconditional" not in conditioning.lower()
    )
    result = {
        "schema": "hhc_vs_interact2ar_joint_paper_acceptance_v2",
        "status": "complete",
        "method": args.method,
        "summary": str(args.summary.resolve()),
        "acceptance": str(args.acceptance.resolve()),
        "checks": checks,
        "numerical_all_joint_metrics_pass": bool(all_pass),
        "generation_conditioning": conditioning,
        "paper_gt_first_frame_protocol_matched": bool(first_frame_matched),
        "fair_paper_superiority_claim_pass": bool(all_pass and first_frame_matched),
        "evaluated_metric_scope": list(METRICS),
        "transition_metrics_excluded": {
            "metrics": ["PJ", "AUJ"],
            "reason": (
                "The paper evaluates these on 64 separately composed sequences "
                "of 8 motions, not on ordinary joint-evaluator test generations; "
                "the released evaluate.py does not implement them."
            ),
        },
        "interpretation": (
            "The numerical gate compares means against the autoregressive "
            "Interact2AR row. A fair same-protocol claim additionally requires "
            "the paper's ground-truth first-frame conditioning."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output),
        "numerical_pass": all_pass,
        "fair_claim_pass": all_pass and first_frame_matched,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
