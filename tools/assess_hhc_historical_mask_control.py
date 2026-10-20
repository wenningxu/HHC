#!/usr/bin/env python3
"""Assess a historical-corruption single-factor MGT diagnostic at 50k."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.assess_hhc_token_window_control import (
    MATCHED_ARGUMENTS,
    artifact,
    atomic_json,
    candidate_at_step,
    early_best,
    load_manifest,
    load_selection,
)


MASK_MATCHED_ARGUMENTS = tuple(dict.fromkeys((
    *MATCHED_ARGUMENTS,
    # The token-window assessor deliberately omits its own treatment factor;
    # it is a mandatory matched variable for this corruption-only control.
    "training_token_length_contract",
)))

EXPECTED_FORMAL = {
    "training_mask_contract": "paper_kinematic",
    "training_loss_contract": "paper_masked_only",
    "training_token_length_contract": "paper_50",
    "optimizer_contract": "paper_warmup_cosine",
    "conditioning_contract": "paper_text_cfg",
    "training_validation_contract": "paper_isolated_common",
    "paper_architecture": True,
}


def validate_formal_contract(formal_args: dict) -> None:
    mismatches = {
        key: {"expected": value, "actual": formal_args.get(key)}
        for key, value in EXPECTED_FORMAL.items()
        if formal_args.get(key) != value
    }
    if mismatches:
        raise ValueError(f"formal reference contract changed: {mismatches}")


def validate_single_factor_contract(
    formal_args: dict, control_args: dict,
) -> None:
    validate_formal_contract(formal_args)
    if control_args.get("training_mask_contract") != "historical_flat_random_replace":
        raise ValueError("control does not use recovered historical corruption")
    mismatches = {
        key: {"formal": formal_args.get(key), "control": control_args.get(key)}
        for key in MASK_MATCHED_ARGUMENTS
        if key != "training_mask_contract"
        and formal_args.get(key) != control_args.get(key)
    }
    if mismatches:
        raise ValueError(
            f"historical-mask control is not single-factor: {mismatches}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-selection", type=Path, required=True)
    parser.add_argument("--control-selection", type=Path)
    parser.add_argument("--comparison-step", type=int, default=50000)
    parser.add_argument("--minimum-absolute-improvement", type=float, default=0.1)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    formal = load_selection(args.formal_selection)
    if int(formal["expected_final_step"]) != 300000:
        raise ValueError("formal reference is not the complete 300k run")
    formal_manifest_path, formal_manifest = load_manifest(formal)
    formal_args = formal_manifest["resolved_arguments"]
    validate_formal_contract(formal_args)

    formal_early = early_best(formal, args.comparison_step)
    formal_endpoint = candidate_at_step(formal, args.comparison_step)
    payload = {
        "schema": "hhc_historical_mask_control_assessment_v1",
        "status": "diagnostic_required",
        "purpose": (
            "single-factor diagnosis of iterative-sampling/training-corruption "
            "alignment; historical corruption cannot replace the formal paper mask"
        ),
        "comparison_step": args.comparison_step,
        "formal_paper_mask_early_best": formal_early,
        "formal_paper_mask_matched_endpoint": formal_endpoint,
        "historical_mask_control": None,
        "decision": "run_historical_flat_random_replace_50k_control",
        "thresholds": {
            "minimum_absolute_fid_improvement": args.minimum_absolute_improvement,
            "minimum_relative_fid_improvement": args.minimum_relative_improvement,
        },
        "artifacts": {
            "formal_selection": artifact(args.formal_selection),
            "formal_training_manifest": artifact(formal_manifest_path),
            "implementation": artifact(Path(__file__)),
        },
    }

    if args.control_selection is not None:
        control = load_selection(args.control_selection)
        if int(control["expected_final_step"]) != args.comparison_step:
            raise ValueError("historical-mask control did not end at comparison step")
        control_manifest_path, control_manifest = load_manifest(control)
        control_args = control_manifest["resolved_arguments"]
        validate_single_factor_contract(formal_args, control_args)

        formal_fid = float(formal_early["validation_fid"])
        control_fid = float(control["selected_validation_fid"])
        absolute = formal_fid - control_fid
        relative = absolute / formal_fid
        supported = (
            absolute >= args.minimum_absolute_improvement
            and relative >= args.minimum_relative_improvement
        )
        control_endpoint = candidate_at_step(control, args.comparison_step)
        formal_endpoint_fid = float(formal_endpoint["validation_fid"])
        control_endpoint_fid = float(control_endpoint["validation_fid"])
        endpoint_absolute = formal_endpoint_fid - control_endpoint_fid
        endpoint_relative = endpoint_absolute / formal_endpoint_fid
        payload.update({
            "status": "complete",
            "historical_mask_control": {
                "selected_step": int(control["selected_step"]),
                "selected_validation_fid": control_fid,
                "absolute_fid_improvement": absolute,
                "relative_fid_improvement": relative,
                "substantial_gap_explained": supported,
                "matched_endpoint": {
                    "step": args.comparison_step,
                    "formal_validation_fid": formal_endpoint_fid,
                    "control_validation_fid": control_endpoint_fid,
                    "absolute_fid_improvement": endpoint_absolute,
                    "relative_fid_improvement": endpoint_relative,
                },
            },
            "decision_basis": (
                "minimum fixed-protocol validation FID over identical 10k-spaced "
                "candidates through 50k; the matched endpoint is disclosed separately"
            ),
            "decision": (
                "historical_corruption_explains_substantial_early_fid_gap"
                if supported else
                "historical_corruption_does_not_explain_substantial_early_fid_gap"
            ),
        })
        payload["artifacts"].update({
            "control_selection": artifact(args.control_selection),
            "control_training_manifest": artifact(control_manifest_path),
        })

    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
