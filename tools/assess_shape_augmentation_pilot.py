#!/usr/bin/env python3
"""Fail closed unless the matched shape-optimization pilot improves physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = (
    ROOT / "configs/interx_shape_augmentation/pilot_physical_gate_v1.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def assess(comparison: dict, gate: dict) -> tuple[dict, bool]:
    # The grounded-chain five-pair gate was frozen before this assessment
    # distinction was implemented.  Preserve that immutable JSON (and its
    # preregistered hash) and map its explicit stage to the only action its
    # decision text permits.  New gates may state the field directly.
    authorization_on_pass = gate.get("authorization_on_pass")
    if authorization_on_pass is None:
        authorization_on_pass = (
            "pilot25"
            if gate.get("stage")
            == "preregistered_grounded_chain_fixed_root_five_pair_gate"
            else "full_augmentation"
        )
    if authorization_on_pass not in {"pilot25", "full_augmentation"}:
        raise ValueError(
            "unsupported shape-pilot authorization_on_pass: "
            f"{authorization_on_pass!r}"
        )
    checks = {
        "comparison_schema_matches": (
            comparison.get("schema") == gate["comparison_schema"]
        ),
        "sample_count_matches": (
            comparison.get("sample_count") == gate["expected_sample_count"]
            and len(comparison.get("sample_keys", []))
            == gate["expected_sample_count"]
            and len(set(comparison.get("sample_keys", [])))
            == gate["expected_sample_count"]
        ),
    }
    decisions = {}
    comparisons = comparison.get("comparisons", {})
    for metric, thresholds in gate["criteria"].items():
        record = comparisons.get(metric, {})
        mean_delta = record.get("paired_delta_after_minus_before_mean")
        ci = record.get("paired_delta_bootstrap_95ci")
        metric_checks = {
            "mean_delta_finite": finite_number(mean_delta),
            "mean_delta_at_or_below_limit": (
                finite_number(mean_delta)
                and float(mean_delta) <= float(thresholds["maximum_mean_delta"])
            ),
        }
        if "maximum_bootstrap_95ci_upper" in thresholds:
            metric_checks.update({
                "bootstrap_95ci_is_finite_pair": (
                    isinstance(ci, list)
                    and len(ci) == 2
                    and all(finite_number(value) for value in ci)
                    and float(ci[0]) <= float(ci[1])
                ),
                "bootstrap_95ci_upper_at_or_below_limit": (
                    isinstance(ci, list)
                    and len(ci) == 2
                    and finite_number(ci[1])
                    and float(ci[1]) <= float(
                        thresholds["maximum_bootstrap_95ci_upper"]
                    )
                ),
            })
        decisions[metric] = {
            "thresholds": thresholds,
            "observed": {
                "before_mean": record.get("before_mean"),
                "after_mean": record.get("after_mean"),
                "paired_delta_after_minus_before_mean": mean_delta,
                "paired_delta_bootstrap_95ci": ci,
                "wilcoxon_two_sided_p": record.get("wilcoxon_two_sided_p"),
            },
            "checks": metric_checks,
            "pass": all(metric_checks.values()),
        }
    passed = all(checks.values()) and all(
        decision["pass"] for decision in decisions.values()
    )
    authorized_action = authorization_on_pass if passed else None
    return {
        "checks": checks,
        "metric_decisions": decisions,
        "authorization_on_pass": authorization_on_pass,
        "authorized_action": authorized_action,
        "pilot25_authorized": passed and authorization_on_pass == "pilot25",
        "full_augmentation_authorized": (
            passed and authorization_on_pass == "full_augmentation"
        ),
    }, passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison_path = args.comparison.resolve()
    gate_path = args.gate.resolve()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("schema") != (
        "hhc_interx_shape_augmentation_pilot_physical_gate_v1"
    ) or gate.get("status") != "active":
        raise ValueError("shape augmentation pilot gate is not active v1")
    assessment, passed = assess(comparison, gate)
    payload = {
        "schema": "hhc_shape_augmentation_pilot_assessment_v1",
        "status": "pass" if passed else "fail",
        **assessment,
        "comparison": artifact(comparison_path),
        "gate": artifact(gate_path),
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
