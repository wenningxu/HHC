#!/usr/bin/env python3
"""Fail closed unless a future training launch obeys online W&B telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/online_wandb_training_policy.json"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument(
        "--metric-profile",
        choices=(
            "masked_token_classification",
            "diffusion_regression",
            "vq_autoencoder",
            "vq_autoencoder_body_root_hands",
        ),
        default="masked_token_classification",
    )
    parser.add_argument(
        "--telemetry-source", type=Path,
        default=ROOT / "utils/training_telemetry.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if policy.get("schema") != "hhc_online_wandb_training_policy_v1":
        raise ValueError("unexpected online W&B policy schema")
    if policy.get("status") != "active" or policy.get("mode") != "online":
        raise ValueError("online W&B policy is not active")

    launcher_source = args.launcher.read_text(encoding="utf-8")
    training_source = args.training_source.read_text(encoding="utf-8")
    telemetry_source = args.telemetry_source.read_text(encoding="utf-8")
    metric_source = training_source + "\n" + telemetry_source
    profile = policy.get("metric_profiles", {}).get(args.metric_profile)
    if not profile:
        raise ValueError(f"missing metric profile: {args.metric_profile}")
    required_metrics = (
        profile["required_training_metrics"]
        + profile["required_validation_metrics"]
    )
    missing_metrics = [
        metric for metric in required_metrics if metric not in metric_source
    ]
    interval = int(policy["logging_interval_optimizer_steps"])
    interval_patterns = (
        rf"telemetry_interval_optimizer_steps\s*=\s*{interval}\b",
        rf"TELEMETRY_INTERVAL_OPTIMIZER_STEPS\s*=\s*{interval}\b",
        rf"--telemetry-interval(?:-optimizer-steps)?[^\n]*default\s*=\s*{interval}\b",
    )
    checks = {
        "launcher_explicitly_online": "--wandb-mode online" in launcher_source,
        "launcher_does_not_train_offline": not any(
            line.strip().startswith("--wandb-mode offline")
            for line in launcher_source.splitlines()
        ),
        "all_required_metric_names_present": not missing_metrics,
        "training_invokes_gradient_weight_telemetry": (
            "parameter_and_gradient_telemetry(" in training_source
        ),
        "per_module_gradient_weight_telemetry_enabled": (
            "parameter_and_gradient_telemetry(" in training_source
            and "include_top_level_modules=False" not in training_source
            and "module.named_children()" in telemetry_source
            and "train/modules/" in telemetry_source
        ),
        # Lightning-based trainers use WandbLogger, while the standalone VQ
        # loop owns the run directly through wandb.init/wandb.log.  Both are
        # valid online backends; accepting only the former incorrectly rejects
        # the VQ implementation even though every event is sent to W&B.
        "wandb_logging_backend_present": (
            "WandbLogger" in training_source
            or (
                re.search(r"\bwandb\.init\s*\(", training_source) is not None
                and re.search(r"\bwandb\.log\s*\(", training_source) is not None
            )
        ),
        "stable_online_run_id_present": (
            "wandb_run_id" in training_source
            and re.search(r"\bid\s*=\s*(?:args\.)?wandb_run_id\b", training_source)
            is not None
        ),
        "online_resume_reuses_run_id": (
            re.search(r"\bresume\s*=", training_source) is not None
            and "allow" in training_source
        ),
        "nonfinite_fail_closed": "nonfinite training telemetry" in metric_source,
        "telemetry_interval_matches_policy": any(
            re.search(pattern, training_source) for pattern in interval_patterns
        ),
        "global_l2_and_max_abs_required": all(
            metric in policy["required_training_metrics"]
            for metric in (
                "train/gradient_l2_norm",
                "train/gradient_max_abs",
                "train/weight_l2_norm",
                "train/weight_max_abs",
            )
        ),
        "global_gradient_weight_diagnostics_required": all(
            metric in policy["required_training_metrics"]
            for metric in (
                "train/gradient_to_weight_l2_ratio",
                "train/gradient_value_count",
                "train/weight_value_count",
            )
        ),
        "measurement_semantics_frozen": all(
            policy.get(field)
            for field in (
                "parameter_scope",
                "norm_aggregation",
                "gradient_measurement_point",
                "weight_measurement_point",
                "major_module_norm_policy",
                "major_module_norm_effective_from",
                "training_progress_axis",
                "validation_coverage_policy",
            )
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    payload = {
        "schema": "hhc_online_wandb_training_contract_audit_v1",
        "status": status,
        "metric_profile": args.metric_profile,
        "checks": checks,
        "missing_required_metrics": missing_metrics,
        "policy": artifact(POLICY),
        "launcher": artifact(args.launcher),
        "training_source": artifact(args.training_source),
        "telemetry_source": artifact(args.telemetry_source),
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
