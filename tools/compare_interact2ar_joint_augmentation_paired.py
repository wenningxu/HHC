#!/usr/bin/env python3
"""Paired comparison of two Interact2AR joint-evaluator result directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


WRAPPERS = ("JointsFull", "JointsBody", "JointsHands")
METRICS = ("fid", "matching", "top1", "top2", "top3", "diversity", "multimodality")
HIGHER_IS_BETTER = {"top1", "top2", "top3", "multimodality"}
COMPARISON_SCHEMAS = {
    "shape_augmentation": "hhc_interact2ar_joint_shape_augmentation_paired_v1",
    "random_shape_augmentation": (
        "hhc_interact2ar_joint_random_shape_augmentation_paired_v1"
    ),
    "optimized_shape_augmentation": (
        "hhc_interact2ar_joint_optimized_shape_augmentation_paired_v1"
    ),
    "optimization_effect_with_fixed_amass_donors": (
        "hhc_interact2ar_joint_shape_optimization_effect_paired_v1"
    ),
    "shape_condition_ablation": (
        "hhc_interact2ar_joint_shape_condition_ablation_paired_v1"
    ),
}
COMPARISON_CANDIDATE_ROLES = {
    "shape_augmentation": "shape_augmented_candidate",
    "random_shape_augmentation": "random_shape_augmented_candidate",
    "optimized_shape_augmentation": "optimized_shape_augmented_candidate",
    "optimization_effect_with_fixed_amass_donors": (
        "optimized_candidate_against_same_donor_random_control"
    ),
    "shape_condition_ablation": "no_shape_condition_candidate",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def replication_set_artifact(root: Path, expected: int) -> dict:
    root = root.resolve()
    expected_names = [f'replication_{index:02d}.json' for index in range(expected)]
    observed_names = sorted(path.name for path in root.glob('replication_*.json'))
    if observed_names != expected_names:
        raise ValueError(
            f'{root}: replication artifact set differs: '
            f'expected={expected_names}, observed={observed_names}'
        )
    digest = hashlib.sha256()
    total_bytes = 0
    for name in expected_names:
        path = root / name
        size = path.stat().st_size
        file_hash = sha256_file(path)
        encoded = path.name.encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'big'))
        digest.update(encoded)
        digest.update(size.to_bytes(8, 'big'))
        digest.update(bytes.fromhex(file_hash))
        total_bytes += size
    return {
        'path': str(root),
        'replication_files': expected,
        'total_bytes': total_bytes,
        'ordered_name_size_sha256_digest': digest.hexdigest(),
    }


def load_records(root: Path, expected: int) -> list[dict]:
    paths = [root / f"replication_{index:02d}.json" for index in range(expected)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing replication records: {missing}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for index, record in enumerate(records):
        if (
            record.get("schema") != "interact2ar_joint_evaluator_replication_v1"
            or record.get("status") != "complete"
            or record.get("replication") != index
            or record.get("samples") != 1696
            or record.get("mm_samples") != 100
            or record.get("mm_repeats") != 30
            or set(record.get("metrics", {})) != set(WRAPPERS)
        ):
            raise ValueError(f"{root}: invalid replication {index}")
    if len(list(root.glob("replication_*.json"))) != expected:
        raise ValueError(f"{root}: unexpected extra replication records")
    return records


def request_identity(record: dict) -> tuple[str, str]:
    return (
        record.get("main_generation_metadata", {}).get("requests_sha256", ""),
        record.get("mm_generation_metadata", {}).get("requests_sha256", ""),
    )


def paired_identity(record: dict) -> tuple:
    return (
        request_identity(record),
        record.get("requests", {}).get("sha256", ""),
        record.get("raw_ground_truth", {}).get("sha256", ""),
        record.get("evaluator_provenance"),
    )


def paired_summary(deltas: np.ndarray, rng: np.random.Generator) -> dict:
    if not np.isfinite(deltas).all():
        raise FloatingPointError("non-finite paired delta")
    draws = rng.integers(0, len(deltas), size=(100_000, len(deltas)))
    boot = deltas[draws].mean(axis=1)
    mean = float(deltas.mean())
    p_value = (
        1.0 if np.all(deltas == 0)
        else float(wilcoxon(deltas, alternative="two-sided").pvalue)
    )
    return {
        "mean_improvement": mean,
        "standard_error": float(deltas.std(ddof=1) / math.sqrt(len(deltas))),
        "bootstrap_95_percent_ci": [
            float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        ],
        "positive_replications": int((deltas > 0).sum()),
        "negative_replications": int((deltas < 0).sum()),
        "zero_replications": int((deltas == 0).sum()),
        "improvement_supported": bool(np.quantile(boot, 0.025) > 0),
        "wilcoxon_two_sided_p": p_value,
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, float(values[index]) * (len(values) - rank))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def metric_improvement(metric: str, baseline: dict, augmented: dict) -> float:
    if metric == "diversity":
        # The papers compare generated diversity with GT; overshooting is not an
        # improvement. Positive means augmentation moves closer to paired GT.
        baseline_error = abs(baseline[metric] - baseline["diversity_gt"])
        augmented_error = abs(augmented[metric] - augmented["diversity_gt"])
        return baseline_error - augmented_error
    if metric in HIGHER_IS_BETTER:
        return augmented[metric] - baseline[metric]
    return baseline[metric] - augmented[metric]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument(
        "--comparison-kind",
        choices=tuple(COMPARISON_SCHEMAS),
        default="shape_augmentation",
    )
    parser.add_argument(
        "--claim-scope",
        choices=(
            "matched_training",
            "diagnostic_unmatched_training_budget",
            "full_training_budget_test_guided_selection_and_validation_rng_differs",
        ),
        default="matched_training",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_records(args.baseline.resolve(), args.replications)
    augmented = load_records(args.augmented.resolve(), args.replications)
    pairing = [paired_identity(left) == paired_identity(right)
               and all(paired_identity(left)[:3])
               for left, right in zip(baseline, augmented)]
    if not all(pairing):
        raise ValueError(
            "baseline and augmentation request/GT/evaluator provenance are not exactly paired"
        )

    rng = np.random.default_rng(20260907)
    summaries = {}
    hypotheses = []
    for wrapper in WRAPPERS:
        summaries[wrapper] = {}
        for metric in METRICS:
            deltas = np.asarray([
                metric_improvement(
                    metric, left["metrics"][wrapper], right["metrics"][wrapper]
                )
                for left, right in zip(baseline, augmented)
            ], dtype=np.float64)
            summaries[wrapper][metric] = paired_summary(deltas, rng)
            hypotheses.append((
                wrapper, metric,
                summaries[wrapper][metric]["wilcoxon_two_sided_p"],
            ))

    adjusted = holm_bonferroni([item[2] for item in hypotheses])
    for (wrapper, metric, _), value in zip(hypotheses, adjusted):
        summaries[wrapper][metric]["holm_familywise_p"] = value
        summaries[wrapper][metric]["holm_significant_at_0.05"] = value < 0.05

    primary = [
        summaries[wrapper][metric]
        for wrapper in WRAPPERS
        for metric in ("diversity", "multimodality")
    ]

    schema = COMPARISON_SCHEMAS[args.comparison_kind]
    candidate_role = COMPARISON_CANDIDATE_ROLES[args.comparison_kind]
    payload = {
        "schema": schema,
        "status": "complete",
        "comparison_kind": args.comparison_kind,
        "claim_scope": args.claim_scope,
        "causal_training_ablation_claim_allowed": args.claim_scope == "matched_training",
        "candidate_role": candidate_role,
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.augmented.resolve()),
        # Retain this alias for compatibility with existing augmentation result
        # readers; candidate_role is authoritative for interpretation.
        "augmented": str(args.augmented.resolve()),
        "replications": args.replications,
        "pairing": (
            "same replication ID, frozen main/MM request hashes, request artifact SHA256, "
            "raw-GT SHA256, and evaluator provenance"
        ),
        "delta_convention": "positive always means the candidate improved over baseline",
        "diversity_convention": "reduction in absolute distance to paired GT diversity",
        "bootstrap_seed": 20260907,
        "bootstrap_draws": 100_000,
        "test": "two-sided paired Wilcoxon signed-rank",
        "multiplicity_correction": "Holm-Bonferroni across 3 wrappers x 7 metrics",
        "primary_shape_metrics": ["diversity", "multimodality"],
        "primary_all_improve": all(
            record["mean_improvement"] > 0 for record in primary
        ),
        "primary_all_ci_exclude_zero": all(
            record["bootstrap_95_percent_ci"][0] > 0 for record in primary
        ),
        "metrics": summaries,
        "evidence": {
            "baseline_replications": replication_set_artifact(
                args.baseline, args.replications
            ),
            "candidate_replications": replication_set_artifact(
                args.augmented, args.replications
            ),
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "bytes": Path(__file__).stat().st_size,
                "sha256": sha256_file(Path(__file__)),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(
        args.output.suffix + f'.tmp.{os.getpid()}'
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding='utf-8',
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
