#!/usr/bin/env python
"""Aggregate official and exhaustive-pair Inter-X multimodality audits."""

import argparse
import glob
import json
import math
import os
import re

import numpy as np


REPLICATION_PATTERN = re.compile(r"replication_(\d{2})")


def replication_map(pattern):
    records = {}
    for path in glob.glob(pattern):
        match = REPLICATION_PATTERN.search(os.path.basename(path))
        if match is None:
            continue
        replication = int(match.group(1))
        if replication in records:
            raise ValueError(f"duplicate replication {replication}: {path}")
        records[replication] = path
    return records


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    population_std = float(values.std())
    return {
        "ci95": float(1.96 * population_std / math.sqrt(len(values))),
        "mean": float(values.mean()),
        "population_std": population_std,
        "values": values.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="replication metrics glob")
    parser.add_argument("--audits", required=True, help="multimodality audit glob")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-replications", type=int, default=20)
    parser.add_argument(
        "--allow-legacy-provenance", action="store_true",
        help="Permit old metric/audit JSONs that predate generated-HDF5 hashes.",
    )
    args = parser.parse_args()

    metric_paths = replication_map(args.metrics)
    audit_paths = replication_map(args.audits)
    replications = sorted(set(metric_paths) & set(audit_paths))
    if set(metric_paths) != set(audit_paths):
        raise ValueError(
            "metric/audit replication mismatch: "
            f"metrics={sorted(metric_paths)}, audits={sorted(audit_paths)}"
        )
    if len(replications) != args.expected_replications:
        raise ValueError(
            f"expected {args.expected_replications} replications, got {len(replications)}"
        )

    official_values = []
    exhaustive_values = []
    per_replication = []
    action_prompt_means = {}
    action_labels = {}
    for replication in replications:
        with open(metric_paths[replication], "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        with open(audit_paths[replication], "r", encoding="utf-8") as handle:
            audit = json.load(handle)
        if metrics.get("mm_samples") != 100 or metrics.get("mm_repeats") != 30:
            raise ValueError(f"replication {replication}: expected official 100 x 30 MM")
        if audit.get("samples") != 100 or audit.get("all_pairs_per_prompt") != 435:
            raise ValueError(f"replication {replication}: expected 100 x 435 pair audit")
        if os.path.abspath(metrics["requests"]) != os.path.abspath(audit["requests"]):
            raise ValueError(f"replication {replication}: request files differ")
        metric_mm_hash = metrics.get("mm_generations_sha256")
        audit_mm_hash = audit.get("mm_generations_sha256")
        if metric_mm_hash is None or audit_mm_hash is None:
            if not args.allow_legacy_provenance:
                raise ValueError(
                    f"replication {replication}: missing MM HDF5 hash; regenerate "
                    "reports or pass --allow-legacy-provenance for an audit-only summary"
                )
        elif metric_mm_hash != audit_mm_hash:
            raise ValueError(
                f"replication {replication}: metric/audit MM HDF5 hashes differ"
            )

        official = float(metrics["multimodality"])
        exhaustive = float(audit["mean_all_prompt_pairs"])
        official_values.append(official)
        exhaustive_values.append(exhaustive)
        per_replication.append({
            "all_435_pairs": exhaustive,
            "official_10_pairs": official,
            "official_minus_all": official - exhaustive,
            "replication": replication,
        })
        for prompt in audit["prompts"]:
            if "action_id" not in prompt or "action_label" not in prompt:
                raise ValueError(
                    f"replication {replication}: audit predates action-label fields; "
                    "regenerate it with analyze_fair_interx_multimodality.py"
                )
            action_id = prompt["action_id"]
            if action_id is None:
                raise ValueError(
                    f"replication {replication}: cannot parse action from {prompt.get('name')!r}"
                )
            action_prompt_means.setdefault(action_id, []).append(
                float(prompt["mean_all_435_pairs"])
            )
            action_labels[action_id] = prompt["action_label"]

    official_values = np.asarray(official_values)
    exhaustive_values = np.asarray(exhaustive_values)
    differences = official_values - exhaustive_values
    action_groups = []
    for action_id in sorted(action_prompt_means):
        values = np.asarray(action_prompt_means[action_id], dtype=np.float64)
        action_groups.append({
            "action_id": action_id,
            "action_label": action_labels[action_id],
            "mean_all_pairs": float(values.mean()),
            "prompt_mean_p05": float(np.percentile(values, 5)),
            "prompt_mean_p50": float(np.percentile(values, 50)),
            "prompt_mean_p95": float(np.percentile(values, 95)),
            "prompt_occurrences": int(len(values)),
        })

    max_error_index = int(np.abs(differences).argmax())
    result = {
        "action_groups": action_groups,
        "all_pairs_per_prompt": 435,
        "exhaustive_all_pairs": statistics(exhaustive_values),
        "official_10_pairs": statistics(official_values),
        "official_minus_all": statistics(differences),
        "max_absolute_pair_subsampling_error": {
            "absolute_error": float(abs(differences[max_error_index])),
            "replication": replications[max_error_index],
            "signed_error": float(differences[max_error_index]),
        },
        "per_replication": per_replication,
        "legacy_provenance_allowed": bool(args.allow_legacy_provenance),
        "replications": len(replications),
        "root_mean_square_pair_subsampling_error": float(
            np.sqrt(np.mean(np.square(differences)))
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in {"action_groups", "per_replication"}
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
