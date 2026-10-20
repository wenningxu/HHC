#!/usr/bin/env python3
"""Fail closed unless a completed grounded-chain run preserves its roots/legs."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "_completion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = manifest.get("completed_names", [])
    checks = {
        "manifest_complete": manifest.get("complete") is True,
        "sample_count_exact": (
            manifest.get("requested_samples")
            == manifest.get("completed_samples")
            == len(names)
            == args.expected_samples
        ),
        "unique_names": len(names) == len(set(names)),
        "fixed_root_policy": manifest.get("optimizer", {}).get(
            "optimization_variable_policy"
        ) == "grounded_chain_fixed_root",
        "mmm_ground_alignment": manifest.get("optimizer", {}).get(
            "ground_alignment_policy"
        ) == "mmm_per_person_joint_min_restore_raw",
    }
    source_checks = {}
    for raw_path, expected_hash in manifest.get("implementation", {}).items():
        path = Path(raw_path)
        source_checks[str(path)] = (
            path.is_file() and sha256_file(path) == expected_hash
        )
    checks["all_sources_match"] = all(source_checks.values())

    sample_checks = {}
    for name in names:
        sidecar_path = run_dir / f"{name}.json"
        p1_path = run_dir / f"{name}_p1.npz"
        p2_path = run_dir / f"{name}_p2.npz"
        if not all(path.is_file() for path in (sidecar_path, p1_path, p2_path)):
            sample_checks[name] = False
            continue
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        invariants = sidecar.get("diagnostics", {}).get(
            "optimization_variable_invariants", {}
        )
        exact = True
        for person in ("p1", "p2"):
            record = invariants.get(person, {})
            exact &= record.get("policy") == "grounded_chain_fixed_root"
            exact &= record.get("root_orientation_max_abs_delta") == 0.0
            exact &= record.get("grounded_chain_pose_max_abs_delta") == 0.0
            exact &= record.get(
                "root_translation_delta_temporal_max_abs_deviation"
            ) == 0.0
            exact &= record.get("root_translation_constant_delta") == [0.0] * 3
        exact &= sidecar.get("output_p1_sha256") == sha256_file(p1_path)
        exact &= sidecar.get("output_p2_sha256") == sha256_file(p2_path)
        sample_checks[name] = bool(exact)
    checks["all_sample_invariants_and_hashes_match"] = all(
        sample_checks.values()
    ) and len(sample_checks) == args.expected_samples
    passed = all(checks.values())
    payload = {
        "schema": "hhc_grounded_chain_run_invariants_audit_v1",
        "status": "pass" if passed else "fail",
        "checks": checks,
        "sample_checks": sample_checks,
        "source_checks": source_checks,
        "manifest": artifact(manifest_path),
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
