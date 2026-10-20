#!/usr/bin/env python3
"""Fail closed unless the Inter-X flat-hand SDF contract is current."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_artifact(record: dict, label: str) -> dict:
    path = Path(record.get("path", "")).resolve()
    require(path.is_file(), f"{label} is missing: {path}")
    observed = sha256_file(path)
    require(observed == record.get("sha256"), f"{label} hash is stale: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": observed,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-audit", type=Path,
        default=Path("output/volumetric_smpl_interx_flat_hand_geometry_audit.json"),
    )
    parser.add_argument(
        "--query-assessment", type=Path,
        default=Path("output/volumetric_smpl_flat_hand_query_chunk_assessment.json"),
    )
    parser.add_argument("--query-chunk", type=int, default=10475)
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/interx_volumetric_smpl_contract_gate.json"),
    )
    args = parser.parse_args()

    geometry_path = args.geometry_audit.resolve()
    assessment_path = args.query_assessment.resolve()
    geometry = load(geometry_path)
    assessment = load(assessment_path)

    representation = geometry.get("representation_contract", {})
    geometry_checks = geometry.get("checks", {})
    require(
        geometry.get("schema")
        == "hhc_volumetric_smpl_interx_flat_hand_geometry_audit_v1"
        and geometry.get("status") == "pass"
        and geometry_checks
        and all(geometry_checks.values()),
        "flat-hand geometry audit is not a complete pass",
    )
    require(
        representation.get("dataset") == "Inter-X"
        and representation.get("pose_layout") == "55-joint axis-angle"
        and representation.get("hand_pose_convention") == "flat_hand_mean"
        and representation.get("smplx_flat_hand_mean") is True,
        "flat-hand representation contract is wrong",
    )

    verified = {
        "geometry_audit": verify_artifact(
            {
                "path": str(geometry_path),
                "sha256": sha256_file(geometry_path),
            },
            "geometry audit",
        ),
        "query_assessment": verify_artifact(
            {
                "path": str(assessment_path),
                "sha256": sha256_file(assessment_path),
            },
            "query assessment",
        ),
        "geometry_implementation": verify_artifact(
            geometry["implementation"], "geometry-audit implementation"
        ),
        "production_backend": verify_artifact(
            geometry["backend_implementation"], "production backend"
        ),
        "body_models": {
            gender: verify_artifact(record, f"{gender} SMPL-X body model")
            for gender, record in geometry["body_models"].items()
        },
        "volumetric_checkpoints": {
            gender: verify_artifact(record, f"{gender} VolumetricSMPL checkpoint")
            for gender, record in geometry["volumetric_checkpoints"].items()
        },
        "geometry_motions": [
            verify_artifact(record["motion"], f"geometry motion {index}")
            for index, record in enumerate(geometry["records"])
        ],
    }
    require(
        {record.get("gender") for record in geometry["records"]}
        == {"male", "female"},
        "geometry audit must cover male and female real motions",
    )

    assessment_checks = assessment.get("checks", {})
    require(
        assessment.get("status") == "pass"
        and assessment_checks
        and all(assessment_checks.values()),
        "query-chunk assessment is not a complete pass",
    )
    assessment_impl = assessment.get("implementation", {})
    verified["query_implementations"] = {
        name: verify_artifact(record, f"query {name} implementation")
        for name, record in assessment_impl.items()
    }
    require(
        verified["query_implementations"]["production"]["sha256"]
        == verified["production_backend"]["sha256"],
        "geometry and query audits bind different production backends",
    )
    verified["query_benchmarks"] = [
        verify_artifact(record["benchmark"], f"query benchmark {index}")
        for index, record in enumerate(assessment.get("motions", []))
    ]

    candidates = {
        int(record["query_chunk"]): record
        for record in assessment.get("passing_common_candidates", [])
    }
    require(
        args.query_chunk in candidates,
        f"query chunk {args.query_chunk} is not authorized on every real motion",
    )
    selected = candidates[args.query_chunk]
    thresholds = assessment.get("thresholds", {})
    require(
        selected["maximum_loss_relative_difference"]
        <= thresholds["maximum_loss_relative_difference"]
        and selected["maximum_gradient_relative_difference"]
        <= thresholds["maximum_gradient_relative_difference"]
        and selected["minimum_speedup"] >= thresholds["minimum_speedup"]
        and selected["maximum_peak_allocated_bytes"]
        <= thresholds["maximum_peak_allocated_gib"] * 1024 ** 3,
        "selected query chunk no longer satisfies its frozen thresholds",
    )

    payload = {
        "schema": "hhc_interx_volumetric_smpl_contract_gate_v1",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_chunk": args.query_chunk,
        "representation_contract": representation,
        "selected_candidate": selected,
        "thresholds": thresholds,
        "verified_artifacts": verified,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "bytes": Path(__file__).stat().st_size,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
