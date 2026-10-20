#!/usr/bin/env python
"""Validate complete optimizer shards and freeze one aggregate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "hhc_interx_shape_augmentation_shard_aggregate_v1"
RUN_SCHEMA = "hhc_interx_shape_augmentation_run_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_equal(reference: dict, candidate: dict, fields, label: str) -> None:
    for field in fields:
        if candidate.get(field) != reference.get(field):
            raise ValueError(f"{label}: shard field differs: {field}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir", type=Path, action="append", required=True,
        help="Completed optimizer shard directory; repeat in any order.",
    )
    parser.add_argument("--expected-samples", type=int, default=6238)
    parser.add_argument("--expected-shards", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate_shards(shard_dirs, expected_samples: int, expected_shards: int):
    """Return an aggregate payload after strict content/provenance checks."""
    if expected_samples <= 0 or expected_shards <= 0:
        raise ValueError("expected counts must be positive")
    shard_dirs = [Path(path).resolve() for path in shard_dirs]
    if len(shard_dirs) != expected_shards:
        raise ValueError(
            f"expected {expected_shards} shard directories, "
            f"received {len(shard_dirs)}"
        )
    if len(shard_dirs) != len(set(shard_dirs)):
        raise ValueError("duplicate shard directory")

    invariant_fields = (
        "schema", "caption_binding_policy", "contact_manifest",
        "contact_manifest_sha256", "implementation", "optimizer", "plan",
        "plan_records", "plan_records_sha256", "plan_sha256",
        "source_motion", "source_motion_sha256",
    )
    reference = None
    by_position = {}
    by_name = {}
    shard_records = []
    for shard_dir in shard_dirs:
        manifest_path = shard_dir / "_completion_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        label = str(manifest_path)
        if manifest.get("schema") != RUN_SCHEMA:
            raise ValueError(f"{label}: unexpected run schema")
        if manifest.get("complete") is not True:
            raise ValueError(f"{label}: shard is incomplete")
        requested = int(manifest.get("requested_samples", -1))
        completed = int(manifest.get("completed_samples", -1))
        names = manifest.get("completed_names")
        selection = manifest.get("selection", {})
        selected_names = selection.get("selected_names")
        positions = selection.get("full_selection_positions")
        if not (
            isinstance(names, list)
            and isinstance(selected_names, list)
            and isinstance(positions, list)
            and requested == completed == len(names)
            and names == selected_names
            and len(names) == len(positions)
        ):
            raise ValueError(f"{label}: names/counts/positions are inconsistent")
        if reference is None:
            reference = manifest
        else:
            require_equal(reference, manifest, invariant_fields, label)
            for key in ("selection_seed", "positive_timing_only"):
                if selection.get(key) != reference["selection"].get(key):
                    raise ValueError(f"{label}: selection field differs: {key}")
        local_positions = []
        for name, raw_position in zip(names, positions):
            position = int(raw_position)
            if position in by_position:
                raise ValueError(f"duplicate selection position {position}")
            if name in by_name:
                raise ValueError(f"duplicate augmented name {name}")
            sidecar = shard_dir / f"{name}.json"
            p1 = shard_dir / f"{name}_p1.npz"
            p2 = shard_dir / f"{name}_p2.npz"
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if int(metadata.get("selection_position", -1)) != position:
                raise ValueError(f"{name}: sidecar selection position differs")
            if metadata.get("output_p1_sha256") != sha256_file(p1):
                raise ValueError(f"{name}: p1 hash mismatch")
            if metadata.get("output_p2_sha256") != sha256_file(p2):
                raise ValueError(f"{name}: p2 hash mismatch")
            by_position[position] = name
            by_name[name] = position
            local_positions.append(position)
        if local_positions != sorted(local_positions):
            raise ValueError(f"{label}: shard positions are not ordered")
        shard_records.append({
            "directory": str(shard_dir),
            "completion_manifest": artifact(manifest_path),
            "contract_sha256": manifest.get("contract_sha256"),
            "samples": len(names),
            "first_position": min(local_positions),
            "last_position": max(local_positions),
        })

    expected_positions = list(range(expected_samples))
    if sorted(by_position) != expected_positions:
        missing = sorted(set(expected_positions) - set(by_position))
        extra = sorted(set(by_position) - set(expected_positions))
        raise ValueError(
            f"shards do not cover exact positions 0..{expected_samples - 1}; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    shard_records.sort(key=lambda item: item["first_position"])
    for previous, current in zip(shard_records, shard_records[1:]):
        if previous["last_position"] + 1 != current["first_position"]:
            raise ValueError("sorted shards are not contiguous")

    completed_names = [by_position[position] for position in expected_positions]
    return {
        "schema": SCHEMA,
        "status": "complete",
        "complete": True,
        "requested_samples": expected_samples,
        "completed_samples": expected_samples,
        "completed_names": completed_names,
        "selection": {
            "full_selection_positions": expected_positions,
            "selected_names": completed_names,
            "selection_seed": reference["selection"]["selection_seed"],
            "positive_timing_only": reference["selection"][
                "positive_timing_only"
            ],
        },
        "optimizer": reference["optimizer"],
        "caption_binding_policy": reference["caption_binding_policy"],
        "contact_manifest": reference["contact_manifest"],
        "contact_manifest_sha256": reference["contact_manifest_sha256"],
        "implementation": reference["implementation"],
        "plan": reference["plan"],
        "plan_records": reference["plan_records"],
        "plan_records_sha256": reference["plan_records_sha256"],
        "plan_sha256": reference["plan_sha256"],
        "source_motion": reference["source_motion"],
        "source_motion_sha256": reference["source_motion_sha256"],
        "shards": shard_records,
        "aggregate_content_sha256": canonical_sha256({
            "completed_names": completed_names,
            "optimizer": reference["optimizer"],
            "plan_sha256": reference["plan_sha256"],
            "shards": shard_records,
        }),
        "implementation_artifact": artifact(Path(__file__)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    payload = aggregate_shards(
        args.shard_dir, args.expected_samples, args.expected_shards,
    )
    atomic_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "shards": len(payload["shards"]),
        "completed_samples": payload["completed_samples"],
        "output": str(args.output.resolve()),
        "aggregate_content_sha256": payload["aggregate_content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
