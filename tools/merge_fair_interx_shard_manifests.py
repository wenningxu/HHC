#!/usr/bin/env python3
"""Validate disjoint fair-evaluation shards and write one canonical manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def comparable(record: dict) -> dict:
    result = dict(record)
    result.pop("replication_range", None)
    result.pop("implementation_source_stability_verified_at_utc", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-dir", type=Path, required=True)
    parser.add_argument("--ranges", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ranges = []
    paths = []
    records = []
    for value in args.ranges:
        try:
            start_text, end_text = value.split(":", 1)
            start, end = int(start_text), int(end_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid range {value!r}; expected START:END") from exc
        if not 0 <= start < end <= 20:
            raise ValueError(f"invalid replication range {value!r}")
        path = args.method_dir / f"run_manifest_{start:02d}_{end:02d}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("replication_range") != [start, end]:
            raise ValueError(f"{path}: embedded replication range differs")
        if record.get("implementation_source_stability_verified_at_completion") is not True:
            raise ValueError(f"{path}: source stability was not verified")
        ranges.append((start, end))
        paths.append(path)
        records.append(record)

    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError(f"ranges are not a gap-free partition at {cursor}: {ranges}")
        cursor = end
    if cursor != 20:
        raise ValueError(f"ranges do not cover all 20 replications: {ranges}")
    reference = comparable(records[0])
    for path, record in zip(paths[1:], records[1:]):
        if comparable(record) != reference:
            raise ValueError(f"{path}: shard identity/provenance differs")

    required_suffixes = (
        "main.h5", "mm.h5", "generation_timing.json", "multimodality_audit.json"
    )
    for replication in range(20):
        required = [args.method_dir / f"replication_{replication:02d}_{suffix}"
                    for suffix in required_suffixes]
        required.append(args.method_dir / f"replication_{replication:02d}.json")
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)

    merged = dict(records[0])
    merged["replication_range"] = [0, 20]
    merged["parallel_generation"] = {
        "schema": "fair_interx_disjoint_replication_shards_v1",
        "ranges": [list(item) for item in sorted(ranges)],
        "shard_manifests": [artifact(path) for path in paths],
        "shared_identity_and_provenance_verified": True,
    }
    merged["implementation_source_stability_verified_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "ranges": sorted(ranges)}, sort_keys=True))


if __name__ == "__main__":
    main()
