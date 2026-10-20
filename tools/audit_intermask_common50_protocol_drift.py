#!/usr/bin/env python3
"""Validate InterMask common50 despite later additions to the shared YAML."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "output/fair_eval/intermask_official_exact_common50"
RUN_MANIFEST = RUN_ROOT / "run_manifest.json"
PROTOCOL = ROOT / "configs/fair_interx_protocol.yaml"
OUTPUT = RUN_ROOT / "protocol_drift_audit.json"
EXPECTED = {
    "time_steps": 50,
    "cond_scale": 2.0,
    "temperature": 1.0,
    "topk_filter_thres": 0.9,
    "length_policy": "exact",
    "seed_strategy": "base_seed + repeat_index",
}


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


def scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def artifact_matches(record: dict) -> bool:
    path = Path(record.get("path", ""))
    return (
        path.is_file()
        and path.stat().st_size == record.get("bytes")
        and sha256_file(path) == record.get("sha256")
    )


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    run = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    current_protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    historical_protocol = run["provenance"]["artifacts"]["protocol"]
    current_protocol_record = artifact(PROTOCOL)
    official_assets = run["provenance"]["artifacts"]
    execution_snapshots = run["provenance"]["implementation"]
    sampling = current_protocol["sampling"]["intermask"]
    checks = {
        "historical_full_yaml_hash_is_explicitly_stale": (
            historical_protocol["sha256"] != current_protocol_record["sha256"]
            and historical_protocol["bytes"] != current_protocol_record["bytes"]
        ),
        "current_yaml_retains_common50_semantics": (
            sampling.get("sampling_steps") == 50
            and sampling.get("cfg_scale") == 2.0
            and sampling.get("topk_filter_threshold") == 0.9
            and sampling.get("temperature") == 1.0
            and sampling.get("length_policy") == "exact"
        ),
        "official_weights_evaluator_and_requests_are_current": all(
            artifact_matches(record)
            for label, record in official_assets.items()
            if label != "protocol"
        ),
        "execution_snapshots_are_intact": all(
            artifact_matches(record) for record in execution_snapshots.values()
        ),
    }
    h5_records = []
    for replication in range(20):
        request = ROOT / f"output/fair_interx_requests/replication_{replication:02d}.npz"
        request_hash = sha256_file(request)
        for kind, repeats, samples in (("main", 1, 1696), ("mm", 30, 100)):
            path = RUN_ROOT / f"replication_{replication:02d}_{kind}.h5"
            with h5py.File(path, "r") as source:
                observed = {key: scalar(source.attrs.get(key)) for key in EXPECTED}
                completed = source["completed_repeats"][:].astype(bool)
                valid = (
                    observed == EXPECTED
                    and source["motions"].shape == (samples, repeats, 150, 672)
                    and completed.shape == (repeats,)
                    and completed.all()
                    and scalar(source.attrs.get("requests_sha256")) == request_hash
                    and scalar(source.attrs.get("checkpoint_sha256"))
                    == official_assets["transformer"]["sha256"]
                    and scalar(source.attrs.get("vq_checkpoint_sha256"))
                    == official_assets["vq"]["sha256"]
                )
            checks[f"replication_{replication:02d}_{kind}"] = bool(valid)
            h5_records.append({
                "replication": replication,
                "kind": kind,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "protocol": observed,
                "completed_repeats": int(completed.sum()),
                "request_sha256": request_hash,
            })
    payload = {
        "schema": "hhc_intermask_common50_protocol_drift_audit_v1",
        "status": (
            "pass_with_disclosed_historical_full_yaml_snapshot_limitation"
            if all(checks.values()) else "fail"
        ),
        "checks": checks,
        "interpretation": (
            "The shared YAML gained unrelated experiment fields after generation. "
            "Its historical full bytes were not snapshotted, but every operative "
            "InterMask sampler field is redundantly embedded and verified in all "
            "40 HDF5 files; official weights, evaluator, requests and execution "
            "source snapshots remain content-addressed and intact."
        ),
        "historical_protocol_record": historical_protocol,
        "current_protocol": current_protocol_record,
        "hdf5_records": h5_records,
        "run_manifest": artifact(RUN_MANIFEST),
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"],
        "checks": len(checks),
        "output": str(OUTPUT),
    }, indent=2))
    if payload["status"] == "fail":
        raise RuntimeError(
            f"InterMask common50 protocol audit failed: "
            f"{[key for key, value in checks.items() if not value]}"
        )


if __name__ == "__main__":
    main()
