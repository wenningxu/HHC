#!/usr/bin/env python3
"""Prove the completed CoShMDM exports used the standard 6D dependency."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COSH_ROOT = Path("/mnt/8T-HDD/wenning/coshmdm-code")
ROTATION_RELATIVE = "utils/rotation_conversions.py"
ADAPTER = COSH_ROOT / "utils/interx_official_adapter.py"
DEFAULT_MANIFEST = (
    ROOT / "output/fair_eval/coshmdm_interx_retrained_2daycap_seed123/"
    "run_manifest.json"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
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
    parser.add_argument("--run-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/coshmdm_rotation_dependency_provenance_audit.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    repository = manifest["provenance"]["repositories"]["coshmdm"]
    revision = repository["revision"]
    status = repository["status"]
    committed = subprocess.run(
        ["git", "show", f"{revision}:{ROTATION_RELATIVE}"],
        cwd=COSH_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    current = (COSH_ROOT / ROTATION_RELATIVE).read_bytes()

    namespace = {"__name__": "cosh_rotation_provenance_audit"}
    source = compile(current, str(COSH_ROOT / ROTATION_RELATIVE), "exec")
    exec(source, namespace)
    identity = torch.eye(3, dtype=torch.float64)
    encoded = namespace["matrix_to_rotation_6d"](identity)
    decoded = namespace["rotation_6d_to_matrix"](encoded)
    roundtrip_error = float(torch.max(torch.abs(decoded - identity)))

    adapter_record = manifest["provenance"]["implementation"][
        "official_adapter"
    ]
    checks = {
        "rotation_dependency_clean_in_frozen_run_status": not any(
            ROTATION_RELATIVE in line for line in status
        ),
        "current_rotation_dependency_matches_frozen_revision": (
            current == committed
        ),
        "frozen_rotation_dependency_is_standard_identity_roundtrip": (
            encoded.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            and roundtrip_error == 0.0
        ),
        "adapter_snapshot_is_content_addressed": (
            adapter_record.get("content_addressed_snapshot") is True
            and adapter_record.get("sha256") == sha256_file(ADAPTER)
            and Path(adapter_record["path"]).is_file()
            and sha256_file(Path(adapter_record["path"]))
            == adapter_record["sha256"]
        ),
        "completed_method_is_coshmdm": manifest.get("method") == "coshmdm",
        "completed_replication_range_is_20": (
            manifest.get("replication_range") == [0, 20]
            and manifest.get("samples_per_replication") == 1696
        ),
    }
    payload = {
        "schema": "hhc_coshmdm_rotation_dependency_provenance_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "repository_revision": revision,
        "rotation_dependency": {
            "path": str((COSH_ROOT / ROTATION_RELATIVE).resolve()),
            "current_sha256": sha256_bytes(current),
            "frozen_revision_sha256": sha256_bytes(committed),
            "identity_rotation6d": encoded.tolist(),
            "identity_roundtrip_max_abs_error": roundtrip_error,
        },
        "interpretation": (
            "The completed CoShMDM export used its repository's clean standard "
            "row-concatenated 6D conversion; TiMotion alone requires replay."
        ),
        "run_manifest": artifact(args.run_manifest),
        "adapter": artifact(ADAPTER),
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
