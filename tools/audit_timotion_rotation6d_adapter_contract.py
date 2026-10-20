#!/usr/bin/env python3
"""Freeze the TiMotion adapter's 6D-convention mismatch as auditable evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
HHC_ROTATIONS = ROOT / "utils/rotation_conversions.py"
TIMOTION_ROTATIONS = Path(
    "/mnt/8T-HDD/wenning/TIMotion/utils/rotation_conversions.py"
)
PREPARER = ROOT / "tools/prepare_coshmdm_interx.py"
EXPORTER = ROOT / "tools/export_timotion_official_interx.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def tensor_payload(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/timotion_rotation6d_adapter_contract_audit.json",
    )
    args = parser.parse_args()

    hhc = load_module("hhc_rotation_contract", HHC_ROTATIONS)
    timotion = load_module("timotion_rotation_contract", TIMOTION_ROTATIONS)
    identity = torch.eye(3, dtype=torch.float64)
    encoded = hhc.matrix_to_rotation_6d(identity)
    expected = hhc.rotation_6d_to_matrix(encoded)
    observed = timotion.rotation_6d_to_matrix(encoded)
    expected_error = float(torch.max(torch.abs(expected - identity)))
    observed_error = float(torch.max(torch.abs(observed - identity)))
    observed_orthogonality_error = float(torch.max(torch.abs(
        observed.transpose(-1, -2) @ observed - identity
    )))

    exporter_source = EXPORTER.read_text(encoding="utf-8")
    preparer_source = PREPARER.read_text(encoding="utf-8")
    checks = {
        "prepared_data_uses_hhc_standard_matrix_to_rotation_6d": (
            "from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d"
            in preparer_source
        ),
        "exporter_inserts_timotion_before_importing_rotation_conversions": (
            "sys.path.insert(0, TIMOTION_ROOT)" in exporter_source
            and "from utils.rotation_conversions import" in exporter_source
        ),
        "standard_identity_roundtrip_exact": expected_error == 0.0,
        "timotion_decoder_misreads_standard_identity": observed_error >= 0.5,
        "timotion_result_is_not_a_rotation_matrix": (
            observed_orthogonality_error >= 0.5
        ),
    }
    payload = {
        "schema": "hhc_timotion_rotation6d_adapter_contract_audit_v1",
        "status": "bug_confirmed" if all(checks.values()) else "inconclusive",
        "checks": checks,
        "identity_matrix": tensor_payload(identity),
        "standard_identity_rotation6d": tensor_payload(encoded),
        "standard_roundtrip_matrix": tensor_payload(expected),
        "timotion_roundtrip_matrix": tensor_payload(observed),
        "standard_max_abs_error": expected_error,
        "timotion_max_abs_error": observed_error,
        "timotion_orthogonality_max_abs_error": observed_orthogonality_error,
        "impact": (
            "_fit_shape_specific_root decodes the standard row-concatenated "
            "body rotations with TiMotion's interleaved-index decoder, so its "
            "posed skeleton and fitted root rotation/translation are invalid"
        ),
        "scope": (
            "read-only diagnosis; the active TiMotion run/export source is not "
            "modified and its frozen validation history remains comparable"
        ),
        "sources": {
            "hhc_rotation_conversions": artifact(HHC_ROTATIONS),
            "timotion_rotation_conversions": artifact(TIMOTION_ROTATIONS),
            "interx_preparer": artifact(PREPARER),
            "active_exporter": artifact(EXPORTER),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
