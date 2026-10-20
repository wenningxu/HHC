#!/usr/bin/env python3
"""Formal TiMotion exporter using the dataset's standard 6D convention.

The active training callback remains bound to the historical exporter.  This
post-training wrapper changes only the shape-specific root-fit decoder, keeps
the exact TiMotion sampler, and gives resumable HDF5 files their own immutable
exporter identity.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import h5py
import torch
import torch.nn.functional as functional

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
import export_timotion_official_interx as base


ROOT = Path(__file__).resolve().parents[1]
BUG_AUDIT = ROOT / "output/timotion_rotation6d_adapter_contract_audit.json"
FIX_AUDIT = ROOT / "output/timotion_cosh_adapter_equivalence_audit_rotationfix.json"
FORMAL_FIX_AUDIT = (
    ROOT / "output/timotion_cosh_adapter_equivalence_audit_standard_formal.json"
)
FORMAL_ADAPTER = (
    "22-body rotations + shape-specific all-joint rigid root fit v3; "
    "standard row-concatenated rotation6d; identity hands"
)
ROTATION_CONTRACT = (
    "a1=d6[...,0:3], a2=d6[...,3:6], Gram-Schmidt, row-stack"
)


def standard_rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def timotion_to_official_interx(features, skeleton=None):
    """Expose the formal adapter for isolated equivalence/runtime tests."""
    original_decoder = base.rotation_6d_to_matrix
    try:
        base.rotation_6d_to_matrix = standard_rotation_6d_to_matrix
        return base.timotion_to_official_interx(features, skeleton=skeleton)
    finally:
        base.rotation_6d_to_matrix = original_decoder


def argument_value(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"required wrapper argument missing: {name}") from error


def load_prerequisites() -> tuple[dict, dict, dict]:
    bug = json.loads(BUG_AUDIT.read_text(encoding="utf-8"))
    fixed = json.loads(FIX_AUDIT.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL_FIX_AUDIT.read_text(encoding="utf-8"))
    base_hash = base.sha256_file(base.__file__)
    wrapper_hash = base.sha256_file(__file__)
    if (
        bug.get("status") != "bug_confirmed"
        or not all(bug.get("checks", {}).values())
        or bug.get("sources", {}).get("active_exporter", {}).get("sha256")
        != base_hash
    ):
        raise ValueError("historical TiMotion adapter-bug audit is stale")
    if (
        fixed.get("status") != "pass"
        or not all(fixed.get("checks", {}).values())
        or fixed.get("contract", {}).get("timotion_rotation_contract")
        != "standard"
        or fixed.get("sources", {}).get("timotion_exporter", {}).get("sha256")
        != base_hash
    ):
        raise ValueError("standard-adapter equivalence audit is stale")
    if (
        formal.get("status") != "pass"
        or not all(formal.get("checks", {}).values())
        or formal.get("contract", {}).get("timotion_rotation_contract")
        != "standard"
        or Path(formal.get("contract", {}).get("timotion_adapter", "")).resolve()
        != Path(__file__).resolve()
        or formal.get("sources", {}).get("timotion_exporter", {}).get("sha256")
        != wrapper_hash
    ):
        raise ValueError("formal standard-exporter equivalence audit is stale")
    identity = torch.eye(3, dtype=torch.float64)
    encoded = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64,
    )
    if not torch.equal(standard_rotation_6d_to_matrix(encoded), identity):
        raise ValueError("standard 6D identity runtime check failed")
    return bug, fixed, formal


def main() -> None:
    bug, fixed, formal = load_prerequisites()
    output = Path(argument_value("--output")).resolve()
    wrapper_hash = base.sha256_file(__file__)
    base_hash = base.sha256_file(base.__file__)
    transient_adapter = (
        "22-body rotations + shape-specific all-joint rigid root fit v2; "
        "identity hands"
    )

    if output.is_file():
        with h5py.File(output, "r") as existing:
            if existing.attrs.get("exporter_sha256") != wrapper_hash:
                raise ValueError(
                    "standard exporter refuses an HDF5 created by another exporter"
                )
            adapter = existing.attrs.get("adapter")
            if adapter not in (transient_adapter, FORMAL_ADAPTER):
                raise ValueError("unknown TiMotion standard-export adapter metadata")
            if adapter == FORMAL_ADAPTER and (
                existing.attrs.get("rotation6d_contract") != ROTATION_CONTRACT
                or not bool(existing.attrs.get("formal_result_eligible", False))
                or existing.attrs.get("base_exporter_sha256") != base_hash
            ):
                raise ValueError("formal standard-export metadata is incomplete")

    original_decoder = base.rotation_6d_to_matrix
    original_file = base.__file__
    original_validate = base.validate_resume_attributes

    def validate_standard_resume(source, expected):
        expected = dict(expected)
        if source.attrs.get("adapter") == FORMAL_ADAPTER:
            expected["adapter"] = FORMAL_ADAPTER
        original_validate(source, expected)

    try:
        base.rotation_6d_to_matrix = standard_rotation_6d_to_matrix
        # The base main function reads its module-global __file__ when binding
        # exporter/adapter hashes.  Give this wrapper a distinct resumable ID.
        base.__file__ = str(Path(__file__).resolve())
        base.validate_resume_attributes = validate_standard_resume
        base.main()
    finally:
        base.rotation_6d_to_matrix = original_decoder
        base.__file__ = original_file
        base.validate_resume_attributes = original_validate

    with h5py.File(output, "r+") as handle:
        if (
            "completed_repeats" not in handle
            or not handle["completed_repeats"][:].astype(bool).all()
        ):
            raise ValueError("standard TiMotion export returned incomplete")
        if handle.attrs.get("exporter_sha256") != wrapper_hash:
            raise ValueError("standard TiMotion exporter identity drifted")
        handle.attrs["adapter"] = FORMAL_ADAPTER
        handle.attrs["rotation6d_contract"] = ROTATION_CONTRACT
        handle.attrs["formal_result_eligible"] = True
        handle.attrs["base_exporter_sha256"] = base_hash
        handle.attrs["historical_adapter_bug_audit_sha256"] = (
            base.sha256_file(BUG_AUDIT)
        )
        handle.attrs["standard_adapter_equivalence_audit_sha256"] = (
            base.sha256_file(FIX_AUDIT)
        )
        handle.attrs["formal_exporter_equivalence_audit_sha256"] = (
            base.sha256_file(FORMAL_FIX_AUDIT)
        )
        handle.flush()

    print(json.dumps({
        "output": str(output),
        "status": "complete_standard_rotation6d",
        "exporter_sha256": wrapper_hash,
        "base_exporter_sha256": base_hash,
        "bug_audit_status": bug["status"],
        "equivalence_status": fixed["status"],
        "formal_equivalence_status": formal["status"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
