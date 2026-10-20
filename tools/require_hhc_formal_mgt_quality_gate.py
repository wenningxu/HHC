#!/usr/bin/env python3
"""Fail closed unless a content-bound current-paper MGT quality gate passed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate", type=Path,
        default=Path("output/hhc_formal_mgt_quality_gate.json"),
    )
    args = parser.parse_args()
    require(args.gate.is_file(), f"formal MGT quality gate missing: {args.gate}")
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    require(gate.get("schema") == "hhc_formal_mgt_quality_gate_v1", "wrong gate schema")
    require(gate.get("status") == "pass", "formal MGT quality gate did not pass")
    require(gate.get("shape_mgt_training_authorized") is True, "shape MGT training not authorized")
    for family in (
        "candidate_protocol_checks", "reference_protocol_checks",
        "training_contract_checks", "quality_checks",
    ):
        checks = gate.get(family, {})
        require(isinstance(checks, dict) and checks, f"missing check family: {family}")
        require(all(value is True for value in checks.values()), f"failed checks in {family}")
    evidence = gate.get("evidence", {})
    require(isinstance(evidence, dict) and evidence, "gate evidence is missing")
    for role, record in evidence.items():
        path = Path(record.get("path", ""))
        require(path.is_file(), f"gate evidence missing: {role}")
        require(path.stat().st_size == record.get("bytes"), f"gate evidence size changed: {role}")
        require(sha256_file(path) == record.get("sha256"), f"gate evidence hash changed: {role}")
    print(json.dumps({
        "status": "pass",
        "gate": str(args.gate.resolve()),
        "candidate_fid": gate["observed"]["candidate_fid"],
        "reference_fid": gate["observed"]["reference_fid"],
        "selected_checkpoint_step": gate["observed"]["selected_checkpoint_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
