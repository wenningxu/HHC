#!/usr/bin/env python3
"""Preflight the historical/original side of a hybrid shape token cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hybrid_shape_token_cache import (  # noqa: E402
    TOKEN_CACHE_CONTRACT,
    partition_expected_names,
    sha256_named_files,
)


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


def read_names(path: Path) -> set[str]:
    rows = [row.strip() for row in path.read_text(encoding="utf-8").splitlines()
            if row.strip()]
    if len(rows) != len(set(rows)):
        raise ValueError(f"duplicate sample names in {path}")
    return set(rows)


def eligible(names: set[str], motion_path: Path) -> set[str]:
    with h5py.File(motion_path, "r") as motions:
        return {
            name for name in names
            if name in motions and 30 <= motions[name].shape[0] < 1000
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
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument(
        "--val-split", type=Path,
        default=ROOT / "dataset/interx/splits/val.txt",
    )
    parser.add_argument("--historical-code-dir", type=Path, required=True)
    parser.add_argument("--historical-contact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    overlay = args.overlay_root.resolve()
    train_split = overlay / "splits/train.txt"
    all_motion = overlay / "motions/all.h5"
    materialization = overlay / "materialization_manifest.json"
    val_split = args.val_split.resolve()
    train_names = read_names(train_split)
    val_names = read_names(val_split)
    if train_names & val_names:
        raise ValueError("shape training and official validation splits overlap")
    eligible_train = eligible(train_names, all_motion)
    eligible_val = eligible(val_names, all_motion)
    expected = eligible_train | eligible_val
    original, augmented, manifest = partition_expected_names(
        expected, materialization
    )

    code_dir = args.historical_code_dir.resolve()
    contact_dir = args.historical_contact_dir.resolve()
    missing_code = sorted(
        name for name in original if not (code_dir / f"{name}.npy").is_file()
    )
    missing_contact = sorted(
        name for name in original
        if not (contact_dir / f"{name}.npy").is_file()
    )
    checks = {
        "train_and_val_are_disjoint": not (train_names & val_names),
        "augmented_names_are_train_only": augmented <= eligible_train,
        "official_validation_contains_only_original_names": not (
            augmented & eligible_val
        ),
        "partition_covers_every_eligible_name": (
            original | augmented == expected and not (original & augmented)
        ),
        "manifest_augmented_count_matches_partition": (
            manifest.get("counts", {}).get("augmented") == len(augmented)
        ),
        "historical_code_cache_is_complete_for_original_subset": not missing_code,
        "historical_contact_cache_is_complete_for_original_subset": (
            not missing_contact
        ),
    }
    payload = {
        "schema": "hhc_hybrid_shape_token_cache_preflight_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "token_cache_contract": TOKEN_CACHE_CONTRACT,
        "checks": checks,
        "counts": {
            "eligible_train": len(eligible_train),
            "eligible_val": len(eligible_val),
            "original": len(original),
            "augmented": len(augmented),
            "expected_total": len(expected),
            "missing_historical_code": len(missing_code),
            "missing_historical_contact": len(missing_contact),
        },
        "missing": {
            "historical_code_first_five": missing_code[:5],
            "historical_contact_first_five": missing_contact[:5],
        },
        "hashes": {
            "historical_original_code_named_sha256": (
                sha256_named_files(code_dir, original) if not missing_code else None
            ),
            "historical_original_contact_named_sha256": (
                sha256_named_files(contact_dir, original)
                if not missing_contact else None
            ),
        },
        "artifacts": {
            "train_split": artifact(train_split),
            "val_split": artifact(val_split),
            "all_motion": artifact(all_motion),
            "materialization": artifact(materialization),
            "hybrid_helper": artifact(
                ROOT / "utils/hybrid_shape_token_cache.py"
            ),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
