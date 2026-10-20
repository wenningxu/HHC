#!/usr/bin/env python3
"""Audit licensed assets needed for a HumanML3D-train AMASS shape pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def amass_path(source: str, amass_root: Path) -> Path:
    prefix = "./pose_data/"
    if not source.startswith(prefix) or not source.endswith(".npy"):
        raise ValueError(f"unexpected HumanML3D source path: {source}")
    return amass_root / (source[len(prefix):-4] + ".npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--humanml3d-root", type=Path,
                        default=Path("/mnt/8T-HDD/HumanML3D"))
    parser.add_argument("--amass-root", type=Path,
                        default=Path("/mnt/8T-HDD/HumanML3D/amass_data"))
    parser.add_argument("--transfer-root", type=Path,
                        default=Path("/mnt/8T-HDD/wenning/smplx-transfer"))
    parser.add_argument("--smplx-model-root", type=Path,
                        default=Path("/home/wenning/projects/MMM/body_models"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.humanml3d_root.resolve()
    index_path = root / "index.csv"
    split_path = root / "HumanML3D/train.txt"
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = {Path(row["new_name"]).stem: row for row in csv.DictReader(handle)}
    split_names = [line.strip() for line in split_path.read_text().splitlines()
                   if line.strip()]
    canonical_names = sorted({name[1:] if name.startswith("M") else name
                              for name in split_names})
    missing_index = [name for name in canonical_names if name not in rows]
    if missing_index:
        raise ValueError(f"train names absent from index.csv: {missing_index[:10]}")

    amass_sources = {}
    humanact_names = []
    for name in canonical_names:
        source = rows[name]["source_path"]
        if "humanact12" in source.lower():
            humanact_names.append(name)
            continue
        source_path = amass_path(source, args.amass_root.resolve())
        amass_sources[str(source_path)] = source
    source_paths = sorted(map(Path, amass_sources))
    present = [path for path in source_paths if path.is_file()]
    missing = [path for path in source_paths if not path.is_file()]
    datasets = Counter(
        path.relative_to(args.amass_root.resolve()).parts[0] for path in source_paths
    )

    required = {
        "smplh_male": root / "body_models/smplh/male/model.npz",
        "smplh_female": root / "body_models/smplh/female/model.npz",
        "smplh_to_smplx_correspondence": (
            args.transfer_root.resolve()
            / "transfer_data/smplh2smplx_deftrafo_setup.pkl"
        ),
        "smplx_male": args.smplx_model_root.resolve()
            / "smplx/SMPLX_MALE.npz",
        "smplx_female": args.smplx_model_root.resolve()
            / "smplx/SMPLX_FEMALE.npz",
    }
    required_artifacts = {key: artifact(path) for key, path in required.items()}
    ready = not missing and all(item["present"] for item in required_artifacts.values())
    payload = {
        "schema": "hhc_humanml3d_amass_shape_asset_audit_v1",
        "status": "pass" if ready else "missing_assets",
        "humanml3d": {
            "root": str(root),
            "index": artifact(index_path),
            "train_split": artifact(split_path),
            "train_entries_including_mirrors": len(split_names),
            "canonical_train_entries_after_deduplicating_mirrors": len(canonical_names),
            "humanact12_entries_excluded": len(humanact_names),
            "unique_amass_source_sequences_required": len(source_paths),
        },
        "amass": {
            "root": str(args.amass_root.resolve()),
            "representation": "SMPL+H G",
            "datasets_and_required_sequence_counts": dict(sorted(datasets.items())),
            "present_sequences": len(present),
            "missing_sequences": len(missing),
            "first_missing_examples": [str(path) for path in missing[:20]],
        },
        "required_model_and_transfer_assets": required_artifacts,
        "invariants": {
            "humanml3d_train_only": True,
            "mirrored_clips_do_not_duplicate_shape_identities": True,
            "humanact12_excluded_because_it_is_not_an_amass_shape_source": True,
            "direct_smplh_beta_to_smplx_copy_forbidden": True,
        },
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({
        "status": payload["status"],
        "canonical_train_entries": len(canonical_names),
        "required_amass_sequences": len(source_paths),
        "present_amass_sequences": len(present),
        "missing_amass_sequences": len(missing),
        "missing_required_assets": [
            key for key, item in required_artifacts.items() if not item["present"]
        ],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
