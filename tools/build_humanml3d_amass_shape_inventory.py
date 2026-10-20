#!/usr/bin/env python3
"""Build a leakage-free SMPL+H shape inventory from HumanML3D-train AMASS files.

This is the input inventory for the *official* SMPL+H -> SMPL-X transfer.  It
deliberately does not reinterpret SMPL+H beta coefficients as SMPL-X betas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_gender(value: object) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"gender must be scalar, got shape {array.shape}")
    scalar = array.reshape(()).item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    gender = str(scalar).strip().lower()
    aliases = {"m": "male", "f": "female"}
    gender = aliases.get(gender, gender)
    if gender not in {"male", "female"}:
        raise ValueError(f"unsupported AMASS gender {gender!r}")
    return gender


def source_to_amass_path(source: str, amass_root: Path) -> Path:
    prefix = "./pose_data/"
    if not source.startswith(prefix) or not source.endswith(".npy"):
        raise ValueError(f"unexpected HumanML3D source path: {source}")
    return amass_root / (source[len(prefix):-4] + ".npz")


def beta_identity(gender: str, betas: np.ndarray) -> str:
    # Hash canonical little-endian float64 bytes.  No rounding is used: AMASS
    # clips from the same fitted identity normally carry exactly equal betas.
    canonical = np.asarray(betas, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(gender.encode("ascii") + b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def load_train_sources(humanml3d_root: Path, amass_root: Path) -> tuple[list[Path], dict]:
    index_path = humanml3d_root / "index.csv"
    train_path = humanml3d_root / "HumanML3D/train.txt"
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = {Path(row["new_name"]).stem: row for row in csv.DictReader(handle)}
    train_names = [line.strip() for line in train_path.read_text().splitlines()
                   if line.strip()]
    canonical_names = sorted({name[1:] if name.startswith("M") else name
                              for name in train_names})
    absent = [name for name in canonical_names if name not in rows]
    if absent:
        raise ValueError(f"HumanML3D train ids absent from index.csv: {absent[:10]}")

    source_paths = set()
    excluded_humanact12 = 0
    for name in canonical_names:
        source = rows[name]["source_path"]
        if "humanact12" in source.lower():
            excluded_humanact12 += 1
            continue
        source_paths.add(source_to_amass_path(source, amass_root).resolve())
    metadata = {
        "train_entries_including_mirrors": len(train_names),
        "canonical_train_entries": len(canonical_names),
        "humanact12_entries_excluded": excluded_humanact12,
        "unique_amass_source_sequences": len(source_paths),
        "index_sha256": sha256_file(index_path),
        "train_split_sha256": sha256_file(train_path),
    }
    return sorted(source_paths), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--humanml3d-root", type=Path,
                        default=Path("/mnt/8T-HDD/HumanML3D"))
    parser.add_argument("--amass-root", type=Path,
                        default=Path("/mnt/8T-HDD/HumanML3D/amass_data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-smplh-betas", type=int, default=16)
    args = parser.parse_args()
    if args.num_smplh_betas != 16:
        raise ValueError("AMASS SMPL+H G donor contract requires exactly 16 betas")

    humanml3d_root = args.humanml3d_root.resolve()
    amass_root = args.amass_root.resolve()
    output_dir = args.output_dir.resolve()
    source_paths, split_metadata = load_train_sources(humanml3d_root, amass_root)
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(source_paths)} HumanML3D-train AMASS files are missing; "
            f"first missing: {missing[0]}"
        )

    identities: dict[str, dict] = {}
    identity_sources: dict[str, list[str]] = defaultdict(list)
    rejected = []
    for source_path in source_paths:
        try:
            with np.load(source_path, allow_pickle=False) as archive:
                if "betas" not in archive or "gender" not in archive:
                    raise KeyError("AMASS archive must contain betas and gender")
                raw_betas = np.asarray(archive["betas"], dtype=np.float64).reshape(-1)
                if raw_betas.size < args.num_smplh_betas:
                    raise ValueError(f"expected >=16 betas, got {raw_betas.size}")
                betas = raw_betas[:args.num_smplh_betas]
                if not np.isfinite(betas).all():
                    raise ValueError("non-finite betas")
                gender = normalize_gender(archive["gender"])
            identity_id = beta_identity(gender, betas)
            identities.setdefault(identity_id, {
                "gender": gender,
                "betas": betas,
                "representative_source": str(source_path),
            })
            identity_sources[identity_id].append(str(source_path))
        except (KeyError, ValueError, OSError) as error:
            rejected.append({"source": str(source_path), "reason": str(error)})

    if rejected:
        preview = "; ".join(f"{item['source']}: {item['reason']}"
                            for item in rejected[:3])
        raise ValueError(f"rejected {len(rejected)} AMASS sources: {preview}")
    if not identities:
        raise ValueError("no valid AMASS identities found")

    ordered_ids = sorted(identities)
    betas = np.stack([identities[key]["betas"] for key in ordered_ids]).astype(np.float32)
    genders = np.asarray([identities[key]["gender"] for key in ordered_ids], dtype="U6")
    counts = np.asarray([len(identity_sources[key]) for key in ordered_ids], dtype=np.int32)
    atomic_npz(output_dir / "smplh_shape_inventory.npz",
               identity_id=np.asarray(ordered_ids, dtype="U64"),
               gender=genders, betas=betas, source_sequence_count=counts)
    inventory_artifact = {
        "path": str(output_dir / "smplh_shape_inventory.npz"),
        "bytes": (output_dir / "smplh_shape_inventory.npz").stat().st_size,
        "sha256": sha256_file(output_dir / "smplh_shape_inventory.npz"),
    }

    records = []
    for key in ordered_ids:
        record = identities[key]
        records.append({
            "identity_id": key,
            "gender": record["gender"],
            "representative_source": record["representative_source"],
            "source_sequence_count": len(identity_sources[key]),
            "sources": sorted(identity_sources[key]),
        })
    manifest = {
        "schema": "hhc_humanml3d_amass_smplh_shape_inventory_v1",
        "status": "ready_for_official_smplh_to_smplx_transfer",
        "invariants": {
            "humanml3d_train_only": True,
            "mirrors_deduplicated": True,
            "humanact12_excluded": True,
            "smplh_betas_are_not_smplx_betas": True,
            "deduplication": "exact gender plus canonical float64 first-16 beta bytes",
        },
        "source": {
            "humanml3d_root": str(humanml3d_root),
            "amass_root": str(amass_root),
            **split_metadata,
        },
        "inventory": {
            "npz": inventory_artifact,
            "identity_count": len(ordered_ids),
            "gender_counts": dict(sorted(Counter(genders.tolist()).items())),
            "source_sequence_count": len(source_paths),
            "beta_dimension": int(betas.shape[1]),
            "identity_records": records,
        },
    }
    atomic_json(output_dir / "smplh_shape_inventory_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "identity_count": len(ordered_ids),
        "gender_counts": manifest["inventory"]["gender_counts"],
        "manifest": str(output_dir / "smplh_shape_inventory_manifest.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
