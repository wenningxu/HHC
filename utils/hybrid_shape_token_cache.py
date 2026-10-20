"""Hybrid cache helpers for a single-factor Inter-X shape augmentation.

Original train/validation samples must retain the byte-exact token and
contact arrays used by the recovered historical MMM run.  Only names created
by the augmentation materializer may be encoded with the deterministic
historical-root tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable


TOKEN_CACHE_CONTRACT = (
    "hybrid_historical_original_plus_deterministic_augmented_v1"
)
MATERIALIZATION_SCHEMA = "hhc_interx_shape_augmentation_materialization_v1"


def _unique_names(values: Iterable[str], label: str) -> set[str]:
    names = list(values)
    if any(
        not isinstance(name, str) or not name or name.strip() != name
        for name in names
    ):
        raise ValueError(f"{label} contains an invalid sample name")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate sample names")
    return set(names)


def partition_expected_names(
    expected_names: Iterable[str], materialization_manifest: str | Path,
) -> tuple[set[str], set[str], dict]:
    """Partition eligible cache names using the bound materialization manifest."""
    path = Path(materialization_manifest).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MATERIALIZATION_SCHEMA:
        raise ValueError("unexpected shape materialization schema")
    if manifest.get("status") != "complete":
        raise ValueError("shape materialization is not complete")
    records = manifest.get("asset_records")
    if not isinstance(records, list) or not records:
        raise ValueError("shape materialization has no asset records")

    expected = _unique_names(expected_names, "expected cache names")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("shape materialization contains a malformed asset record")
    augmented = _unique_names(
        (record.get("augmented_name") for record in records),
        "materialized augmented names",
    )
    unexpected_augmented = sorted(augmented - expected)
    if unexpected_augmented:
        raise ValueError(
            "materialization contains augmented names outside the eligible "
            f"cache scope: {unexpected_augmented[:5]}"
        )
    original = expected - augmented
    if not original:
        raise ValueError("hybrid cache has no original samples")
    if not augmented:
        raise ValueError("hybrid cache has no augmented samples")
    if original & augmented or original | augmented != expected:
        raise AssertionError("hybrid cache partition is not exact")
    return original, augmented, manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_named_files(directory: str | Path, names: Iterable[str]) -> str:
    """Hash names and bytes with the same framing used by MGT manifests."""
    root = Path(directory)
    digest = hashlib.sha256()
    for name in sorted(set(names)):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with (root / f"{name}.npy").open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def import_original_pairs_byte_exact(
    original_names: Iterable[str],
    historical_code_dir: str | Path,
    historical_contact_dir: str | Path,
    destination_code_dir: str | Path,
    destination_contact_dir: str | Path,
) -> int:
    """Atomically import original pairs, refusing conflicting resume files."""
    names = sorted(set(original_names))
    code_source = Path(historical_code_dir).resolve()
    contact_source = Path(historical_contact_dir).resolve()
    code_destination = Path(destination_code_dir).resolve()
    contact_destination = Path(destination_contact_dir).resolve()

    sources: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for name in names:
        code = code_source / f"{name}.npy"
        contact = contact_source / f"{name}.npy"
        if not code.is_file():
            missing.append(f"code:{name}")
        if not contact.is_file():
            missing.append(f"contact:{name}")
        sources.extend(((code, code_destination / code.name),
                        (contact, contact_destination / contact.name)))
    if missing:
        raise FileNotFoundError(
            "historical MMM cache is incomplete: " + ", ".join(missing[:10])
        )

    code_destination.mkdir(parents=True, exist_ok=True)
    contact_destination.mkdir(parents=True, exist_ok=True)
    for source, destination in sources:
        if destination.exists():
            if _sha256_file(destination) != _sha256_file(source):
                raise RuntimeError(
                    f"existing hybrid cache file conflicts with MMM: {destination}"
                )
            continue
        temporary = destination.with_name(
            destination.name + f".tmp.{os.getpid()}"
        )
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    return len(names)


def audit_hybrid_cache_subsets(
    original_names: Iterable[str],
    augmented_names: Iterable[str],
    historical_code_dir: str | Path,
    historical_contact_dir: str | Path,
    cache_code_dir: str | Path,
    cache_contact_dir: str | Path,
) -> dict:
    """Prove exact original reuse and complete, disjoint augmented coverage."""
    original = set(original_names)
    augmented = set(augmented_names)
    if not original or not augmented or original & augmented:
        raise ValueError("hybrid original/augmented partitions are invalid")

    cache_code = Path(cache_code_dir)
    cache_contact = Path(cache_contact_dir)
    observed_code = {path.stem for path in cache_code.glob("*.npy")}
    observed_contact = {path.stem for path in cache_contact.glob("*.npy")}
    expected = original | augmented
    if observed_code != expected or observed_contact != expected:
        raise RuntimeError(
            "hybrid cache scope mismatch: "
            f"code_missing={sorted(expected - observed_code)[:5]}, "
            f"code_extra={sorted(observed_code - expected)[:5]}, "
            f"contact_missing={sorted(expected - observed_contact)[:5]}, "
            f"contact_extra={sorted(observed_contact - expected)[:5]}"
        )

    source_code_hash = sha256_named_files(historical_code_dir, original)
    source_contact_hash = sha256_named_files(historical_contact_dir, original)
    cache_original_code_hash = sha256_named_files(cache_code, original)
    cache_original_contact_hash = sha256_named_files(cache_contact, original)
    checks = {
        "partitions_are_nonempty_disjoint_and_complete": (
            bool(original) and bool(augmented) and not (original & augmented)
        ),
        "original_code_cache_is_byte_exact_mmm": (
            cache_original_code_hash == source_code_hash
        ),
        "original_contact_cache_is_byte_exact_mmm": (
            cache_original_contact_hash == source_contact_hash
        ),
        "cache_scope_is_exact": (
            observed_code == expected and observed_contact == expected
        ),
    }
    return {
        "schema": "hhc_hybrid_shape_token_cache_subset_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "token_cache_contract": TOKEN_CACHE_CONTRACT,
        "checks": checks,
        "counts": {
            "original": len(original),
            "augmented": len(augmented),
            "total": len(expected),
        },
        "hashes": {
            "historical_original_code": source_code_hash,
            "cached_original_code": cache_original_code_hash,
            "historical_original_contact": source_contact_hash,
            "cached_original_contact": cache_original_contact_hash,
            "cached_augmented_code": sha256_named_files(cache_code, augmented),
            "cached_augmented_contact": sha256_named_files(
                cache_contact, augmented
            ),
        },
    }


__all__ = [
    "TOKEN_CACHE_CONTRACT",
    "audit_hybrid_cache_subsets",
    "import_original_pairs_byte_exact",
    "partition_expected_names",
    "sha256_named_files",
]
