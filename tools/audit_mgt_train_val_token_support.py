#!/usr/bin/env python3
"""Audit whether held-out MGT tokens are outside the training code support.

This is a cache/data diagnostic only.  It never decodes motion, evaluates a
generation, or uses the test split.  A tiny unseen-token fraction rules out a
train/validation code-ID mismatch as an explanation for a large denoising gap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


PARTS = ("body", "hand", "root")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_tokens(
    directory: Path, names: list[str], num_codes: int
) -> tuple[list[np.ndarray], list[str]]:
    arrays, missing = [], []
    for name in names:
        path = directory / f"{name}.npy"
        if not path.is_file():
            missing.append(name)
            continue
        values = np.load(path, allow_pickle=False)
        if values.size % 3:
            raise ValueError(f"{path}: token count is not divisible by three")
        values = values.reshape(-1, 3)
        if values.size and (values.min() < 0 or values.max() >= num_codes):
            raise ValueError(f"{path}: token ID outside [0, {num_codes})")
        arrays.append(values)
    return arrays, missing


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-unseen-token-fraction", type=float, default=0.001)
    args = parser.parse_args()

    manifest = json.loads(args.training_manifest.read_text())
    resolved = manifest["resolved_arguments"]
    directory = Path(resolved["codebook_dir"]).resolve()
    num_codes = int(resolved["nb_code"])
    train_split = Path(resolved["train_split_file"]).resolve()
    val_split = Path(resolved["val_split_file"]).resolve()
    train, train_missing = load_tokens(
        directory, split_names(train_split), num_codes
    )
    val, val_missing = load_tokens(directory, split_names(val_split), num_codes)
    cache_manifest = manifest.get("cache_manifest", {})
    expected_train = int(cache_manifest.get("eligible_train_samples", len(train)))
    expected_val = int(cache_manifest.get("eligible_val_samples", len(val)))
    if len(train) != expected_train or len(val) != expected_val:
        raise RuntimeError({
            "loaded_train": len(train), "expected_train": expected_train,
            "loaded_validation": len(val), "expected_validation": expected_val,
            "train_missing": train_missing, "validation_missing": val_missing,
        })
    if not train or not val:
        raise RuntimeError("empty train or validation token cache")

    parts = {}
    failures = []
    for channel, part in enumerate(PARTS):
        train_values = np.concatenate([values[:, channel] for values in train])
        val_values = np.concatenate([values[:, channel] for values in val])
        train_unique = np.unique(train_values)
        val_unique = np.unique(val_values)
        unseen_ids = np.setdiff1d(val_unique, train_unique)
        unseen_mask = np.isin(val_values, unseen_ids)
        fraction = float(unseen_mask.mean())
        parts[part] = {
            "train_tokens": int(train_values.size),
            "validation_tokens": int(val_values.size),
            "train_unique_codes": int(train_unique.size),
            "validation_unique_codes": int(val_unique.size),
            "validation_unique_codes_unseen_in_train": int(unseen_ids.size),
            "validation_tokens_unseen_in_train": int(unseen_mask.sum()),
            "validation_unseen_token_fraction": fraction,
            "unseen_code_ids": unseen_ids.astype(int).tolist(),
        }
        if fraction > args.maximum_unseen_token_fraction:
            failures.append({"part": part, "fraction": fraction})

    payload = {
        "schema": "hhc_mgt_train_val_token_support_audit_v1",
        "status": "fail" if failures else "pass",
        "interpretation": (
            "Validation code support is covered by training; unseen IDs cannot "
            "explain a large train/validation denoising gap."
            if not failures else
            "Validation contains a material fraction of token IDs unseen in training."
        ),
        "training_manifest": {
            "path": str(args.training_manifest.resolve()),
            "sha256": sha256_file(args.training_manifest),
        },
        "codebook_directory": str(directory),
        "train_split": {"path": str(train_split), "sha256": sha256_file(train_split)},
        "validation_split": {"path": str(val_split), "sha256": sha256_file(val_split)},
        "train_samples": len(train),
        "validation_samples": len(val),
        "split_rows_excluded_by_frozen_cache_eligibility": {
            "training": train_missing,
            "validation": val_missing,
        },
        "maximum_unseen_token_fraction": args.maximum_unseen_token_fraction,
        "parts": parts,
        "threshold_failures": failures,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
