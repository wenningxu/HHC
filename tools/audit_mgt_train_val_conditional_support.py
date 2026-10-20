#!/usr/bin/env python3
"""Audit train/validation support for MGT cascade-conditioning tuples.

The cascade predicts body tokens after root, then hand tokens after root and
body.  A near-complete marginal code vocabulary can therefore hide a large
shift in the conditional tuples actually presented to those stages.  This
diagnostic is data-only and never reads the test split or model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
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


def load_split(directory: Path, split: Path, num_codes: int):
    records = []
    missing = []
    for name in split_names(split):
        path = directory / f"{name}.npy"
        if not path.is_file():
            missing.append(name)
            continue
        tokens = np.load(path, allow_pickle=False)
        if tokens.size % 3:
            raise ValueError(f"{path}: token count is not divisible by three")
        tokens = tokens.reshape(-1, 3).astype(np.int64, copy=False)
        if not tokens.size or tokens.min() < 0 or tokens.max() >= num_codes:
            raise ValueError(f"{path}: empty or out-of-range token sequence")
        records.append((name, tokens))
    return records, missing


def apply_training_loader_contract(
        records, minimum_payload_tokens: int, maximum_payload_tokens: int):
    """Mirror the active token dataset's bucket filter and payload truncation."""
    if minimum_payload_tokens <= 0 or maximum_payload_tokens < minimum_payload_tokens:
        raise ValueError("invalid MGT token-loader payload bounds")
    active = []
    excluded_below_bucket = []
    truncated = 0
    for name, tokens in records:
        if len(tokens) < minimum_payload_tokens:
            excluded_below_bucket.append(name)
            continue
        truncated += int(len(tokens) > maximum_payload_tokens)
        active.append((name, tokens[:maximum_payload_tokens]))
    return active, excluded_below_bucket, truncated


def token_window_contract(resolved: dict) -> tuple[int, int, str]:
    contract = resolved.get("training_token_length_contract")
    tensor_windows = {
        "paper_50": 50,
        "original_mmm_38": 38,
        "historical_38": 38,
    }
    if contract not in tensor_windows:
        raise ValueError(f"unsupported token-length contract: {contract!r}")
    tensor_window = tensor_windows[contract]
    if contract == "paper_50" and int(resolved.get("block_size", -1)) != 51:
        raise ValueError("paper_50 manifest does not declare block_size=51")
    return tensor_window, tensor_window - 1, contract


def tuple_counter(records, channels: tuple[int, ...]) -> Counter:
    counter = Counter()
    for _, tokens in records:
        counter.update(map(tuple, tokens[:, channels].tolist()))
    return counter


def transition_counter(records, channel: int) -> Counter:
    counter = Counter()
    for _, tokens in records:
        values = tokens[:, channel]
        if len(values) > 1:
            counter.update(zip(values[:-1].tolist(), values[1:].tolist()))
    return counter


def support_summary(train: Counter, validation: Counter) -> dict:
    total = sum(validation.values())
    seen = sum(count for key, count in validation.items() if key in train)
    return {
        "training_unique": len(train),
        "validation_unique": len(validation),
        "validation_occurrences": total,
        "validation_occurrences_seen_in_training": seen,
        "validation_occurrence_coverage": seen / total if total else 0.0,
        "validation_unique_seen_in_training": sum(key in train for key in validation),
        "validation_unique_coverage": (
            sum(key in train for key in validation) / len(validation)
            if validation else 0.0
        ),
    }


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
    args = parser.parse_args()

    manifest = json.loads(args.training_manifest.read_text())
    resolved = manifest["resolved_arguments"]
    directory = Path(resolved["codebook_dir"]).resolve()
    train_split = Path(resolved["train_split_file"]).resolve()
    val_split = Path(resolved["val_split_file"]).resolve()
    num_codes = int(resolved["nb_code"])
    train_cached, train_missing = load_split(directory, train_split, num_codes)
    val_cached, val_missing = load_split(directory, val_split, num_codes)

    cache = manifest.get("cache_manifest", {})
    expected_train = int(cache.get("eligible_train_samples", len(train_cached)))
    expected_val = int(cache.get("eligible_val_samples", len(val_cached)))
    if len(train_cached) != expected_train or len(val_cached) != expected_val:
        raise RuntimeError({
            "loaded_train": len(train_cached), "expected_train": expected_train,
            "loaded_validation": len(val_cached), "expected_validation": expected_val,
            "training_missing": train_missing, "validation_missing": val_missing,
        })
    tensor_window, maximum_payload_tokens, length_contract = (
        token_window_contract(resolved)
    )
    minimum_payload_tokens = 20
    train, train_below_bucket, train_truncated = apply_training_loader_contract(
        train_cached, minimum_payload_tokens, maximum_payload_tokens
    )
    val, val_below_bucket, val_truncated = apply_training_loader_contract(
        val_cached, minimum_payload_tokens, maximum_payload_tokens
    )

    tuple_specs = {
        "root_to_body_pair": (2, 0),
        "root_body_to_hand_triplet": (2, 0, 1),
        "body_hand_pair": (0, 1),
        "all_part_triplet": (0, 1, 2),
    }
    tuples = {
        name: support_summary(
            tuple_counter(train, channels), tuple_counter(val, channels)
        )
        for name, channels in tuple_specs.items()
    }
    transitions = {
        part: support_summary(
            transition_counter(train, channel), transition_counter(val, channel)
        )
        for channel, part in enumerate(PARTS)
    }
    marginal = {
        part: support_summary(
            tuple_counter(train, (channel,)), tuple_counter(val, (channel,))
        )
        for channel, part in enumerate(PARTS)
    }
    payload = {
        "schema": "hhc_mgt_train_val_conditional_support_audit_v1",
        "status": "complete",
        "purpose": (
            "diagnose cascade conditional-combination shift without using test data"
        ),
        "interpretation_policy": (
            "Low tuple coverage with high marginal coverage supports a conditional "
            "composition/generalization gap rather than a code-ID mismatch."
        ),
        "training_manifest": {
            "path": str(args.training_manifest.resolve()),
            "sha256": sha256_file(args.training_manifest),
        },
        "codebook_directory": str(directory),
        "loader_contract": {
            "training_token_length_contract": length_contract,
            "tensor_window_including_end": tensor_window,
            "minimum_payload_tokens": minimum_payload_tokens,
            "maximum_payload_tokens": maximum_payload_tokens,
            "filter_then_truncate": True,
        },
        "training_split": {
            "path": str(train_split), "sha256": sha256_file(train_split),
            "cached_samples": len(train_cached),
            "loaded_samples": len(train), "excluded_rows": train_missing,
            "excluded_below_training_bucket": train_below_bucket,
            "truncated_to_payload_window": train_truncated,
        },
        "validation_split": {
            "path": str(val_split), "sha256": sha256_file(val_split),
            "cached_samples": len(val_cached),
            "loaded_samples": len(val), "excluded_rows": val_missing,
            "excluded_below_training_bucket": val_below_bucket,
            "truncated_to_payload_window": val_truncated,
        },
        "marginal_code_support": marginal,
        "same_frame_conditional_tuple_support": tuples,
        "within_part_transition_support": transitions,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
