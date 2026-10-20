#!/usr/bin/env python3
"""Audit MGT training-token windows against original MMM and fair Inter-X.

This is a data/configuration audit.  It deliberately does not claim that a
window mismatch caused a metric difference; it establishes whether a matched
38-token control is required before attributing the gap to the loss or mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MMM_ROOT = Path("/home/wenning/projects/MMM")
ORIGINAL_MMM_DATASET = Path(
    "/home/wenning/projects/MMM/dataset/dataset_interx_TM.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
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


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_from_root(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def original_default_window(source: str) -> int:
    marker = "class Text2MotionDatasetV2HHI_Token"
    start = source.index(marker)
    fragment = source[start:start + 600]
    default_marker = "max_motion_length="
    value_start = fragment.index(default_marker) + len(default_marker)
    digits = []
    for character in fragment[value_start:]:
        if character.isdigit():
            digits.append(character)
        elif digits:
            break
    if not digits:
        raise ValueError("cannot resolve original MMM token-window default")
    return int("".join(digits))


def distribution(values: list[int]) -> dict:
    array = np.asarray(values, dtype=np.int64)
    counts = Counter(values)
    return {
        "count": int(array.size),
        "minimum": int(array.min()),
        "maximum": int(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "counts_30_through_60": {
            str(length): int(counts[length]) for length in range(30, 61)
        },
    }


def window_effect(lengths: list[int], window: int, downsample: int) -> dict:
    # The released loader reserves one position for <END>.
    payload_limit = window - 1
    effective = [min(length, payload_limit) for length in lengths]
    discarded = [max(0, length - payload_limit) for length in lengths]
    return {
        "tensor_window_including_end": int(window),
        "motion_payload_limit_tokens": int(payload_limit),
        "motion_payload_horizon_frames": int(payload_limit * downsample),
        "samples_truncated": int(sum(value > 0 for value in discarded)),
        "samples_truncated_fraction": float(
            sum(value > 0 for value in discarded) / len(discarded)
        ),
        "payload_tokens_discarded": int(sum(discarded)),
        "mean_effective_payload_tokens": float(np.mean(effective)),
        "maximum_effective_payload_tokens": int(max(effective)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument(
        "--requests",
        type=Path,
        default=ROOT / "output/fair_interx_requests/replication_00.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    resolved = manifest["resolved_arguments"]
    codebook_dir = resolve_from_root(resolved["codebook_dir"])
    train_split = resolve_from_root(resolved["train_split_file"])
    val_split = resolve_from_root(resolved["val_split_file"])
    downsample = 2 ** int(resolved.get("down_t", 2))

    split_names = {
        "train": read_split(train_split),
        "validation": read_split(val_split),
    }
    length_by_name = {}
    for split, names in split_names.items():
        for name in names:
            token_path = codebook_dir / f"{name}.npy"
            if not token_path.is_file():
                continue
            tokens = np.load(token_path, mmap_mode="r")
            if tokens.size % 3:
                raise ValueError(f"token cache is not three-part: {token_path}")
            length_by_name[name] = int(tokens.size // 3)
    expected_cached = int(manifest["cache_manifest"]["cached_code_samples"])
    if len(length_by_name) != expected_cached:
        raise ValueError(
            f"resolved {len(length_by_name)} cache rows, expected {expected_cached}"
        )

    original_source_bytes = ORIGINAL_MMM_DATASET.read_bytes()
    original_source = original_source_bytes.decode("utf-8")
    git_revision = subprocess.run(
        ["git", "-C", str(ORIGINAL_MMM_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    git_source_bytes = subprocess.run(
        [
            "git", "-C", str(ORIGINAL_MMM_ROOT), "show",
            f"{git_revision}:dataset/dataset_interx_TM.py",
        ],
        check=True, capture_output=True,
    ).stdout
    git_status = subprocess.run(
        [
            "git", "-C", str(ORIGINAL_MMM_ROOT), "status", "--short", "--",
            "dataset/dataset_interx_TM.py",
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    original_window = original_default_window(original_source)
    current_window = int(resolved["block_size"]) - 1
    lengths = list(length_by_name.values())

    with np.load(args.requests, allow_pickle=False) as requests:
        frame_lengths = requests["lengths"].astype(np.int64).tolist()
    requested_token_lengths = [
        int(math.ceil(frames / downsample)) for frames in frame_lengths
    ]

    split_stats = {}
    for split, names in split_names.items():
        split_lengths = [length_by_name[name] for name in names if name in length_by_name]
        split_stats[split] = {
            "source_rows": len(names),
            "cached_rows": len(split_lengths),
            "token_lengths": distribution(split_lengths),
            "original_38_effect": window_effect(
                split_lengths, original_window, downsample
            ),
            "current_50_effect": window_effect(
                split_lengths, current_window, downsample
            ),
        }

    checks = {
        "manifest_is_current_formal_paper_50": (
            resolved["training_token_length_contract"] == "paper_50"
            and current_window == 50
        ),
        "original_mmm_loader_default_is_38": original_window == 38,
        "original_mmm_dataset_matches_git_head_blob": (
            original_source_bytes == git_source_bytes and not git_status
        ),
        "fair_requests_are_at_most_150_frames": max(frame_lengths) <= 150,
        "fair_requests_require_at_most_38_motion_tokens": (
            max(requested_token_lengths) <= 38
        ),
        "cache_contains_sequences_beyond_current_payload_horizon": (
            max(lengths) > current_window - 1
        ),
        "current_window_exceeds_original_and_fair_horizon": (
            current_window > original_window
            and (current_window - 1) * downsample > max(frame_lengths)
        ),
    }
    payload = {
        "schema": "hhc_mgt_token_window_alignment_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "finding": (
            "A matched 38-token training-window control is required. The "
            "current formal run used a 50-position loader tensor (49 motion "
            "tokens plus END), while the original MMM loader default is 38 "
            "positions and the fair evaluator observes at most 150 frames."
        ),
        "causal_interpretation": (
            "configuration mismatch established; causal contribution to FID "
            "is unproven until a seed/data/mask/loss-matched 38-token control"
        ),
        "all_cached_token_lengths": distribution(lengths),
        "window_effects": {
            "original_mmm_38": window_effect(lengths, original_window, downsample),
            "current_formal_50": window_effect(lengths, current_window, downsample),
        },
        "split_statistics": split_stats,
        "fair_requests": {
            "samples": len(frame_lengths),
            "frame_lengths": distribution(frame_lengths),
            "motion_token_lengths_ceil": distribution(requested_token_lengths),
            "downsample": downsample,
        },
        "contracts": {
            "original_mmm_tensor_window_including_end": original_window,
            "current_formal_tensor_window_including_end": current_window,
            "current_manifest_token_length_contract": resolved[
                "training_token_length_contract"
            ],
            "loader_reserves_one_end_position": True,
            "original_mmm_git_revision": git_revision,
            "original_mmm_git_blob_sha256": sha256_bytes(git_source_bytes),
            "original_mmm_dataset_git_status": git_status,
        },
        "artifacts": {
            "training_manifest": artifact(args.training_manifest),
            "original_mmm_dataset_source": artifact(ORIGINAL_MMM_DATASET),
            "train_split": artifact(train_split),
            "validation_split": artifact(val_split),
            "requests": artifact(args.requests),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
