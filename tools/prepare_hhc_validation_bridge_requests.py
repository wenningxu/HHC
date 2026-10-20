#!/usr/bin/env python3
"""Freeze the exact sample order used by HHC's in-training validation.

The shared validation request is intentionally suitable for cross-method
checkpoint selection, but its samples are stored in evaluator (motion-length)
order.  HHC's training hook samples in the DataLoader's collated order before
the evaluator sorts embeddings.  With a stochastic sampler, changing that
within-batch order assigns the same CUDA random draws to different samples and
prevents an exact replay of the recorded validation FID.

This tool only permutes the already frozen, byte-bound validation request back
to the audited training-loader order.  It does not change the sample set,
captions, crops, shapes, contact timing, or evaluator embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PER_SAMPLE_KEYS = (
    "names", "captions", "lengths", "text_embeddings", "gt_embeddings",
    "betas", "gender", "contact",
)
ORDERING_CONTRACT = (
    "exact_hhc_training_validation_loader_batch_and_within_batch_order"
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
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def ordered_records_sha256(records: list[dict]) -> str:
    serialized = json.dumps(
        records, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_exact_order_payload(source: dict[str, np.ndarray], selection: dict):
    records = selection.get("records", [])
    expected_samples = int(selection.get("evaluated_samples", -1))
    if expected_samples <= 0 or len(records) != expected_samples:
        raise ValueError("training validation selection has an invalid record count")
    if selection.get("ordered_request_records_sha256") != ordered_records_sha256(records):
        raise ValueError("training validation record order hash is stale")

    missing = set(PER_SAMPLE_KEYS) - set(source)
    if missing:
        raise ValueError(f"shared validation request lacks arrays: {sorted(missing)}")
    source_names = source["names"].astype(str).tolist()
    if len(source_names) != expected_samples or len(set(source_names)) != len(source_names):
        raise ValueError("shared validation request names are not one-to-one")
    source_index = {name: index for index, name in enumerate(source_names)}
    ordered_names = [str(record["name"]) for record in records]
    if len(set(ordered_names)) != len(ordered_names):
        raise ValueError("training validation selection contains duplicate names")
    if set(ordered_names) != set(source_names):
        raise ValueError("shared and training validation requests use different samples")

    indices = np.asarray([source_index[name] for name in ordered_names], dtype=np.int64)
    output = {key: np.asarray(source[key])[indices] for key in PER_SAMPLE_KEYS}
    for position, record in enumerate(records):
        if (
            str(output["names"][position]) != str(record["name"])
            or str(output["captions"][position]) != str(record["caption"])
            or int(output["lengths"][position]) != int(record["length"])
        ):
            raise ValueError(
                f"shared request fields differ from training record at {position}"
            )
        expected_batch = position // int(selection["batch_size"])
        if int(record["batch"]) != expected_batch:
            raise ValueError("training validation records are not contiguous full batches")
    output["mm_indices"] = np.arange(min(100, expected_samples), dtype=np.int64)
    output["seed"] = np.asarray(int(selection["seed"]), dtype=np.int64)
    output["source_common_request_indices"] = indices
    return output, indices


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--common-request-manifest", type=Path,
        default=Path("output/coshmdm_interx_val_requests/manifest.json"),
    )
    parser.add_argument(
        "--training-selection", type=Path,
        default=Path("output/mgt_validation_selection_manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("output/hhc_mgt_validation_bridge_requests"),
    )
    args = parser.parse_args()

    common_manifest_path = args.common_request_manifest.resolve()
    selection_path = args.training_selection.resolve()
    common_manifest = json.loads(common_manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        common_manifest.get("schema") != "hhc_interx_validation_request_v1"
        or common_manifest.get("status") != "complete"
        or common_manifest.get("samples") != 544
        or common_manifest.get("seed") != 500123
    ):
        raise ValueError("shared validation request manifest is incomplete")
    if (
        selection.get("schema") != "hhc_mgt_validation_selection_v1"
        or selection.get("status") != "pass"
        or selection.get("evaluated_samples") != 544
        or selection.get("batch_size") != 32
        or selection.get("drop_last") is not True
        or selection.get("shuffle") is not True
        or selection.get("seed") != 500123
        or selection.get("repeat_deterministic") is not True
    ):
        raise ValueError("training validation ordering audit is incomplete")
    common_request_record = common_manifest.get("request", {})
    common_request_path = Path(common_request_record.get("path", "")).resolve()
    if (
        not common_request_path.is_file()
        or sha256_file(common_request_path) != common_request_record.get("sha256")
    ):
        raise ValueError("shared validation request hash changed")
    bound_selection = common_manifest.get("formal_hhc_validation_selection", {})
    if (
        Path(bound_selection.get("path", "")).resolve() != selection_path
        or bound_selection.get("sha256") != sha256_file(selection_path)
    ):
        raise ValueError("shared validation request binds another record selection")

    with np.load(common_request_path, allow_pickle=False) as source_file:
        source = {key: source_file[key] for key in source_file.files}
    output, indices = build_exact_order_payload(source, selection)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "validation.npz"
    temporary = request_path.with_suffix(f".npz.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **output)
    os.replace(temporary, request_path)

    identity = np.arange(len(indices), dtype=np.int64)
    payload = {
        "schema": "hhc_mgt_validation_bridge_request_v1",
        "status": "complete",
        "purpose": "exact replay of HHC in-training validation only",
        "test_split_used": False,
        "samples": 544,
        "batch_size": 32,
        "seed": 500123,
        "sampling_order_contract": ORDERING_CONTRACT,
        "ordered_sampling_records_match_training_selection": True,
        "same_record_set_as_common_validation_request": True,
        "common_request_was_not_already_in_training_sampling_order": bool(
            not np.array_equal(indices, identity)
        ),
        "permutation_fixed_points": int((indices == identity).sum()),
        "source_common_request": artifact(common_request_path),
        "source_common_request_manifest": artifact(common_manifest_path),
        "formal_hhc_validation_selection": artifact(selection_path),
        "request": artifact(request_path),
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(output_dir / "manifest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
