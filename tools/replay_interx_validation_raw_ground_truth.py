#!/usr/bin/env python3
"""Replay the exact frozen Inter-X validation loader into raw 672D motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import dataset_interx_TM
from utils.motion_process import MotionNormalizerTorchInterX
from utils.word_vectorizer import WordVectorizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request-manifest", type=Path,
        default=Path("output/interx_official_val_requests/manifest.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/interx_validation_raw_ground_truth.h5"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("output/interx_validation_raw_ground_truth_manifest.json"),
    )
    args = parser.parse_args()

    request_manifest_path = args.request_manifest.resolve()
    request_manifest = json.loads(request_manifest_path.read_text())
    if (request_manifest.get("schema")
            != "hhc_interx_method_neutral_validation_request_v1"
            or request_manifest.get("status") != "complete"
            or request_manifest.get("batch_size") != 32
            or request_manifest.get("purpose") != (
                "held-out official Inter-X checkpoint selection for all "
                "methods; never a reported test row"
            )):
        raise ValueError("invalid frozen validation request manifest")
    request_path = Path(request_manifest["request"]["path"]).resolve()
    if artifact(request_path) != request_manifest["request"]:
        raise ValueError("frozen validation request is stale")
    source_impl = Path(
        request_manifest["request_generation_implementation"]["path"]
    )
    if artifact(source_impl) != request_manifest["request_generation_implementation"]:
        raise ValueError("validation-loader implementation changed since freezing")

    request = np.load(request_path, allow_pickle=False)
    expected_names = request["names"].astype(str)
    expected_captions = request["captions"].astype(str)
    expected_lengths = request["lengths"].astype(np.int64)
    seed = int(request["seed"])

    split = Path(request_manifest["split"]["path"])
    motion_file = Path(request_manifest["motions"]["path"])
    if artifact(split) != request_manifest["split"]:
        raise ValueError("validation split is stale")
    if artifact(motion_file) != request_manifest["motions"]:
        raise ValueError("validation motion file is stale")

    seed_everything(seed)
    vectorizer = WordVectorizer(str(ROOT / "dataset/interx/glove"), "hhi_vab")
    loader = dataset_interx_TM.DATALoader(
        "hhi", 32, str(split), vectorizer, str(motion_file),
        unit_length=4, shuffle=True,
    )
    normalizer = MotionNormalizerTorchInterX("raw")
    # Match the freeze script's reset immediately before DataLoader iteration.
    seed_everything(seed)

    names_out, captions_out, lengths_out = [], [], []
    raw_out, betas_out, gender_out, contact_out = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            (_, _, captions, _, pose, lengths, _, names,
             betas, gender, contact) = batch
            raw_pose = normalizer.backward(pose.float()).cpu().numpy()
            order = np.argsort(lengths.numpy().tolist())[::-1].copy()
            names_out.extend(np.asarray(names)[order].tolist())
            captions_out.extend(np.asarray(captions)[order].tolist())
            lengths_out.extend(lengths.numpy()[order].tolist())
            raw_out.append(raw_pose[order])
            betas_out.append(betas.numpy()[order])
            gender_out.append(gender.numpy()[order])
            contact_out.append(contact.numpy()[order])

    actual_names = np.asarray(names_out).astype(str)
    actual_captions = np.asarray(captions_out).astype(str)
    actual_lengths = np.asarray(lengths_out, dtype=np.int64)
    raw = np.concatenate(raw_out).astype(np.float32)
    actual_betas = np.concatenate(betas_out).astype(np.float32)
    actual_gender = np.concatenate(gender_out).astype(np.int64)
    actual_contact = np.concatenate(contact_out).astype(np.int8)
    mismatches = {
        "names": not np.array_equal(actual_names, expected_names),
        "captions": not np.array_equal(actual_captions, expected_captions),
        "lengths": not np.array_equal(actual_lengths, expected_lengths),
        "betas": not np.array_equal(actual_betas, request["betas"]),
        "gender": not np.array_equal(actual_gender, request["gender"]),
        "contact": not np.array_equal(actual_contact, request["contact"]),
    }
    if any(mismatches.values()):
        raise ValueError(f"validation replay differs from frozen request: {mismatches}")
    if raw.shape != (len(expected_names), 150, 672) or not np.isfinite(raw).all():
        raise ValueError(f"invalid replayed raw motion {raw.shape}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    with h5py.File(temporary, "w") as target:
        # This artifact is read repeatedly during checkpoint selection. LZF
        # preserves the exact float32 payload while avoiding gzip's large
        # one-time CPU cost for only modest extra disk use (~220 MB raw).
        target.create_dataset("motions", data=raw[:, None], compression="lzf")
        target.create_dataset(
            "names", data=actual_names.tolist(),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        target.create_dataset("lengths", data=actual_lengths)
        target.attrs["requests_sha256"] = sha256_file(request_path)
        target.attrs["request_manifest_sha256"] = sha256_file(
            request_manifest_path
        )
        target.attrs["seed"] = seed
        target.attrs["representation"] = "raw_interx_150x672"
        target.attrs["compression"] = "lzf_lossless"
    os.replace(temporary, output)

    payload = {
        "schema": "hhc_interx_validation_raw_ground_truth_v1",
        "status": "complete",
        "samples": len(expected_names),
        "seed": seed,
        "request_manifest": artifact(request_manifest_path),
        "request": artifact(request_path),
        "raw_ground_truth": artifact(output),
        "source_split": artifact(split),
        "source_motions": artifact(motion_file),
        "freeze_implementation": artifact(source_impl),
        "replay_implementation": artifact(Path(__file__)),
        "exact_replay_checks": {key: not value for key, value in mismatches.items()},
    }
    atomic_json(args.manifest.resolve(), payload)
    print(json.dumps({
        "status": "complete", "samples": len(expected_names),
        "output": str(output), "manifest": str(args.manifest.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
