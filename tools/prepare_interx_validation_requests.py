#!/usr/bin/env python3
"""Freeze one held-out Inter-X validation request for baseline selection.

This is deliberately separate from the 20 frozen test replications.  It uses
the same official evaluator/data path, but its only role is selecting among
checkpoints produced inside the user-capped CoShMDM training window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dataset import dataset_interx_TM
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
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
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/coshmdm_interx_val_requests"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=500123)
    parser.add_argument("--expected-samples", type=int, default=544)
    args = parser.parse_args()

    split = ROOT / "dataset/interx/splits/val.txt"
    motions = ROOT / "dataset/interx/motions/val.h5"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "validation.npz"
    manifest_path = output_dir / "manifest.json"

    seed_everything(args.seed)
    vectorizer = WordVectorizer(str(ROOT / "dataset/interx/glove"), "hhi_vab")
    loader = dataset_interx_TM.DATALoader(
        "hhi", 32, str(split), vectorizer, str(motions),
        unit_length=4, shuffle=True,
    )
    evaluator = EvaluatorModelWrapper(
        get_opt(str(ROOT / "checkpoints/hhi/Comp_v6_KLD01/opt.txt"), args.device)
    )
    normalizer = MotionNormalizerTorchInterX("raw")

    # Model construction initializes random tensors before loading weights and
    # therefore consumes the global PyTorch RNG.  The formal HHC validation
    # resets its seed immediately before creating the DataLoader iterator; do
    # the same here so shuffle/drop_last retains the identical 544 examples.
    seed_everything(args.seed)

    names_out, captions_out, lengths_out = [], [], []
    text_out, gt_out, betas_out, gender_out, contact_out = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            (word_embeddings, pos_one_hots, captions, sent_len, pose, lengths,
             _, names, betas, gender, contact) = batch
            raw_pose = normalizer.backward(pose.float())
            text_embedding, gt_embedding = evaluator.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, raw_pose, lengths
            )
            order = np.argsort(lengths.numpy().tolist())[::-1].copy()
            names_out.extend(np.asarray(names)[order].tolist())
            captions_out.extend(np.asarray(captions)[order].tolist())
            lengths_out.extend(lengths.numpy()[order].tolist())
            # Evaluator output is already sorted by descending length.
            text_out.append(text_embedding.cpu().numpy())
            gt_out.append(gt_embedding.cpu().numpy())
            betas_out.append(betas.numpy()[order])
            gender_out.append(gender.numpy()[order])
            contact_out.append(contact.numpy()[order])

    names_array = np.asarray(names_out)
    captions_array = np.asarray(captions_out)
    lengths_array = np.asarray(lengths_out, dtype=np.int64)
    if len(names_array) != args.expected_samples:
        raise RuntimeError(
            f"expected {args.expected_samples} complete validation samples, "
            f"got {len(names_array)}"
        )
    payload = {
        "names": names_array,
        "captions": captions_array,
        "lengths": lengths_array,
        "text_embeddings": np.concatenate(text_out),
        "gt_embeddings": np.concatenate(gt_out),
        "betas": np.concatenate(betas_out).astype(np.float32),
        "gender": np.concatenate(gender_out).astype(np.int64),
        "contact": np.concatenate(contact_out).astype(np.int8),
        "mm_indices": np.arange(min(100, len(names_array)), dtype=np.int64),
        "seed": np.asarray(args.seed),
    }

    mgt_selection_contract = ROOT / "output/mgt_validation_selection_manifest.json"
    if not mgt_selection_contract.is_file():
        raise FileNotFoundError(mgt_selection_contract)
    selection = json.loads(mgt_selection_contract.read_text())
    expected_records = {
        record["name"]: (record["caption"], int(record["length"]))
        for record in selection["records"]
    }
    actual_records = {
        name: (caption, int(length))
        for name, caption, length in zip(
            names_array.tolist(), captions_array.tolist(), lengths_array.tolist()
        )
    }
    if actual_records != expected_records:
        missing = sorted(set(expected_records) - set(actual_records))
        unexpected = sorted(set(actual_records) - set(expected_records))
        field_mismatches = sorted(
            name for name in set(actual_records) & set(expected_records)
            if actual_records[name] != expected_records[name]
        )
        raise RuntimeError({
            "validation_request_does_not_match_formal_hhc_selection": True,
            "missing_names": missing,
            "unexpected_names": unexpected,
            "caption_or_length_mismatches": field_mismatches,
        })
    temporary = request_path.with_suffix(f".npz.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, request_path)

    manifest = {
        "schema": "hhc_interx_validation_request_v1",
        "status": "complete",
        "purpose": "CoShMDM checkpoint selection only; never a reported test row",
        "seed": args.seed,
        "batch_size": 32,
        "samples": len(names_array),
        "request": artifact(request_path),
        "split": artifact(split),
        "motions": artifact(motions),
        "evaluator_options": artifact(ROOT / "checkpoints/hhi/Comp_v6_KLD01/opt.txt"),
        "evaluator_weights": artifact(ROOT / "checkpoints/hhi/text_mot_match/model/finest.tar"),
        "formal_hhc_validation_selection": artifact(mgt_selection_contract),
        "formal_hhc_validation_record_set_exact": True,
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
