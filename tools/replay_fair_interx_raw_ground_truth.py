#!/usr/bin/env python
"""Replay frozen Inter-X requests and export their exact raw 672-D GT crops.

The frozen request files intentionally contain evaluator embeddings rather than
raw motion.  A joint-space evaluator needs the raw crop from every replication,
so this script replays the original official-loader RNG contract and refuses to
write an HDF5 file unless all request fields and the legacy evaluator embeddings
match the frozen request.
"""

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
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.motion_process import MotionNormalizerTorchInterX
from utils.word_vectorizer import WordVectorizer

EXACT_FIELDS = (
    "names", "captions", "lengths", "betas", "gender", "contact",
    "mm_indices", "seed",
)
EMBEDDING_FIELDS = ("text_embeddings", "gt_embeddings", "gt_body_embeddings")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def artifact_matches(record, path=None):
    if not isinstance(record, dict):
        return False
    resolved = Path(path if path is not None else record.get("path", "")).resolve()
    return (
        resolved.is_file()
        and record.get("path") == str(resolved)
        and record.get("bytes") == resolved.stat().st_size
        and record.get("sha256") == sha256_file(resolved)
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compare_exact(label, actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            f"{label} schema mismatch: actual {actual.shape}/{actual.dtype}, "
            f"expected {expected.shape}/{expected.dtype}"
        )
    if not np.array_equal(actual, expected):
        mismatch = np.flatnonzero(np.asarray(actual != expected).reshape(-1))
        first = int(mismatch[0]) if len(mismatch) else -1
        raise ValueError(f"{label} does not exactly replay at flat index {first}")


@torch.inference_mode()
def replay_one(loader, evaluator, normalizer, request_path, output_path, replication,
               embedding_atol):
    frozen = np.load(request_path, allow_pickle=False)
    seed = int(frozen["seed"])
    seed_everything(seed)

    raw_batches = []
    values = {key: [] for key in EXACT_FIELDS if key not in ("mm_indices", "seed")}
    values.update({key: [] for key in EMBEDDING_FIELDS})

    for batch in loader:
        (word_embeddings, pos_one_hots, captions, sent_len, pose, lengths,
         _, names, betas, gender, contact) = batch
        raw_pose = normalizer.backward(pose.float())
        text_embedding, gt_embedding = evaluator.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, raw_pose, lengths
        )
        body_pose = raw_pose.reshape(raw_pose.shape[0], raw_pose.shape[1], 56, 12).clone()
        identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=body_pose.dtype)
        for sample_index, valid_length in enumerate(lengths.tolist()):
            body_pose[sample_index, :valid_length, 22:55, :6] = identity
            body_pose[sample_index, :valid_length, 22:55, 6:12] = identity
        _, gt_body_embedding = evaluator.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len,
            body_pose.reshape_as(raw_pose), lengths,
        )
        order = np.argsort(lengths.numpy().tolist())[::-1].copy()
        raw_batches.append(raw_pose.cpu().numpy()[order])
        values["names"].extend(np.asarray(names)[order].tolist())
        values["captions"].extend(np.asarray(captions)[order].tolist())
        values["lengths"].extend(lengths.numpy()[order].tolist())
        values["betas"].append(betas.numpy()[order])
        values["gender"].append(gender.numpy()[order])
        values["contact"].append(contact.numpy()[order])
        values["text_embeddings"].append(text_embedding.cpu().numpy())
        values["gt_embeddings"].append(gt_embedding.cpu().numpy())
        values["gt_body_embeddings"].append(gt_body_embedding.cpu().numpy())

    actual = {
        "names": np.asarray(values["names"]),
        "captions": np.asarray(values["captions"]),
        "lengths": np.asarray(values["lengths"], dtype=np.int64),
        "betas": np.concatenate(values["betas"]).astype(np.float32),
        "gender": np.concatenate(values["gender"]).astype(np.int64),
        "contact": np.concatenate(values["contact"]).astype(np.int8),
        "text_embeddings": np.concatenate(values["text_embeddings"]).astype(np.float32),
        "gt_embeddings": np.concatenate(values["gt_embeddings"]).astype(np.float32),
        "gt_body_embeddings": np.concatenate(values["gt_body_embeddings"]).astype(np.float32),
        "seed": np.asarray(seed, dtype=np.int64),
    }
    base_seed = seed - replication
    generator = np.random.default_rng(base_seed + 10_000 + replication)
    actual["mm_indices"] = generator.permutation(len(actual["names"]))[:100].astype(np.int64)

    for key in EXACT_FIELDS:
        compare_exact(key, actual[key], frozen[key])

    embedding_differences = {}
    for key in EMBEDDING_FIELDS:
        if actual[key].shape != frozen[key].shape:
            raise ValueError(f"{key} shape mismatch")
        difference = np.abs(actual[key] - frozen[key])
        maximum = float(difference.max(initial=0.0))
        if not np.isfinite(difference).all() or maximum > embedding_atol:
            raise ValueError(
                f"{key} replay max abs difference {maximum} exceeds {embedding_atol}"
            )
        embedding_differences[key] = {
            "maximum_absolute_difference": maximum,
            "mean_absolute_difference": float(difference.mean()),
            "within_absolute_tolerance": True,
        }

    raw_motion = np.concatenate(raw_batches).astype(np.float32)
    if raw_motion.shape != (len(frozen["names"]), 150, 672):
        raise ValueError(f"unexpected raw GT shape {raw_motion.shape}")
    if not np.isfinite(raw_motion).all():
        raise FloatingPointError("replayed raw GT contains NaN or Inf")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output_path) + ".partial")
    with h5py.File(partial, "w") as target:
        target.create_dataset(
            "motions", data=raw_motion[:, None], chunks=(1, 1, 150, 672),
            compression="lzf",
        )
        target.create_dataset("lengths", data=frozen["lengths"])
        target.create_dataset("names", data=np.asarray(frozen["names"], dtype="S"))
        target.attrs["schema"] = "fair_interx_raw_gt_replay_v1"
        target.attrs["requests"] = str(Path(request_path).resolve())
        target.attrs["requests_sha256"] = sha256_file(request_path)
        target.attrs["seed"] = seed
        target.attrs["replication"] = replication
        target.attrs["embedding_absolute_tolerance"] = embedding_atol
        target.attrs["source"] = "exact official-loader replay of frozen Inter-X request"
    os.replace(partial, output_path)
    return {
        "replication": replication,
        "seed": seed,
        "samples": len(frozen["names"]),
        "exact_fields": list(EXACT_FIELDS),
        "embedding_comparisons": embedding_differences,
        "request": artifact(request_path),
        "raw_ground_truth": artifact(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-dir", type=Path, default=ROOT / "output/fair_interx_requests")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "output/interact2ar_joint_evaluator_ground_truth",
    )
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument(
        "--device", choices=("cpu",), default="cpu",
        help=(
            "Frozen requests were created by the official CPU evaluator; "
            "CPU replay is required for the 1e-6 embedding identity gate."
        ),
    )
    parser.add_argument("--embedding-atol", type=float, default=1e-6)
    args = parser.parse_args()
    if args.replications != 20:
        raise ValueError("paper-aligned fair protocol requires exactly 20 replications")
    if os.environ.get("OMP_NUM_THREADS") != "8":
        raise ValueError(
            "frozen requests were created with OMP_NUM_THREADS=8; "
            "raw-GT replay must restore the same evaluator runtime"
        )

    report_path = args.output_dir / "replay_audit.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        records = existing.get("records", [])
        valid = (
            existing.get("schema") == "fair_interx_raw_gt_replay_suite_v1"
            and existing.get("status") == "pass"
            and existing.get("replications") == 20
            and existing.get("samples_per_replication") == 1696
            and existing.get("evaluator_device") == "cpu"
            and existing.get("omp_num_threads") == 8
            and len(records) == 20
            and artifact_matches(existing.get("implementation", {}), __file__)
        )
        for replication, record in enumerate(records):
            valid = valid and (
                record.get("replication") == replication
                and record.get("samples") == 1696
                and artifact_matches(
                    record.get("request", {}),
                    args.requests_dir / f"replication_{replication:02d}.npz",
                )
                and artifact_matches(
                    record.get("raw_ground_truth", {}),
                    args.output_dir / f"replication_{replication:02d}.h5",
                )
            )
        if not valid:
            raise RuntimeError(
                f"existing raw-GT replay suite is stale or malformed: {report_path}"
            )
        print(json.dumps({
            "status": "resume_skip", "audit": str(report_path.resolve()),
            "replications": 20,
        }, sort_keys=True))
        return

    vectorizer = WordVectorizer(str(ROOT / "dataset/interx/glove"), "hhi_vab")
    loader = dataset_interx_TM.DATALoader(
        "hhi", 32,
        str(ROOT / "dataset/interx/splits/test.txt"), vectorizer,
        str(ROOT / "dataset/interx/motions/test.h5"),
        unit_length=4, shuffle=True,
    )
    evaluator = EvaluatorModelWrapper(
        get_opt(str(ROOT / "checkpoints/hhi/Comp_v6_KLD01/opt.txt"), args.device)
    )
    normalizer = MotionNormalizerTorchInterX("raw")
    records = []
    for replication in range(args.replications):
        request_path = args.requests_dir / f"replication_{replication:02d}.npz"
        output_path = args.output_dir / f"replication_{replication:02d}.h5"
        record = replay_one(
            loader, evaluator, normalizer, request_path, output_path,
            replication, args.embedding_atol,
        )
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    report = {
        "schema": "fair_interx_raw_gt_replay_suite_v1",
        "status": "pass",
        "scope": "twenty_frozen_replications_for_interact2ar_joint_evaluators",
        "replications": args.replications,
        "samples_per_replication": 1696,
        "embedding_absolute_tolerance": args.embedding_atol,
        "evaluator_device": args.device,
        "omp_num_threads": 8,
        "records": records,
        "implementation": artifact(__file__),
        "legacy_evaluator_checkpoint": artifact(
            ROOT / "checkpoints/hhi/text_mot_match/model/finest.tar"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + f".tmp.{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    os.replace(temporary, report_path)
    print(json.dumps({"status": "pass", "audit": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
