#!/usr/bin/env python3
"""Numerically compare the fair scorer's embedding order to the author loader path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

import score_interact2ar_joint_evaluators as fair


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--requests", type=Path,
        default=Path("output/fair_interx_requests/replication_00.npz"),
    )
    parser.add_argument(
        "--motions", type=Path,
        default=Path(
            "output/fair_eval/intermask_official_exact_common50/"
            "replication_00_main.h5"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/interact2ar_joint_scorer_parity_audit.json"),
    )
    args = parser.parse_args()
    if args.batch_size != 32:
        raise ValueError("formal parity audit uses the released batch size 32")

    request = np.load(args.requests, allow_pickle=False)
    names = request["names"].astype(str)[:args.batch_size]
    captions = request["captions"].astype(str)[:args.batch_size]
    lengths = request["lengths"].astype(np.int64)[:args.batch_size]
    vectorizer = fair.WordVectorizer(
        str(fair.INTERACT2AR_ROOT / "data/glove"), "hhi_vab"
    )
    words, pos, sent_lens = fair.vectorize_captions(
        names, captions, fair.INTERACT2AR_ROOT / "data/texts_processed", vectorizer
    )
    with h5py.File(args.motions, "r") as source:
        fair.validate_h5(
            source, request["names"].astype(str), request["lengths"], 1,
            "parity source",
        )
        motions = np.asarray(
            source["motions"][:args.batch_size, 0], dtype=np.float32
        )

    old_cwd = Path.cwd()
    os.chdir(fair.INTERACT2AR_ROOT)
    try:
        wrappers = {
            name: fair.EvaluatorModelWrapper(
                SimpleNamespace(), args.device, checkpoint,
                representation="joints", part=part, normalized=False,
            )
            for name, (part, checkpoint) in fair.WRAPPERS.items()
        }
        fair_text = {
            name: fair.embed_text(
                wrapper, words, pos, sent_lens, args.batch_size
            )
            for name, wrapper in wrappers.items()
        }
        fair_motion = {
            name: fair.embed_motion(
                wrapper, motions[:, None], lengths, args.batch_size
            )
            for name, wrapper in wrappers.items()
        }
        shared_motion = fair.embed_motion_all_wrappers(
            wrappers, motions[:, None], lengths, args.batch_size,
        )

        # The released generated-motion collate first sorts by caption length;
        # get_co_embeddings then sorts that batch by motion length. Restore the
        # resulting rows to the original frozen-request order before comparing.
        caption_order = np.argsort(sent_lens)[::-1].copy()
        motion_order = np.argsort(lengths[caption_order])[::-1].copy()
        final_order = caption_order[motion_order]
        inverse = np.empty_like(final_order)
        inverse[final_order] = np.arange(len(final_order))
        author_text = {}
        author_motion = {}
        for name, wrapper in wrappers.items():
            text_embedding, motion_embedding = wrapper.get_co_embeddings(
                torch.from_numpy(words[caption_order]),
                torch.from_numpy(pos[caption_order]),
                torch.from_numpy(sent_lens[caption_order]).long(),
                torch.from_numpy(motions[caption_order]),
                torch.from_numpy(lengths[caption_order]).long(),
            )
            author_text[name] = text_embedding[inverse].cpu().numpy()
            author_motion[name] = motion_embedding[inverse].cpu().numpy()
    finally:
        os.chdir(old_cwd)

    differences = {}
    checks = {}
    for name in fair.WRAPPERS:
        text_difference = np.abs(fair_text[name] - author_text[name])
        motion_difference = np.abs(fair_motion[name] - author_motion[name])
        shared_difference = np.abs(shared_motion[name] - fair_motion[name])
        differences[name] = {
            "text": float(text_difference.max(initial=0.0)),
            "motion": float(motion_difference.max(initial=0.0)),
            "shared_joint_vs_independent_motion": float(
                shared_difference.max(initial=0.0)
            ),
        }
        checks[f"{name}_text_embedding_shape_matches"] = (
            fair_text[name].shape == author_text[name].shape
        )
        checks[f"{name}_motion_embedding_shape_matches"] = (
            fair_motion[name].shape == author_motion[name].shape
        )
        checks[f"{name}_text_matches_author_loader_wrapper"] = (
            differences[name]["text"] <= args.atol
        )
        checks[f"{name}_motion_matches_author_loader_wrapper"] = (
            differences[name]["motion"] <= args.atol
        )
        checks[f"{name}_shared_joint_matches_independent_wrapper"] = (
            differences[name]["shared_joint_vs_independent_motion"] <= args.atol
        )
        matching, top_k = fair.batched_retrieval(
            fair_text[name], fair_motion[name], args.batch_size
        )
        distance = fair.euclidean_distance_matrix(
            author_text[name], author_motion[name]
        )
        author_matching = float(np.trace(distance) / args.batch_size)
        author_top_k = fair.calculate_top_k(
            np.argsort(distance, axis=1), 3
        ).sum(axis=0) / args.batch_size
        differences[name]["matching"] = abs(matching - author_matching)
        differences[name]["top_k"] = float(np.max(np.abs(top_k - author_top_k)))
        checks[f"{name}_retrieval_metrics_match"] = (
            differences[name]["matching"] <= args.atol
            and differences[name]["top_k"] <= args.atol
        )
    payload = {
        "schema": "interact2ar_joint_scorer_parity_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "batch_size": args.batch_size,
        "absolute_tolerance": args.atol,
        "maximum_absolute_difference": differences,
        "ordering_contract": (
            "released caption-length collate then wrapper motion-length sort, "
            "inverted back to frozen-request order"
        ),
        "artifacts": {
            "requests": artifact(args.requests),
            "motions": artifact(args.motions),
            "author_wrapper": artifact(
                fair.INTERACT2AR_ROOT / "src/evaluation/evaluator_wrapper.py"
            ),
            "author_dataset": artifact(
                fair.INTERACT2AR_ROOT / "src/data/dataset.py"
            ),
            "fair_scorer": artifact(
                Path(__file__).with_name("score_interact2ar_joint_evaluators.py")
            ),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
