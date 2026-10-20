#!/usr/bin/env python
"""Audit per-prompt multimodality under the frozen Inter-X protocol.

The headline evaluator samples ten repeat pairs.  This audit additionally
computes all 30 choose 2 pair distances per prompt, separating genuine prompt
variation from variance caused by the ten-pair Monte Carlo estimate.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np

from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from tools.score_fair_interx_generations import (
    embed_motions,
    embedding_lengths,
    sha256_file,
    validate_generation_file,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--mm-generations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--repeat-limit", type=int,
        help=(
            "Diagnostic only: analyze the first N durably completed repeats. "
            "Omit this option for the formal 30-repeat protocol."
        ),
    )
    parser.add_argument(
        "--action-labels",
        default="/mnt/8T-HDD/wenning/Inter-X/datasets/action_setting.txt",
    )
    args = parser.parse_args()

    action_labels = []
    if os.path.isfile(args.action_labels):
        with open(args.action_labels, "r", encoding="utf-8") as handle:
            action_labels = [line.strip() for line in handle if line.strip()]

    request = np.load(args.requests, allow_pickle=False)
    all_names = request["names"].astype(str)
    all_captions = request["captions"].astype(str)
    all_lengths = request["lengths"].astype(np.int64)
    indices = request["mm_indices"].astype(np.int64)
    names = all_names[indices]
    captions = all_captions[indices]
    lengths = all_lengths[indices]

    evaluator = EvaluatorModelWrapper(
        get_opt("checkpoints/hhi/Comp_v6_KLD01/opt.txt", args.device)
    )
    with h5py.File(args.mm_generations, "r") as source:
        request_sha256 = sha256_file(args.requests)
        recorded_request_sha256 = source.attrs.get("requests_sha256")
        if (recorded_request_sha256 is not None and
                recorded_request_sha256 != request_sha256):
            raise ValueError(
                "MM generation request hash does not match the supplied request"
            )
        repeat_count = 30 if args.repeat_limit is None else args.repeat_limit
        if not 2 <= repeat_count <= 30:
            raise ValueError("--repeat-limit must be in [2, 30]")
        if args.repeat_limit is None:
            validate_generation_file(source, names, lengths, 30, "MM")
        else:
            stored_repeats = int(source["motions"].shape[1])
            if source["motions"].shape[0] != len(lengths) or stored_repeats < repeat_count:
                raise ValueError(
                    "partial MM diagnostic requires at least the requested "
                    "number of stored repeats"
                )
            if not np.array_equal(source["lengths"][:], lengths):
                raise ValueError("partial MM lengths mismatch")
            if not np.array_equal(source["names"][:].astype(str), names):
                raise ValueError("partial MM sample-name/order mismatch")
            completed = source["completed_repeats"][:].astype(bool)
            if completed.shape != (stored_repeats,) or not completed[:repeat_count].all():
                raise ValueError(
                    f"first {repeat_count} repeats are not all durably complete"
                )
        effective_lengths = embedding_lengths(source, lengths)
        embeddings = np.stack([
            embed_motions(
                evaluator,
                source["motions"][:, repeat],
                effective_lengths,
                args.batch_size,
            )
            for repeat in range(repeat_count)
        ], axis=1)

    first, second = np.triu_indices(repeat_count, k=1)
    prompt_records = []
    all_pair_distances = []
    for prompt_index, embedding in enumerate(embeddings):
        distances = np.linalg.norm(
            embedding[first] - embedding[second], axis=-1
        )
        all_pair_distances.append(distances)
        action_match = re.search(r"A(\d{3})", names[prompt_index])
        action_id = int(action_match.group(1)) if action_match else None
        action_label = (
            action_labels[action_id]
            if action_id is not None and action_id < len(action_labels)
            else None
        )
        prompt_records.append({
            "action_id": action_id,
            "action_label": action_label,
            "caption": captions[prompt_index],
            "effective_length": int(effective_lengths[prompt_index]),
            "mean_all_435_pairs": float(distances.mean()),
            "median_all_435_pairs": float(np.median(distances)),
            "name": names[prompt_index],
            "p05_all_435_pairs": float(np.percentile(distances, 5)),
            "p95_all_435_pairs": float(np.percentile(distances, 95)),
            "requested_length": int(lengths[prompt_index]),
            "std_all_435_pairs": float(distances.std()),
        })

    flattened = np.concatenate(all_pair_distances)
    action_groups = []
    action_ids = {record["action_id"] for record in prompt_records}
    for action_id in sorted(
        action_ids, key=lambda value: -1 if value is None else value
    ):
        action_prompts = [
            record for record in prompt_records if record["action_id"] == action_id
        ]
        prompt_means = np.asarray([
            record["mean_all_435_pairs"] for record in action_prompts
        ])
        action_groups.append({
            "action_id": action_id,
            "action_label": action_prompts[0]["action_label"],
            "mean_all_pairs": float(prompt_means.mean()),
            "prompt_mean_p05": float(np.percentile(prompt_means, 5)),
            "prompt_mean_p95": float(np.percentile(prompt_means, 95)),
            "prompts": int(len(action_prompts)),
        })
    prompt_records.sort(key=lambda record: record["mean_all_435_pairs"])
    result = {
        "action_groups": action_groups,
        "all_pairs_per_prompt": int(repeat_count * (repeat_count - 1) // 2),
        "diagnostic_partial_repeats": args.repeat_limit is not None,
        "effective_length_policy": (
            "generated_lengths"
            if not np.array_equal(effective_lengths, lengths)
            else "requested_lengths"
        ),
        "mean_all_prompt_pairs": float(flattened.mean()),
        "median_all_prompt_pairs": float(np.median(flattened)),
        "mm_generations": os.path.abspath(args.mm_generations),
        "mm_generations_sha256": sha256_file(args.mm_generations),
        "prompts": prompt_records,
        "requests": os.path.abspath(args.requests),
        "requests_sha256": request_sha256,
        "generation_request_hash_verified": recorded_request_sha256 is not None,
        "samples": int(len(prompt_records)),
        "repeats": int(repeat_count),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in {"prompts", "action_groups"}
    }, indent=2))


if __name__ == "__main__":
    main()
