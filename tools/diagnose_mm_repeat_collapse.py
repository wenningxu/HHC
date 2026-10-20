#!/usr/bin/env python3
"""Measure exact-repeat collapse in frozen multimodality generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be LABEL=H5")
    label, path = value.split("=", 1)
    return label, Path(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {
        'path': str(path), 'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def inspect(path: Path, pair_count: int, seed: int, repeat_count: int = 30) -> dict:
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as source:
        motions = source["motions"]
        lengths = source["lengths"][:]
        completed = source.get('completed_repeats')
        if completed is None or completed.shape != (repeat_count,) or not completed[:].astype(bool).all():
            raise ValueError(f"{path}: repeats are incomplete")
        if motions.ndim < 3 or not motions.shape[0] or repeat_count < 2 or motions.shape[1] != repeat_count:
            raise ValueError(f"{path}: expected nonempty motions with {repeat_count} repeats")
        if (lengths.shape != (motions.shape[0],) or not np.issubdtype(lengths.dtype, np.integer)
                or np.any(lengths <= 0) or np.any(lengths > motions.shape[2])):
            raise ValueError(f"{path}: invalid motion lengths")
        maximum_pairs = repeat_count * (repeat_count - 1) // 2
        if not 1 <= pair_count <= maximum_pairs:
            raise ValueError("pairs-per-prompt exceeds available repeat pairs")
        pairs = np.asarray([(i, j) for i in range(repeat_count) for j in range(i + 1, repeat_count)])
        sampled_pairs = pairs[rng.choice(len(pairs), pair_count, replace=False)]
        exact_pairs = 0
        total_pairs = 0
        unique_counts = []
        sampled_rms = []
        prompt_exact_pair_rates = []
        for prompt, length in enumerate(lengths):
            values = np.asarray(motions[prompt, :, :length], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: non-finite valid motion values at prompt {prompt}")
            digests = [hashlib.sha256(item.tobytes()).digest() for item in values]
            counts = {}
            for digest in digests:
                counts[digest] = counts.get(digest, 0) + 1
            prompt_exact = sum(count * (count - 1) // 2 for count in counts.values())
            exact_pairs += prompt_exact
            total_pairs += maximum_pairs
            unique_counts.append(len(counts))
            prompt_exact_pair_rates.append(prompt_exact / maximum_pairs)
            for first, second in sampled_pairs:
                difference = values[first] - values[second]
                sampled_rms.append(float(np.sqrt(np.mean(difference * difference))))
        attributes = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in source.attrs.items()
            if key in ("seed", "seed_strategy", "sampler", "sampling_steps", "temperature", "cfg_scale",
                       "confidence_noise_scale", "confidence_noise_scope", "beta_policy")
        }
    return {
        "path": str(path.resolve()),
        "input_artifact": artifact(path),
        "samples": int(len(lengths)),
        "repeats": repeat_count,
        "generation_attributes": attributes,
        "exact_pair_rate": exact_pairs / total_pairs,
        "unique_repeats_per_prompt": {
            "mean": float(np.mean(unique_counts)),
            "median": float(np.median(unique_counts)),
            "minimum": int(np.min(unique_counts)),
            "maximum": int(np.max(unique_counts)),
        },
        "prompt_exact_pair_rate": {
            "mean": float(np.mean(prompt_exact_pair_rates)),
            "median": float(np.median(prompt_exact_pair_rates)),
            "p95": float(np.percentile(prompt_exact_pair_rates, 95)),
        },
        "sampled_raw_pair_rms": {
            "pairs_per_prompt": pair_count,
            "mean": float(np.mean(sampled_rms)),
            "median": float(np.median(sampled_rms)),
            "p95": float(np.percentile(sampled_rms, 95)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs-per-prompt", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.repeats < 2 or not 1 <= args.pairs_per_prompt <= args.repeats * (args.repeats - 1) // 2:
        raise ValueError("invalid pairs-per-prompt/repeats")
    methods = {
        label: inspect(path.resolve(), args.pairs_per_prompt, args.seed, args.repeats)
        for label, path in args.input
    }
    payload = {
        "schema": "hhc_mm_repeat_collapse_diagnostic_v1",
        "status": "complete",
        "pair_sampling_seed": args.seed,
        "methods": methods,
        "implementation": artifact(Path(__file__)),
        "interpretation": (
            "Exact duplicate motions across distinct repeat seeds indicate a "
            "collapsed discrete conditional sampler/model distribution, not "
            "reuse of one stored repeat."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(
        args.output.suffix + f'.tmp.{os.getpid()}'
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding='utf-8',
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
