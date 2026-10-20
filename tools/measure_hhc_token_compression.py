#!/usr/bin/env python3
"""Measure lossless token repetition in the historical38 train/val cache."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

from audit_mgt_train_val_conditional_support import (
    apply_training_loader_contract, atomic_json, load_split, sha256_file,
    token_window_contract,
)


def channel_stats(records, channel):
    counts, total, repeats, runs = Counter(), 0, 0, 0
    lengths = []
    for _, tokens in records:
        values = tokens[:, channel].tolist()
        lengths.append(len(values))
        total += len(values)
        counts.update(values)
        repeats += sum(a == b for a, b in zip(values, values[1:]))
        runs += 1 + sum(a != b for a, b in zip(values, values[1:])) if values else 0
    entropy = -sum((n / total) * math.log2(n / total) for n in counts.values()) if total else 0.0
    assert total >= runs >= 0 and repeats == total - runs
    return {"tokens": total, "samples": len(records),
            "length_min": min(lengths, default=0), "length_max": max(lengths, default=0),
            "length_mean": sum(lengths) / len(lengths) if lengths else 0.0,
            "used_codes": len(counts), "entropy_bits": entropy,
            "effective_vocabulary": 2 ** entropy if total else 0.0,
            "adjacent_exact_repeat_rate": repeats / max(total - len(records), 1),
            "rle_runs": runs, "rle_theoretical_token_reduction": (total - runs) / total if total else 0.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training-manifest", type=Path)
    p.add_argument("--output", type=Path, required=False)
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        stats = channel_stats([("synthetic", np.array([[1], [1], [2], [2]]))], 0)
        assert stats["entropy_bits"] == 1.0 and stats["rle_runs"] == 2
        assert stats["adjacent_exact_repeat_rate"] == 2 / 3
        assert stats["rle_theoretical_token_reduction"] == 0.5
        print("self-check: ok")
        if args.training_manifest is None:
            return
    if args.training_manifest is None or args.output is None:
        p.error("--training-manifest and --output are required unless --self-check is standalone")
    manifest = json.loads(args.training_manifest.read_text())
    resolved = manifest["resolved_arguments"]
    directory = Path(resolved["codebook_dir"])
    train_raw, train_missing = load_split(directory, Path(resolved["train_split_file"]), int(resolved["nb_code"]))
    val_raw, val_missing = load_split(directory, Path(resolved["val_split_file"]), int(resolved["nb_code"]))
    window, maximum, contract = token_window_contract(resolved)
    train, train_short, train_truncated = apply_training_loader_contract(train_raw, 20, maximum)
    val, val_short, val_truncated = apply_training_loader_contract(val_raw, 20, maximum)
    payload = {"schema": "hhc_token_compression_statistics_v1", "status": "complete",
               "interpretation": "RLE is a lossless storage bound; modeling it requires durations.",
               "training_manifest": {"path": str(args.training_manifest.resolve()), "sha256": sha256_file(args.training_manifest)},
               "loader_contract": {"name": contract, "window_including_end": window, "payload_max": maximum, "minimum": 20},
               "splits": {"train": {"cached": len(train_raw), "loaded": len(train), "missing": len(train_missing), "below_min": len(train_short), "truncated": train_truncated},
                          "val": {"cached": len(val_raw), "loaded": len(val), "missing": len(val_missing), "below_min": len(val_short), "truncated": val_truncated}},
               "channels": {part: {"train": channel_stats(train, i), "val": channel_stats(val, i)}
                           for i, part in enumerate(("body", "hand", "root"))}}
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__": main()
