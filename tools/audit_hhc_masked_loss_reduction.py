#!/usr/bin/env python3
"""Disclose the unreported HHC masked-NLL reduction and alternatives."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from utils.motion_masks import masked_token_nll


INTERMASK_ROOT = Path("/mnt/8T-HDD/wenning/InterMask")
INTERMASK_REVISION = "5100c555de9839b325d0f3d6904669698c5c87f5"
INTERMASK_SOURCE_HASHES = {
    "models/mask_transformer/transformer.py": (
        "652f5639e22414816ac86008a22e8211b04f1275afa01032fadedc34a2389379"
    ),
    "models/mask_transformer/tools.py": (
        "fd6bb95b594147d92223114f9a6e01a92abd003d6509389793669c798b959f36"
    ),
}


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


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def audit_formal_mask_weighting(seed: int = 90123, repeats: int = 8) -> dict:
    """Measure the reduction's real weighting on the frozen formal cache.

    A global masked-token mean weights a sample in proportion to its masked
    count.  The active per-sample reduction gives every non-empty sample equal
    total weight.  We report the population-normalized global weights; actual
    batch-96 weights fluctuate narrowly around these values.
    """
    split = ROOT / "dataset/interx/splits/train.txt"
    cache = ROOT / "checkpoints/vq_precleanup_raw12288_formal_mgt_v1/codebook"
    names = [
        value.strip() for value in split.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    raw_lengths = []
    loaded_names = []
    for name in names:
        path = cache / f"{name}.npy"
        if not path.is_file():
            continue
        tokens = np.load(path, mmap_mode="r")
        if tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[2] != 3:
            raise ValueError(f"unexpected token cache shape for {path}: {tokens.shape}")
        if tokens.shape[1] < 20:
            continue
        loaded_names.append(name)
        # paper_50 reserves one of its 50 tensor positions for END.
        raw_lengths.append(min(int(tokens.shape[1]), 49))
    lengths = np.asarray(raw_lengths, dtype=np.int64)
    if len(lengths) != 8617:
        raise ValueError(
            f"formal training cache expected 8617 active rows, found {len(lengths)}"
        )
    rng = np.random.default_rng(seed)
    raw_taus = rng.random((repeats, len(lengths), 3))
    ordered = np.sort(raw_taus, axis=-1)
    # Tensor channel order is Body, Hand, Root; tau ordering is
    # tau_hand < tau_body < tau_root.
    taus = ordered[..., [1, 0, 2]]
    ratios = np.cos(taus * (np.pi * 0.5))
    counts_by_part = np.rint(
        ratios * lengths.reshape(1, -1, 1)
    ).astype(np.int64)
    counts_by_part = np.maximum(counts_by_part, 1)
    counts_by_part = np.minimum(
        counts_by_part, lengths.reshape(1, -1, 1)
    )
    total_counts = counts_by_part.sum(axis=-1)
    relative_global_weight = total_counts / total_counts.mean(axis=1, keepdims=True)
    repeated_lengths = np.broadcast_to(lengths, total_counts.shape)
    correlation = float(np.corrcoef(
        repeated_lengths.reshape(-1), total_counts.reshape(-1)
    )[0, 1])
    return {
        "contract": "paper_50/paper_kinematic, 8 deterministic mask draws per sample",
        "seed": seed,
        "draws_per_sample": repeats,
        "active_training_samples": int(len(lengths)),
        "sample_names_sha256": hashlib.sha256(
            "\n".join(loaded_names).encode("utf-8")
        ).hexdigest(),
        "retained_motion_token_length": summarize(lengths),
        "masked_tokens_per_sample": summarize(total_counts),
        "global_mean_relative_sample_weight": summarize(relative_global_weight),
        "masked_count_correlation_with_retained_length": correlation,
        "interpretation": (
            "per-sample masked mean assigns relative weight 1 to every sample; "
            "InterMask-style global mean assigns approximately the reported "
            "masked-count-proportional relative weight"
        ),
        "sources": {
            "training_split": artifact(split),
            "token_cache_manifest": artifact(cache / "_cache_manifest.json"),
        },
    }


def main() -> None:
    # Deliberately unequal masked counts and token difficulties make the two
    # plausible reductions observably different.
    logits = torch.zeros(2, 3, 3, 4, dtype=torch.float64)
    target = torch.zeros(2, 3, 3, dtype=torch.long)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[1, :2, :] = True
    logits[0, 0, 0, 0] = -3.0
    logits[1, :2, :, 0] = 3.0

    current = masked_token_nll(logits, target, mask)
    losses = F.cross_entropy(
        logits[mask], target[mask], reduction="none"
    )
    per_sample = torch.stack([
        F.cross_entropy(logits[index][mask[index]], target[index][mask[index]])
        for index in range(2)
    ]).mean()
    global_token_mean = losses.mean()
    intermask_transformer = (
        INTERMASK_ROOT / "models/mask_transformer/transformer.py"
    )
    intermask_tools = INTERMASK_ROOT / "models/mask_transformer/tools.py"
    transformer_source = intermask_transformer.read_text(encoding="utf-8")
    tools_source = intermask_tools.read_text(encoding="utf-8")
    intermask_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=INTERMASK_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    intermask_source_status = subprocess.run(
        [
            "git", "status", "--porcelain", "--",
            *INTERMASK_SOURCE_HASHES,
        ],
        cwd=INTERMASK_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    checks = {
        "current_is_per_sample_masked_mean_then_batch_mean": bool(
            torch.equal(current, per_sample)
        ),
        "current_differs_from_global_masked_token_mean": bool(
            not torch.isclose(current, global_token_mean)
        ),
        "intermask_revision_matches_frozen_upstream": (
            intermask_revision == INTERMASK_REVISION
        ),
        "intermask_loss_sources_are_clean": intermask_source_status == "",
        "intermask_source_hashes_match_frozen_upstream": all(
            sha256_file(INTERMASK_ROOT / relative) == expected
            for relative, expected in INTERMASK_SOURCE_HASHES.items()
        ),
        "intermask_labels_only_masked_positions": (
            "labels = torch.where(mask, ids, self.mask_id)"
            in transformer_source
        ),
        "intermask_ignores_unmasked_labels": (
            "cal_performance(logits, labels, ignore_index=self.mask_id)"
            in transformer_source
        ),
        "intermask_cross_entropy_uses_default_global_mean": (
            "F.cross_entropy(pred, labels, ignore_index=ignore_index)"
            in tools_source
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)

    payload = {
        "schema": "hhc_masked_loss_reduction_disclosure_v1",
        "status": "pass",
        "paper_scope": (
            "Eq. 3 specifies masked-token negative log likelihood but does "
            "not specify variable-length batch reduction"
        ),
        "active_contract": (
            "mean NLL over masked tokens independently per sample, then mean "
            "over batch"
        ),
        "comparison_contract": (
            "released InterMask cross_entropy uses one global mean over all "
            "non-ignored masked labels"
        ),
        "diagnostic_priority_if_fid_remains_high": (
            "train a controlled global-masked-token-mean arm; do not change "
            "the active formal run in place"
        ),
        "measurement": {
            "active": float(current),
            "per_sample_then_batch": float(per_sample),
            "global_masked_token_mean": float(global_token_mean),
            "masked_counts": [int(mask[index].sum()) for index in range(2)],
        },
        "formal_cache_weighting": audit_formal_mask_weighting(),
        "checks": checks,
        "sources": {
            "hhc_masked_loss": artifact(ROOT / "utils/motion_masks.py"),
            "intermask_repository": {
                "path": str(INTERMASK_ROOT.resolve()),
                "revision": intermask_revision,
                "expected_revision": INTERMASK_REVISION,
                "source_status": intermask_source_status,
            },
            "intermask_transformer": artifact(intermask_transformer),
            "intermask_loss": artifact(intermask_tools),
            "implementation": artifact(Path(__file__)),
        },
    }
    output = ROOT / "output/hhc_masked_loss_reduction_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
