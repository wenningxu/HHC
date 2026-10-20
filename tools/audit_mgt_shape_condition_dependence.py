#!/usr/bin/env python3
"""Diagnose whether HHC-M memorizes motion through per-sample body shape.

The audit never samples or decodes motion and never reads the test split.  For
the same text, target tokens, masks and checkpoint it compares the exact
Root->Body->Hands teacher-forced cascade contexts under three shape inputs:
the requested shape, a deterministic derangement of shapes across samples,
and neutral (zero) betas with the requested gender.  Running it once on train
and once on validation distinguishes legitimate shape sensitivity from a
train-only shape-identity shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import clip
import numpy as np
import torch
import torch.nn.functional as F

from dataset.dataset_interx_TM import Text2MotionDatasetV2HHI_Token
from tools.audit_mgt_mask_exposure_gap import make_batch, sha256_file
from train.train_t2i_trans import MaskGITLightning
from utils.runtime import generate_src_mask


CASCADE_STAGES = (
    ("root_from_fully_masked", 2),
    ("body_after_gt_root", 0),
    ("hand_after_gt_root_body", 1),
)
SHAPE_CONDITIONS = (
    "requested",
    "permuted_beta_and_gender",
    "zero_beta_requested_gender",
)


def metric_store() -> dict:
    return {
        condition: {
            stage: {"loss_sum": 0.0, "correct": 0, "tokens": 0}
            for stage, _ in CASCADE_STAGES
        }
        for condition in SHAPE_CONDITIONS
    }


def add_metrics(store: dict, condition: str, stage: str, channel: int,
                logits: torch.Tensor, target: torch.Tensor,
                valid: torch.Tensor) -> None:
    selected_logits = logits[..., channel, :][valid].float()
    selected_target = target[..., channel][valid]
    count = int(selected_target.numel())
    record = store[condition][stage]
    record["loss_sum"] += float(F.cross_entropy(
        selected_logits, selected_target, reduction="sum"
    ).item())
    record["correct"] += int(
        (selected_logits.argmax(dim=-1) == selected_target).sum().item()
    )
    record["tokens"] += count


def finalize(store: dict) -> dict:
    for conditions in store.values():
        for record in conditions.values():
            tokens = int(record["tokens"])
            record["cross_entropy"] = record.pop("loss_sum") / max(tokens, 1)
            record["accuracy_percent"] = (
                100.0 * record.pop("correct") / max(tokens, 1)
            )
    aggregates = {}
    for condition, stages in store.items():
        tokens = sum(int(row["tokens"]) for row in stages.values())
        weighted_loss = sum(
            float(row["cross_entropy"]) * int(row["tokens"])
            for row in stages.values()
        )
        weighted_correct = sum(
            float(row["accuracy_percent"]) * int(row["tokens"]) / 100.0
            for row in stages.values()
        )
        aggregates[condition] = {
            "tokens": tokens,
            "cross_entropy": weighted_loss / max(tokens, 1),
            "accuracy_percent": 100.0 * weighted_correct / max(tokens, 1),
        }
    requested = aggregates["requested"]
    deltas = {}
    for condition in SHAPE_CONDITIONS[1:]:
        candidate = aggregates[condition]
        deltas[condition] = {
            "accuracy_percent_candidate_minus_requested": (
                candidate["accuracy_percent"] - requested["accuracy_percent"]
            ),
            "cross_entropy_candidate_minus_requested": (
                candidate["cross_entropy"] - requested["cross_entropy"]
            ),
        }
    return {"stages": store, "aggregate": aggregates, "deltas": deltas}


def record_shape_signature(record: dict) -> str:
    """Hash the exact shape condition consumed by the transformer."""
    betas = np.asarray(record["betas"], dtype=np.float32).reshape(-1)
    gender = np.asarray(record["gender"], dtype=np.int64).reshape(-1)
    digest = hashlib.sha256()
    digest.update(betas.astype("<f4", copy=False).tobytes())
    digest.update(gender.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def donor_mapping(names: list[str], signatures: dict[str, str]) -> tuple[dict[str, str], str]:
    """Return a stable derangement whose donor has a different exact shape.

    Inter-X sample names are grouped by actor pair.  A one-position rotation
    over sorted names therefore often changes the motion name without changing
    beta or gender at all.  Group by the exact consumed shape signature and
    rotate by the largest group size, which yields a cross-signature
    derangement whenever one signature does not occupy more than half of the
    selected sample.
    """
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("shape permutation requires distinct samples")
    if set(signatures) != set(names):
        raise ValueError("shape signatures do not cover the selected samples")
    groups: dict[str, list[str]] = {}
    for name in sorted(names):
        groups.setdefault(signatures[name], []).append(name)
    ordered_groups = sorted(groups.items())
    grouped_names = [
        name for _, group_names in ordered_groups for name in group_names
    ]
    largest_group = max(len(group_names) for _, group_names in ordered_groups)
    if largest_group * 2 > len(grouped_names):
        raise ValueError(
            "selected sample is too shape-imbalanced for a cross-shape "
            f"derangement: largest_group={largest_group}, total={len(grouped_names)}"
        )
    donors = grouped_names[largest_group:] + grouped_names[:largest_group]
    mapping = dict(zip(grouped_names, donors, strict=True))
    if any(name == mapping[name] for name in grouped_names):
        raise AssertionError("shape donor mapping contains a name fixed point")
    if any(signatures[name] == signatures[mapping[name]] for name in grouped_names):
        raise AssertionError("shape donor mapping retained an identical shape")
    serialized = "\n".join(f"{name}\t{mapping[name]}" for name in names)
    return mapping, hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@torch.inference_mode()
def audit(args: argparse.Namespace, manifest: dict, dataset, names: list[str],
          donors: dict[str, str]) -> dict:
    resolved = manifest["resolved_arguments"]
    model_args = Namespace(**resolved)
    model_args.resume_trans = str(args.checkpoint.resolve())
    model_args.resume_training_ckpt = None
    if not model_args.paper_architecture:
        raise ValueError("shape-dependence audit requires the paper architecture")
    model = MaskGITLightning(model_args, None, None).to(args.device).eval()
    trans = model.trans_encoder
    max_length = int(model_args.block_size) - 1
    mask_id = int(model_args.nb_code) + 2
    totals = metric_store()

    for start in range(0, len(names), args.batch_size):
        batch_names = names[start:start + args.batch_size]
        captions, target, lengths, betas, genders = make_batch(
            dataset, batch_names, max_length
        )
        donor_names = [donors[name] for name in batch_names]
        _, _, _, donor_betas, donor_genders = make_batch(
            dataset, donor_names, max_length
        )
        target = target.to(args.device)
        lengths = lengths.to(args.device)
        betas = betas.to(args.device)
        genders = genders.to(args.device)
        donor_betas = donor_betas.to(args.device)
        donor_genders = donor_genders.to(args.device)
        valid = generate_src_mask(max_length, lengths)
        valid3 = valid.unsqueeze(-1).expand(-1, -1, 3)
        text = clip.tokenize(captions, truncate=True).to(args.device)
        clip_global, clip_words = model.clip_model(text)

        fully_masked = torch.where(valid3, mask_id, target)
        body_input = fully_masked.clone()
        body_input[..., 2] = torch.where(
            valid, target[..., 2], body_input[..., 2]
        )
        hand_input = body_input.clone()
        hand_input[..., 0] = torch.where(
            valid, target[..., 0], hand_input[..., 0]
        )
        stage_inputs = {
            "root_from_fully_masked": fully_masked,
            "body_after_gt_root": body_input,
            "hand_after_gt_root_body": hand_input,
        }
        condition_shapes = {
            "requested": (betas, genders),
            "permuted_beta_and_gender": (donor_betas, donor_genders),
            "zero_beta_requested_gender": (torch.zeros_like(betas), genders),
        }
        for condition, (condition_betas, condition_genders) in (
                condition_shapes.items()):
            for stage, channel in CASCADE_STAGES:
                logits = trans(
                    stage_inputs[stage], clip_global, src_mask=valid,
                    word_emb=clip_words, betas=condition_betas,
                    gender=condition_genders,
                )
                add_metrics(
                    totals, condition, stage, channel, logits, target, valid
                )

    del model
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return finalize(totals)


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=90123)
    parser.add_argument(
        "--split-role", choices=("validation", "training"),
        default="validation",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    resolved = manifest["resolved_arguments"]
    split_file = (
        resolved["val_split_file"]
        if args.split_role == "validation"
        else resolved["train_split_file"]
    )
    dataset = Text2MotionDatasetV2HHI_Token(
        split_file, resolved["codebook_dir"],
        unit_length=2 ** resolved["down_t"],
        codebook_size=resolved["nb_code"],
        text_dir="dataset/interx/texts_processed",
        max_motion_length=resolved["block_size"] - 1,
        contact_dir=str(Path(resolved["cache_root"]) / "contact_token_label"),
        beta_dir="dataset/interx/betas", strict_assets=True,
    )
    active_names = sorted(dataset.name_list[dataset.pointer:])
    if len(active_names) < args.samples:
        raise RuntimeError(
            f"requested {args.samples} samples, found {len(active_names)}"
        )
    selection_rng = np.random.default_rng(args.seed)
    names = sorted(selection_rng.choice(
        active_names, size=args.samples, replace=False
    ).tolist())
    signatures = {
        name: record_shape_signature(dataset.data_dict[name]) for name in names
    }
    donors, mapping_sha256 = donor_mapping(names, signatures)
    metrics = audit(args, manifest, dataset, names, donors)
    payload = {
        "schema": "hhc_mgt_shape_condition_dependence_audit_v1",
        "status": "complete",
        "purpose": (
            "paired train/validation shape-shortcut diagnosis; no test data "
            "and no model selection"
        ),
        "split_role": args.split_role,
        "split_file": str(Path(split_file).resolve()),
        "split_file_sha256": sha256_file(split_file),
        "samples": len(names),
        "sample_selection": "uniform_without_replacement_over_active_split",
        "active_split_samples": len(active_names),
        "sample_names_sha256": hashlib.sha256(
            "\n".join(names).encode("utf-8")
        ).hexdigest(),
        "shape_derangement": {
            "method": (
                "group by exact float32 beta+int64 gender signature and "
                "rotate by the largest signature-group size"
            ),
            "fixed_points": 0,
            "same_shape_signature_pairs": 0,
            "unique_shape_signatures": len(set(signatures.values())),
            "mapping_sha256": mapping_sha256,
        },
        "seed": args.seed,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
        },
        "training_manifest": {
            "path": str(args.training_manifest.resolve()),
            "sha256": sha256_file(args.training_manifest),
        },
        "metrics": metrics,
        "interpretation_policy": (
            "A substantially larger requested-vs-permuted benefit on train "
            "than validation supports a train-only shape-identity shortcut. "
            "This diagnostic cannot replace decoded-motion FID."
        ),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
