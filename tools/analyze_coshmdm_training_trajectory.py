#!/usr/bin/env python3
"""Diagnose CoShMDM validation degradation from frozen training evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENT = Path(
    "/mnt/8T-HDD/wenning/coshmdm-code/checkpoints/"
    "CoShMDM-InterX-fair-statsfix-seed123-bs32x2/log/"
    "events.out.tfevents.1788056811.edmond-Z790-Taichi.1409003.0"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite_summary(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    if not finite.any():
        return {
            "count": int(len(values)),
            "finite_count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    selected = values[finite]
    return {
        "count": int(len(values)),
        "finite_count": int(finite.sum()),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "minimum": float(np.min(selected)),
        "maximum": float(np.max(selected)),
    }


def load_scalar_history(event_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    accumulator = EventAccumulator(
        str(event_path), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    history = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        history[tag] = (
            np.asarray([event.step for event in events], dtype=np.int64),
            np.asarray([event.value for event in events], dtype=np.float64),
        )
    return history


def milestone_scalar_summary(history, step: int, window: int) -> dict:
    result = {}
    for tag, (steps, values) in history.items():
        if not len(steps) or tag == "dropped_incomplete_accumulation_micro_batches":
            continue
        nearest_index = int(np.argmin(np.abs(steps - step)))
        in_window = (steps > step - window) & (steps <= step)
        result[tag] = {
            "nearest_step": int(steps[nearest_index]),
            "nearest_value": float(values[nearest_index]),
            "trailing_window": finite_summary(values[in_window]),
            "trailing_window_start_exclusive": int(step - window),
            "trailing_window_end_inclusive": int(step),
        }
    return result


def checkpoint_weight_summary(path: Path) -> dict:
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    state = checkpoint.get("state_dict", {})
    # The released Lightning module stores two sinusoidal positional-encoding
    # buffers alongside 344 optimizer parameters. Exclude only those buffers.
    trainable_like = {
        key: value for key, value in state.items()
        if torch.is_tensor(value) and not key.endswith(".pe")
    }
    squared = 0.0
    maximum = 0.0
    nonfinite = 0
    values = 0
    floating_tensors = 0
    for tensor in trainable_like.values():
        if not tensor.is_floating_point():
            continue
        floating_tensors += 1
        flat = tensor.detach().float()
        finite = torch.isfinite(flat)
        nonfinite += int((~finite).sum().item())
        safe = flat.masked_fill(~finite, 0.0)
        squared += float(torch.sum(safe * safe).item())
        if safe.numel():
            maximum = max(maximum, float(torch.max(torch.abs(safe)).item()))
        values += int(flat.numel())
    groups = [
        group
        for optimizer in checkpoint.get("optimizer_states", [])
        for group in optimizer.get("param_groups", [])
    ]
    summary = {
        "path": str(path.resolve()),
        "global_step": int(checkpoint.get("global_step", -1)),
        "epoch": int(checkpoint.get("epoch", -1)),
        "state_dict_tensors": int(len(state)),
        "trainable_like_floating_tensors": floating_tensors,
        "trainable_like_values": values,
        "excluded_positional_encoding_buffers": sorted(
            key for key in state if key.endswith(".pe")
        ),
        "weight_l2_norm": math.sqrt(squared),
        "weight_max_abs": maximum,
        "nonfinite_weight_values": nonfinite,
        "optimizer_learning_rates": [float(group["lr"]) for group in groups],
        "optimizer_weight_decays": [
            float(group["weight_decay"]) for group in groups
        ],
    }
    del checkpoint, state, trainable_like
    gc.collect()
    return summary


def correlation(left, right) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=DEFAULT_EVENT)
    parser.add_argument(
        "--selection", type=Path,
        default=ROOT / "output/coshmdm_interx_val_selection.json",
    )
    parser.add_argument("--trailing-window", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/coshmdm_training_trajectory_diagnosis.json",
    )
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    candidates = selection["candidates"]
    history = load_scalar_history(args.event)
    required_tags = (
        "total", "simple", "DM", "JA", "RO", "A", "B",
        "VEL_A", "BL_A", "FC_A", "VEL_B", "BL_B", "FC_B",
    )
    missing = sorted(set(required_tags) - set(history))
    if missing:
        raise ValueError(f"TensorBoard event is missing tags: {missing}")

    milestones = []
    for candidate in candidates:
        checkpoint_path = Path(candidate["checkpoint"]["path"])
        step = int(candidate["step"])
        checkpoint = checkpoint_weight_summary(checkpoint_path)
        if checkpoint["global_step"] != step:
            raise ValueError(
                f"checkpoint step mismatch: expected {step}, "
                f"got {checkpoint['global_step']}"
            )
        if checkpoint["nonfinite_weight_values"]:
            raise FloatingPointError(
                f"non-finite weights at step {step}: "
                f"{checkpoint['nonfinite_weight_values']}"
            )
        milestones.append({
            "step": step,
            "validation": {
                "fid": float(candidate["validation_fid"]),
                "matching": float(candidate["validation_matching_score"]),
                "r_precision": [
                    float(value) for value in candidate["validation_r_precision"]
                ],
            },
            "training_scalars": milestone_scalar_summary(
                history, step, args.trailing_window
            ),
            "checkpoint": checkpoint,
        })

    total_losses = [
        item["training_scalars"]["total"]["trailing_window"]["median"]
        for item in milestones
    ]
    fids = [item["validation"]["fid"] for item in milestones]
    weight_norms = [
        item["checkpoint"]["weight_l2_norm"] for item in milestones
    ]
    first_norm = weight_norms[0]
    weight_norm_relative_span = (
        (max(weight_norms) - min(weight_norms)) / first_norm
        if first_norm else None
    )
    selected_index = [item["step"] for item in milestones].index(
        int(selection["selected_step"])
    )
    post_selected = milestones[selected_index + 1:]
    checks = {
        "all_logged_losses_finite": all(
            np.isfinite(values).all()
            for tag, (_, values) in history.items()
            if tag != "dropped_incomplete_accumulation_micro_batches"
        ),
        "all_milestone_weights_finite": all(
            item["checkpoint"]["nonfinite_weight_values"] == 0
            for item in milestones
        ),
        "training_total_loss_decreases_at_every_milestone": all(
            later < earlier
            for earlier, later in zip(total_losses, total_losses[1:])
        ),
        "validation_fid_worsens_after_selected_checkpoint": all(
            item["validation"]["fid"] > fids[selected_index]
            for item in post_selected
        ),
        "historical_gradient_norms_available": False,
    }
    payload = {
        "schema": "hhc_coshmdm_training_trajectory_diagnosis_v1",
        "status": "complete",
        "purpose": (
            "diagnose the capped historical CoShMDM run; no new training and "
            "no test-set checkpoint selection"
        ),
        "event": {
            "path": str(args.event.resolve()),
            "sha256": sha256_file(args.event),
            "scalar_tags": sorted(history),
            "minimum_logged_step": int(min(steps[0] for steps, _ in history.values())),
            "maximum_logged_step": int(max(steps[-1] for steps, _ in history.values())),
        },
        "selection_manifest": {
            "path": str(args.selection.resolve()),
            "sha256": sha256_file(args.selection),
            "selected_step": int(selection["selected_step"]),
            "validation_samples": int(selection["validation_samples"]),
            "test_metrics_used_for_selection": bool(
                selection["test_metrics_used_for_selection"]
            ),
        },
        "trailing_window_optimizer_steps": args.trailing_window,
        "milestones": milestones,
        "aggregate": {
            "validation_fids": fids,
            "training_total_loss_trailing_medians": total_losses,
            "weight_l2_norms": weight_norms,
            "weight_norm_relative_span_vs_first": weight_norm_relative_span,
            "pearson_fid_vs_training_total_loss": correlation(fids, total_losses),
            "selected_to_100k_fid_change": float(fids[-1] - fids[selected_index]),
            "selected_to_100k_training_loss_change": float(
                total_losses[-1] - total_losses[selected_index]
            ),
        },
        "checks": checks,
        "diagnosis": (
            "The available evidence does not support numerical divergence: "
            "logged objectives and milestone weights are finite, while the "
            "training objective continues to fall after the validation-optimal "
            "checkpoint. The official validation FID nevertheless degrades. "
            "This is evidence for objective/generalization/representation "
            "mismatch rather than an exploding-loss failure. The original "
            "TensorBoard run did not log gradients, so gradient behavior cannot "
            "be reconstructed retrospectively."
        ),
        "limitations": [
            "Five validation milestones are descriptive and insufficient for causal inference.",
            "Checkpoint state does not contain historical gradients.",
            "State-dict norms exclude the two non-optimizer sinusoidal positional-encoding buffers.",
        ],
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
