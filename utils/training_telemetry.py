"""Low-overhead, fail-closed telemetry for future online W&B training."""

from __future__ import annotations

import math
import time

import torch


# Literal names keep launch-contract audits independent of the implementation
# detail that `_norm_metrics` also builds module-prefixed variants dynamically.
GLOBAL_NORM_METRIC_NAMES = (
    "train/gradient_l2_norm",
    "train/gradient_max_abs",
    "train/weight_l2_norm",
    "train/weight_max_abs",
    "train/gradient_to_weight_l2_ratio",
    "train/gradient_value_count",
    "train/weight_value_count",
    "train/nonfinite_gradients",
    "train/nonfinite_weights",
)


def _tensor_collection_stats(tensors):
    squared_norm = None
    max_abs = None
    nonfinite = 0
    values = 0
    for tensor in tensors:
        if tensor is None:
            continue
        detached = tensor.detach()
        values += detached.numel()
        finite = torch.isfinite(detached)
        nonfinite += int((~finite).sum().item())
        safe = detached.float().masked_fill(~finite, 0.0)
        part_squared = torch.sum(safe * safe)
        part_max = torch.max(torch.abs(safe)) if safe.numel() else safe.new_zeros(())
        squared_norm = (
            part_squared if squared_norm is None else squared_norm + part_squared
        )
        max_abs = part_max if max_abs is None else torch.maximum(max_abs, part_max)
    if squared_norm is None:
        squared_norm = torch.zeros((), dtype=torch.float32)
        max_abs = torch.zeros((), dtype=torch.float32)
    return {
        "l2_norm": torch.sqrt(squared_norm),
        "max_abs": max_abs,
        "nonfinite": nonfinite,
        "values": values,
    }


def _norm_metrics(prefix, weights, gradients):
    weight_l2 = weights["l2_norm"]
    gradient_l2 = gradients["l2_norm"]
    ratio = gradient_l2 / torch.clamp(weight_l2, min=1e-30)
    return {
        f"{prefix}/gradient_l2_norm": gradient_l2,
        f"{prefix}/gradient_max_abs": gradients["max_abs"],
        f"{prefix}/weight_l2_norm": weight_l2,
        f"{prefix}/weight_max_abs": weights["max_abs"],
        f"{prefix}/gradient_to_weight_l2_ratio": ratio,
        f"{prefix}/gradient_value_count": float(gradients["values"]),
        f"{prefix}/weight_value_count": float(weights["values"]),
        f"{prefix}/nonfinite_gradients": float(gradients["nonfinite"]),
        f"{prefix}/nonfinite_weights": float(weights["nonfinite"]),
    }


def parameter_and_gradient_telemetry(
    module,
    fail_on_nonfinite=True,
    include_top_level_modules=True,
):
    """Return detached global parameter/gradient statistics.

    A caller must invoke this at the policy interval after backward and AMP
    unscale, but before gradient clipping and the optimizer step. Statistics
    cover every trainable parameter and use float32 accumulation. By default,
    the same statistics are also emitted for every trainable top-level child
    under ``train/modules/<name>/``. The global metrics remain authoritative:
    the module breakdown is diagnostic and can omit root-owned parameters.
    Any non-finite count is an integrity failure.
    """
    parameters = [
        parameter for parameter in module.parameters()
        if parameter.requires_grad
    ]
    weights = _tensor_collection_stats(parameters)
    gradients = _tensor_collection_stats(
        parameter.grad for parameter in parameters
    )
    if fail_on_nonfinite and (weights["nonfinite"] or gradients["nonfinite"]):
        raise FloatingPointError(
            "nonfinite training telemetry: "
            f"weights={weights['nonfinite']} gradients={gradients['nonfinite']}"
        )
    metrics = _norm_metrics("train", weights, gradients)
    if include_top_level_modules:
        for name, child in module.named_children():
            child_parameters = [
                parameter for parameter in child.parameters()
                if parameter.requires_grad
            ]
            if not child_parameters:
                continue
            child_weights = _tensor_collection_stats(child_parameters)
            child_gradients = _tensor_collection_stats(
                parameter.grad for parameter in child_parameters
            )
            metrics.update(_norm_metrics(
                f"train/modules/{name}", child_weights, child_gradients
            ))
    return metrics


class GradientClipTelemetry:
    """Apply gradient clipping every step and report its sparse telemetry.

    ``clip`` must replace a trainer's direct ``clip_grad_norm_`` call so the
    cumulative hit rate includes every optimizer update, not only logging
    steps. Set ``emit_metrics`` at the normal sparse telemetry interval.
    """

    def __init__(self, threshold):
        self.threshold = float(threshold)
        if not math.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("gradient clip threshold must be finite and positive")
        self.events = 0
        self.hits = 0

    def clip(self, module, emit_metrics=False):
        parameters = [
            parameter for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.threshold,
            norm_type=2.0,
            error_if_nonfinite=True,
        )
        triggered = bool(float(pre_clip_norm.detach()) > self.threshold)
        self.events += 1
        self.hits += int(triggered)
        if not emit_metrics:
            return {}
        post_clip = _tensor_collection_stats(
            parameter.grad for parameter in parameters
        )
        return {
            "train/gradient_l2_norm_post_clip": post_clip["l2_norm"],
            "train/gradient_clip_threshold": self.threshold,
            "train/gradient_clip_triggered": float(triggered),
            "train/gradient_clip_hit_rate": self.hits / self.events,
        }


class ThroughputMeter:
    """Measure samples/second between sparse telemetry events."""

    def __init__(self):
        self.last_time = time.perf_counter()
        self.samples = 0

    def update(self, samples):
        self.samples += int(samples)

    def measure_and_reset(self):
        now = time.perf_counter()
        elapsed = max(now - self.last_time, 1e-12)
        rate = self.samples / elapsed
        if not math.isfinite(rate):
            raise FloatingPointError("nonfinite training throughput telemetry")
        self.last_time = now
        self.samples = 0
        return rate


def gpu_peak_memory_gib(device):
    if torch.device(device).type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / float(1024 ** 3)
