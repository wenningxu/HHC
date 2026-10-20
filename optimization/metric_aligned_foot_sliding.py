"""Metric-aligned foot constraints for preregistered shape augmentation.

This module is intentionally not wired into the production optimizer while
the serial active-contact-mean pilot is running.  It implements the exact
candidate formula frozen in
``metric_aligned_foot_sliding_pilot_v1.json`` without changing loss weights.
"""

from __future__ import annotations

import torch


FOOT_JOINT_INDICES = (7, 10, 8, 11)
HEIGHT_THRESHOLDS_M = (0.12, 0.05, 0.12, 0.05)


def grounded_foot_mask(joints: torch.Tensor) -> torch.Tensor:
    """Return the detached grounded predicate used by the official metric."""
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("joints must have shape [T,J,3]")
    if joints.shape[0] < 2:
        return torch.zeros(
            (0, len(FOOT_JOINT_INDICES)),
            dtype=torch.bool,
            device=joints.device,
        )
    feet = joints[:-1, FOOT_JOINT_INDICES]
    thresholds = joints.new_tensor(HEIGHT_THRESHOLDS_M)
    return (feet[..., 1] < thresholds).detach()


def freeze_initial_grounded_foot_mask(joints: torch.Tensor) -> torch.Tensor:
    """Freeze initial grounded frames so lifting a foot cannot evade loss."""
    with torch.no_grad():
        return grounded_foot_mask(joints).clone()


def initial_or_current_grounded_horizontal_mean_speed(
    joints: torch.Tensor,
    initial_grounded_mask: torch.Tensor,
) -> torch.Tensor:
    """Match the gate's horizontal first-order foot-sliding reduction.

    The active set is the union of the frozen iteration-0 grounded mask and
    the current grounded mask.  The first component prevents threshold
    evasion; the second directly covers every transition counted by the
    official final-motion metric.
    """
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("joints must have shape [T,J,3]")
    feet = joints[:, FOOT_JOINT_INDICES]
    horizontal_velocity = feet[1:, :, (0, 2)] - feet[:-1, :, (0, 2)]
    expected_shape = horizontal_velocity.shape[:-1]
    if tuple(initial_grounded_mask.shape) != tuple(expected_shape):
        raise ValueError(
            "initial grounded mask shape differs from foot velocity: "
            f"{tuple(initial_grounded_mask.shape)} vs {tuple(expected_shape)}"
        )
    active = initial_grounded_mask.to(
        device=joints.device, dtype=torch.bool
    ) | grounded_foot_mask(joints)
    if not bool(active.any()):
        return horizontal_velocity.sum() * 0.0
    speed = torch.linalg.vector_norm(horizontal_velocity, dim=-1)
    return speed[active].mean()
