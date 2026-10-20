"""Auditable foot-contact predicates for shape-optimization ablations."""

from __future__ import annotations

import torch


FOOT_JOINT_INDICES = (7, 10, 8, 11)
HEIGHT_THRESHOLDS_M = (0.12, 0.05, 0.12, 0.05)
SQUARED_SPEED_THRESHOLD = 0.001


def released_dynamic_contact_mask(joints: torch.Tensor) -> torch.Tensor:
    """Reproduce the released optimizer's current foot-contact predicate."""
    feet = joints[:, FOOT_JOINT_INDICES]
    velocity = feet[1:] - feet[:-1]
    height = feet[:-1, :, 1]
    thresholds = joints.new_tensor(HEIGHT_THRESHOLDS_M)
    return (
        torch.sum(velocity ** 2, dim=-1) < SQUARED_SPEED_THRESHOLD
    ) & (height < thresholds)


def freeze_initial_contact_mask(joints: torch.Tensor) -> torch.Tensor:
    """Evaluate the released predicate once and detach it from optimization."""
    with torch.no_grad():
        return released_dynamic_contact_mask(joints).detach().clone()


def skate_loss_with_mask(
    joints: torch.Tensor,
    contact_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the released three-axis mean-squared velocity loss to one mask."""
    feet = joints[:, FOOT_JOINT_INDICES]
    velocity = feet[1:] - feet[:-1]
    if contact_mask.shape != velocity.shape[:-1]:
        raise ValueError(
            'foot contact mask shape differs from foot velocity: '
            f'{tuple(contact_mask.shape)} vs {tuple(velocity.shape[:-1])}'
        )
    contact_mask = contact_mask.to(device=joints.device, dtype=torch.bool)
    if not bool(contact_mask.any()):
        return velocity.sum() * 0.0
    return torch.mean(velocity[contact_mask] ** 2)


def released_dynamic_skate_loss(joints: torch.Tensor) -> torch.Tensor:
    """Exact factored form of the current per-person optimizer skate loss."""
    return skate_loss_with_mask(joints, released_dynamic_contact_mask(joints))
