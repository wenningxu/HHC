from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimization.metric_aligned_foot_sliding import (
    FOOT_JOINT_INDICES,
    freeze_initial_grounded_foot_mask,
    grounded_foot_mask,
    initial_or_current_grounded_horizontal_mean_speed,
)


def joints_with_feet(frames=3):
    joints = torch.zeros(frames, 12, 3)
    joints[:, FOOT_JOINT_INDICES, 1] = 1.0
    return joints


def test_matches_horizontal_first_order_metric_on_grounded_transitions():
    joints = joints_with_feet().requires_grad_(True)
    with torch.no_grad():
        joints[:, 7, 1] = 0.0
        joints[1, 7, 0] = 3.0
        joints[2, 7, 0] = 7.0
    frozen = freeze_initial_grounded_foot_mask(joints)
    loss = initial_or_current_grounded_horizontal_mean_speed(joints, frozen)
    torch.testing.assert_close(loss, torch.tensor(3.5))
    loss.backward()
    assert joints.grad is not None
    assert torch.isfinite(joints.grad).all()


def test_frozen_initial_mask_prevents_lifting_threshold_evasion():
    initial = joints_with_feet()
    initial[:, 10, 1] = 0.0
    frozen = freeze_initial_grounded_foot_mask(initial)
    current = initial.clone().requires_grad_(True)
    with torch.no_grad():
        current[:, 10, 1] = 2.0
        current[1, 10, 2] = 0.25
        current[2, 10, 2] = 0.75
    assert not bool(grounded_foot_mask(current)[:, 1].any())
    loss = initial_or_current_grounded_horizontal_mean_speed(current, frozen)
    torch.testing.assert_close(loss, torch.tensor(0.375))


def test_current_grounded_union_captures_new_contacts():
    initial = joints_with_feet()
    frozen = freeze_initial_grounded_foot_mask(initial)
    assert not bool(frozen.any())
    current = initial.clone()
    current[:, 8, 1] = 0.0
    current[1, 8, 0] = 0.2
    current[2, 8, 0] = 0.6
    loss = initial_or_current_grounded_horizontal_mean_speed(current, frozen)
    torch.testing.assert_close(loss, torch.tensor(0.3))


def test_empty_active_set_is_differentiable_zero():
    joints = joints_with_feet().requires_grad_(True)
    frozen = freeze_initial_grounded_foot_mask(joints)
    loss = initial_or_current_grounded_horizontal_mean_speed(joints, frozen)
    assert loss.item() == 0.0
    loss.backward()
    assert joints.grad is not None
    assert torch.count_nonzero(joints.grad) == 0


if __name__ == "__main__":
    test_matches_horizontal_first_order_metric_on_grounded_transitions()
    test_frozen_initial_mask_prevents_lifting_threshold_evasion()
    test_current_grounded_union_captures_new_contacts()
    test_empty_active_set_is_differentiable_zero()
