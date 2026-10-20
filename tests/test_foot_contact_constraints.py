import sys
import json
import ast
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization.foot_contact_constraints import (
    freeze_initial_contact_mask,
    released_dynamic_contact_mask,
    released_dynamic_skate_loss,
    skate_loss_with_mask,
)


def legacy_formula(joints):
    indices = [7, 10, 8, 11]
    velocity = joints[1:, indices] - joints[:-1, indices]
    height = joints[:-1, indices, 1]
    thresholds = joints.new_tensor([0.12, 0.05, 0.12, 0.05])
    contact = (
        torch.sum(velocity ** 2, dim=-1) < 0.001
    ) & (height < thresholds)
    if not torch.any(contact):
        return velocity.sum() * 0.0, contact
    return torch.mean(velocity[contact] ** 2), contact


def test_factored_dynamic_loss_is_exactly_the_released_formula():
    generator = torch.Generator().manual_seed(123)
    joints = torch.randn((12, 55, 3), generator=generator) * 0.01
    expected_loss, expected_mask = legacy_formula(joints)
    actual_mask = released_dynamic_contact_mask(joints)
    actual_loss = released_dynamic_skate_loss(joints)
    assert torch.equal(actual_mask, expected_mask)
    assert torch.equal(actual_loss, expected_loss)


def test_frozen_mask_keeps_penalizing_motion_that_escapes_dynamic_mask():
    joints = torch.zeros((3, 12, 3), dtype=torch.float32)
    frozen = freeze_initial_contact_mask(joints)
    moved = joints.clone().requires_grad_(True)
    moved.data[1:, (7, 10, 8, 11), 0] = 0.04
    moved.data[2:, (7, 10, 8, 11), 0] = 0.08
    assert not bool(released_dynamic_contact_mask(moved).any())
    loss = skate_loss_with_mask(moved, frozen)
    assert float(loss) > 0.0
    loss.backward()
    assert moved.grad is not None
    assert float(torch.linalg.vector_norm(moved.grad)) > 0.0


def test_empty_frozen_mask_returns_differentiable_zero():
    joints = torch.zeros((2, 12, 3), requires_grad=True)
    empty = torch.zeros((1, 4), dtype=torch.bool)
    loss = skate_loss_with_mask(joints, empty)
    assert float(loss) == 0.0
    loss.backward()
    assert joints.grad is not None


def test_frozen_mask_pilot_keeps_the_original_authorization_gate():
    preregistration = json.loads((
        ROOT / 'configs/interx_shape_augmentation/'
        'frozen_initial_foot_contact_pilot_v1.json'
    ).read_text(encoding='utf-8'))
    gate = json.loads((
        ROOT / 'configs/interx_shape_augmentation/'
        'pilot_physical_gate_v1.json'
    ).read_text(encoding='utf-8'))
    assert preregistration['status'] == 'conditionally_preregistered'
    assert preregistration['created_before_released_pilot_completion'] is True
    assert preregistration['selection']['sample_count'] == 25
    assert preregistration['authorization_gate']['criteria'] == gate['criteria']
    frozen = preregistration['frozen_unchanged_settings']
    assert frozen['foot_joint_indices'] == [7, 10, 8, 11]
    assert frozen['foot_height_thresholds_m'] == [0.12, 0.05, 0.12, 0.05]
    assert frozen['foot_squared_3d_speed_threshold'] == 0.001
    assert frozen['foot_skate_weight'] == 100.0


def test_frozen_mask_policy_is_threaded_through_the_optimizer_runner():
    losses = (ROOT / 'optimization/losses.py').read_text(encoding='utf-8')
    optimizer = (ROOT / 'optimize/optimize_full_body.py').read_text(
        encoding='utf-8'
    )
    runner = (ROOT / 'tools/run_interx_shape_augmentation.py').read_text(
        encoding='utf-8'
    )
    for path, source in (
        ('optimization/losses.py', losses),
        ('optimize/optimize_full_body.py', optimizer),
        ('tools/run_interx_shape_augmentation.py', runner),
    ):
        ast.parse(source, filename=path)
        assert 'foot_contact_mask_policy' in source
    assert "default='released_dynamic'" in runner
    assert "choices=('released_dynamic', 'frozen_initial')" in runner
    assert 'freeze_initial_contact_mask(j1)' in losses
    assert 'freeze_initial_contact_mask(j2)' in losses
    assert 'self.frozen_foot_contact_masks[1]' in losses
    assert 'self.frozen_foot_contact_masks[0]' in losses


def test_frozen_pilot_finalizer_preserves_comparisons_on_a_failed_gate():
    source = (ROOT / 'tools/finalize_frozen_foot_pilot.sh').read_text(
        encoding='utf-8'
    )
    causal = '--before "${released_physical}" --after "${frozen_physical}"'
    assessment = 'tools/assess_shape_augmentation_pilot.py'
    assert source.index(causal) < source.index(assessment)
    assert '|| assessment_status=$?' in source
    assert 'exit "${assessment_status}"' in source


if __name__ == '__main__':
    test_factored_dynamic_loss_is_exactly_the_released_formula()
    test_frozen_mask_keeps_penalizing_motion_that_escapes_dynamic_mask()
    test_empty_frozen_mask_returns_differentiable_zero()
    test_frozen_mask_pilot_keeps_the_original_authorization_gate()
    test_frozen_mask_policy_is_threaded_through_the_optimizer_runner()
    test_frozen_pilot_finalizer_preserves_comparisons_on_a_failed_gate()
