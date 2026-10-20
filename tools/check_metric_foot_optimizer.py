"""Small CPU check: fast grounded feet receive gradients; other aux terms stay fixed."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.losses import DyadicInteractionLoss


def loss_shell(policy):
    loss = DyadicInteractionLoss.__new__(DyadicInteractionLoss)
    torch.nn.Module.__init__(loss)
    loss.foot_contact_mask_policy = policy
    loss.aux_temporal_weighting = 'released_broadcast'
    loss._initial_grounded_foot_masks = {}
    loss.lfoot_idx, loss.rfoot_idx = [7, 10], [8, 11]
    return loss


if __name__ == '__main__':
    policy = 'initial_or_current_grounded_horizontal_mean_speed'
    feet = [7, 10, 8, 11]
    joints = torch.zeros(3, 55, 3)
    joints[:, feet, 0] = torch.arange(3)[:, None] * .1
    joints.requires_grad_()
    loss = loss_shell(policy)
    value = loss._metric_aligned_skate_loss(joints, 'A')
    torch.testing.assert_close(value, torch.tensor(.1))
    value.backward()
    assert torch.isfinite(joints.grad).all() and joints.grad.abs().sum() > 0
    translated = joints.detach() + torch.tensor([0., 2., 0.])
    torch.testing.assert_close(loss._metric_aligned_skate_loss(translated, 'A'), value)
    lifted = joints.detach().clone()
    lifted[:, feet, 1] = 1
    torch.testing.assert_close(loss._metric_aligned_skate_loss(lifted, 'A'), value)
    assert loss_shell(policy)._metric_aligned_skate_loss(lifted, 'A') == 0
    assert loss_shell(policy)._metric_aligned_skate_loss(joints[:1], 'A') == 0

    slow = joints.detach() * .1
    params = {'pose_body': torch.zeros(3, 21, 3), 'trans': torch.zeros(3, 3)}
    verts = torch.zeros(3, 10, 3)
    args = (params, params, params, params, slow, slow, verts, verts)
    old = loss_shell('released_dynamic')._calc_aux_losses(*args, temporal_mask=torch.ones(3))
    new = loss_shell(policy)._calc_aux_losses(*args, temporal_mask=torch.ones(3))
    for index in (0, 1, 2, 4):
        assert torch.equal(old[index], new[index]), index
    torch.testing.assert_close(old[3], torch.tensor(2 * .01 ** 2 / 3))
    torch.testing.assert_close(new[3], torch.tensor(.02))
    print('PASS: fast-foot gradient, floor shift, frozen mask, empty/T1 masks, unchanged other aux losses')
