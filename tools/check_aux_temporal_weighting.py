"""CPU check for DyadicInteractionLoss auxiliary temporal weighting."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.losses import DyadicInteractionLoss


def shell(mode):
    loss = DyadicInteractionLoss.__new__(DyadicInteractionLoss)
    torch.nn.Module.__init__(loss)
    loss.aux_temporal_weighting = mode
    loss.foot_contact_mask_policy = "released_dynamic"
    loss.lfoot_idx, loss.rfoot_idx = [7, 10], [8, 11]
    loss._initial_grounded_foot_masks = {}
    return loss


def inputs(t=4, requires_grad=False):
    pose = torch.arange(t * 21 * 3, dtype=torch.float64).reshape(t, 21, 3) / 100
    root = torch.arange(t * 3, dtype=torch.float64).reshape(t, 3) / 10
    trans = torch.arange(t * 3, dtype=torch.float64).reshape(t, 3) / 20
    params = {"pose_body": pose, "root_orient": root, "trans": trans}
    init = {key: value * 0.3 for key, value in params.items()}
    if requires_grad:
        params = {key: value.clone().requires_grad_() for key, value in params.items()}
    joints = torch.arange(t * 55 * 3, dtype=torch.float64).reshape(t, 55, 3).square() / 1e7
    joints.requires_grad_(requires_grad)
    verts = torch.ones(t, 10, 3, dtype=torch.float64)
    return params, init, joints, verts


def explicit(params, init, joints, mask, per_frame):
    w = 1 + 2 * mask.view(-1, 1, 1)
    if per_frame:
        weights = w.flatten()
        reg = sum(scale * sum(weights[t] * (params[key][t] - init[key][t]).square().mean()
                              for t in range(len(weights))) / len(weights)
                  for key, scale in (("pose_body", 1), ("root_orient", 2), ("trans", 5)))
        smooth = sum(weights[t] * (joints[t] - joints[t-1]).square().mean()
                     for t in range(1, len(weights))) / (len(weights) - 1)
        acc = (sum(weights[t] * (joints[t] - 2*joints[t-1] + joints[t-2]).square().sum()
                   for t in range(2, len(weights))) / (weights[2:].sum().float() + 1e-6)
               if len(weights) >= 3 else joints.new_zeros(()))
        return 2 * reg, 2 * smooth, 2 * acc
    reg = 0
    for key, scale in (("pose_body", 1), ("root_orient", 2), ("trans", 5)):
        weight = w.reshape(w.shape[0], *([1] * (params[key].ndim - 1))) if per_frame else w
        reg = reg + scale * torch.mean((params[key] - init[key]) ** 2 * weight)
    def smooth_acc(x):
        vel = x[1:] - x[:-1]
        wv = w[1:].reshape(w.shape[0] - 1, 1, 1) if per_frame else w[1:].unsqueeze(-1)
        smooth = torch.mean(vel.square() * wv)
        acc = x[2:] - 2 * x[1:-1] + x[:-2]
        wa = w[2:].reshape(w.shape[0] - 2, 1, 1) if per_frame else w[2:].unsqueeze(-1)
        acc_loss = torch.sum(acc.square() * wa) / (wa.sum().float() + 1e-6) if x.shape[0] >= 3 else x.new_zeros(())
        return smooth, acc_loss
    smooth, acc = smooth_acc(joints)
    return reg * 2, smooth * 2, acc * 2


if __name__ == "__main__":
    mask = torch.tensor([0., 1., 0., 1.], dtype=torch.float64)
    args = inputs()
    for mode, per_frame in (("released_broadcast", False), ("per_frame", True)):
        got = shell(mode)._calc_aux_losses(args[0], args[0], args[1], args[1], args[2], args[2], args[3], args[3], mask)
        expected = explicit(args[0], args[1], args[2], mask, per_frame)
        for index in range(3):
            torch.testing.assert_close(got[index], expected[index], rtol=1e-10, atol=1e-12)
    p, i, j, v = inputs(requires_grad=True)
    out = shell("per_frame")._calc_aux_losses(p, p, i, i, j, j, v, v, mask)
    sum(out[:3]).backward()
    assert all(torch.isfinite(x.grad).all() for x in p.values())
    p_ref, i_ref, j_ref, _ = inputs(requires_grad=True)
    sum(explicit(p_ref, i_ref, j_ref, mask, True)).backward()
    for key in p:
        torch.testing.assert_close(p[key].grad, p_ref[key].grad, rtol=1e-10, atol=1e-12)
    assert j.grad is not None and torch.isfinite(j.grad).all() and j.grad.abs().sum() > 0
    torch.testing.assert_close(j.grad, j_ref.grad, rtol=1e-10, atol=1e-12)
    assert out[2] > 0
    for t, m in ((4, mask), (2, torch.tensor([0., 1.], dtype=torch.float64))):
        p, i, j, v = inputs(t)
        got = shell("per_frame")._calc_aux_losses(p, p, i, i, j, j, v, v, m)
        assert all(torch.isfinite(x) for x in got[:3])
        if t == 2:
            assert got[2] == 0
    p, i, j, v = inputs()
    got = shell("per_frame")._calc_aux_losses(p, p, i, i, j, j, v, v, None)
    assert all(torch.isfinite(x) for x in got[:3])
    for actual, reference in zip(got[:3], explicit(p, i, j, torch.zeros(4), True)):
        torch.testing.assert_close(actual, reference)
    print("PASS: old/per-frame aux formulas, gradients, mask=None, T=2")
