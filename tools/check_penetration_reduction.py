"""Check frame-sum reduction without changing nearest distances or thresholds."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.losses import DyadicInteractionLoss


def loss_shell(reduction):
    loss = DyadicInteractionLoss.__new__(DyadicInteractionLoss)
    torch.nn.Module.__init__(loss)
    loss.device = 'cuda:1'
    loss.signed_distance_backend = 'nearest_vertex'
    loss.penetration_loss_reduction = reduction
    return loss


if __name__ == '__main__':
    a = torch.tensor([[[-.05, 0., 0.], [-.1, .1, 0.]]], device='cuda:1').repeat(2, 1, 1)
    b = torch.tensor([[[0., 0., 0.], [0., .1, 0.]]], device='cuda:1').repeat(2, 1, 1)
    n = torch.zeros_like(a)
    n[..., 0] = 1
    old, new = loss_shell('penetrating_vertex_mean'), loss_shell('frame_sum')
    torch.testing.assert_close(old._calc_penetration_loss(a, b, n, n), a.new_tensor(.0125))
    a.requires_grad_()
    value = new._calc_penetration_loss(a, b, n, n)
    torch.testing.assert_close(value, a.new_tensor(.025))
    expected = new._calc_penetration_metric(a, b, n, n) / a.shape[0]
    torch.testing.assert_close(value, expected)
    gradient, = torch.autograd.grad(value, (a,))
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    repeat_frames = new._calc_penetration_loss(a.repeat(2, 1, 1), b.repeat(2, 1, 1),
                                               n.repeat(2, 1, 1), n.repeat(2, 1, 1))
    torch.testing.assert_close(repeat_frames, value)
    repeat_vertices = new._calc_penetration_loss(a.repeat(1, 2, 1), b,
                                                 n.repeat(1, 2, 1), n)
    torch.testing.assert_close(repeat_vertices, value * 2)
    clear = a.detach() + a.new_tensor([1., 0., 0.])
    assert new._calc_penetration_loss(clear, b, n, n) == 0
    print('PASS: released value, frame sum, gradient, frame/density scaling, zero collision')
