"""Run directly: python tests/test_diagnostic_attraction_weight.py."""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.check_shape_ground_after_optimization import with_attraction_weight, gradient_snapshot


def check():
    def factory(device):
        return SimpleNamespace(device=device, weights={'attract': 1., 'ground': 200., 'skate': 100.})
    for value in (1., 10.):
        loss = with_attraction_weight(factory, value, device='cpu')
        assert loss.device == 'cpu'
        assert loss.weights == {'attract': value, 'ground': 200., 'skate': 100.}
    for invalid in (0., -1., float('nan'), float('inf')):
        try:
            with_attraction_weight(factory, invalid, device='cpu')
        except AssertionError:
            continue
        raise AssertionError('invalid weight accepted')
    x = torch.tensor([1., 2.], requires_grad=True)
    terms = {'attract': x.sum(), 'opposed': -2*x.sum(), 'total': -x.sum()}
    snapshot = gradient_snapshot(terms, {'x': x})
    assert snapshot['parameters']['x']['opposed']['cosine_attract'] < -.99999
    assert x.grad is None
    terms['total'].backward()
    assert torch.equal(x.grad, torch.tensor([-1., -1.]))


if __name__ == '__main__':
    check()
