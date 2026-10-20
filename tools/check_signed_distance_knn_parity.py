"""Check one-direction KNN against the installed original Chamfer wrapper."""
import argparse
import sys
from pathlib import Path

import torch
from chamfer_distance import ChamferDistance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.losses import point2point_signed


def reference(x, y, normals):
    _, _, index, _ = ChamferDistance()(x, y)
    expanded = index[..., None].expand(-1, -1, 3).long()
    target = y.gather(1, expanded)
    vector = x - target
    distance = vector.norm(dim=-1)
    if normals is not None:
        distance = distance * (vector * normals.gather(1, expanded)).sum(-1).sign()
    return distance, index, vector, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    torch.manual_seed(20260911)
    for count_x, count_y in ((17, 23), (1, 1), (10475, 10475)):
        x = torch.randn(2, count_x, 3, device=args.device)
        y = torch.randn(2, count_y, 3, device=args.device)
        # Ties and exactly coincident vertices exercise nearest-index conventions.
        x[:, 0] = y[:, 0]
        if count_y > 1:
            y[:, 1] = y[:, 0]
        for signed in (False, True):
            normal = torch.randn_like(y) if signed else None
            outcomes = []
            for function in (reference, lambda a, b, n: point2point_signed(
                    a, b, y_normals=n, return_vector=True)):
                a, b = x.detach().requires_grad_(), y.detach().requires_grad_()
                result = function(a, b, normal)
                loss = result[0].square().mean() + result[2].square().mean()
                gradients = torch.autograd.grad(loss, (a, b))
                outcomes.append((result, gradients))
            for before, after in zip(outcomes[0][0], outcomes[1][0]):
                assert torch.equal(before, after), (count_x, count_y, signed)
            for before, after in zip(outcomes[0][1], outcomes[1][1]):
                torch.testing.assert_close(before, after, atol=1e-7, rtol=1e-6)
            print(count_x, count_y, signed, 'exact values/indices; gradients match', flush=True)


if __name__ == '__main__':
    main()
