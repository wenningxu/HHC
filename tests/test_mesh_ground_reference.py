import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optimization.mesh_ground_support as m


def run():
    obj = object.__new__(m.MeshGroundSupportLoss)
    obj.support_reduction = 'mean'
    obj.temporal_weight = 0.
    obj.horizontal_weight = 0.
    obj.swapped = False
    obj.reference_params = (torch.tensor(1.), torch.tensor(2.))
    obj.rid_to_vid_tensors = {1: torch.tensor([0]), 42: torch.tensor([1]),
                              65: torch.tensor([1]), 67: torch.tensor([0])}
    obj.weights = {'ground': 20.}
    obj._compute_geometry = lambda p: (torch.zeros(1, 2, 3) + p, None, None)
    obj._calc_aux_losses_base = lambda *a: ('a', 'old-ground')
    # Patch the inherited method on this instance; no model construction.
    import optimization.losses as losses
    old = losses.DyadicInteractionLoss._calc_aux_losses
    losses.DyadicInteractionLoss._calc_aux_losses = lambda self, *a: self._calc_aux_losses_base(*a)
    try:
        verts = (torch.zeros(1, 2, 3),) * 2
        args = (None, None, torch.tensor(9.), torch.tensor(8.), None, None, *verts)
        out = obj._calc_aux_losses(*args)
        assert out[:-1] == ('a',) and float(out[-1]) == 0
        assert torch.equal(obj._mesh_support_reference[0][0], torch.ones(1, 2))
        cached = obj._mesh_support_reference
        obj.reference_params = (torch.tensor(3.), torch.tensor(4.))
        assert torch.equal(obj._calc_aux_losses(*args)[-1], out[-1])
        assert obj._mesh_support_reference is cached
        obj2 = object.__new__(m.MeshGroundSupportLoss)
        obj2.__dict__.update(obj.__dict__); obj2.swapped = True
        obj2._mesh_support_reference = None
        obj2.__dict__.pop('_mesh_support_reference')
        obj2._calc_aux_losses(*args)
        assert torch.equal(obj2._mesh_support_reference[0][0], torch.full((1, 2), 4.))
        obj3 = object.__new__(m.MeshGroundSupportLoss); obj3.__dict__.update(obj.__dict__)
        obj3.reference_params = None; obj3.swapped = True; obj3.__dict__.pop('_mesh_support_reference')
        obj3._calc_aux_losses(*args)
        assert torch.equal(obj3._mesh_support_reference[0][0], torch.full((1, 2), 9.))
        # Horizontal references are built from the selected (and swapped) people.
        h = object.__new__(m.MeshGroundSupportLoss)
        h.__dict__.update(obj.__dict__); h.horizontal_weight = .5; h.swapped = True
        h.__dict__.pop('_mesh_support_reference')
        h._calc_aux_losses_base = lambda *a: (1., 2., 3., 4., 'old-ground')
        def geometry(p):
            v = torch.zeros(3, 2, 3)
            v[:, 0 if float(p) == 4. else 1, 1] = .02
            j = torch.zeros(3, 12, 3)
            j[:, [7, 10, 8, 11], 0] = torch.arange(3.)[:, None] * p
            return v, j, None
        h._compute_geometry = geometry
        hv = (torch.zeros(3, 2, 3),) * 2
        ha = (None, None, torch.tensor(9.), torch.tensor(8.),
              geometry(torch.tensor(9.))[1], geometry(torch.tensor(8.))[1], *hv)
        out_h = h._calc_aux_losses(*ha)
        assert out_h[:-1] == (1., 2., 3., 4.)
        assert h._horizontal_reference[0][0][0, 0].item() == 4.
        assert h._horizontal_reference[1][0][0, 0].item() == 3.
        assert torch.equal(h._horizontal_reference[0][1], torch.tensor([[False, False, True, True]]).expand(2, -1))
        assert torch.equal(h._horizontal_reference[1][1], torch.tensor([[True, True, False, False]]).expand(2, -1))
        assert abs(float(out_h[-1]) - 5.) < 1e-7
        cached_h = h._horizontal_reference
        assert torch.equal(h._calc_aux_losses(*ha)[-1], out_h[-1])
        assert h._horizontal_reference is cached_h
    finally:
        losses.DyadicInteractionLoss._calc_aux_losses = old


if __name__ == '__main__':
    run(); print('PASS')
