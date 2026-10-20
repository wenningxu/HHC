import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diagnose_shape_contact_coordinates import relative_relation_timings

def run():
    from unittest.mock import patch
    from optimization.mesh_ground_support import MeshGroundSupportLoss
    from optimization.losses import DyadicInteractionLoss
    obj = object.__new__(MeshGroundSupportLoss)
    obj._relation_timings = torch.eye(2); obj.is_anchored = False
    verts = torch.zeros(2, 1, 3, requires_grad=True); calls = []
    pairs = [{'id': 0}, {'id': 1}]
    def base(self, a, b, na, nb, relations, timing):
        calls.append((relations, timing.clone()))
        return (a[:, 0, 0] * timing).sum() + timing[1], timing.sum()
    with patch.object(DyadicInteractionLoss, '_calc_semantic_contact_loss', base):
        values = obj._calc_semantic_contact_loss(verts, verts, verts, verts, pairs, torch.ones(2))
        assert values[0] == .5 and values[1] == 1 and len(calls) == 2
        assert calls[0][0] == [pairs[0]] and torch.equal(calls[0][1], torch.tensor([1., 0.]))
        assert calls[1][0] == [pairs[1]] and torch.equal(calls[1][1], torch.tensor([0., 1.]))
        values[0].backward(); assert torch.isfinite(verts.grad).all()
        obj.is_anchored = True
        try: obj._calc_semantic_contact_loss(verts, verts, verts, verts, pairs, torch.ones(2))
        except AssertionError: pass
        else: raise AssertionError('anchored pilot accepted')
        obj._relation_timings = None; calls.clear()
        values = obj._calc_semantic_contact_loss(verts, verts, verts, verts, pairs, torch.ones(2))
        assert len(calls) == 1 and calls[0][0] == pairs and values[1] == 2
    manual=torch.tensor([0.,.5,1.])
    assert torch.equal(relative_relation_timings(torch.zeros(3,0),manual,1.),torch.zeros(3,0))
    d=torch.tensor([[.1],[.2],[.3]]); assert torch.allclose(relative_relation_timings(d,manual,1.).squeeze(),manual)
    e=torch.tensor([[.1,.1],[.2,.2],[.3,.3]]); assert torch.allclose(relative_relation_timings(e,manual,1.),manual[:,None].expand_as(e))
    unequal = torch.tensor([[.1, .8], [.5, .2], [.4, .1]])
    weights = relative_relation_timings(unequal, manual, .05)
    assert not weights[0].any()
    assert torch.equal(relative_relation_timings(unequal.flip(1), manual, .05), weights.flip(1))
    assert relative_relation_timings(torch.zeros(0, 2), torch.zeros(0), 1.).shape == (0, 2)
    s=relative_relation_timings(torch.tensor([[.1,.2],[.3,.1]]),torch.ones(2),.1); assert s[0,0]>s[0,1] and s[1,1]>s[1,0]
    assert torch.allclose(relative_relation_timings(torch.tensor([[1.,2.]]),torch.ones(1),1.),relative_relation_timings(torch.tensor([[4.,5.]]),torch.ones(1),1.))
    assert relative_relation_timings(torch.zeros(1,0),torch.zeros(1),1.).shape==(1,0)
    for d0,m,s0 in [(torch.tensor([[-1.]]),torch.ones(1),1.),(torch.tensor([[float('nan')]]),torch.ones(1),1.),(torch.zeros(1,1),torch.ones(1),0.)]:
        try: relative_relation_timings(d0,m,s0)
        except AssertionError: pass
        else: raise AssertionError('invalid input accepted')
    print('PASS')
if __name__=='__main__': run()
