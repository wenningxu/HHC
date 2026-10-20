"""Experimental ground constraints and optional relation times; not production."""
import math
import torch
from optimization.losses import DyadicInteractionLoss
from optimization.metric_aligned_foot_sliding import FOOT_JOINT_INDICES


def horizontal_support_excess(joints, reference_speed, mask):
    speed = torch.linalg.vector_norm(torch.diff(joints[:, FOOT_JOINT_INDICES][..., [0, 2]], dim=0), dim=-1)
    assert speed.shape == reference_speed.shape == mask.shape
    assert torch.isfinite(speed).all() and torch.isfinite(reference_speed).all()
    return torch.relu(speed - reference_speed)[mask].mean() if bool(mask.any()) else speed.sum() * 0


def supported_mesh_ground_loss(vertices, foot_ids, reference_heights, support_mask, reduction='mean', temporal_weight=0.):
    if reduction not in ('mean', 'frame_max', 'height_decay'):
        raise ValueError(reduction)
    penetration = torch.relu(-vertices[..., 1]).amax(dim=1).mean()
    heights = vertices[:, foot_ids, 1]
    if reduction == 'height_decay':
        # Keep the moving reference height, not y=0; decay beyond 1cm over 2cm.
        weights = torch.exp(-torch.relu(reference_heights - .01) / .02)
        support = ((heights - reference_heights).abs() * weights).amax(dim=1).mean()
    elif bool(support_mask.any()):
        errors = (heights - reference_heights).abs()
        support = (errors[support_mask].mean() if reduction == 'mean' else
                   errors.masked_fill(~support_mask, 0).amax(dim=1)[support_mask.any(dim=1)].mean())
    else:
        support = heights.sum() * 0
    temporal = heights.sum() * 0
    if temporal_weight and len(heights) > 2:
        temporal = torch.diff(heights - reference_heights, n=2, dim=0).abs().amax(dim=1).mean()
    return penetration + support + temporal_weight * temporal


class MeshGroundSupportLoss(DyadicInteractionLoss):
    """Keep every non-ground term; retain initial near-plane foot vertices."""
    def __init__(self, *args, support_reduction='mean', ground_weight=20., reference_params=None, temporal_weight=0., horizontal_weight=0., relation_timing_scale=0., **kwargs):
        if support_reduction not in ('mean', 'frame_max', 'height_decay'):
            raise ValueError(support_reduction)
        if not math.isfinite(ground_weight) or ground_weight <= 0:
            raise ValueError('ground_weight must be finite and positive')
        if reference_params is not None and len(reference_params) != 2:
            raise ValueError('expected two reference people in input order')
        if not math.isfinite(temporal_weight) or temporal_weight < 0:
            raise ValueError('temporal_weight must be finite and nonnegative')
        if not math.isfinite(horizontal_weight) or horizontal_weight < 0:
            raise ValueError('horizontal_weight must be finite and nonnegative')
        if horizontal_weight and reference_params is None:
            raise ValueError('horizontal support requires original-shape reference')
        if not math.isfinite(relation_timing_scale) or relation_timing_scale < 0:
            raise ValueError('relation_timing_scale must be finite and nonnegative')
        if relation_timing_scale and reference_params is None:
            raise ValueError('relation timing requires original-shape reference')
        super().__init__(*args, **kwargs)
        self.support_reduction = support_reduction
        self.weights['ground'] = ground_weight
        self.reference_params = reference_params
        self.temporal_weight = temporal_weight
        self.horizontal_weight = horizontal_weight
        self.relation_timing_scale = relation_timing_scale
        self._relation_timings = None

    def _determine_role_assignment(self, vA, vB, nA, nB, interaction_pairs, temporal_mask):
        # Choose roles with the unchanged global timing before creating pseudo-times.
        super()._determine_role_assignment(vA, vB, nA, nB, interaction_pairs, temporal_mask)
        if not self.relation_timing_scale or len(interaction_pairs) < 2:
            return
        from pytorch3d.ops import knn_points
        from tools.diagnose_shape_contact_coordinates import relative_relation_timings
        with torch.no_grad():
            refs = self.reference_params[::-1] if self.swapped else self.reference_params
            vertices = [self._compute_geometry(p)[0] for p in refs]
            distances = []
            for pair in interaction_pairs:
                ids = [torch.unique(torch.cat([self.rid_to_vid_tensors[r] for r in pair[key]]))
                       for key in ('p1_ids', 'p2_ids')]
                x, y = [v[:, idx].contiguous() for v, idx in zip(vertices, ids)]
                distances.append(knn_points(x, y, K=1).dists[..., 0].amin(dim=1).clamp_min(0).sqrt())
            manual = temporal_mask.reshape(-1) if temporal_mask is not None else vA.new_ones(len(vA))
            self._relation_timings = relative_relation_timings(torch.stack(distances, dim=1), manual, self.relation_timing_scale)

    def _calc_semantic_contact_loss(self, verts_A, verts_B, norms_A, norms_B, interaction_pairs, temporal_mask):
        timings = getattr(self, '_relation_timings', None)
        if timings is None:
            return super()._calc_semantic_contact_loss(verts_A, verts_B, norms_A, norms_B, interaction_pairs, temporal_mask)
        # Per-pair anchored caches use original indices; this pilot stops before anchoring.
        assert not self.is_anchored, 'relation timing pilot does not support anchored optimization'
        assert timings.shape == (len(verts_A), len(interaction_pairs))
        values = [super(MeshGroundSupportLoss, self)._calc_semantic_contact_loss(
            verts_A, verts_B, norms_A, norms_B, [pair], timings[:, i])
            for i, pair in enumerate(interaction_pairs)]
        return tuple(sum(value[k] for value in values) / len(values) for k in (0, 1))

    def _calc_aux_losses(self, params_A, params_B, params_A_init, params_B_init,
                         joints_A, joints_B, verts_A, verts_B, temporal_mask=None):
        original = super()._calc_aux_losses(params_A, params_B, params_A_init,
            params_B_init, joints_A, joints_B, verts_A, verts_B, temporal_mask)
        if not hasattr(self, '_mesh_support_reference'):
            self._mesh_foot_ids = torch.unique(torch.cat([
                self.rid_to_vid_tensors[r] for r in (1, 42, 65, 67)]))
            reference = (params_A_init, params_B_init)
            if self.reference_params is not None:
                reference = self.reference_params[::-1] if self.swapped else self.reference_params
            with torch.no_grad():
                geometry = [self._compute_geometry(p) for p in reference]
                refs = [v[:, self._mesh_foot_ids, 1].detach() for v, _, _ in geometry]
                if self.horizontal_weight:
                    feet = [torch.unique(torch.cat([self.rid_to_vid_tensors[r] for r in group]))
                            for group in ((1, 67), (42, 65))]
                    self._horizontal_reference = []
                    for v, j, _ in geometry:
                        heights = torch.stack([v[:, ids, 1].min(1).values for ids in feet], 1)
                        mask = (heights[:-1] <= .01).repeat_interleave(2, dim=1)
                        speed = torch.linalg.vector_norm(torch.diff(j[:, FOOT_JOINT_INDICES][..., [0, 2]], dim=0), dim=-1)
                        self._horizontal_reference.append((speed.detach(), mask.detach()))
            assert all(torch.isfinite(h).all() for h in refs)
            assert all(h.shape == v[:, self._mesh_foot_ids, 1].shape for h, v in zip(refs, (verts_A, verts_B)))
            self._mesh_support_reference = [(h, h <= .01) for h in refs]
        ground = sum(supported_mesh_ground_loss(v, self._mesh_foot_ids, h, mask, self.support_reduction, self.temporal_weight)
                     for v, (h, mask) in zip((verts_A, verts_B), self._mesh_support_reference))
        if self.horizontal_weight:
            ground = ground + self.horizontal_weight * sum(
                horizontal_support_excess(j, speed, mask)
                for j, (speed, mask) in zip((joints_A, joints_B), self._horizontal_reference))
        return (*original[:-1], ground)


if __name__ == '__main__':
    test_joints = torch.zeros(3, 12, 3, requires_grad=True)
    moved = test_joints + torch.arange(3.).view(3, 1, 1) * torch.tensor([.02, 0., 0.])
    ref_speed = torch.full((2, 4), .01); active = torch.ones(2, 4, dtype=torch.bool)
    excess = horizontal_support_excess(moved, ref_speed, active)
    assert abs(float(excess) - .01) < 1e-7
    excess.backward(); assert torch.isfinite(test_joints.grad).all()
    assert float(horizontal_support_excess(moved, ref_speed * 3, active)) == 0
    assert float(horizontal_support_excess(moved, ref_speed, ~active)) == 0
    v = torch.zeros(2, 100, 3, requires_grad=True)
    ids = torch.tensor([0, 1]); ref = torch.zeros(2, 2); mask = ref == 0
    assert float(supported_mesh_ground_loss(v, ids, ref, mask)) == 0
    shifted = v + torch.tensor([0., .02, 0.])
    value = supported_mesh_ground_loss(shifted, ids, ref, mask)
    assert abs(float(value) - .02) < 1e-7
    value.backward(); assert torch.isfinite(v.grad).all()
    below = torch.zeros(2, 100, 3); below[:, -1, 1] = -.01
    assert abs(float(supported_mesh_ground_loss(below, ids, ref, mask)) - .01) < 1e-7
    local = torch.zeros(2, 100, 3); local[:, 0, 1] = .02
    local.requires_grad_()
    assert abs(float(supported_mesh_ground_loss(local, ids, ref, mask)) - .01) < 1e-7
    worst = supported_mesh_ground_loss(local, ids, ref, mask, 'frame_max')
    assert abs(float(worst) - .02) < 1e-7
    worst.backward(); assert torch.isfinite(local.grad).all()
    assert float(supported_mesh_ground_loss(local, ids, ref, ~mask, 'frame_max')) == 0
    high_ref = torch.full((2, 2), .05)
    high = torch.full((2, 100, 3), .06, requires_grad=True)
    soft = supported_mesh_ground_loss(high, ids, high_ref, high_ref <= .01, 'height_decay')
    assert abs(float(soft) - .01 * math.exp(-2)) < 1e-7
    soft.backward(); assert torch.isfinite(high.grad).all()
    spike = torch.zeros(3, 2, 3, requires_grad=True)
    spike_ref = torch.zeros(3, 2)
    spike_input = spike + torch.tensor([0., .01, 0.]).view(3, 1, 1) * torch.tensor([0., 1., 0.])
    base_loss = supported_mesh_ground_loss(spike_input, ids, spike_ref, spike_ref == 0)
    smooth_loss = supported_mesh_ground_loss(spike_input, ids, spike_ref, spike_ref == 0, temporal_weight=1.)
    assert abs(float(smooth_loss - base_loss) - .02) < 1e-7
    smooth_loss.backward(); assert torch.isfinite(spike.grad).all()
    print('PASS: zero reference, support height, finite gradients, no vertex dilution')
