"""Offline correction of sparse denominators; NEVER used by archived custom run.

Retains the custom GT distance masks and grouped valid-frame reduction.
This does not make the complete objective equivalent to official InterGen.
"""
import torch
from dataset.interhuman_root_rotation import fk_positions, rotation_matrix

def selected_pair_mean(values, selected):
    """Mean over selected joint pairs per frame, zero when none are selected."""
    return (values*selected).sum(dim=(-2,-1))/selected.sum(dim=(-2,-1)).clamp_min(1)


def selected_contact_person_sum(values, selected):
    """XYZ mean, selected feet mean per person, then sum the two people."""
    xyz_mean=values.mean(dim=-1)
    per_person=(xyz_mean*selected).sum(-1)/selected.sum(-1).clamp_min(1)
    return per_person.sum(-1)


def rotation_objective(prediction, target, offsets, lengths, timesteps, mean, std, reference_micro_batch=4):
    b, t = prediction.shape[:2]
    assert b % reference_micro_batch == 0
    prediction = prediction.reshape(b, t, 2, 268).float()
    target = target.reshape_as(prediction).float()
    raw, gt = prediction*std+mean, target*std+mean
    valid = torch.arange(t, device=prediction.device)[None] < lengths[:, None]
    geometric = valid & (timesteps[:, None] <= 700)

    def avg(values, mask=valid):
        per_frame = values.flatten(2).mean(-1)
        # Preserve the original average of losses from microbatches of4.
        numerator = (per_frame*mask).reshape(-1, reference_micro_batch, per_frame.shape[1]).sum(dim=(1, 2))
        denominator = mask.reshape(-1, reference_micro_batch, mask.shape[1]).sum(dim=(1, 2)).clamp_min(1)
        return (numerator/denominator).mean()

    local = raw[..., 132:258].reshape(b, t, 2, 21, 6)
    local_gt = gt[..., 132:258].reshape_as(local)
    fk = fk_positions(raw, offsets)
    positions = gt[..., :66].reshape_as(fk)
    losses = {
        'rotation6d': avg((prediction[..., 132:258]-target[..., 132:258]).square()),
        'root6d': avg((prediction[..., 262:]-target[..., 262:]).square()),
        'rotation_matrix': .5*avg((rotation_matrix(local)-rotation_matrix(local_gt)).square()),
        'root_matrix': .5*avg((rotation_matrix(raw[..., 262:])-rotation_matrix(gt[..., 262:])).square()),
        'fk_position': 4*avg((fk-positions).square(), geometric),
        'root_position': avg(((raw[..., :3]-gt[..., :3])/std[:3]).square()),
        # Keep evaluator's original channels trainable, but not the main target.
        'aux_position': .1*avg((prediction[..., :66]-target[..., :66]).square()),
        'aux_velocity': .1*avg((prediction[..., 66:132]-target[..., 66:132]).square()),
        'aux_contact': .1*avg((prediction[..., 258:262]-target[..., 258:262]).square()),
    }
    pair_mask = geometric[:, 1:] & geometric[:, :-1]
    velocity = fk[:, 1:]-fk[:, :-1]
    losses['fk_velocity'] = 30*avg((velocity-(positions[:, 1:]-positions[:, :-1])).square(), pair_mask)
    feet = velocity[..., [7, 10, 8, 11], :]
    contact = gt[:, :-1, :, 258:262]
    losses['fk_contact'] = 30*avg(selected_contact_person_sum(feet.square(), contact)[..., None], pair_mask)
    distance = torch.cdist(fk[:, :, 0], fk[:, :, 1])
    gt_distance = torch.cdist(positions[:, :, 0], positions[:, :, 1])
    losses['fk_distance'] = 3*avg(selected_pair_mean((distance-gt_distance).square(), gt_distance < 1)[..., None], geometric)
    losses['fk_affinity'] = 3*avg(selected_pair_mean(distance.square(), gt_distance < .1)[..., None], geometric)
    losses['total'] = sum(losses.values())
    return losses

