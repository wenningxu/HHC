"""Same rotation/FK equations with reference micro4 valid-frame reductions."""
import torch
from dataset.interhuman_root_rotation import fk_positions, rotation_matrix


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
    contact = gt[:, :-1, :, 258:262, None]
    losses['fk_contact'] = 30*avg(feet.square()*contact, pair_mask)
    distance = torch.cdist(fk[:, :, 0], fk[:, :, 1])
    gt_distance = torch.cdist(positions[:, :, 0], positions[:, :, 1])
    losses['fk_distance'] = 3*avg((distance-gt_distance).square()*(gt_distance < 1), geometric)
    losses['fk_affinity'] = 3*avg(distance.square()*(gt_distance < .1), geometric)
    losses['total'] = sum(losses.values())
    return losses

