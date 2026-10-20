"""Rotation-primary InterGen objective; geometry gradients go through FK."""
import torch
from dataset.interhuman_root_rotation import fk_positions, rotation_matrix


def rotation_objective(prediction, target, offsets, lengths, timesteps, mean, std):
    b, t = prediction.shape[:2]
    prediction = prediction.reshape(b, t, 2, 268).float()
    target = target.reshape_as(prediction).float()
    raw, gt = prediction*std+mean, target*std+mean
    valid = torch.arange(t, device=prediction.device)[None] < lengths[:, None]
    geometric = valid & (timesteps[:, None] <= 700)

    def avg(values, mask=valid):
        per_frame = values.flatten(2).mean(-1)
        return (per_frame*mask).sum()/mask.sum().clamp_min(1)

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


def self_check():
    from dataset.interhuman_root_rotation import RootRotationData
    torch.set_num_threads(2)
    data = RootRotationData()
    x, offsets = data.pair(data.ids[0], mirror=True, anchor=2, swap=True, max_frames=32)
    target = torch.from_numpy(x)[None]
    pred = (target+.01*torch.randn_like(target)).requires_grad_()
    mean, std = torch.zeros(268), torch.ones(268)
    losses = rotation_objective(pred, target, torch.from_numpy(offsets)[None], torch.tensor([len(x)]), torch.tensor([10]), mean, std)
    losses['fk_position'].backward(retain_graph=True)
    assert pred.grad[..., 132:258].abs().sum() > 0
    assert pred.grad[..., 262:].abs().sum() > 0
    assert pred.grad[..., 3:132].abs().sum() == 0, 'FK loss must not bypass rotations via position head'
    pred.grad.zero_()
    losses['total'].backward()
    assert torch.isfinite(pred.grad).all()
    # Padding must have no contribution to any objective, including differences.
    padded = torch.cat([pred.detach(), torch.randn_like(pred)], 1).requires_grad_()
    target_pad = torch.cat([target, torch.randn_like(target)], 1)
    longer = rotation_objective(padded, target_pad, torch.from_numpy(offsets)[None], torch.tensor([len(x)]), torch.tensor([10]), mean, std)
    assert torch.allclose(longer['total'], losses['total'], atol=1e-6)
    longer['total'].backward()
    assert padded.grad[:, len(x):].abs().sum() == 0
    print({'status': 'passed', 'losses': {k: float(v.detach()) for k, v in losses.items()}}, flush=True)


if __name__ == '__main__':
    self_check()
