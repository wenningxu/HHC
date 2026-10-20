"""Inter-X-style paired layout, native InterHuman SMPL semantics (not SMPL-X).

Archive: [T,23,6] axis angles (22 joints) + translation row; full source FPS.
Model view: [T,23,12], each joint has [p1_rot6d,p2_rot6d] (HHC row convention).
Last row: [p1_translation,p1_backward_velocity,p2_translation,p2_backward_velocity].
Velocities are metres/frame, first frame zero. No centering or coordinate change.
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints

from utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix

SCHEMA = 'interhuman_native_smpl_interx_layout_v1'
FEATURE_DIM = 23 * 12
BODY_SLICE = slice(0, 22 * 12)
ROOT_SLICE = slice(22 * 12, 23 * 12)


def pack_raw(raw):
    people = []
    for name in ('person1', 'person2'):
        person = raw[name]
        length = len(person['trans'])
        assert person['root_orient'].shape == (length, 3)
        assert person['pose_body'].shape == (length, 63)
        assert person['betas'].shape == (10,)
        people.append(np.concatenate([person['root_orient'][:, None],
                                      person['pose_body'].reshape(length, 21, 3),
                                      person['trans'][:, None]], axis=1))
    axis_angle = np.concatenate(people, axis=-1)
    betas = np.stack([raw[p]['betas'] for p in ('person1', 'person2')])
    assert np.isfinite(axis_angle).all() and np.isfinite(betas).all()
    return axis_angle, betas


def unpack_raw(axis_angle, betas):
    if axis_angle.ndim != 3 or axis_angle.shape[1:] != (23, 6) or betas.shape != (2, 10):
        raise ValueError('Expected paired [T,23,6] SMPL archive and [2,10] betas')
    return {name: {'root_orient': axis_angle[:, 0, 3*i:3*i+3].copy(),
                   'pose_body': axis_angle[:, 1:22, 3*i:3*i+3].reshape(len(axis_angle), 63).copy(),
                   'trans': axis_angle[:, 22, 3*i:3*i+3].copy(), 'betas': betas[i].copy()}
            for i, name in enumerate(('person1', 'person2'))}


def refresh_velocity(motion):
    """Recompute redundant root velocities, including after cropping."""
    result = motion.clone()
    for offset in (0, 6):
        result[..., 0, 22, offset+3:offset+6] = 0
        result[..., 1:, 22, offset+3:offset+6] = (
            result[..., 1:, 22, offset:offset+3] - result[..., :-1, 22, offset:offset+3])
    return result


def model_view(axis_angle, frame_rate):
    if len(axis_angle) == 0:
        raise ValueError('Empty source motion has no model view')
    if not np.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError('Invalid source FPS')
    indices = np.arange(0, len(axis_angle), float(frame_rate)/30).astype(np.int64)
    source = torch.from_numpy(axis_angle[indices]).float()
    rotations = source[:, :22].reshape(len(indices), 22, 2, 3)
    matrices = batch_rodrigues(rotations.reshape(-1, 3)).reshape(len(indices), 22, 2, 3, 3)
    motion = torch.zeros(len(indices), 23, 12)
    motion[:, :22] = matrix_to_rotation_6d(matrices).reshape(len(indices), 22, 12)
    motion[:, 22, :3] = source[:, 22, :3]
    motion[:, 22, 6:9] = source[:, 22, 3:]
    return refresh_velocity(motion), indices


def body192_from_motion(motion, betas, model):
    """Differentiable native SMPL FK; no GT features/labels are used.

    Input [T,23,12], beta [2,10]; output [2,T,192] in released pre-online axes.
    Predicted redundant velocity fields are not integrated: position is authoritative.
    """
    if motion.ndim != 3 or motion.shape[1:] != (23, 12) or betas.shape != (2, 10):
        raise ValueError('Expected motion [T,23,12] and betas [2,10]')
    if not torch.isfinite(motion).all() or not torch.isfinite(betas).all():
        raise ValueError('Nonfinite motion or shape')
    rot6 = motion[:, :22].reshape(len(motion), 22, 2, 6)
    if torch.any(torch.linalg.vector_norm(torch.cross(rot6[..., :3], rot6[..., 3:], dim=-1), dim=-1) < 1e-8):
        raise ValueError('Degenerate predicted 6D rotation')
    matrices = rotation_6d_to_matrix(rot6)
    output = []
    for person in range(2):
        shaped = model.v_template[None] + blend_shapes(betas[person:person+1], model.shapedirs)
        rest = vertices2joints(model.J_regressor, shaped).expand(len(motion), -1, -1)
        identity = torch.eye(3, dtype=motion.dtype, device=motion.device)
        full_rotation = torch.cat([matrices[:, :, person], identity.expand(len(motion), 2, 3, 3)], dim=1)
        joints, _ = batch_rigid_transform(full_rotation, rest, model.parents)
        positions = joints[:, :22] + motion[:, 22, 6*person:6*person+3][:, None]
        body_rotations = matrices[:, 1:22, person].clone()
        body_rotations[:, -2:] = identity  # Official InterHuman wrist serialization.
        # InterHuman stored rotations are interleaved columns, unlike HHC rot6d.
        output.append(torch.cat([positions.reshape(len(motion), 66),
                                 body_rotations[:, :, :, :2].reshape(len(motion), 126)], dim=-1))
    return torch.stack(output)


def to_interhuman(motion, betas, model, adapter, normalize=False):
    """Convert a generated motion to [T,524] evaluator features (length T-1)."""
    with torch.no_grad():
        pair = body192_from_motion(motion, betas, model).detach().cpu().numpy()
    people = adapter.process_loaded_pair(pair[0], pair[1], normalize=normalize)
    return np.concatenate(people, axis=-1)


class InterHumanSMPLPairDataset(Dataset):
    """Training-ready view with explicit shape and per-view train-only statistics."""
    def __init__(self, root, split='train', window_size=None, normalize=True):
        self.root = Path(root)
        if split not in ('train', 'val', 'test'):
            raise ValueError(split)
        self.window_size, self.normalize = window_size, normalize
        with h5py.File(self.root/'motions.h5', 'r') as handle:
            if handle.attrs['schema'] != SCHEMA or handle.attrs.get('status') != 'complete':
                raise ValueError('Incomplete or incompatible SMPL-pair dataset')
            self.ids = [s for s in (self.root/'split'/f'{split}.txt').read_text().splitlines()
                        if s in handle and 'motion30' in handle[s]
                        and (window_size is None or len(handle[s]['motion30']) >= window_size)]
        stats = np.load(self.root/'normalization.npz')
        self.mean = torch.from_numpy(stats['mean'])
        self.std = torch.from_numpy(stats['std'])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        with h5py.File(self.root/'motions.h5', 'r') as handle:
            group = handle[name]
            motion = torch.from_numpy(group['motion30'][:])
            beta = torch.from_numpy(group['betas'][:])
        if self.window_size is not None:
            start = int(torch.randint(len(motion)-self.window_size+1, ()))
            motion = refresh_velocity(motion[start:start+self.window_size])
        motion = motion.reshape(len(motion), FEATURE_DIM)
        if self.normalize:
            motion = (motion-self.mean)/self.std
        return {'name': name, 'motion': motion, 'betas': beta.reshape(20),
                'body_model_gender': torch.ones(2, dtype=torch.long), 'length': len(motion)}
