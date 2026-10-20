"""Audited InterHuman VQ reconstruction losses.

Equations and weights follow the released InterMask implementation.  Paths to
normalization statistics are explicit so training does not depend on cwd.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


KINEMATIC_CHAINS = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)
FOOT_JOINTS = (7, 10, 8, 11)
FOOT_HEIGHT_THRESHOLDS = (0.12, 0.05, 0.12, 0.05)


def _cont6d_to_matrix(cont6d: torch.Tensor) -> torch.Tensor:
    """Match InterMask's column-oriented 6D rotation conversion exactly."""
    if cont6d.shape[-1] != 6:
        raise ValueError(f'rotation tensor must end in 6D, got {cont6d.shape}')
    x_raw = cont6d[..., :3]
    y_raw = cont6d[..., 3:]
    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    return torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)


class InterHumanVQLoss(nn.Module):
    """Official single-person geometric terms plus VQ commitment."""

    def __init__(
        self,
        recons_loss='l1_smooth',
        mean_path='/mnt/8T-HDD/wenning/InterMask/data/stats/global_mean.npy',
        std_path='/mnt/8T-HDD/wenning/InterMask/data/stats/global_std.npy',
        commit=0.02,
        loss_explicit=1.0,
        loss_vel=100.0,
        loss_bn=5.0,
        loss_geo=0.01,
        loss_fc=500.0,
    ):
        super().__init__()
        if recons_loss == 'l1':
            self.criterion = nn.L1Loss()
        elif recons_loss == 'l1_smooth':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f'unsupported InterHuman reconstruction loss {recons_loss!r}')
        mean = torch.from_numpy(np.load(mean_path).astype(np.float32))
        std = torch.from_numpy(np.load(std_path).astype(np.float32))
        if mean.shape != (262,) or std.shape != (262,) or torch.any(std <= 0):
            raise ValueError(f'invalid InterHuman normalization stats: {mean.shape}, {std.shape}')
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.weights = {
            'commitment': float(commit),
            'explicit_position': float(loss_explicit),
            'position_velocity': float(loss_vel),
            'bone_length': float(loss_bn),
            'rotation_geodesic': float(loss_geo),
            'foot_sliding': float(loss_fc),
        }

    def denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        return motion * self.std + self.mean

    @staticmethod
    def _positions(motion: torch.Tensor) -> torch.Tensor:
        return motion[..., :66].reshape(*motion.shape[:-1], 22, 3)

    def _bone_lengths(self, motion: torch.Tensor) -> torch.Tensor:
        positions = self._positions(motion)
        bones = []
        for chain in KINEMATIC_CHAINS:
            for start, end in zip(chain[:-1], chain[1:]):
                bones.append(
                    (positions[..., start, :] - positions[..., end, :])
                    .norm(dim=-1, keepdim=True)
                )
        return torch.cat(bones, dim=-1)

    def _rotation_geodesic(self, target: torch.Tensor, prediction: torch.Tensor):
        target_rotation = target[..., 132:258].reshape(*target.shape[:-1], -1, 6)
        predicted_rotation = prediction[..., 132:258].reshape(
            *prediction.shape[:-1], -1, 6
        )
        target_matrix = _cont6d_to_matrix(target_rotation).reshape(-1, 3, 3)
        predicted_matrix = _cont6d_to_matrix(predicted_rotation).reshape(-1, 3, 3)
        relative = torch.bmm(target_matrix, predicted_matrix.transpose(1, 2))
        cosine = (relative[:, 0, 0] + relative[:, 1, 1] + relative[:, 2, 2] - 1) / 2
        return torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)).mean()

    def _foot_sliding(self, target: torch.Tensor, prediction: torch.Tensor):
        target_positions = self._positions(self.denormalize(target))
        predicted_positions = self._positions(self.denormalize(prediction))
        target_velocity = (
            target_positions[:, 1:, FOOT_JOINTS, :]
            - target_positions[:, :-1, FOOT_JOINTS, :]
        )
        predicted_velocity = (
            predicted_positions[:, 1:, FOOT_JOINTS, :]
            - predicted_positions[:, :-1, FOOT_JOINTS, :]
        )
        target_height = target_positions[:, :-1, FOOT_JOINTS, 1]
        squared_speed = target_velocity.square().sum(dim=-1)
        speed_threshold = target_velocity.new_full((4,), 0.001)
        height_threshold = target_velocity.new_tensor(FOOT_HEIGHT_THRESHOLDS)
        contact = (squared_speed < speed_threshold) & (target_height < height_threshold)
        if not torch.any(contact):
            return prediction.new_zeros(())
        return self.criterion(predicted_velocity[contact], torch.zeros_like(predicted_velocity[contact]))

    def forward(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        commitment: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if target.shape != prediction.shape or target.ndim != 3 or target.shape[-1] != 262:
            raise ValueError(
                f'InterHuman target/prediction must match [B,T,262], got '
                f'{target.shape}, {prediction.shape}'
            )
        terms = {
            'reconstruction': self.criterion(prediction[..., :258], target[..., :258]),
            'explicit_position': self.criterion(prediction[..., :66], target[..., :66]),
            'position_velocity': self.criterion(
                prediction[:, 1:, :66] - prediction[:, :-1, :66],
                target[:, 1:, :66] - target[:, :-1, :66],
            ),
            'bone_length': self.criterion(
                self._bone_lengths(prediction), self._bone_lengths(target)
            ),
            'rotation_geodesic': self._rotation_geodesic(target, prediction),
            'foot_sliding': self._foot_sliding(target, prediction),
            'commitment': commitment,
        }
        total = terms['reconstruction']
        for name, weight in self.weights.items():
            total = total + weight * terms[name]
        return {'total': total, **terms}


class InterHumanPairVQLoss(nn.Module):
    """Apply official 262D losses symmetrically to a paired 524D HHC motion.

    Per-person geometric terms are averaged.  The one joint VQ commitment term
    is then added exactly once, preventing its weight from changing merely
    because the two official person representations were concatenated.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.single = InterHumanVQLoss(**kwargs)

    @property
    def weights(self):
        return self.single.weights

    def forward(self, target, prediction, commitment):
        if target.shape != prediction.shape or target.ndim != 3 or target.shape[-1] != 524:
            raise ValueError(
                f'InterHuman paired target/prediction must match [B,T,524], got '
                f'{target.shape}, {prediction.shape}'
            )
        person_results = [
            self.single(
                target[..., offset:offset + 262],
                prediction[..., offset:offset + 262],
                commitment.new_zeros(()),
            )
            for offset in (0, 262)
        ]
        names = (
            'reconstruction', 'explicit_position', 'position_velocity',
            'bone_length', 'rotation_geodesic', 'foot_sliding',
        )
        terms = {
            name: torch.stack([result[name] for result in person_results]).mean()
            for name in names
        }
        terms['commitment'] = commitment
        total = terms['reconstruction']
        for name, weight in self.weights.items():
            total = total + weight * terms[name]
        return {'total': total, **terms}
