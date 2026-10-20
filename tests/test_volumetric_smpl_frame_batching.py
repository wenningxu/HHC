from types import SimpleNamespace
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimization.volumetric_smpl_backend import VolumetricSMPLBackend
from optimization.volumetric_smpl_batching import (
    _query_frames_preserving_serial_sampling,
    penetration_squared_sum_frame_batched,
)


class AnalyticBatchBackend(VolumetricSMPLBackend):
    """Cheap differentiable stand-in for serial/batched packing tests."""

    def __init__(self):
        nn.Module.__init__(self)
        self.models = {
            'male': SimpleNamespace(
                volume=SimpleNamespace(bbox_padding=1.0)
            )
        }

    def _query_frame(self, query_points, target_output, target_gender):
        del target_gender
        target_center_x = target_output.vertices[..., 0].mean(
            dim=1, keepdim=True
        )
        return query_points[..., 0] - target_center_x


class StochasticFakeVolume:
    """Minimal volume whose target encoder consumes RNG per frame."""

    def __init__(self):
        self.bbox_padding = 1.0
        self.n_samples = 5
        self.partitioner = self
        self.impl_code = None

    def compute_bone_trans(self, full_pose, joints):
        del full_pose, joints
        identity = torch.eye(4).reshape(1, 1, 4, 4)
        return identity

    def get_bbox_bounds(self, vertices, bone_trans):
        del bone_trans
        return (
            vertices.amin(dim=1).reshape(1, 1, 1, 3),
            vertices.amax(dim=1).reshape(1, 1, 1, 3),
        )

    def sample_mesh_points(self, vertices, samples):
        center = vertices.mean(dim=1).reshape(1, 1, 1, 3)
        return center + torch.rand(1, 1, samples, 3)

    def encoder(self, samples):
        return samples.mean(dim=1)

    def query(self, points):
        latent_x = self.impl_code['latent_code'][:, 0, 0].unsqueeze(1)
        return points[..., 0] + latent_x


class StochasticFakeBackend:
    def __init__(self):
        self.models = {
            'male': SimpleNamespace(volume=StochasticFakeVolume())
        }

    @staticmethod
    def normalize_gender(gender):
        return gender


def test_frame_batching_preserves_penetration_value_and_gradients():
    backend = AnalyticBatchBackend()
    query_base = torch.tensor([
        [[-0.5, 0.0, 0.0], [0.2, 0.0, 0.0], [3.0, 0.0, 0.0]],
        [[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [3.0, 0.0, 0.0]],
        [[-0.3, 0.0, 0.0], [0.6, 0.0, 0.0], [3.0, 0.0, 0.0]],
    ])
    target_base = torch.tensor([
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        [[-1.1, -1.0, -1.0], [0.9, 1.0, 1.0]],
        [[-0.8, -1.0, -1.0], [1.2, 1.0, 1.0]],
    ])
    temporal = torch.tensor([1.0, 0.0, 1.0])

    def run(batched):
        query = query_base.clone().requires_grad_(True)
        target = target_base.clone().requires_grad_(True)
        output = SimpleNamespace(
            vertices=target,
            joints=target[:, :1],
            full_pose=target[:, :1],
        )
        if batched:
            value = penetration_squared_sum_frame_batched(
                backend,
                query,
                output,
                'male',
                temporal,
                shallow_threshold=10.0,
                interaction_threshold=10.0,
                frame_batch_size=3,
            )
        else:
            value = backend.penetration_squared_sum(
                query,
                output,
                'male',
                temporal,
                shallow_threshold=10.0,
                interaction_threshold=10.0,
            )
        value.backward()
        return value.detach(), query.grad.detach(), target.grad.detach()

    serial = run(False)
    batched = run(True)
    for expected, actual in zip(serial, batched):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_backend_routes_selected_frame_batch_size_without_changing_result():
    backend = AnalyticBatchBackend()
    query = torch.tensor([
        [[-0.5, 0.0, 0.0], [0.2, 0.0, 0.0]],
        [[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]],
    ])
    target = torch.tensor([
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
    ])
    output = SimpleNamespace(
        vertices=target,
        joints=target[:, :1],
        full_pose=target[:, :1],
    )
    temporal = torch.ones(2)
    serial = backend.penetration_squared_sum(
        query, output, 'male', temporal,
        shallow_threshold=10.0,
        interaction_threshold=10.0,
    )
    batched = backend.penetration_squared_sum(
        query, output, 'male', temporal,
        shallow_threshold=10.0,
        interaction_threshold=10.0,
        frame_batch_size=8,
    )
    torch.testing.assert_close(batched, serial, rtol=0.0, atol=0.0)

    try:
        backend.penetration_squared_sum(
            query, output, 'male', temporal, frame_batch_size=0
        )
    except ValueError as error:
        assert 'positive' in str(error)
    else:
        raise AssertionError('non-positive frame_batch_size was accepted')


def test_batched_query_preserves_serial_per_frame_random_draws_and_gradients():
    backend = StochasticFakeBackend()
    query_base = torch.arange(24, dtype=torch.float32).reshape(3, 8, 1)
    query_base = torch.cat((query_base, query_base * 0, query_base * 0), dim=-1)
    vertex_base = torch.arange(45, dtype=torch.float32).reshape(3, 5, 3) / 10

    def run(frame_batched):
        query = query_base.clone().requires_grad_(True)
        vertices = vertex_base.clone().requires_grad_(True)
        output = SimpleNamespace(
            vertices=vertices,
            joints=vertices[:, :1],
            full_pose=vertices[:, :1],
        )
        torch.manual_seed(1927)
        if frame_batched:
            result = _query_frames_preserving_serial_sampling(
                backend, query, output, 'male'
            )
        else:
            rows = []
            for index in range(query.shape[0]):
                frame = slice(index, index + 1)
                rows.append(_query_frames_preserving_serial_sampling(
                    backend,
                    query[frame],
                    SimpleNamespace(
                        vertices=output.vertices[frame],
                        joints=output.joints[frame],
                        full_pose=output.full_pose[frame],
                    ),
                    'male',
                ))
            result = torch.cat(rows, dim=0)
        result.sum().backward()
        return result.detach(), query.grad.detach(), vertices.grad.detach()

    serial = run(False)
    batched = run(True)
    for expected, actual in zip(serial, batched):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if __name__ == '__main__':
    test_frame_batching_preserves_penetration_value_and_gradients()
    test_backend_routes_selected_frame_batch_size_without_changing_result()
    test_batched_query_preserves_serial_per_frame_random_draws_and_gradients()
