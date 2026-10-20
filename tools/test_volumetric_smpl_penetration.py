#!/usr/bin/env python
"""Validate exact, checkpointed Eq. 6 against a direct full-vertex query."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimization.volumetric_smpl_backend import VolumetricSMPLBackend


def make_params(frames, device, translation_x):
    return {
        'pose_body': torch.zeros(frames, 63, device=device, requires_grad=True),
        'pose_lhand': torch.zeros(frames, 45, device=device, requires_grad=True),
        'pose_rhand': torch.zeros(frames, 45, device=device, requires_grad=True),
        'root_orient': torch.zeros(frames, 3, device=device, requires_grad=True),
        'trans': torch.tensor(
            [[translation_x, 0.0, 0.0]], device=device
        ).expand(frames, -1).clone().requires_grad_(True),
        'betas': torch.zeros(1, 10, device=device),
        'gender': 'male',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--translation-x', type=float, default=0.15)
    parser.add_argument('--query-chunk', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output')
    args = parser.parse_args()

    device = torch.device(args.device)
    backend = VolumetricSMPLBackend(
        device=device,
        genders=('male',),
        max_query_points_per_call=args.query_chunk,
    )
    source_params = make_params(args.frames, device, args.translation_x)
    target_params = make_params(args.frames, device, 0.0)

    backend.begin_iteration(args.seed)
    source_output, _ = backend.body(source_params)
    target_output, target_gender = backend.body(target_params)
    temporal_mask = torch.ones(args.frames, device=device)
    target_vertices = target_output.vertices.detach()
    raw_min = target_vertices.amin(dim=1)
    raw_max = target_vertices.amax(dim=1)
    margin = (raw_max - raw_min) * (
        float(backend.models['male'].volume.bbox_padding) - 1.0
    ) * 0.5
    candidate_mask = (
        (source_output.vertices.detach() >= (raw_min - margin)[:, None])
        & (source_output.vertices.detach() <= (raw_max + margin)[:, None])
    ).all(dim=-1)

    started = time.perf_counter()
    checkpointed_loss = backend.penetration_squared_sum(
        source_output.vertices,
        target_output,
        target_gender,
        temporal_mask,
        shallow_threshold=0.05,
        interaction_threshold=10.0,
    )
    forward_seconds = time.perf_counter() - started
    checkpointed_loss.backward()
    backward_seconds = time.perf_counter() - started - forward_seconds

    gradient_tensors = {
        'source_trans': source_params['trans'].grad,
        'target_trans': target_params['trans'].grad,
        'source_pose': source_params['pose_body'].grad,
        'target_pose': target_params['pose_body'].grad,
    }
    finite_gradients = {
        name: value is not None and bool(torch.isfinite(value).all())
        for name, value in gradient_tensors.items()
    }
    if not all(finite_gradients.values()):
        raise RuntimeError(f'non-finite gradients: {finite_gradients}')

    # Reset the random mesh sampler and compare against querying every vertex.
    with torch.no_grad():
        backend.begin_iteration(args.seed)
        direct_sdf = torch.cat([
            backend._query_frame(
                source_output.vertices[frame_index:frame_index + 1].detach(),
                backend._slice_output(
                    backend.detached_output(target_output),
                    slice(frame_index, frame_index + 1),
                ),
                target_gender,
            )
            for frame_index in range(args.frames)
        ])
        direct_active = (direct_sdf < 0) & (direct_sdf.abs() < 10.0)
        direct_loss = (direct_sdf[direct_active] ** 2).sum()
    if not torch.allclose(
        checkpointed_loss.detach(), direct_loss, rtol=1e-5, atol=1e-7
    ):
        raise RuntimeError(
            'AABB-pruned checkpointed loss differs from direct full query: '
            f'{checkpointed_loss.item()} vs {direct_loss.item()}'
        )

    report = {
        'device': str(device),
        'frames': args.frames,
        'vertices_per_frame': int(source_output.vertices.shape[1]),
        'aabb_candidate_counts': [int(value) for value in candidate_mask.sum(1)],
        'active_penetration_counts': [int(value) for value in direct_active.sum(1)],
        'checkpointed_loss': float(checkpointed_loss.detach()),
        'direct_full_vertex_loss': float(direct_loss),
        'forward_seconds': forward_seconds,
        'backward_seconds': backward_seconds,
        'finite_gradients': finite_gradients,
        'query_chunk': args.query_chunk,
        'seed': args.seed,
        'translation_x': args.translation_x,
        'checkpoint': backend.checkpoint_records['male'],
    }
    print(report)
    if args.output:
        target = Path(args.output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + f'.tmp.{os.getpid()}')
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        os.replace(temporary, target)


if __name__ == '__main__':
    main()
