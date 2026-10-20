#!/usr/bin/env python
"""Execution smoke for the paper's differentiable VolumetricSMPL backend."""

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.volumetric_smpl_backend import VolumetricSMPLBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', default='body_models')
    parser.add_argument('--gender', choices=('male', 'female'), default='male')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--query-count', type=int, default=32)
    parser.add_argument('--query-chunk', type=int, default=1024)
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    if args.query_count < 1 or args.query_chunk < 1 or args.frames < 1:
        raise ValueError('query count, query chunk, and frames must be positive')

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    backend = VolumetricSMPLBackend(
        model_root=args.model_root,
        device=device,
        genders=(args.gender,),
        max_query_points_per_call=args.query_chunk,
    )

    body_pose = torch.zeros(args.frames, 63, device=device, requires_grad=True)
    left_hand_pose = torch.zeros(args.frames, 45, device=device, requires_grad=True)
    right_hand_pose = torch.zeros(args.frames, 45, device=device, requires_grad=True)
    global_orient = torch.zeros(args.frames, 3, device=device, requires_grad=True)
    transl = torch.zeros(args.frames, 3, device=device, requires_grad=True)
    betas = torch.zeros(1, 10, device=device)

    params = {
        'pose_body': body_pose,
        'pose_lhand': left_hand_pose,
        'pose_rhand': right_hand_pose,
        'root_orient': global_orient,
        'trans': transl,
        'betas': betas,
        'gender': args.gender,
    }
    backend.begin_iteration()
    output, gender = backend.body(params)
    query_count = min(args.query_count, output.vertices.shape[1])
    fixed_queries = output.vertices[:, :query_count].detach().clone()
    fixed_queries[..., 0] += 0.01
    query_started = time.perf_counter()
    signed_distance = backend.signed_distance(fixed_queries, output, gender)
    query_seconds = time.perf_counter() - query_started
    if signed_distance.shape != (args.frames, query_count):
        raise RuntimeError(
            f'unexpected signed-distance shape: {tuple(signed_distance.shape)}'
        )
    if not torch.isfinite(signed_distance).all():
        raise RuntimeError('VolumetricSMPL returned NaN or Inf')

    signed_distance.square().mean().backward()
    gradients = {
        'body_pose': body_pose.grad,
        'left_hand_pose': left_hand_pose.grad,
        'right_hand_pose': right_hand_pose.grad,
        'global_orient': global_orient.grad,
        'transl': transl.grad,
    }
    finite = {
        name: gradient is not None and bool(torch.isfinite(gradient).all())
        for name, gradient in gradients.items()
    }
    if not all(finite.values()):
        raise RuntimeError(f'non-finite or absent parameter gradients: {finite}')
    if not any(bool(torch.count_nonzero(gradient)) for gradient in gradients.values()):
        raise RuntimeError('VolumetricSMPL query produced only zero gradients')

    # Eq. 4 freezes the target character and should still propagate through
    # the source query points via activation checkpointing.
    for parameter in (
        body_pose, left_hand_pose, right_hand_pose, global_orient, transl
    ):
        parameter.grad = None
    backend.begin_iteration(seed=args.seed)
    source_output, source_gender = backend.body(params)
    source_only_sdf = backend.signed_distance(
        source_output.vertices[:, :query_count],
        backend.detached_output(source_output),
        source_gender,
    )
    source_only_sdf.square().mean().backward()
    source_only_gradients = {
        'body_pose': body_pose.grad,
        'left_hand_pose': left_hand_pose.grad,
        'right_hand_pose': right_hand_pose.grad,
        'global_orient': global_orient.grad,
        'transl': transl.grad,
    }
    source_only_finite = {
        name: gradient is not None and bool(torch.isfinite(gradient).all())
        for name, gradient in source_only_gradients.items()
    }
    if not all(source_only_finite.values()):
        raise RuntimeError(
            f'non-finite source-only contact gradients: {source_only_finite}'
        )

    checkpoint_record = backend.checkpoint_records[args.gender]
    checkpoint = Path(checkpoint_record['path'])
    report = {
        'device': str(device),
        'gender': args.gender,
        'seed': args.seed,
        'frames': args.frames,
        'query_shape': tuple(signed_distance.shape),
        'query_chunk': args.query_chunk,
        'query_seconds': query_seconds,
        'sdf_min': float(signed_distance.min().detach()),
        'sdf_max': float(signed_distance.max().detach()),
        'finite_gradients': finite,
        'nonzero_gradient_counts': {
            name: int(torch.count_nonzero(gradient))
            for name, gradient in gradients.items()
        },
        'source_only_checkpoint_finite_gradients': source_only_finite,
        'volumetric_smpl_version': importlib.metadata.version('VolumetricSMPL'),
        'checkpoint': str(checkpoint.resolve()),
        'checkpoint_bytes': checkpoint_record['bytes'],
        'checkpoint_sha256': checkpoint_record['sha256'],
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
