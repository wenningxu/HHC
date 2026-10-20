#!/usr/bin/env python3
"""Compare exact serial and experimental frame-batched Eq. 6 execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimization.volumetric_smpl_backend import VolumetricSMPLBackend
from optimization.volumetric_smpl_batching import (
    penetration_squared_sum_frame_batched,
)


GRADIENT_KEYS = (
    'root_orient', 'trans', 'pose_body', 'pose_lhand', 'pose_rhand',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_params(path, frames, device):
    with np.load(path, allow_pickle=False) as archive:
        poses = archive['poses'][:frames].astype(np.float32)
        gender = str(archive['gender'].item())
        arrays = {
            'root_orient': poses[:, 0],
            'pose_body': poses[:, 1:22].reshape(frames, 63),
            'pose_lhand': poses[:, 25:40].reshape(frames, 45),
            'pose_rhand': poses[:, 40:55].reshape(frames, 45),
            'trans': archive['trans'][:frames].astype(np.float32),
        }
        betas = archive['betas'].reshape(1, 10).astype(np.float32)
    params = {
        key: torch.as_tensor(value, device=device).clone().requires_grad_(True)
        for key, value in arrays.items()
    }
    params['betas'] = torch.as_tensor(betas, device=device)
    params['gender'] = gender
    return params


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_once(backend, p1_path, p2_path, frames, device, frame_batch_size):
    params_a = load_params(p1_path, frames, device)
    params_b = load_params(p2_path, frames, device)
    temporal = torch.ones(frames, dtype=torch.float32, device=device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    backend.begin_iteration(seed=1234)
    synchronize(device)
    started = time.perf_counter()
    output_a, gender_a = backend.body(params_a)
    output_b, gender_b = backend.body(params_b)
    if frame_batch_size == 1:
        query = backend.penetration_squared_sum
        kwargs = {}
        mode = 'production_serial'
    else:
        query = lambda *positional, **keywords: (
            penetration_squared_sum_frame_batched(
                backend,
                *positional,
                **keywords,
                frame_batch_size=frame_batch_size,
            )
        )
        kwargs = {}
        mode = 'experimental_frame_batched'
    loss = query(
        output_b.vertices,
        output_a,
        gender_a,
        temporal,
        shallow_threshold=0.05,
        interaction_threshold=10.0,
        **kwargs,
    ) + query(
        output_a.vertices,
        output_b,
        gender_b,
        temporal,
        shallow_threshold=0.05,
        interaction_threshold=10.0,
        **kwargs,
    )
    loss.backward()
    synchronize(device)
    elapsed = time.perf_counter() - started
    gradients = {}
    for person, params in (('p1', params_a), ('p2', params_b)):
        for key in GRADIENT_KEYS:
            gradient = params[key].grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(f'absent/non-finite gradient: {person}/{key}')
            gradients[f'{person}/{key}'] = gradient.detach().cpu()
    peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == 'cuda' else 0
    )
    return {
        'mode': mode,
        'frame_batch_size': int(frame_batch_size),
        'loss': float(loss.detach().cpu()),
        'elapsed_seconds': elapsed,
        'peak_allocated_bytes': int(peak),
        'gradients': gradients,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p1', type=Path, required=True)
    parser.add_argument('--p2', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--frame-batch-sizes', type=int, nargs='+', default=(2, 4))
    parser.add_argument(
        '--query-chunk', type=int, default=10475,
        help='Promoted flat-hand serial optimizer query chunk.',
    )
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.query_chunk < 1:
        raise ValueError('frames and query chunk must be positive')
    if any(size <= 1 for size in args.frame_batch_sizes):
        raise ValueError('experimental frame batch sizes must exceed one')

    device = torch.device(args.device)
    genders = set()
    available_frames = []
    for path in (args.p1, args.p2):
        with np.load(path, allow_pickle=False) as archive:
            genders.add(str(archive['gender'].item()))
            available_frames.append(int(archive['poses'].shape[0]))
    frames = min(args.frames, *available_frames)
    backend = VolumetricSMPLBackend(
        model_root='body_models',
        device=device,
        genders=tuple(sorted(genders)),
        max_query_points_per_call=args.query_chunk,
    )

    serial = run_once(backend, args.p1, args.p2, frames, device, 1)
    experimental = []
    for size in args.frame_batch_sizes:
        try:
            experimental.append(
                run_once(backend, args.p1, args.p2, frames, device, size)
            )
        except torch.cuda.OutOfMemoryError as error:
            peak = (
                torch.cuda.max_memory_allocated(device)
                if device.type == 'cuda' else 0
            )
            experimental.append({
                'mode': 'experimental_frame_batched',
                'frame_batch_size': int(size),
                'status': 'cuda_out_of_memory',
                'error_type': type(error).__name__,
                'error_summary': str(error).splitlines()[0],
                'peak_allocated_bytes': int(peak),
            })
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    serial_gradients = serial.pop('gradients')
    comparisons = []
    for result in experimental:
        if result.get('status') == 'cuda_out_of_memory':
            comparisons.append({
                **result,
                'speedup_over_serial': None,
                'loss_absolute_difference': None,
                'loss_relative_difference': None,
                'gradient_max_absolute_difference': None,
                'gradient_relative_to_serial_max': None,
            })
            continue
        gradients = result.pop('gradients')
        gradient_max_abs = max(
            float((gradients[key] - serial_gradients[key]).abs().max())
            for key in serial_gradients
        )
        gradient_reference_max_abs = max(
            float(serial_gradients[key].abs().max())
            for key in serial_gradients
        )
        loss_abs = abs(result['loss'] - serial['loss'])
        comparisons.append({
            **result,
            'speedup_over_serial': (
                serial['elapsed_seconds'] / result['elapsed_seconds']
            ),
            'loss_absolute_difference': loss_abs,
            'loss_relative_difference': loss_abs / max(abs(serial['loss']), 1e-12),
            'gradient_max_absolute_difference': gradient_max_abs,
            'gradient_relative_to_serial_max': (
                gradient_max_abs / max(gradient_reference_max_abs, 1e-12)
            ),
        })

    production_path = ROOT / 'optimization/volumetric_smpl_backend.py'
    experimental_path = ROOT / 'optimization/volumetric_smpl_batching.py'
    payload = {
        'schema': 'hhc_volumetric_smpl_frame_batch_benchmark_v1',
        'status': 'benchmark_complete_not_promoted',
        'device': str(device),
        'frames': frames,
        'query_chunk': args.query_chunk,
        'inputs': {
            'p1': {'path': str(args.p1.resolve()), 'sha256': sha256_file(args.p1)},
            'p2': {'path': str(args.p2.resolve()), 'sha256': sha256_file(args.p2)},
        },
        'serial': serial,
        'experimental': comparisons,
        'promotion_gate': {
            'maximum_loss_relative_difference': 1e-5,
            'maximum_gradient_relative_difference': 1e-4,
            'minimum_speedup': 1.25,
            'fixed_query_chunk': args.query_chunk,
            'interpretation': (
                'All numerical gates and CUDA memory safety must pass on more than one '
                'motion before editing the production optimization backend.'
            ),
        },
        'implementation': {
            'production': {
                'path': str(production_path),
                'sha256': sha256_file(production_path),
            },
            'experimental': {
                'path': str(experimental_path),
                'sha256': sha256_file(experimental_path),
            },
            'benchmark': {
                'path': str(Path(__file__).resolve()),
                'sha256': sha256_file(Path(__file__).resolve()),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
