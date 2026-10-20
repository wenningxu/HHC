#!/usr/bin/env python
"""Prove frozen VolumetricSMPL backend reuse is numerically transparent."""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimization.volumetric_smpl_backend import VolumetricSMPLBackend
from optimize.optimize_full_body import optimize_full_body


SCHEMA = 'hhc_shared_volumetric_backend_equivalence_v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_person(frames, gender, translation_x):
    return {
        'pose_body': np.zeros((frames, 63), dtype=np.float32),
        'pose_lhand': np.zeros((frames, 45), dtype=np.float32),
        'pose_rhand': np.zeros((frames, 45), dtype=np.float32),
        'root_orient': np.zeros((frames, 3), dtype=np.float32),
        'trans': np.tile(
            np.asarray([[translation_x, 0.0, 0.0]], dtype=np.float32),
            (frames, 1),
        ),
        'betas': np.zeros((1, 10), dtype=np.float32),
        'gender': gender,
    }


def run_once(args, backend):
    return optimize_full_body(
        make_person(args.frames, 'male', 0.0),
        make_person(args.frames, 'female', 0.15),
        temporal_mask=np.ones(args.frames, dtype=np.float32),
        num_epochs=args.epochs,
        freeze_epoch=max(0, args.epochs // 2),
        device=args.device,
        volumetric_smpl_seed=args.seed,
        volumetric_smpl_query_chunk=args.query_chunk,
        volumetric_smpl_backend=backend,
        return_diagnostics=True,
    )


def compare_outputs(left, right):
    records = {}
    maximum = 0.0
    bitwise = True
    finite = True
    for person_index in (0, 1):
        left_person = left[person_index]
        right_person = right[person_index]
        if set(left_person) != set(right_person):
            raise ValueError('optimizer output keys changed between backend modes')
        for key in sorted(left_person):
            left_array = np.asarray(left_person[key])
            right_array = np.asarray(right_person[key])
            if left_array.shape != right_array.shape:
                raise ValueError(f'output shape mismatch for person {person_index}/{key}')
            equal = bool(np.array_equal(left_array, right_array))
            field_finite = bool(
                np.isfinite(left_array).all()
                and np.isfinite(right_array).all()
            )
            difference = float(np.max(np.abs(
                left_array.astype(np.float64) - right_array.astype(np.float64)
            ))) if left_array.size else 0.0
            records[f'p{person_index + 1}/{key}'] = {
                'bitwise_equal': equal,
                'finite': field_finite,
                'max_abs_difference': difference,
            }
            maximum = max(maximum, difference)
            bitwise = bitwise and equal
            finite = finite and field_finite
    return {
        'bitwise_equal': bitwise,
        'finite': finite,
        'max_abs_difference': maximum,
        'fields': records,
    }


def loss_history_is_finite(diagnostics):
    return bool(
        diagnostics.get('loss_history')
        and all(
            np.isfinite(float(value))
            for row in diagnostics['loss_history']
            for key, value in row.items()
            if key not in ('iteration', 'contact_stage')
        )
    )


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--query-chunk', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument(
        '--output', type=Path,
        default=Path(
            'output/optimization_backend_audit/'
            'shared_backend_equivalence_cpu_f3_e1.json'
        ),
    )
    args = parser.parse_args()
    if args.frames < 3 or args.epochs < 1 or args.threads < 1:
        raise ValueError(
            'frames must be at least 3 and epochs/threads must be positive'
        )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    shared_backend = VolumetricSMPLBackend(
        model_root='body_models', device=args.device,
        max_query_points_per_call=args.query_chunk,
    )
    fresh = run_once(args, None)
    shared_first = run_once(args, shared_backend)
    shared_second = run_once(args, shared_backend)
    fresh_vs_shared = compare_outputs(fresh, shared_first)
    repeated_shared = compare_outputs(shared_first, shared_second)
    finite_loss_histories = {
        'fresh': loss_history_is_finite(fresh[2]),
        'shared_first': loss_history_is_finite(shared_first[2]),
        'shared_second': loss_history_is_finite(shared_second[2]),
    }
    checkpoint_records = shared_backend.checkpoint_records
    checkpoint_hashes_valid = all(
        sha256_file(record['path']) == record['sha256']
        for record in checkpoint_records.values()
    )
    status = (
        'pass' if (
            fresh_vs_shared['bitwise_equal']
            and repeated_shared['bitwise_equal']
            and fresh_vs_shared['finite']
            and repeated_shared['finite']
            and all(finite_loss_histories.values())
            and checkpoint_hashes_valid
            and fresh[2]['volumetric_smpl_backend_reused'] is False
            and shared_first[2]['volumetric_smpl_backend_reused'] is True
            and shared_second[2]['volumetric_smpl_backend_reused'] is True
        ) else 'fail'
    )
    payload = {
        'schema': SCHEMA,
        'status': status,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'contract': {
            'device': args.device,
            'frames': args.frames,
            'epochs': args.epochs,
            'query_chunk': args.query_chunk,
            'seed': args.seed,
            'threads': args.threads,
            'deterministic_algorithms': True,
        },
        'fresh_vs_shared': fresh_vs_shared,
        'shared_first_vs_second': repeated_shared,
        'finite_loss_histories': finite_loss_histories,
        'checkpoint_hashes_valid': checkpoint_hashes_valid,
        'volumetric_smpl_checkpoints': checkpoint_records,
        'diagnostic_flags': {
            'fresh_backend_reused': fresh[2][
                'volumetric_smpl_backend_reused'
            ],
            'shared_first_backend_reused': shared_first[2][
                'volumetric_smpl_backend_reused'
            ],
            'shared_second_backend_reused': shared_second[2][
                'volumetric_smpl_backend_reused'
            ],
        },
        'implementation': {
            str(path): sha256_file(path) for path in (
                Path(__file__).resolve(),
                ROOT / 'tools/run_interx_shape_augmentation.py',
                ROOT / 'optimize/optimize_full_body.py',
                ROOT / 'optimization/losses.py',
                ROOT / 'optimization/volumetric_smpl_backend.py',
            )
        },
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != 'pass':
        raise RuntimeError('shared VolumetricSMPL backend changed optimizer output')


if __name__ == '__main__':
    main()
