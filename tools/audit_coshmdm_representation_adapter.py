#!/usr/bin/env python
"""Audit CoShMDM's shape-specific 262-D to official Inter-X adapter."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COSH_ROOT = Path('/mnt/8T-HDD/wenning/coshmdm-code')
sys.path.insert(0, str(COSH_ROOT))

from utils.interx_projection import project_official_motion


SCHEMA = 'hhc_coshmdm_representation_adapter_audit_v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def summarize(rows):
    values = np.asarray([
        [
            row['all_rmse'], row['root_rotation6d_rmse'],
            row['translation_rmse_m'], row['body_rotation6d_rmse'],
            row['max_abs_difference'],
        ]
        for row in rows
    ], dtype=np.float64)
    names = (
        'all_rmse', 'root_rotation6d_rmse', 'translation_rmse_m',
        'body_rotation6d_rmse', 'max_abs_difference',
    )
    return {
        name: {
            'mean': float(values[:, index].mean()),
            'p95': float(np.quantile(values[:, index], 0.95)),
            'maximum': float(values[:, index].max()),
        }
        for index, name in enumerate(names)
    }


def update_difference_accumulator(accumulator, values):
    values = np.asarray(values, dtype=np.float64)
    accumulator['sum_squared'] += float(np.sum(values ** 2))
    accumulator['count'] += int(values.size)
    accumulator['maximum'] = max(
        accumulator['maximum'], float(np.max(np.abs(values)))
    )


def finish_difference_accumulator(accumulator):
    return {
        'rmse': float(np.sqrt(
            accumulator['sum_squared'] / accumulator['count']
        )),
        'maximum': accumulator['maximum'],
        'values': accumulator['count'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--requests', default='output/fair_interx_requests/replication_00.npz'
    )
    parser.add_argument(
        '--ground-truth',
        default='output/fair_interx_ground_truth/replication_00.h5',
    )
    parser.add_argument(
        '--skeleton-root', default='dataset/interx/skeleton'
    )
    parser.add_argument(
        '--output',
        default='output/coshmdm_representation_adapter_audit.json',
    )
    args = parser.parse_args()

    request = np.load(args.requests, allow_pickle=False)
    names = request['names'].astype(str)
    lengths = request['lengths'].astype(np.int64)
    thresholds = {
        'all_rmse_mean_max': 1e-3,
        'all_rmse_p95_max': 1e-3,
        'all_rmse_sample_max': 1e-2,
        'root_rotation6d_rmse_sample_max': 1e-3,
        'translation_rmse_m_sample_max': 5e-2,
        'body_rotation6d_rmse_sample_max': 1e-3,
        'max_abs_difference_max': 1e-1,
        'nonroot_body_rotation_max_abs': 0.0,
    }
    used_groups = sorted({int(name[1:4]) for name in names})
    skeleton_artifacts = {
        str(group): artifact(Path(args.skeleton_root) / f'{group}.npy')
        for group in used_groups
    }
    rows = []
    raw_projection_difference = {
        name: {'sum_squared': 0.0, 'count': 0, 'maximum': 0.0}
        for name in (
            'root_rotation6d', 'nonroot_body_rotation6d',
            'hand_rotation6d', 'root_translation_m', 'all',
        )
    }
    started = time.time()
    with h5py.File(args.ground_truth, 'r') as ground_truth:
        if not np.array_equal(ground_truth['names'][:].astype(str), names):
            raise ValueError('GT names/order differ from frozen requests')
        if not np.array_equal(ground_truth['lengths'][:], lengths):
            raise ValueError('GT lengths differ from frozen requests')
        for index, (name, length) in enumerate(zip(names, lengths)):
            length = int(length)
            skeleton = np.load(
                Path(args.skeleton_root) / f'{int(name[1:4])}.npy'
            )
            raw = ground_truth['motions'][index, 0, :length]
            projected_once = project_official_motion(
                raw, skeleton, length
            )
            projected_twice = project_official_motion(
                projected_once[:length], skeleton, length
            )
            raw_structured = np.asarray(raw, dtype=np.float32).reshape(
                length, 56, 12
            )
            projected_structured = projected_once[:length].reshape(
                length, 56, 12
            )
            raw_difference = (
                projected_structured - raw_structured
            ).astype(np.float64)
            update_difference_accumulator(
                raw_projection_difference['root_rotation6d'],
                np.concatenate([
                    raw_difference[:, 0, :6].reshape(-1),
                    raw_difference[:, 0, 6:12].reshape(-1),
                ]),
            )
            update_difference_accumulator(
                raw_projection_difference['nonroot_body_rotation6d'],
                np.concatenate([
                    raw_difference[:, 1:22, :6].reshape(-1),
                    raw_difference[:, 1:22, 6:12].reshape(-1),
                ]),
            )
            update_difference_accumulator(
                raw_projection_difference['hand_rotation6d'],
                np.concatenate([
                    raw_difference[:, 22:55, :6].reshape(-1),
                    raw_difference[:, 22:55, 6:12].reshape(-1),
                ]),
            )
            update_difference_accumulator(
                raw_projection_difference['root_translation_m'],
                np.concatenate([
                    raw_difference[:, 55, :3].reshape(-1),
                    raw_difference[:, 55, 6:9].reshape(-1),
                ]),
            )
            update_difference_accumulator(
                raw_projection_difference['all'], raw_difference
            )
            difference = (
                projected_once[:length] - projected_twice[:length]
            ).reshape(length, 56, 12).astype(np.float64)
            root = np.concatenate([
                difference[:, 0, :6].reshape(-1),
                difference[:, 0, 6:12].reshape(-1),
            ])
            translation = np.concatenate([
                difference[:, 55, :3].reshape(-1),
                difference[:, 55, 6:9].reshape(-1),
            ])
            body = np.concatenate([
                difference[:, :22, :6].reshape(-1),
                difference[:, :22, 6:12].reshape(-1),
            ])
            rows.append({
                'index': int(index),
                'name': name,
                'length': length,
                'all_rmse': float(np.sqrt(np.mean(difference ** 2))),
                'root_rotation6d_rmse': float(np.sqrt(np.mean(root ** 2))),
                'translation_rmse_m': float(
                    np.sqrt(np.mean(translation ** 2))
                ),
                'body_rotation6d_rmse': float(
                    np.sqrt(np.mean(body ** 2))
                ),
                'max_abs_difference': float(np.max(np.abs(difference))),
            })
            if (index + 1) % 250 == 0:
                print({'audited': index + 1, 'total': len(names)}, flush=True)

    summary = summarize(rows)
    raw_projection_summary = {
        name: finish_difference_accumulator(accumulator)
        for name, accumulator in raw_projection_difference.items()
    }
    pass_conditions = {
        'sample_count': len(rows) == 1696,
        'all_rmse_mean': (
            summary['all_rmse']['mean']
            <= thresholds['all_rmse_mean_max']
        ),
        'all_rmse_p95': (
            summary['all_rmse']['p95']
            <= thresholds['all_rmse_p95_max']
        ),
        'all_rmse_max': (
            summary['all_rmse']['maximum']
            <= thresholds['all_rmse_sample_max']
        ),
        'root_rotation6d_rmse_max': (
            summary['root_rotation6d_rmse']['maximum']
            <= thresholds['root_rotation6d_rmse_sample_max']
        ),
        'translation_rmse_m_max': (
            summary['translation_rmse_m']['maximum']
            <= thresholds['translation_rmse_m_sample_max']
        ),
        'body_rotation6d_rmse_max': (
            summary['body_rotation6d_rmse']['maximum']
            <= thresholds['body_rotation6d_rmse_sample_max']
        ),
        'max_abs_difference': (
            summary['max_abs_difference']['maximum']
            <= thresholds['max_abs_difference_max']
        ),
        'nonroot_body_rotations_preserved_exactly': (
            raw_projection_summary['nonroot_body_rotation6d']['maximum']
            <= thresholds['nonroot_body_rotation_max_abs']
        ),
    }
    worst = sorted(
        rows, key=lambda row: row['all_rmse'], reverse=True
    )[:20]
    implementation = {
        str(path.resolve()): sha256_file(path) for path in (
            Path(__file__),
            COSH_ROOT / 'utils/interx_official_adapter.py',
            COSH_ROOT / 'utils/interx_projection.py',
            COSH_ROOT / 'tools/export_official_interx.py',
            ROOT / 'tools/export_timotion_official_interx.py',
            ROOT / 'tools/prepare_coshmdm_interx.py',
        )
    }
    payload = {
        'schema': SCHEMA,
        'status': 'pass' if all(pass_conditions.values()) else 'fail',
        'adapter_contract': (
            'shape-specific all-joint least-squares rigid root fit v2; '
            'full requested length preserved by terminal velocity padding; '
            'no GT motion is used by generation exporters'
        ),
        'samples': len(rows),
        'thresholds': thresholds,
        'pass_conditions': pass_conditions,
        'projection_once_vs_twice': summary,
        'raw_to_projection_information_loss': raw_projection_summary,
        'irreversible_or_unmodelled_fields': [
            'sequence-initial global root rotation',
            'sequence-initial global root translation and floor offset',
            '33 hand-joint local rotations per person',
        ],
        'worst_samples': worst,
        'elapsed_seconds': time.time() - started,
        'inputs': {
            'requests': artifact(args.requests),
            'ground_truth': artifact(args.ground_truth),
            'skeletons': skeleton_artifacts,
        },
        'implementation': implementation,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, output)
    print(json.dumps({
        'output': str(output),
        'status': payload['status'],
        'samples': payload['samples'],
        'projection_once_vs_twice': summary,
    }, indent=2))
    if payload['status'] != 'pass':
        raise RuntimeError('CoShMDM representation adapter audit failed')


if __name__ == '__main__':
    main()
