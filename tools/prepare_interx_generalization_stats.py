#!/usr/bin/env python
"""Compute leakage-free raw-672D normalization for Inter-X holdouts."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import rotation_conversions as geometry


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_ids(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def to_rot_6d(motion):
    motion = torch.from_numpy(motion.astype(np.float32, copy=False))
    translation = motion[:, -1, :]
    pose = motion[:, :-1, :]
    encoded = torch.cat([
        geometry.matrix_to_rotation_6d(
            geometry.axis_angle_to_matrix(pose[:, :, 3 * person:3 * person + 3])
        )
        for person in range(2)
    ], dim=2)
    padded_translation = torch.zeros(
        (encoded.shape[0], encoded.shape[2]), dtype=encoded.dtype
    )
    for person in range(2):
        padded_translation[:, 6 * person:6 * person + 3] = (
            translation[:, 3 * person:3 * person + 3]
        )
    return torch.cat((encoded, padded_translation[:, None]), dim=1).reshape(
        encoded.shape[0], -1
    ).numpy()


def compute_stats(handle, names):
    total = np.zeros(672, dtype=np.float64)
    square_total = np.zeros(672, dtype=np.float64)
    frames = 0
    for name in names:
        if name not in handle:
            raise KeyError(f'{name} is absent from train motion HDF5')
        features = to_rot_6d(handle[name][:]).astype(np.float64)
        total += features.sum(axis=0)
        square_total += np.square(features).sum(axis=0)
        frames += len(features)
    mean64 = total / frames
    variance64 = np.maximum(square_total / frames - np.square(mean64), 0.0)
    mean = mean64.astype(np.float32)
    std = np.sqrt(variance64).astype(np.float32)
    zero = std == 0
    std[zero] = 1.0
    return mean, std, {
        'sequences': len(names), 'frames': frames,
        'feature_dimensions': 672,
        'zero_variance_dimensions_pinned_to_one': int(zero.sum()),
    }


def save_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'wb') as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)
    return {'path': str(path.resolve()), 'shape': list(array.shape),
            'dtype': str(array.dtype), 'sha256': sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=Path('configs/interx_generalization/manifest.json'))
    parser.add_argument('--motion-file', type=Path,
                        default=Path('dataset/interx/motions/train.h5'))
    parser.add_argument('--reference-mean', type=Path,
                        default=Path('dataset/interx/Interx_Mean.npy'))
    parser.add_argument('--reference-std', type=Path,
                        default=Path('dataset/interx/Interx_Std.npy'))
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    motion_path = args.motion_file.resolve()
    official_train = Path(manifest['official_splits']['train']['path'])
    if sha256_file(official_train) != manifest['official_splits']['train']['sha256']:
        raise ValueError('official train split changed after generalization freeze')
    reports = {}
    with h5py.File(motion_path, 'r') as motions:
        reference_mean, reference_std, reference_counts = compute_stats(
            motions, load_ids(official_train)
        )
        existing_mean = np.load(args.reference_mean, allow_pickle=False)
        existing_std = np.load(args.reference_std, allow_pickle=False)
        mean_error = float(np.max(np.abs(reference_mean - existing_mean)))
        std_error = float(np.max(np.abs(reference_std - existing_std)))
        if mean_error > 2e-6 or std_error > 2e-6:
            raise ValueError(
                'raw feature conversion does not reproduce official train stats: '
                f'mean={mean_error}, std={std_error}'
            )
        for label, experiment in manifest['experiments'].items():
            split_record = experiment['partitions']['train_seen']
            split_path = Path(split_record['path'])
            if sha256_file(split_path) != split_record['sha256']:
                raise ValueError(f'{label} train split changed')
            mean, std, counts = compute_stats(motions, load_ids(split_path))
            stats_dir = split_path.parent / 'stats'
            reports[label] = {
                **counts,
                'stats_prefix': f'generalization_{label}',
                'mean': save_array(
                    stats_dir / f'generalization_{label}_Mean.npy', mean
                ),
                'std': save_array(
                    stats_dir / f'generalization_{label}_Std.npy', std
                ),
                'train_split_sha256': split_record['sha256'],
            }
    result = {
        'schema': 'hhc_interx_generalization_train_only_stats_v1',
        'status': 'pass',
        'split_manifest': {'path': str(manifest_path),
                           'sha256': sha256_file(manifest_path)},
        'motion_file': {'path': str(motion_path)},
        'reference_check': {
            **reference_counts,
            'mean_max_absolute_error': mean_error,
            'std_max_absolute_error': std_error,
            'existing_mean_sha256': sha256_file(args.reference_mean),
            'existing_std_sha256': sha256_file(args.reference_std),
        },
        'experiments': reports,
        'implementation_sha256': sha256_file(__file__),
    }
    output = manifest_path.parent / 'stats_manifest.json'
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
