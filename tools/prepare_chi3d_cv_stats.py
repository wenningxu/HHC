#!/usr/bin/env python
"""Compute train-only normalization statistics for frozen CHI3D CV folds."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import rotation_conversions as geometry


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_ids(path):
    return [
        line.strip()
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def to_rot_6d(motion):
    motion = torch.from_numpy(motion.astype(np.float32, copy=False))
    translation = motion[:, -1, :]
    pose = motion[:, :-1, :]
    rotations = []
    for person in range(2):
        rotations.append(
            geometry.matrix_to_rotation_6d(
                geometry.axis_angle_to_matrix(
                    pose[:, :, 3 * person:3 * person + 3]
                )
            )
        )
    encoded = torch.cat(rotations, dim=2)
    padded_translation = torch.zeros(
        (encoded.shape[0], encoded.shape[2]), dtype=encoded.dtype
    )
    for person in range(2):
        padded_translation[:, 6 * person:6 * person + 3] = (
            translation[:, 3 * person:3 * person + 3]
        )
    encoded = torch.cat((encoded, padded_translation[:, None]), dim=1)
    return encoded.reshape(encoded.shape[0], -1).numpy()


def compute_stats(handle, ids):
    features = []
    for sample_id in ids:
        if sample_id not in handle:
            raise KeyError(f'{sample_id} is absent from the CHI3D motion HDF5')
        features.append(to_rot_6d(handle[sample_id][:]))
    features = np.concatenate(features, axis=0)
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    zero_variance = std == 0
    std[zero_variance] = 1.0
    return mean, std, {
        'feature_dimensions': int(features.shape[1]),
        'frames': int(features.shape[0]),
        'sequences': len(ids),
        'zero_variance_dimensions_pinned_to_one': int(zero_variance.sum()),
    }


def save_array(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    return {
        'dtype': str(value.dtype),
        'path': str(path.resolve()),
        'sha256': sha256_file(path),
        'shape': list(value.shape),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cv-dir', default='configs/chi3d_cv')
    parser.add_argument(
        '--motion-file',
        default='/home/wenning/projects/MMM/dataset/chi3d/motions/chi3d-shape.h5',
    )
    parser.add_argument(
        '--reference-stats-dir',
        default='/home/wenning/projects/MMM/dataset/chi3d/stats',
        help='Existing all-training-data stats used only to verify feature conversion.',
    )
    args = parser.parse_args()

    cv_dir = Path(args.cv_dir).resolve()
    split_manifest_path = cv_dir / 'manifest.json'
    with open(split_manifest_path, encoding='utf-8') as handle:
        split_manifest = json.load(handle)
    source_split = Path(split_manifest['source']['path'])
    if sha256_file(source_split) != split_manifest['source']['sha256']:
        raise ValueError('source split hash no longer matches the frozen CV manifest')

    motion_file = Path(args.motion_file).resolve()
    reference_dir = Path(args.reference_stats_dir).resolve()
    reports = {}
    with h5py.File(motion_file, 'r') as motion_h5:
        source_mean, source_std, source_counts = compute_stats(
            motion_h5, load_ids(source_split)
        )
        expected_mean = np.load(reference_dir / 'Chi3D_Mean_root.npy', allow_pickle=False)
        expected_std = np.load(reference_dir / 'Chi3D_Std_root.npy', allow_pickle=False)
        mean_max_error = float(np.abs(source_mean - expected_mean).max())
        std_max_error = float(np.abs(source_std - expected_std).max())
        if mean_max_error > 1e-6 or std_max_error > 1e-6:
            raise ValueError(
                'feature conversion does not reproduce the existing full-train stats: '
                f'mean error={mean_max_error}, std error={std_max_error}'
            )

        for held_out, fold in sorted(split_manifest['folds'].items()):
            train_record = fold['splits']['train']
            train_split = Path(train_record['path'])
            if sha256_file(train_split) != train_record['sha256']:
                raise ValueError(f'{held_out} train split hash no longer matches manifest')
            train_ids = load_ids(train_split)
            mean, std, counts = compute_stats(motion_h5, train_ids)
            stats_dir = train_split.parent / 'stats'
            reports[held_out] = {
                **counts,
                'mean': save_array(stats_dir / 'Chi3D_Mean_root.npy', mean),
                'std': save_array(stats_dir / 'Chi3D_Std_root.npy', std),
                'train_split_sha256': train_record['sha256'],
            }

    result = {
        'feature_conversion_reference_check': {
            **source_counts,
            'existing_mean_sha256': sha256_file(reference_dir / 'Chi3D_Mean_root.npy'),
            'existing_std_sha256': sha256_file(reference_dir / 'Chi3D_Std_root.npy'),
            'mean_max_absolute_error': mean_max_error,
            'status': 'pass',
            'std_max_absolute_error': std_max_error,
        },
        'folds': reports,
        'motion_file': {
            'path': str(motion_file),
            'sha256': sha256_file(motion_file),
        },
        'split_manifest': {
            'path': str(split_manifest_path),
            'sha256': sha256_file(split_manifest_path),
        },
        'status': 'pass',
    }
    output = cv_dir / 'stats_manifest.json'
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({**result, 'manifest_path': str(output)}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
