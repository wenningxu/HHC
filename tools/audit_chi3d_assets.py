#!/usr/bin/env python
"""Audit local CHI3D assets and prevent train data being reported as test data."""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


SUBJECT_TO_GROUP = {'s02': 'G2', 's03': 'G3', 's04': 'G4'}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    return {
        'bytes': path.stat().st_size,
        'path': str(path.resolve()),
        'sha256': sha256_file(path),
    }


def parse_sample_id(sample_id):
    fields = sample_id.split('_')
    if len(fields) < 3 or not fields[0].startswith('G'):
        raise ValueError(f'invalid CHI3D sample ID: {sample_id!r}')
    return fields[0], '_'.join(fields[1:-1]), fields[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--root',
        default='/home/wenning/projects/MMM/dataset/chi3d',
        help='Root of the local processed CHI3D assets.',
    )
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    train_split = root / 'splits' / 'train.txt'
    motion_h5 = root / 'motions' / 'chi3d-shape.h5'
    joint_h5 = root / 'motions' / 'chi3d-shape-joints.h5'
    required = [train_split, motion_h5, joint_h5]
    required.extend(
        root / 'interaction_labels' / f'{subject}_interaction_contact_signature.json'
        for subject in SUBJECT_TO_GROUP
    )
    required.extend(root / 'betas' / f'{group[-1]}.npy' for group in SUBJECT_TO_GROUP.values())
    required.extend(
        root / 'betas_group' / f'{group[-1]}.npy'
        for group in SUBJECT_TO_GROUP.values()
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing required CHI3D assets: {missing}')

    train_ids = [
        line.strip()
        for line in train_split.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError('CHI3D train split contains duplicate sample IDs')
    train_set = set(train_ids)

    group_counts = collections.Counter()
    action_counts = collections.Counter()
    group_action_counts = collections.defaultdict(collections.Counter)
    for sample_id in train_ids:
        group, action, _ = parse_sample_id(sample_id)
        group_counts[group] += 1
        action_counts[action] += 1
        group_action_counts[group][action] += 1

    h5_reports = {}
    h5_key_sets = {}
    for path in (motion_h5, joint_h5):
        with h5py.File(path, 'r') as handle:
            keys = set(handle.keys())
            shapes = collections.Counter(str(tuple(handle[key].shape)) for key in keys)
            dtypes = collections.Counter(str(handle[key].dtype) for key in keys)
        h5_key_sets[path.name] = keys
        h5_reports[path.name] = {
            **file_record(path),
            'dataset_count': len(keys),
            'dataset_shapes': dict(sorted(shapes.items())),
            'dataset_dtypes': dict(sorted(dtypes.items())),
            'keys_equal_train_split': keys == train_set,
            'missing_train_ids': sorted(train_set - keys),
            'unexpected_ids': sorted(keys - train_set),
        }

    label_reports = {}
    mapped_label_ids = set()
    for subject, group in SUBJECT_TO_GROUP.items():
        path = root / 'interaction_labels' / f'{subject}_interaction_contact_signature.json'
        with open(path, encoding='utf-8') as handle:
            labels = json.load(handle)
        if not isinstance(labels, dict):
            raise ValueError(f'{path} must contain a JSON object')
        mapped_ids = set()
        for label_id in labels:
            fields = label_id.rsplit(' ', 1)
            if len(fields) != 2:
                raise ValueError(f'invalid contact-label sample ID: {label_id!r}')
            mapped_ids.add(f'{group}_{fields[0].replace(" ", "_")}_{fields[1]}')
        mapped_label_ids.update(mapped_ids)
        expected_ids = {sample_id for sample_id in train_set if sample_id.startswith(f'{group}_')}
        label_reports[subject] = {
            **file_record(path),
            'mapped_group': group,
            'record_count': len(labels),
            'mapped_ids_equal_group_split': mapped_ids == expected_ids,
            'missing_group_ids': sorted(expected_ids - mapped_ids),
            'unexpected_ids': sorted(mapped_ids - expected_ids),
        }

    beta_reports = {}
    for kind in ('betas', 'betas_group'):
        beta_reports[kind] = {}
        for group in SUBJECT_TO_GROUP.values():
            path = root / kind / f'{group[-1]}.npy'
            value = np.load(path, allow_pickle=False)
            beta_reports[kind][group] = {
                **file_record(path),
                'dtype': str(value.dtype),
                'finite': bool(np.isfinite(value).all()),
                'shape': list(value.shape),
            }

    test_split = root / 'splits' / 'test.txt'
    consistency_checks = {
        'contact_labels_equal_train_split': mapped_label_ids == train_set,
        'joint_h5_keys_equal_motion_h5_keys': (
            h5_key_sets[joint_h5.name] == h5_key_sets[motion_h5.name]
        ),
        'motion_h5_keys_equal_train_split': h5_key_sets[motion_h5.name] == train_set,
    }
    if not all(consistency_checks.values()):
        raise ValueError(f'CHI3D asset consistency check failed: {consistency_checks}')
    if not all(
        record['finite']
        for kind in beta_reports.values()
        for record in kind.values()
    ):
        raise ValueError('CHI3D beta assets contain NaN or Inf')

    result = {
        'asset_classification': 'official_train_subject_pairs_only',
        'beta_assets': beta_reports,
        'consistency_checks': consistency_checks,
        'contact_labels': label_reports,
        'counts': {
            'actions': dict(sorted(action_counts.items())),
            'groups': dict(sorted(group_counts.items())),
            'group_actions': {
                group: dict(sorted(counts.items()))
                for group, counts in sorted(group_action_counts.items())
            },
            'train_samples': len(train_ids),
        },
        'files': {
            'hdf5': h5_reports,
            'train_split': file_record(train_split),
        },
        'official_test_assets': {
            'local_test_split_exists': test_split.is_file(),
            'local_test_split_path': str(test_split),
            'usable_for_official_test_evaluation': False,
        },
        'protocol_guardrail': {
            'allowed_local_generalization_protocol': (
                'predeclared leave-one-group-out cross-validation over G2/G3/G4; '
                'report as custom CV, never as official CHI3D test'
            ),
            'in_sample_diagnostic': (
                'all 373 samples may be used only for training-set reconstruction or '
                'GT-optimization diagnostics and must be labelled in-sample'
            ),
            'official_test_requirement': (
                'authorized official test package plus challenge-server submission '
                'or equivalent official evaluation access'
            ),
        },
        'root': str(root),
        'status': 'pass',
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
