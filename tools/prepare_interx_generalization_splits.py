#!/usr/bin/env python
"""Freeze leakage-free held-out shape-group and action Inter-X protocols."""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np


NAME_PATTERN = re.compile(r'^G(\d+)T\d+A(\d+)R\d+$')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_name(name):
    match = NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f'unexpected Inter-X sample name: {name}')
    return f'G{match.group(1)}', f'A{match.group(2)}'


def deterministic_holdout(values, count, namespace):
    ranked = sorted(
        values,
        key=lambda value: hashlib.sha256(
            f'{namespace}:{value}'.encode('utf-8')
        ).hexdigest(),
    )
    return sorted(ranked[:count])


def atomic_text(path, values):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(''.join(f'{value}\n' for value in values), encoding='utf-8')
    os.replace(temporary, path)


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def summarize(names, beta_dir):
    group_counts = Counter()
    action_counts = Counter()
    genders = Counter()
    beta_rows = []
    unique_shapes = set()
    for name in names:
        group, action = parse_name(name)
        group_counts[group] += 1
        action_counts[action] += 1
        with np.load(beta_dir / f'{name}.npz', allow_pickle=True) as shape:
            betas = shape['betas'].reshape(2, 10).astype(np.float64)
            raw_gender = shape['gender'].reshape(-1).tolist()
        for person in range(2):
            gender = str(raw_gender[person])
            genders[gender] += 1
            beta_rows.append(betas[person])
            unique_shapes.add((gender, betas[person].tobytes()))
    beta = np.stack(beta_rows) if beta_rows else np.zeros((0, 10))
    norms = np.linalg.norm(beta, axis=1) if len(beta) else np.zeros(0)
    return {
        'samples': len(names),
        'person_instances': len(names) * 2,
        'groups': len(group_counts),
        'actions': len(action_counts),
        'group_counts': dict(sorted(group_counts.items())),
        'action_counts': dict(sorted(action_counts.items())),
        'gender_person_instances': dict(sorted(genders.items())),
        'unique_gender_beta_shapes': len(unique_shapes),
        'beta_mean': beta.mean(axis=0).tolist() if len(beta) else None,
        'beta_std': beta.std(axis=0).tolist() if len(beta) else None,
        'beta_l2_norm': {
            'mean': float(norms.mean()) if len(norms) else None,
            'std': float(norms.std()) if len(norms) else None,
            'q05_q50_q95': np.quantile(norms, (0.05, 0.5, 0.95)).tolist()
            if len(norms) else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', type=Path,
                        default=Path('dataset/interx/splits'))
    parser.add_argument('--beta-dir', type=Path,
                        default=Path('dataset/interx/betas'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('configs/interx_generalization'))
    parser.add_argument('--heldout-groups', type=int, default=12)
    parser.add_argument('--heldout-actions', type=int, default=8)
    parser.add_argument('--namespace', default='hhc-review-generalization-v1')
    args = parser.parse_args()
    split_dir = args.split_dir.resolve()
    beta_dir = args.beta_dir.resolve()
    official = {
        split: [
            line.strip() for line in (split_dir / f'{split}.txt').read_text().splitlines()
            if line.strip()
        ]
        for split in ('train', 'val', 'test')
    }
    for split, names in official.items():
        if len(names) != len(set(names)):
            raise ValueError(f'duplicate names in official {split}')
    if any(
        set(official[left]) & set(official[right])
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test'))
    ):
        raise ValueError('official splits overlap')
    all_groups = sorted({parse_name(name)[0] for names in official.values() for name in names})
    all_actions = sorted({parse_name(name)[1] for names in official.values() for name in names})
    heldout_groups = deterministic_holdout(
        all_groups, args.heldout_groups, args.namespace + ':shape'
    )
    heldout_actions = deterministic_holdout(
        all_actions, args.heldout_actions, args.namespace + ':action'
    )
    experiments = {
        'shape': (set(heldout_groups), 0),
        'action': (set(heldout_actions), 1),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema': 'hhc_interx_generalization_splits_v1',
        'selection': (
            'SHA-256 rank of semantic IDs under a frozen namespace; no metric, '
            'caption, beta magnitude, or test performance used'
        ),
        'namespace': args.namespace,
        'heldout_groups': heldout_groups,
        'heldout_actions': heldout_actions,
        'official_splits': {
            split: {
                'path': str((split_dir / f'{split}.txt').resolve()),
                'sha256': sha256_file(split_dir / f'{split}.txt'),
                'samples': len(names),
            }
            for split, names in official.items()
        },
        'experiments': {},
    }
    for label, (heldout, field_index) in experiments.items():
        directory = args.output_dir / label
        directory.mkdir(parents=True, exist_ok=True)

        def is_unseen(name):
            return parse_name(name)[field_index] in heldout

        partitions = {
            'train_seen': [name for name in official['train'] if not is_unseen(name)],
            'val_seen': [name for name in official['val'] if not is_unseen(name)],
            'val_unseen_diagnostic': [name for name in official['val'] if is_unseen(name)],
            'test_seen': [name for name in official['test'] if not is_unseen(name)],
            'test_unseen': [name for name in official['test'] if is_unseen(name)],
        }
        train_values = {
            parse_name(name)[field_index] for name in partitions['train_seen']
        }
        if train_values & heldout:
            raise AssertionError(f'{label}: heldout value leaked into training')
        if not partitions['test_unseen'] or not partitions['val_seen']:
            raise ValueError(f'{label}: empty required partition')
        records = {}
        for partition, names in partitions.items():
            path = directory / f'{partition}.txt'
            atomic_text(path, names)
            records[partition] = {
                'path': str(path.resolve()),
                'sha256': sha256_file(path),
                **summarize(names, beta_dir),
            }
        manifest['experiments'][label] = {
            'heldout_values': sorted(heldout),
            'model_selection_partition': 'val_seen',
            'primary_partition': 'test_unseen',
            'test_seen_is_matched_control': True,
            'partitions': records,
        }
    manifest['implementation_sha256'] = sha256_file(__file__)
    atomic_json(args.output_dir / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
