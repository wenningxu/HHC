#!/usr/bin/env python
"""Filter the 20 frozen official requests for seen/unseen review protocols."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_npz(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'wb') as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-manifest', type=Path,
                        default=Path('configs/interx_generalization/manifest.json'))
    parser.add_argument('--requests-dir', type=Path,
                        default=Path('output/fair_interx_requests'))
    parser.add_argument('--output-root', type=Path,
                        default=Path('output/fair_interx_generalization_requests'))
    parser.add_argument('--replications', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--mm-samples', type=int, default=100)
    parser.add_argument('--seed-offset', type=int, default=700000)
    args = parser.parse_args()
    split_manifest_path = args.split_manifest.resolve()
    split_manifest = json.loads(split_manifest_path.read_text(encoding='utf-8'))
    output = {
        'schema': 'hhc_interx_generalization_frozen_requests_v1',
        'source_manifest': str(split_manifest_path),
        'source_manifest_sha256': sha256_file(split_manifest_path),
        'batch_size': args.batch_size,
        'selection': (
            'retain frozen official requests whose IDs belong to the partition, '
            'then truncate the ordered tail to a complete evaluator batch'
        ),
        'partitions': {},
    }
    for experiment_name, experiment in split_manifest['experiments'].items():
        for partition_name in ('test_seen', 'test_unseen'):
            split_record = experiment['partitions'][partition_name]
            split_path = Path(split_record['path'])
            if sha256_file(split_path) != split_record['sha256']:
                raise ValueError(f'{experiment_name}/{partition_name} split changed')
            allowed = set(split_path.read_text().split())
            directory = args.output_root / experiment_name / partition_name
            directory.mkdir(parents=True, exist_ok=True)
            records = []
            for replication in range(args.replications):
                source_path = (
                    args.requests_dir / f'replication_{replication:02d}.npz'
                ).resolve()
                with np.load(source_path, allow_pickle=False) as source:
                    names = source['names'].astype(str)
                    indices = np.asarray([
                        index for index, name in enumerate(names) if name in allowed
                    ], dtype=np.int64)
                    retained_before_drop_last = len(indices)
                    retained = (len(indices) // args.batch_size) * args.batch_size
                    indices = indices[:retained]
                    if retained < args.mm_samples:
                        raise ValueError(
                            f'{experiment_name}/{partition_name}/rep{replication}: '
                            f'only {retained} samples'
                        )
                    payload = {
                        key: source[key][indices]
                        for key in source.files
                        if key not in ('mm_indices', 'seed')
                    }
                    request_seed = int(source['seed'])
                generator = np.random.default_rng(
                    request_seed + args.seed_offset
                )
                payload['mm_indices'] = generator.permutation(retained)[
                    :args.mm_samples
                ]
                payload['seed'] = np.asarray(request_seed)
                target_path = directory / f'replication_{replication:02d}.npz'
                atomic_npz(target_path, payload)
                records.append({
                    'replication': replication,
                    'source': str(source_path),
                    'source_sha256': sha256_file(source_path),
                    'retained_before_drop_last': retained_before_drop_last,
                    'tail_dropped': retained_before_drop_last - retained,
                    'samples': retained,
                    'output': str(target_path.resolve()),
                    'output_sha256': sha256_file(target_path),
                })
            key = f'{experiment_name}/{partition_name}'
            output['partitions'][key] = {
                'split': str(split_path.resolve()),
                'split_sha256': split_record['sha256'],
                'replications': records,
            }
    output['implementation_sha256'] = sha256_file(__file__)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / 'manifest.json', output)
    compact = {
        key: [record['samples'] for record in value['replications']]
        for key, value in output['partitions'].items()
    }
    print(json.dumps({'manifest': str(args.output_root / 'manifest.json'),
                      'sample_counts': compact}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
