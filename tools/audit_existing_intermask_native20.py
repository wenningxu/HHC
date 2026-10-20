#!/usr/bin/env python
"""Freeze the actual HDF5 protocol of the completed InterMask native-20 audit."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


EXPECTED = {
    'time_steps': 20,
    'cond_scale': 2.0,
    'temperature': 1.0,
    'topk_filter_thres': 0.9,
    'length_policy': 'exact',
    'seed_strategy': 'base_seed + repeat_index',
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.generic):
        return value.item()
    return value


def inspect_h5(path, expected_shape, expected_lengths, request, request_hash):
    with h5py.File(path, 'r') as source:
        attributes = {key: scalar(value) for key, value in source.attrs.items()}
        actual_protocol = {key: attributes.get(key) for key in EXPECTED}
        if actual_protocol != EXPECTED:
            raise ValueError(
                f'{path}: sampler protocol {actual_protocol} != {EXPECTED}'
            )
        if tuple(source['motions'].shape) != expected_shape:
            raise ValueError(
                f'{path}: shape {source["motions"].shape} != {expected_shape}'
            )
        completed = source['completed_repeats'][:].astype(bool)
        if completed.shape != (expected_shape[1],) or not completed.all():
            raise ValueError(f'{path}: incomplete repeats {completed.tolist()}')
        generated_lengths_recorded = 'generated_lengths' in source
        if generated_lengths_recorded:
            generated_lengths = source['generated_lengths'][:].astype(np.int64)
            if not np.array_equal(generated_lengths, expected_lengths):
                raise ValueError(
                    f'{path}: generated_lengths differs under exact policy'
                )
        if os.path.abspath(attributes.get('requests', '')) != str(request):
            raise ValueError(f'{path}: frozen request path differs')
        embedded_request_hash = attributes.get('requests_sha256')
        if embedded_request_hash not in (None, request_hash):
            raise ValueError(f'{path}: embedded frozen-request hash differs')
        return {
            'path': str(path),
            'bytes': path.stat().st_size,
            'shape': list(expected_shape),
            'completed_repeats': int(completed.sum()),
            'generated_lengths_recorded': generated_lengths_recorded,
            'exact_lengths_verified': True,
            'protocol': actual_protocol,
            'checkpoint_sha256': attributes.get('checkpoint_sha256'),
            'vq_checkpoint_sha256': attributes.get('vq_checkpoint_sha256'),
            'requests_sha256_embedded': embedded_request_hash,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--directory', type=Path,
        default=Path('output/fair_eval/intermask_official_exact'),
    )
    parser.add_argument(
        '--requests-dir', type=Path,
        default=Path('output/fair_interx_requests'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path(
            'output/fair_eval/intermask_official_exact/'
            'native20_hdf5_protocol_audit.json'
        ),
    )
    args = parser.parse_args()
    directory = args.directory.resolve()
    requests_dir = args.requests_dir.resolve()
    records = []
    for replication in range(20):
        request = requests_dir / f'replication_{replication:02d}.npz'
        request_hash = sha256_file(request)
        with np.load(request, allow_pickle=False) as frozen:
            main_lengths = frozen['lengths'].astype(np.int64)
            mm_lengths = main_lengths[frozen['mm_indices'].astype(np.int64)]
        record = {
            'replication': replication,
            'request': str(request),
            'request_sha256': request_hash,
            'main': inspect_h5(
                directory / f'replication_{replication:02d}_main.h5',
                (1696, 1, 150, 672), main_lengths, request, request_hash,
            ),
            'multimodality': inspect_h5(
                directory / f'replication_{replication:02d}_mm.h5',
                (100, 30, 150, 672), mm_lengths, request, request_hash,
            ),
        }
        records.append(record)
    result = {
        'schema': 'intermask_existing_native20_hdf5_protocol_audit_v1',
        'status': 'pass',
        'scope': (
            'metadata/shape/completeness audit of legacy generated HDF5; '
            'these files predate strict full-file generation hashes'
        ),
        'classification': (
            'author-native-20-steps_exact-length_main20_'
            'mm20-extension_not-common50'
        ),
        'replications': 20,
        'samples_per_replication': 1696,
        'multimodality': {'prompts': 100, 'generations_per_prompt': 30},
        'paper_source': {
            'url': 'https://arxiv.org/html/2410.10010v3',
            'version': 'v3',
            'revised': '2025-03-02',
            'reported_interx_fid': 0.399,
        },
        'paper_replication_policy': {
            'fid_matching_rprecision_diversity': 20,
            'multimodality': 5,
        },
        'generated_replication_policy': {
            'fid_matching_rprecision_diversity': 20,
            'multimodality': 20,
        },
        'paper_exact_scope': (
            'sampler and 20-replication main metrics; MModality is a '
            '20-replication precision extension of the paper 5-replication metric'
        ),
        'sampler_protocol': EXPECTED,
        'records': records,
        'implementation_sha256': sha256_file(__file__),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps({
        'output': str(args.output.resolve()), 'status': 'pass',
        'classification': result['classification'], 'replications': 20,
    }, indent=2))


if __name__ == '__main__':
    main()
