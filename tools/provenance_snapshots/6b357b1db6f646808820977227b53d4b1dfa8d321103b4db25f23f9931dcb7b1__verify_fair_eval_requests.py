#!/usr/bin/env python
"""Validate frozen Inter-X request files and write a checksum manifest."""

import argparse
import hashlib
import json
import os

import numpy as np


EXPECTED_SHAPES = {
    'names': (1696,), 'captions': (1696,), 'lengths': (1696,),
    'text_embeddings': (1696, 512), 'gt_embeddings': (1696, 512),
    'gt_body_embeddings': (1696, 512), 'betas': (1696, 20),
    'gender': (1696, 2), 'contact': (1696, 150), 'mm_indices': (100,),
    'seed': (),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path):
    absolute = os.path.abspath(path)
    return {
        'path': absolute,
        'bytes': os.path.getsize(absolute),
        'sha256': sha256(absolute),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-dir', default='output/fair_interx_requests')
    parser.add_argument('--replications', type=int, default=20)
    parser.add_argument('--base-seed', type=int, default=1234)
    parser.add_argument('--output', default='output/fair_interx_requests/manifest.json')
    args = parser.parse_args()

    records = []
    for replication in range(args.replications):
        path = os.path.join(args.request_dir, f'replication_{replication:02d}.npz')
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        request = np.load(path, allow_pickle=False)
        if set(request.files) != set(EXPECTED_SHAPES):
            raise ValueError(f'{path}: fields {request.files}')
        for key, shape in EXPECTED_SHAPES.items():
            if request[key].shape != shape:
                raise ValueError(f'{path}: {key} expected {shape}, got {request[key].shape}')
        if int(request['seed']) != args.base_seed + replication:
            raise ValueError(f'{path}: seed mismatch')
        if len(np.unique(request['names'])) != 1696:
            raise ValueError(f'{path}: sample names are not unique')
        if len(np.unique(request['mm_indices'])) != 100:
            raise ValueError(f'{path}: MM indices are not unique')
        if request['mm_indices'].min() < 0 or request['mm_indices'].max() >= 1696:
            raise ValueError(f'{path}: MM index out of bounds')
        if request['lengths'].min() < 24 or request['lengths'].max() > 150:
            raise ValueError(f'{path}: invalid motion lengths')
        if not np.isin(request['contact'], (0, 1)).all():
            raise ValueError(f'{path}: contact is not binary')
        for key in ('text_embeddings', 'gt_embeddings', 'gt_body_embeddings', 'betas'):
            if not np.isfinite(request[key]).all():
                raise ValueError(f'{path}: {key} contains non-finite values')
        records.append({
            'replication': replication,
            'seed': int(request['seed']),
            'path': os.path.abspath(path),
            'sha256': sha256(path),
            'bytes': os.path.getsize(path),
        })

    manifest = {
        'protocol': 'interx_official_v1',
        'request_contract_version': 2,
        'replications': args.replications,
        'samples_per_replication': 1696,
        'multimodality': {'prompts': 100, 'repeats': 30, 'pairs': 10},
        'records': records,
        'frozen_input_provenance': {
            label: artifact_record(path)
            for label, path in {
                'protocol': 'configs/fair_interx_protocol.yaml',
                'request_preparer': 'tools/prepare_fair_eval_requests.py',
                'request_verifier': 'tools/verify_fair_eval_requests.py',
                'dataset_loader': 'dataset/dataset_interx_TM.py',
                'normalizer': 'utils/motion_process.py',
                'evaluator_wrapper': 'models/evaluator_wrapper.py',
                'evaluator_options': 'checkpoints/hhi/Comp_v6_KLD01/opt.txt',
                'evaluator_weights': 'checkpoints/hhi/text_mot_match/model/finest.tar',
                'test_split': 'dataset/interx/splits/test.txt',
                'test_motions': 'dataset/interx/motions/test.h5',
                'raw_mean': 'dataset/interx/Interx_Mean.npy',
                'raw_std': 'dataset/interx/Interx_Std.npy',
            }.items()
        },
        'status': 'pass',
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = os.path.abspath(args.output) + f'.tmp.{os.getpid()}'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(temporary, os.path.abspath(args.output))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
