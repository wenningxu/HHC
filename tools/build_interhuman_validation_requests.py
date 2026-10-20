#!/usr/bin/env python3
"""Freeze one official InterHuman val set for checkpoint selection."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from tools.build_interhuman_fair_requests import (
    atomic_json,
    canonical_sha256,
    sample_seed,
    sha256_file,
)


def build_validation_request(adapter, seed=20260130, batch_size=96):
    eligible = adapter.split_ids('val', require_text=True)
    if len(eligible) != 547:
        raise ValueError(f'expected 547 official-loader-eligible val entries, got {len(eligible)}')
    ordered = eligible.copy()
    random.Random(seed).shuffle(ordered)
    evaluated = len(ordered) // batch_size * batch_size
    requests = []
    for position, identifier in enumerate(ordered[:evaluated]):
        per_sample_seed = sample_seed(seed, 0, position, identifier)
        python_rng = random.Random(per_sample_seed)
        numpy_rng = np.random.default_rng(per_sample_seed)
        captions = adapter.texts(identifier)
        caption_index = python_rng.randrange(len(captions))
        raw_length = adapter._pair_raw_length(identifier)
        crop_start = (
            python_rng.randrange(0, raw_length - 300)
            if raw_length > 300 else 0
        )
        raw_frames = min(raw_length, 300)
        requests.append({
            'position': position,
            'identifier': identifier,
            'caption_index': caption_index,
            'caption': captions[caption_index].strip(),
            'crop_start': crop_start,
            'raw_frames': raw_frames,
            'processed_length': raw_frames - 1,
            'left_right_swap': False,
            'anchor_person': 2 if numpy_rng.random() > 0.5 else 1,
            'output_person_swap': bool(numpy_rng.random() > 0.5),
            'sample_seed': per_sample_seed,
        })
    payload = {
        'schema': 'interhuman_validation_requests_v1',
        'split': 'val',
        'seed': seed,
        'batch_size': batch_size,
        'drop_last': True,
        'official_loader_effective_entries': len(eligible),
        'evaluated_entries': len(requests),
        'dropped_entries': len(eligible) - len(requests),
        'requests': requests,
    }
    payload['request_sha256'] = canonical_sha256(requests)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='output/interhuman_validation_requests')
    parser.add_argument('--seed', type=int, default=20260130)
    parser.add_argument('--batch-size', type=int, default=96)
    args = parser.parse_args()
    if args.batch_size != 96:
        raise ValueError('official InterHuman validation protocol requires batch_size=96')
    adapter = OfficialInterHumanAdapter()
    payload = build_validation_request(adapter, args.seed, args.batch_size)
    output_dir = Path(args.output_dir).resolve()
    request_path = output_dir / 'validation.json'
    if request_path.exists():
        if json.loads(request_path.read_text()) != payload:
            raise ValueError(f'refusing to overwrite changed validation request {request_path}')
    else:
        atomic_json(request_path, payload)
    manifest = {
        'schema': 'interhuman_validation_requests_manifest_v1',
        'status': 'complete',
        'purpose': 'held-out validation only; checkpoint selection must never use the formal test replications',
        'protocol': {
            'split': 'val',
            'seed': args.seed,
            'batch_size': 96,
            'drop_last': True,
            'official_loader_effective_val_entries': 547,
            'evaluated_entries': 480,
            'dropped_entries': 67,
        },
        'request': {
            'path': str(request_path),
            'bytes': request_path.stat().st_size,
            'sha256': sha256_file(request_path),
            'request_sha256': payload['request_sha256'],
        },
        'artifacts': {
            'val_split': {
                'path': str(adapter.data_root / 'split/val.txt'),
                'sha256': sha256_file(adapter.data_root / 'split/val.txt'),
            },
            'adapter': {
                'path': str(ROOT / 'dataset/interhuman_hhc_adapter.py'),
                'sha256': sha256_file(ROOT / 'dataset/interhuman_hhc_adapter.py'),
            },
            'normalization_mean': {
                'path': str(adapter.mean_path), 'sha256': sha256_file(adapter.mean_path),
            },
            'normalization_std': {
                'path': str(adapter.std_path), 'sha256': sha256_file(adapter.std_path),
            },
        },
    }
    manifest_path = output_dir / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f'refusing to overwrite changed validation manifest {manifest_path}')
    else:
        atomic_json(manifest_path, manifest)
    print(json.dumps({
        'status': 'complete',
        'manifest': str(manifest_path),
        'effective_val_entries': 547,
        'evaluated_entries': 480,
    }, indent=2))


if __name__ == '__main__':
    main()
