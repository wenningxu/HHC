#!/usr/bin/env python3
"""Freeze reproducible InterHuman evaluator requests without duplicating GT arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def sample_seed(base_seed: int, replication: int, position: int, identifier: str) -> int:
    payload = f'{base_seed}:{replication}:{position}:{identifier}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def build_replication(
    adapter: OfficialInterHumanAdapter,
    effective_ids: list[str],
    replication: int,
    base_seed: int,
    batch_size: int,
) -> dict:
    order_rng = random.Random(base_seed + replication)
    ordered = effective_ids.copy()
    order_rng.shuffle(ordered)
    evaluated_count = len(ordered) // batch_size * batch_size
    ordered = ordered[:evaluated_count]
    requests = []
    for position, identifier in enumerate(ordered):
        seed = sample_seed(base_seed, replication, position, identifier)
        python_rng = random.Random(seed)
        numpy_rng = np.random.default_rng(seed)
        captions = adapter.texts(identifier)
        caption_index = python_rng.randrange(len(captions))
        raw_path = adapter._person_path(identifier, 1)
        raw_length = int(np.load(raw_path, mmap_mode='r').shape[0])
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
            'sample_seed': seed,
        })
    payload = {
        'schema': 'interhuman_fair_requests_replication_v3',
        'replication': replication,
        'replication_seed': base_seed + replication,
        'batch_size': batch_size,
        'drop_last': True,
        'effective_test_entries': len(effective_ids),
        'evaluated_entries': len(requests),
        'dropped_entries': len(effective_ids) - len(requests),
        'requests': requests,
    }
    payload['request_sha256'] = canonical_sha256(requests)
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='output/interhuman_fair_requests_v3')
    parser.add_argument('--replications', type=int, default=20)
    parser.add_argument('--base-seed', type=int, default=1234)
    parser.add_argument('--batch-size', type=int, default=96)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.replications != 20 or args.batch_size != 96:
        raise ValueError('formal InterHuman protocol requires 20 replications and batch_size=96')
    adapter = OfficialInterHumanAdapter()
    split_path = adapter.data_root / 'split/test.txt'
    effective_ids = adapter.split_ids('test', require_text=True)
    if len(effective_ids) != 1098:
        raise ValueError(f'expected 1098 official-loader-eligible InterHuman test pairs, got {len(effective_ids)}')
    output_dir = Path(args.output_dir).resolve()
    entries = []
    for replication in range(args.replications):
        payload = build_replication(
            adapter, effective_ids, replication, args.base_seed, args.batch_size
        )
        destination = output_dir / f'replication_{replication:02d}.json'
        if destination.exists():
            existing = json.loads(destination.read_text())
            if existing != payload:
                raise ValueError(f'refusing to overwrite changed frozen requests: {destination}')
        else:
            atomic_json(destination, payload)
        entries.append({
            'replication': replication,
            'path': str(destination),
            'bytes': destination.stat().st_size,
            'sha256': sha256_file(destination),
            'request_sha256': payload['request_sha256'],
            'evaluated_entries': payload['evaluated_entries'],
        })

    manifest = {
        'schema': 'interhuman_fair_requests_manifest_v3',
        'status': 'complete',
        'supersedes_invalid_manifests': [
            str(ROOT / 'output/interhuman_fair_requests/manifest.json'),
            str(ROOT / 'output/interhuman_fair_requests_v2/manifest.json'),
        ],
        'supersession_reason': 'v1 omitted min_length=15; v2 retained trailing caption whitespace that released __getitem__.strip removes',
        'protocol': {
            'replications': 20,
            'base_seed': args.base_seed,
            'batch_size': 96,
            'drop_last': True,
            'official_test_entries': 1177,
            'motion_effective_test_entries': 1176,
            'official_loader_effective_test_entries': 1098,
            'evaluated_entries_per_replication': 1056,
            'mm_prompts': 100,
            'mm_repeats': 30,
            'mm_pairs': 10,
            'diversity_pairs': 300,
        },
        'request_policy': {
            'ordering': 'official effective test split shuffled by Random(base_seed+replication)',
            'per_sample_randomness': 'SHA256-derived independent seed binds caption, crop, coordinate anchor and final person order',
            'gt_storage': 'not duplicated; replayed from official raw motion with the hash-bound adapter',
            'test_left_right_augmentation': False,
            'official_text_exclusions': adapter.split_text_exclusions('test'),
        },
        'artifacts': {
            'test_split': {
                'path': str(split_path), 'sha256': sha256_file(split_path),
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
        'replications': entries,
    }
    manifest['replication_set_sha256'] = canonical_sha256(entries)
    destination = output_dir / 'manifest.json'
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing != manifest:
            raise ValueError(f'refusing to overwrite changed frozen manifest: {destination}')
    else:
        atomic_json(destination, manifest)
    print(json.dumps({
        'status': 'complete', 'manifest': str(destination),
        'replications': len(entries), 'samples_per_replication': 1056,
    }, indent=2))


if __name__ == '__main__':
    main()
