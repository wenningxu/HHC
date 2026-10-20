#!/usr/bin/env python3
"""Freeze 100 InterHuman multimodality prompts per formal replication."""

from __future__ import annotations

import argparse
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
from tools.build_interhuman_fair_requests import (
    canonical_sha256, sample_seed, sha256_file,
)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def build_mm_replication(adapter, replication, base_seed=1234):
    eligible = adapter.split_ids('test', require_text=True)
    if len(eligible) != 1098:
        raise ValueError(f'expected 1098 official-loader-eligible prompts, got {len(eligible)}')
    order_rng = random.Random(base_seed + replication)
    order_rng.shuffle(eligible)
    source_positions = list(range(len(eligible)))
    selection_rng = random.Random(100000 + base_seed + replication)
    selection_rng.shuffle(source_positions)
    source_positions = source_positions[:100]
    requests = []
    for prompt_index, position in enumerate(source_positions):
        identifier = eligible[position]
        seed = sample_seed(base_seed, replication, position, identifier)
        python_rng = random.Random(seed)
        numpy_rng = np.random.default_rng(seed)
        captions = adapter.texts(identifier)
        caption_index = python_rng.randrange(len(captions))
        raw_length = adapter._pair_raw_length(identifier)
        crop_start = (
            python_rng.randrange(0, raw_length - 300)
            if raw_length > 300 else 0
        )
        raw_frames = min(raw_length, 300)
        requests.append({
            'prompt_index': prompt_index,
            'source_position_in_full_1098_order': position,
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
        'schema': 'interhuman_fair_mm_requests_replication_v1',
        'replication': replication,
        'full_effective_entries': 1098,
        'mm_prompts': 100,
        'mm_repeats': 30,
        'mm_pairs': 10,
        'requests': requests,
    }
    payload['request_sha256'] = canonical_sha256(requests)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='output/interhuman_fair_mm_requests')
    parser.add_argument('--base-seed', type=int, default=1234)
    args = parser.parse_args()
    adapter = OfficialInterHumanAdapter()
    output_dir = Path(args.output_dir).resolve()
    entries = []
    for replication in range(20):
        payload = build_mm_replication(adapter, replication, args.base_seed)
        path = output_dir / f'replication_{replication:02d}.json'
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise ValueError(f'refusing to overwrite changed MM requests: {path}')
        else:
            atomic_json(path, payload)
        entries.append({
            'replication': replication,
            'path': str(path),
            'bytes': path.stat().st_size,
            'sha256': sha256_file(path),
            'request_sha256': payload['request_sha256'],
        })
    main_manifest = ROOT / 'output/interhuman_fair_requests_v3/manifest.json'
    manifest = {
        'schema': 'interhuman_fair_mm_requests_manifest_v1',
        'status': 'complete',
        'protocol': {
            'replications': 20, 'effective_entries': 1098,
            'prompts': 100, 'repeats': 30, 'pairs': 10,
            'embedding_scale': 6, 'distance_divisor': 2,
        },
        'selection': '100 positions without replacement from each full 1098-entry shuffled replication order; not from the 1056-entry drop_last main subset',
        'main_request_manifest': {
            'path': str(main_manifest), 'sha256': sha256_file(main_manifest),
        },
        'adapter': {
            'path': str(ROOT / 'dataset/interhuman_hhc_adapter.py'),
            'sha256': sha256_file(ROOT / 'dataset/interhuman_hhc_adapter.py'),
        },
        'replications': entries,
    }
    manifest['replication_set_sha256'] = canonical_sha256(entries)
    destination = output_dir / 'manifest.json'
    if destination.exists():
        if json.loads(destination.read_text()) != manifest:
            raise ValueError(f'refusing to overwrite changed MM manifest: {destination}')
    else:
        atomic_json(destination, manifest)
    print(json.dumps({
        'status': 'complete', 'manifest': str(destination),
        'replications': 20, 'prompts_per_replication': 100,
    }, indent=2))


if __name__ == '__main__':
    main()

