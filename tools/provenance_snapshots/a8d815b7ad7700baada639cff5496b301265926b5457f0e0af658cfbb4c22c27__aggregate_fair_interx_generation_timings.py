#!/usr/bin/env python
"""Aggregate common-boundary generation wall times over 20 replications."""

import argparse
import glob
import hashlib
import json
import math
import os
import re

import numpy as np


PATTERN = re.compile(r'replication_(\d{2})_generation_timing\.json$')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(values.mean()),
        'ci95_half_width': float(
            1.96 * values.std(ddof=1) / math.sqrt(len(values))
        ),
        'median': float(np.median(values)),
        'p95': float(np.quantile(values, 0.95)),
        'min': float(values.min()),
        'max': float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    paths = sorted(glob.glob(args.inputs))
    if len(paths) != 20:
        raise ValueError(f'exactly 20 timing reports are required, got {len(paths)}')
    records, ids = [], []
    for path in paths:
        match = PATTERN.search(path)
        if match is None:
            raise ValueError(f'invalid timing filename: {path}')
        ids.append(int(match.group(1)))
        with open(path, encoding='utf-8') as handle:
            record = json.load(handle)
        if record.get('schema') != 'fair_interx_generation_wall_timing_v1':
            raise ValueError(f'{path}: unsupported timing schema')
        if record.get('timing_comparable') is not True:
            raise ValueError(
                f'{path}: timing is resume-only and not cross-method comparable'
            )
        for key in ('requests', 'main_generations', 'mm_generations'):
            source = record[key]
            if not os.path.isfile(source):
                raise FileNotFoundError(source)
            if sha256_file(source) != record[f'{key}_sha256']:
                raise ValueError(f'{path}: {key} hash mismatch')
        records.append(record)
    if ids != list(range(20)):
        raise ValueError(f'expected replication IDs 0..19, got {ids}')
    identities = {(record['method'], record['label']) for record in records}
    if len(identities) != 1:
        raise ValueError(f'timing reports mix methods/labels: {identities}')
    method, label = next(iter(identities))
    output = {
        'schema': 'fair_interx_generation_wall_timing_summary_v1',
        'method': method, 'label': label, 'replications': 20,
        'boundary': records[0]['boundary'],
        'main_elapsed_seconds': summarize([
            record['main']['elapsed_seconds'] for record in records
        ]),
        'main_seconds_per_sample_amortized': summarize([
            record['main']['seconds_per_sample_amortized'] for record in records
        ]),
        'multimodality_elapsed_seconds': summarize([
            record['multimodality']['elapsed_seconds'] for record in records
        ]),
        'multimodality_seconds_per_sample_amortized': summarize([
            record['multimodality']['seconds_per_sample_amortized']
            for record in records
        ]),
        'inputs': [
            {'path': os.path.abspath(path), 'sha256': sha256_file(path)}
            for path in paths
        ],
        'implementation_sha256': sha256_file(__file__),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = args.output + f'.tmp.{os.getpid()}'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    os.replace(temporary, args.output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
