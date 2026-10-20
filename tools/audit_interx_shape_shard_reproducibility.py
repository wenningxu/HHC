#!/usr/bin/env python
"""Audit that Inter-X shape optimization is invariant to process sharding."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--shard-dir', action='append', required=True, type=Path)
    parser.add_argument(
        '--fresh-serial-sidecar', action='append', default=[], type=Path,
    )
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def load_sidecars(directory):
    rows = {}
    for path in sorted(directory.glob('*.json')):
        if path.name.startswith('_'):
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        rows[payload['augmented_name']] = (path, payload)
    return rows


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8'
    )
    os.replace(temporary, path)


def main():
    args = parse_args()
    reference = load_sidecars(args.reference_dir)
    shards = {}
    for directory in args.shard_dir:
        overlap = set(shards).intersection(load_sidecars(directory))
        if overlap:
            raise ValueError(f'duplicate shard samples: {sorted(overlap)[:3]}')
        shards.update(load_sidecars(directory))

    comparisons = []
    for name, (path, candidate) in sorted(shards.items()):
        if name not in reference:
            raise ValueError(f'{name}: absent from reference')
        reference_path, baseline = reference[name]
        baseline_history = baseline['diagnostics']['loss_history']
        candidate_history = candidate['diagnostics']['loss_history']
        if len(baseline_history) != len(candidate_history):
            raise ValueError(f'{name}: loss-history length differs')
        comparisons.append({
            'augmented_name': name,
            'reference_sidecar': str(reference_path.resolve()),
            'shard_sidecar': str(path.resolve()),
            'selection_position': candidate.get('selection_position'),
            'volumetric_smpl_seed': candidate.get('volumetric_smpl_seed'),
            'p1_sha256_equal': (
                baseline['output_p1_sha256'] == candidate['output_p1_sha256']
            ),
            'p2_sha256_equal': (
                baseline['output_p2_sha256'] == candidate['output_p2_sha256']
            ),
            'total_loss_max_abs_difference': max(
                abs(float(left['total']) - float(right['total']))
                for left, right in zip(baseline_history, candidate_history)
            ),
            'reference_elapsed_seconds': float(
                baseline['diagnostics']['elapsed_seconds']
            ),
            'shard_elapsed_seconds': float(
                candidate['diagnostics']['elapsed_seconds']
            ),
        })

    fresh_serial_seconds = [
        float(json.loads(path.read_text(encoding='utf-8'))[
            'diagnostics'
        ]['elapsed_seconds'])
        for path in args.fresh_serial_sidecar
    ]
    concurrent_wall_seconds = max(
        row['shard_elapsed_seconds'] for row in comparisons
    )
    serial_reference_seconds = (
        sum(fresh_serial_seconds) if fresh_serial_seconds else
        sum(row['reference_elapsed_seconds'] for row in comparisons)
    )
    passed = bool(comparisons) and all(
        row['p1_sha256_equal']
        and row['p2_sha256_equal']
        and row['total_loss_max_abs_difference'] == 0.0
        for row in comparisons
    )
    payload = {
        'schema': 'hhc_interx_shape_shard_reproducibility_audit_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'pass' if passed else 'fail',
        'seed_policy': (
            'base_seed_plus_100x_full_selection_position_before_slicing'
        ),
        'fixed_failure_mode': (
            'post-slice local enumeration previously changed a sample seed '
            'when offset/limit sharding was used'
        ),
        'formal_frame_batch_size': 8,
        'formal_query_chunk': 10475,
        'comparisons': comparisons,
        'runtime': {
            'serial_reference_seconds': serial_reference_seconds,
            'concurrent_wall_seconds': concurrent_wall_seconds,
            'observed_speedup': (
                serial_reference_seconds / concurrent_wall_seconds
            ),
            'fresh_serial_sidecars': [
                str(path.resolve()) for path in args.fresh_serial_sidecar
            ],
            'note': (
                'Runtime is descriptive only; numerical identity is the '
                'acceptance criterion.'
            ),
        },
        'accepted_topology': 'two_processes_one_gpu_contiguous_shards',
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
