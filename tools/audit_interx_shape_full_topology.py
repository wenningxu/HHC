#!/usr/bin/env python
"""Supersede the short shape benchmark with evidence from the full corpus."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short-audit', required=True, type=Path)
    parser.add_argument('--preflight', required=True, type=Path)
    parser.add_argument('--shard-dir', action='append', required=True, type=Path)
    parser.add_argument('--random-physical', required=True, type=Path)
    parser.add_argument('--optimized-physical', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    short = json.loads(args.short_audit.read_text(encoding='utf-8'))
    preflight = json.loads(args.preflight.read_text(encoding='utf-8'))
    baseline = json.loads(args.random_physical.read_text(encoding='utf-8'))
    optimized = json.loads(args.optimized_physical.read_text(encoding='utf-8'))
    completed = []
    for directory in args.shard_dir:
        completed.extend(sorted(
            path.stem for path in directory.glob('G*.json')
        ))

    before = baseline['aggregate']
    after = optimized['aggregate']
    checks = {
        'short_two_process_audit_had_passed': short.get('status') == 'pass',
        'full_preflight_has_6238_samples': (
            preflight.get('status') == 'pass'
            and preflight.get('selected_samples') == 6238
        ),
        'full_run_reached_real_samples_before_oom': len(completed) >= 3,
        'extreme_eq6_improved': (
            after['paper_eq6_squared_sum_m2_per_frame']['mean']
            < before['paper_eq6_squared_sum_m2_per_frame']['mean']
        ),
        'extreme_penetrating_fraction_improved': (
            after['penetrating_vertex_fraction']['mean']
            < before['penetrating_vertex_fraction']['mean']
        ),
        'extreme_foot_sliding_unchanged': (
            after['mean_foot_sliding_m_per_frame']['mean']
            == before['mean_foot_sliding_m_per_frame']['mean']
        ),
    }
    payload = {
        'schema': 'hhc_interx_shape_full_topology_audit_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'pass' if all(checks.values()) else 'fail',
        'checks': checks,
        'observed_full_run_failure': {
            'exception_class': 'torch.OutOfMemoryError',
            'scope': 'two concurrent frame_batch=8/query_chunk=10475 processes',
            'completed_atomic_samples_preserved': completed,
            'partial_samples_are_not_accepted_without_sidecars': True,
        },
        'decision': {
            'supersedes_short_benchmark_topology': True,
            'formal_topology': 'one_process_per_gpu',
            'frame_batch_size': 8,
            'query_chunk': 10475,
            'seed_policy': short.get('seed_policy'),
            'reason': (
                'full-corpus memory behavior outranks the two-sample '
                'benchmark; numerical method settings remain unchanged'
            ),
        },
        'extreme_sample_physical_metrics': {
            'sample': 'G053T001A009R026__shapeaug_s00000123',
            'random_shape_only': {
                key: value['mean'] for key, value in before.items()
            },
            'optimized': {
                key: value['mean'] for key, value in after.items()
            },
        },
        'inputs': {
            str(path.resolve()): sha256_file(path)
            for path in (
                args.short_audit, args.preflight,
                args.random_physical, args.optimized_physical,
            )
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
