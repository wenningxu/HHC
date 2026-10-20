#!/usr/bin/env python
"""Audit restartable progress of the eight formal Inter-X shape shards."""

import argparse
import hashlib
import json
import math
import os
import statistics
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


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('output'))
    parser.add_argument('--expected-samples', type=int, default=6238)
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    positions = {}
    elapsed = []
    errors = []
    shard_records = []
    implementation_hashes = set()
    for shard in range(args.shards):
        directory = args.output_root / (
            'interx_shape_aug_full_batch8_query10475_seed20260828_'
            f'shard{shard}'
        )
        contract_path = directory / '_run_contract.json'
        if not contract_path.is_file():
            shard_records.append({
                'shard': shard, 'directory': str(directory.resolve()),
                'state': 'not_started', 'completed': 0,
            })
            continue
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
        expected_names = contract['selection']['selected_names']
        expected_positions = contract['selection']['full_selection_positions']
        implementation_hashes.add(
            contract['implementation'][str(
                (Path.cwd() / 'tools/run_interx_shape_augmentation.py').resolve()
            )]
        )
        completed = 0
        for path in directory.glob('G*.json'):
            sidecar = json.loads(path.read_text(encoding='utf-8'))
            name = sidecar['augmented_name']
            position = int(sidecar['selection_position'])
            if name not in expected_names:
                errors.append(f'{path}: name outside shard contract')
                continue
            expected_position = expected_positions[expected_names.index(name)]
            if position != expected_position:
                errors.append(f'{path}: selection position mismatch')
            if position in positions:
                errors.append(f'duplicate completed position {position}')
            positions[position] = name
            history = sidecar.get('diagnostics', {}).get('loss_history', [])
            if len(history) != 100 or any(
                not math.isfinite(float(value))
                for row in history
                for key, value in row.items()
                if key not in ('iteration', 'contact_stage')
            ):
                errors.append(f'{path}: incomplete/nonfinite loss history')
            for person in ('p1', 'p2'):
                npz = directory / f'{name}_{person}.npz'
                if not npz.is_file() or sha256_file(npz) != sidecar.get(
                    f'output_{person}_sha256'
                ):
                    errors.append(f'{path}: {person} output hash mismatch')
            sample_elapsed = float(sidecar['diagnostics']['elapsed_seconds'])
            if not math.isfinite(sample_elapsed) or sample_elapsed <= 0:
                errors.append(f'{path}: invalid elapsed time')
            else:
                elapsed.append(sample_elapsed)
            completed += 1
        completion_path = directory / '_completion_manifest.json'
        completion_complete = False
        if completion_path.is_file():
            completion_manifest = json.loads(
                completion_path.read_text(encoding='utf-8')
            )
            completion_complete = (
                completion_manifest.get('complete') is True
                and completion_manifest.get('completed_samples')
                == len(expected_names)
            )
        shard_records.append({
            'shard': shard,
            'directory': str(directory.resolve()),
            'state': 'complete' if completion_complete else 'running_or_paused',
            'completed': completed,
            'expected': len(expected_names),
            'first_position': min(expected_positions),
            'last_position': max(expected_positions),
            'contract_sha256': contract['contract_sha256'],
        })

    completed_count = len(positions)
    remaining = args.expected_samples - completed_count
    mean_elapsed = statistics.mean(elapsed) if elapsed else None
    payload = {
        'schema': 'hhc_interx_shape_full_progress_audit_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'invalid' if errors else (
            'complete' if completed_count == args.expected_samples
            else 'running_or_paused_valid'
        ),
        'expected_samples': args.expected_samples,
        'completed_samples': completed_count,
        'remaining_samples': remaining,
        'progress_fraction': completed_count / args.expected_samples,
        'completed_position_min': min(positions) if positions else None,
        'completed_position_max': max(positions) if positions else None,
        'implementation_hashes': sorted(implementation_hashes),
        'runtime_seconds': {
            'observations': len(elapsed),
            'mean': mean_elapsed,
            'median': statistics.median(elapsed) if elapsed else None,
            'p95': percentile(elapsed, 0.95),
            'projected_remaining_gpu_days_at_observed_mean': (
                remaining * mean_elapsed / 86400 if mean_elapsed else None
            ),
        },
        'shards': shard_records,
        'errors': errors,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
