#!/usr/bin/env python
"""Compare a fresh official-loader request replay to one frozen request."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


EXACT_FIELDS = (
    'names', 'captions', 'lengths', 'betas', 'gender', 'contact',
    'mm_indices', 'seed',
)
EMBEDDING_FIELDS = (
    'text_embeddings', 'gt_embeddings', 'gt_body_embeddings',
)
EXPECTED_CREATION_RUNTIME = {
    'contract': 'pytorch_official_evaluator_single_device_v1',
    'requested_device': 'cpu',
    'device_type': 'cpu',
    'device_index': None,
    'device_name': 'cpu',
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def compare_requests(frozen_path, replay_path, embedding_atol):
    frozen = np.load(frozen_path, allow_pickle=False)
    replay = np.load(replay_path, allow_pickle=False)
    expected_fields = set(EXACT_FIELDS + EMBEDDING_FIELDS)
    if set(frozen.files) != expected_fields or set(replay.files) != expected_fields:
        raise ValueError('frozen/replay request field contract changed')

    comparisons = {}
    failures = []
    for field in EXACT_FIELDS + EMBEDDING_FIELDS:
        left, right = frozen[field], replay[field]
        shape_equal = left.shape == right.shape
        dtype_equal = left.dtype == right.dtype
        exact = shape_equal and dtype_equal and np.array_equal(left, right)
        record = {
            'shape': list(left.shape),
            'dtype': str(left.dtype),
            'shape_equal': shape_equal,
            'dtype_equal': dtype_equal,
            'exact': bool(exact),
        }
        if field in EMBEDDING_FIELDS and shape_equal:
            difference = np.abs(
                left.astype(np.float64) - right.astype(np.float64)
            )
            record.update({
                'maximum_absolute_difference': float(difference.max()),
                'mean_absolute_difference': float(difference.mean()),
                'different_values': int(np.count_nonzero(difference)),
                'within_absolute_tolerance': bool(
                    difference.max() <= embedding_atol
                ),
            })
            passed = (
                dtype_equal and record['within_absolute_tolerance']
            )
        else:
            passed = exact
        record['status'] = 'pass' if passed else 'fail'
        comparisons[field] = record
        if not passed:
            failures.append(field)

    return comparisons, failures


def validate_creation_provenance(path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    creation = payload.get('request_creation_runtime_recovery', {})
    if not (
        payload.get('schema') == 'fair_interx_request_provenance_recovery_v1'
        and payload.get('status') == 'pass'
        and creation.get('status') == 'completed'
        and creation.get('exit_code') == 0
        and creation.get('runtime') == EXPECTED_CREATION_RUNTIME
        and creation.get('replications_reported_complete') == 20
        and creation.get('samples_per_replication_reported') == 1696
        and '--device cpu' in creation.get('command', '')
    ):
        raise ValueError(
            'frozen request creation provenance does not prove the exact '
            'successful CPU/20-replication contract'
        )
    return creation['runtime']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--frozen', type=Path,
        default=Path('output/fair_interx_requests/replication_00.npz'),
    )
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--embedding-atol', type=float, default=1e-6)
    parser.add_argument(
        '--creation-provenance', type=Path,
        default=Path('output/fair_interx_requests/provenance_recovery.json'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/fair_interx_request_replay_audit.json'),
    )
    args = parser.parse_args()
    if args.embedding_atol <= 0 or args.embedding_atol > 1e-5:
        raise ValueError('embedding tolerance must be in (0, 1e-5]')
    if not args.frozen.is_file() or not args.replay.is_file():
        raise FileNotFoundError('frozen or replay request is missing')
    if not args.creation_provenance.is_file():
        raise FileNotFoundError('frozen request creation provenance is missing')

    creation_runtime = validate_creation_provenance(
        args.creation_provenance
    )

    comparisons, failures = compare_requests(
        args.frozen, args.replay, args.embedding_atol
    )
    payload = {
        'schema': 'fair_interx_request_replay_audit_v1',
        'status': 'pass' if not failures else 'fail',
        'scope': 'replication_00_seed1234_official_loader_cpu_replay',
        'sample_count': 1696,
        'embedding_absolute_tolerance': args.embedding_atol,
        'exact_fields': list(EXACT_FIELDS),
        'tolerance_fields': list(EMBEDDING_FIELDS),
        'backend_disclosure': (
            'The NPZ schema itself does not embed its evaluator device, but '
            'the bound successful creation record proves that all 20 frozen '
            'requests were created on CPU. The fresh replay also uses the '
            'queue-fixed CPU path; FP32 embeddings are tolerance-compared for '
            'backend/thread reduction roundoff while all sampling, caption, '
            'motion-length, shape, gender, contact and multimodality fields '
            'must be bitwise equal.'
        ),
        'request_creation_runtime': creation_runtime,
        'request_creation_provenance': artifact(args.creation_provenance),
        'frozen': artifact(args.frozen),
        'replay': artifact(args.replay),
        'comparisons': comparisons,
        'failures': failures,
        'implementation': artifact(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(
        args.output.suffix + f'.tmp.{os.getpid()}'
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, args.output)
    print(json.dumps({
        'status': payload['status'],
        'failures': failures,
        'maximum_embedding_difference': max(
            comparisons[field]['maximum_absolute_difference']
            for field in EMBEDDING_FIELDS
        ),
        'output': str(args.output),
    }, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
