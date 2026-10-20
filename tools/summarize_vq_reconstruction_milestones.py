#!/usr/bin/env python
"""Summarize preregistered full-test VQ reconstruction checkpoints."""

import argparse
import glob
import json
import os
from pathlib import Path


EXPECTED_STEPS = (5000, 10000, 25000, 150000)
METRICS = (
    'mpjpe_m', 'body_mpjpe_m', 'hand_mpjpe_m',
    'root_translation_error_m', 'mean_rotation_error_rad',
    'mae_raw_features', 'rmse_raw_features',
)


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-glob',
        default='output/vq_paper_eval/*reconstruction.json',
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/vq_paper_eval/reconstruction_milestones.json'),
    )
    args = parser.parse_args()

    by_step = {}
    request_hashes = set()
    ground_truth_hashes = set()
    for filename in sorted(glob.glob(args.input_glob)):
        path = Path(filename).resolve()
        payload = json.loads(path.read_text(encoding='utf-8'))
        step = int(payload['checkpoint_iteration'])
        if step in by_step:
            raise ValueError(f'duplicate reconstruction milestone {step}')
        if payload.get('samples') != 1696:
            raise ValueError(f'{path}: expected 1696 samples')
        if not all(metric in payload for metric in METRICS):
            raise ValueError(f'{path}: reconstruction metrics are incomplete')
        request_hashes.add(payload['requests_sha256'])
        ground_truth_hashes.add(payload['ground_truth_sha256'])
        by_step[step] = {
            'checkpoint': payload['checkpoint'],
            'checkpoint_sha256': payload['checkpoint_sha256'],
            **{metric: float(payload[metric]) for metric in METRICS},
            'codebook_usage': payload['codebook_usage'],
        }
    if tuple(sorted(by_step)) != EXPECTED_STEPS:
        raise ValueError(
            f'expected reconstruction milestones {EXPECTED_STEPS}, '
            f'got {tuple(sorted(by_step))}'
        )
    if len(request_hashes) != 1 or len(ground_truth_hashes) != 1:
        raise ValueError('milestones do not share one frozen request/GT artifact')

    final = by_step[150000]
    result = {
        'schema': 'hhc_vq_reconstruction_milestones_v1',
        'status': 'pass',
        'steps': list(EXPECTED_STEPS),
        'samples_per_step': 1696,
        'requests_sha256': next(iter(request_hashes)),
        'ground_truth_sha256': next(iter(ground_truth_hashes)),
        'milestones': {str(step): by_step[step] for step in EXPECTED_STEPS},
        'minimum_metric_step': {
            metric: min(EXPECTED_STEPS, key=lambda step: by_step[step][metric])
            for metric in METRICS
        },
        'final_vs_25000_percent': {
            metric: 100.0 * (final[metric] / by_step[25000][metric] - 1.0)
            for metric in METRICS
        },
        'selection_note': (
            'The paper fixes the VQ budget at 150000 updates. These test-set '
            'milestones are diagnostics and were not used for checkpoint selection.'
        ),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output.resolve(), result)
    print(json.dumps({
        'output': str(args.output.resolve()),
        'steps': list(EXPECTED_STEPS),
        'final_mpjpe_m': final['mpjpe_m'],
    }))


if __name__ == '__main__':
    main()
