#!/usr/bin/env python3
"""Compare selected HHC and released InterMask VQs on identical val evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


LOWER_IS_BETTER = (
    'fid', 'matching', 'mpjpe_m', 'normalized_feature_mae',
)
HIGHER_IS_BETTER = ('top1', 'top2', 'top3')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compare(hhc, official):
    if hhc.get('schema') != 'hhc_interhuman_vq_official_validation_score_v1':
        raise ValueError('unexpected HHC InterHuman VQ score schema')
    if official.get('schema') != 'interhuman_vq_official_validation_score_v1':
        raise ValueError('unexpected released InterMask VQ score schema')
    for record, label in ((hhc, 'HHC'), (official, 'InterMask')):
        if record.get('status') != 'complete' or record.get('split') != 'val':
            raise ValueError(f'{label} VQ score is not completed val evidence')
        if int(record.get('samples', -1)) != 480:
            raise ValueError(f'{label} VQ score does not contain 480 samples')
    for key in ('request_file_sha256', 'request_content_sha256', 'evaluator_provenance'):
        if hhc.get(key) != official.get(key):
            raise ValueError(f'HHC and InterMask VQ {key} differ')
    for record in (hhc, official):
        for name in LOWER_IS_BETTER + HIGHER_IS_BETTER + ('diversity', 'diversity_gt'):
            if not math.isfinite(float(record[name])):
                raise ValueError(f'non-finite VQ comparison metric: {name}')
    if not math.isclose(float(hhc['diversity_gt']), float(official['diversity_gt']),
                        rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError('HHC and InterMask VQ diversity_gt differ')
    metrics = {}
    for name in LOWER_IS_BETTER + HIGHER_IS_BETTER:
        hhc_value = float(hhc[name])
        official_value = float(official[name])
        direction = 'lower' if name in LOWER_IS_BETTER else 'higher'
        hhc_better = hhc_value < official_value if direction == 'lower' else hhc_value > official_value
        metrics[name] = {
            'hhc': hhc_value,
            'official_intermask': official_value,
            'hhc_minus_official': hhc_value - official_value,
            'better_direction': direction,
            'hhc_better': bool(hhc_better),
        }
    gt = float(official['diversity_gt'])
    hhc_diversity, official_diversity = float(hhc['diversity']), float(official['diversity'])
    metrics['diversity'] = {
        'hhc': hhc_diversity,
        'official_intermask': official_diversity,
        'hhc_minus_official': hhc_diversity - official_diversity,
        'gt_reference': gt,
        'hhc_absolute_gt_error': abs(hhc_diversity - gt),
        'official_absolute_gt_error': abs(official_diversity - gt),
        'better_direction': 'closer_to_gt',
        'hhc_better': abs(hhc_diversity - gt) < abs(official_diversity - gt),
    }
    return {
        'schema': 'hhc_vs_official_intermask_vq_validation_comparison_v1',
        'status': 'complete',
        'scope': 'single_frozen_official_val_set_descriptive_not_test',
        'samples': 480,
        'selection_metric': 'FID',
        'same_request_and_evaluator': True,
        'metrics': metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hhc', required=True)
    parser.add_argument('--official-intermask', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    hhc_path = Path(args.hhc).resolve()
    official_path = Path(args.official_intermask).resolve()
    result = compare(
        json.loads(hhc_path.read_text()), json.loads(official_path.read_text())
    )
    result['inputs'] = {
        'hhc': {'path': str(hhc_path), 'sha256': sha256_file(hhc_path)},
        'official_intermask': {
            'path': str(official_path), 'sha256': sha256_file(official_path),
        },
    }
    result['implementation_sha256'] = sha256_file(Path(__file__).resolve())
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
