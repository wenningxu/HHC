#!/usr/bin/env python
"""Assess original-MMM shape refinement and a foot-preserving projection."""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.calc_pene_from_npz import PeneCalculator
from tools.diagnose_original_mmm_upper_body_projection import project


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-dir', required=True, type=Path)
    parser.add_argument('--optimized-dir', required=True, type=Path)
    parser.add_argument('--contact-dir', required=True, type=Path)
    parser.add_argument('--projected-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.projected_dir.mkdir(parents=True, exist_ok=True)
    calculator = PeneCalculator(device=args.device)
    rows = []
    sidecars = sorted(
        path for path in args.optimized_dir.glob('*.json')
        if not path.name.startswith('_')
    )
    for sidecar in sidecars:
        metadata = json.loads(sidecar.read_text(encoding='utf-8'))
        name = metadata['augmented_name']
        source_name = metadata['source_name']
        contacts_path = args.contact_dir / f'{source_name}.pkl'
        with open(contacts_path, 'rb') as handle:
            contacts = pickle.load(handle)
        random_paths = [
            args.random_dir / f'{name}_p{person}.npz' for person in (1, 2)
        ]
        optimized_paths = [
            args.optimized_dir / f'{name}_p{person}.npz' for person in (1, 2)
        ]
        projected_paths = [
            args.projected_dir / f'{name}_p{person}.npz' for person in (1, 2)
        ]
        for initial, optimized, output in zip(
            random_paths, optimized_paths, projected_paths
        ):
            project(initial, optimized, output, 'constant_delta')
        conditions = {}
        for label, paths in (
            ('random_shape_mismatch', random_paths),
            ('original_mmm_raw_output', optimized_paths),
            ('foot_preserving_projection', projected_paths),
        ):
            conditions[label] = {
                metric: float(calculator.compute_from_npz(
                    paths[0], paths[1], metric=metric,
                    interaction_pairs=(contacts if metric == 'mismatch' else None),
                ))
                for metric in ('mismatch', 'skate', 'pene')
            }
        rows.append({
            'source_name': source_name,
            'augmented_name': name,
            'conditions': conditions,
        })

    labels = (
        'random_shape_mismatch', 'original_mmm_raw_output',
        'foot_preserving_projection',
    )
    metrics = ('mismatch', 'skate', 'pene')
    aggregate = {}
    baseline = labels[0]
    for label in labels:
        aggregate[label] = {}
        for metric in metrics:
            values = np.asarray([
                row['conditions'][label][metric] for row in rows
            ], dtype=np.float64)
            base = np.asarray([
                row['conditions'][baseline][metric] for row in rows
            ], dtype=np.float64)
            delta = values - base
            aggregate[label][metric] = {
                'mean': float(values.mean()),
                'mean_delta_vs_random': float(delta.mean()),
                'improved_samples': int(np.sum(delta < -1e-8)),
                'degraded_samples': int(np.sum(delta > 1e-8)),
                'max_abs_delta_vs_random': float(np.max(np.abs(delta))),
            }
    payload = {
        'schema': 'hhc_original_mmm_shape_pilot_assessment_v1',
        'status': 'diagnostic_not_production',
        'samples': len(rows),
        'aggregate': aggregate,
        'rows': rows,
        'implementation': {
            'optimizer': {
                'path': str(Path('optimize/optimize_full_body.py').resolve()),
                'sha256': sha256_file('optimize/optimize_full_body.py'),
            },
            'losses': {
                'path': str(Path('optimization/losses.py').resolve()),
                'sha256': sha256_file('optimization/losses.py'),
            },
            'assessment': {
                'path': str(Path(__file__).resolve()),
                'sha256': sha256_file(__file__),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps({'output': str(args.output), 'samples': len(rows)}))


if __name__ == '__main__':
    main()
