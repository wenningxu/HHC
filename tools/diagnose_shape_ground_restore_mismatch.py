#!/usr/bin/env python
"""Measure whether per-person ground restoration undoes optimized contact."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.calc_pene_from_npz import (
    PeneCalculator,
    npz_to_amass_dict,
    to_torch_params,
)


def shifted_copy(source, destination, y_delta):
    payload = dict(np.load(source, allow_pickle=True))
    trans = np.asarray(payload['trans'], dtype=np.float32).copy()
    trans[:, 1] += float(y_delta)
    payload['trans'] = trans
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **payload)


def timed_mismatch(calculator, paths, contacts, timing, swapped=False):
    params = [
        to_torch_params(
            npz_to_amass_dict(dict(np.load(path, allow_pickle=True))),
            calculator.device,
        )
        for path in paths
    ]
    max_frames = min(len(timing), *(item['trans'].shape[0] for item in params), 150)
    for item in params:
        for key in ('root_orient', 'pose_body', 'pose_lhand', 'pose_rhand', 'trans'):
            item[key] = item[key][:max_frames]
    geometry = [calculator._compute_geometry(item) for item in params]
    if swapped:
        geometry = geometry[::-1]
    mismatch, _ = calculator._calc_contact_mismatch_per_frame(
        geometry[0][0], geometry[1][0], geometry[0][2], geometry[1][2], contacts,
    )
    weights = torch.as_tensor(
        timing[:max_frames], dtype=mismatch.dtype, device=mismatch.device,
    )
    return float((mismatch * weights).sum() / weights.sum().clamp_min(1.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-dir', required=True, type=Path)
    parser.add_argument('--optimized-dir', required=True, type=Path)
    parser.add_argument('--contact-dir', required=True, type=Path)
    parser.add_argument('--plan-records', required=True, type=Path)
    parser.add_argument('--aligned-dir', required=True, type=Path)
    parser.add_argument('--projected-dir', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    calculator = PeneCalculator(device=args.device)
    plan = {
        row['augmented_name']: row
        for row in (
            json.loads(line) for line in args.plan_records.read_text().splitlines()
            if line.strip()
        )
    }
    rows = []
    sidecars = sorted(
        path for path in args.optimized_dir.glob('*.json')
        if not path.name.startswith('_')
    )
    for sidecar in sidecars:
        metadata = json.loads(sidecar.read_text(encoding='utf-8'))
        name = metadata['augmented_name']
        source_name = metadata['source_name']
        floors = [
            metadata['diagnostics']['ground_alignment']['p1_floor_y_m'],
            metadata['diagnostics']['ground_alignment']['p2_floor_y_m'],
        ]
        with open(args.contact_dir / f'{source_name}.pkl', 'rb') as handle:
            contacts = pickle.load(handle)
        record = plan[name]
        timing = np.load(record['timing_file'], allow_pickle=False).astype(np.float32)
        timing = timing[record['crop_start']:record['crop_stop']]

        conditions = {}
        for label, directory in (
            ('random_raw', args.random_dir),
            ('optimized_restored_raw', args.optimized_dir),
        ):
            paths = [directory / f'{name}_p{person}.npz' for person in (1, 2)]
            conditions[label] = float(calculator.compute_from_npz(
                paths[0], paths[1], metric='mismatch',
                interaction_pairs=contacts,
            ))

        aligned = {}
        for label, directory in (
            ('random_aligned', args.random_dir),
            ('optimized_before_restore', args.optimized_dir),
        ):
            paths = []
            for person, floor in zip((1, 2), floors):
                source = directory / f'{name}_p{person}.npz'
                destination = args.aligned_dir / label / f'{name}_p{person}.npz'
                shifted_copy(source, destination, -float(floor))
                paths.append(destination)
            aligned[label] = float(calculator.compute_from_npz(
                paths[0], paths[1], metric='mismatch',
                interaction_pairs=contacts,
            ))
        conditions.update(aligned)
        random_aligned_paths = [
            args.aligned_dir / 'random_aligned' / f'{name}_p{person}.npz'
            for person in (1, 2)
        ]
        optimized_aligned_paths = [
            args.aligned_dir / 'optimized_before_restore' / f'{name}_p{person}.npz'
            for person in (1, 2)
        ]
        timing_metrics = {
            'random_fixed_role': timed_mismatch(
                calculator, random_aligned_paths, contacts, timing,
            ),
            'random_swapped_role': timed_mismatch(
                calculator, random_aligned_paths, contacts, timing, swapped=True,
            ),
            'optimized_fixed_role': timed_mismatch(
                calculator, optimized_aligned_paths, contacts, timing,
            ),
            'optimized_swapped_role': timed_mismatch(
                calculator, optimized_aligned_paths, contacts, timing, swapped=True,
            ),
        }
        if args.projected_dir is not None:
            projected_paths = [
                args.projected_dir / f'{name}_p{person}.npz'
                for person in (1, 2)
            ]
            conditions['foot_projection_raw'] = float(calculator.compute_from_npz(
                projected_paths[0], projected_paths[1], metric='mismatch',
                interaction_pairs=contacts,
            ))
            projected_aligned_paths = []
            for person, floor in zip((1, 2), floors):
                destination = (
                    args.aligned_dir / 'foot_projection_aligned' /
                    f'{name}_p{person}.npz'
                )
                shifted_copy(projected_paths[person - 1], destination, -float(floor))
                projected_aligned_paths.append(destination)
            timing_metrics['foot_projection_fixed_role'] = timed_mismatch(
                calculator, projected_aligned_paths, contacts, timing,
            )
            timing_metrics['foot_projection_swapped_role'] = timed_mismatch(
                calculator, projected_aligned_paths, contacts, timing, swapped=True,
            )
        rows.append({
            'source_name': source_name,
            'augmented_name': name,
            'floor_y_m': floors,
            'relative_restore_shift_m': float(floors[0] - floors[1]),
            'mismatch': conditions,
            'timing_weighted_mismatch_aligned': timing_metrics,
            'initial_role_preferred_by_timing_weighted_mismatch': (
                'swapped' if timing_metrics['random_swapped_role'] <
                timing_metrics['random_fixed_role'] else 'fixed'
            ),
            'delta_optimized_vs_random_raw': (
                conditions['optimized_restored_raw'] - conditions['random_raw']
            ),
            'delta_optimized_vs_random_aligned': (
                conditions['optimized_before_restore'] - conditions['random_aligned']
            ),
        })

    result = {
        'schema': 'hhc_shape_ground_restore_diagnostic_v1',
        'status': 'diagnostic_not_production',
        'samples': len(rows),
        'rows': rows,
        'aggregate': {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                'relative_restore_shift_m',
                'delta_optimized_vs_random_raw',
                'delta_optimized_vs_random_aligned',
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result['aggregate'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
