#!/usr/bin/env python3
"""Compare original-MMM's own semantic objective before and after refinement."""

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

from eval.calc_pene_from_npz import npz_to_amass_dict
from optimization.losses import DyadicInteractionLoss


def tensor_params(path, device):
    raw = npz_to_amass_dict(dict(np.load(path, allow_pickle=True)))
    result = {
        key: torch.as_tensor(raw[key], dtype=torch.float32, device=device)
        for key in (
            'root_orient', 'pose_body', 'pose_lhand', 'pose_rhand', 'trans',
            'betas',
        )
    }
    result['gender'] = str(np.asarray(raw['gender']).item())
    return result


def semantic(loss, params_a, params_b, contacts, timing):
    geometry_a = loss._compute_geometry(params_a)
    geometry_b = loss._compute_geometry(params_b)
    return loss._calc_semantic_contact_loss(
        geometry_a[0], geometry_b[0], geometry_a[2], geometry_b[2],
        contacts, timing,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aligned-dir', required=True, type=Path)
    parser.add_argument('--optimized-dir', required=True, type=Path)
    parser.add_argument('--contact-dir', required=True, type=Path)
    parser.add_argument('--plan-records', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    records = {
        row['augmented_name']: row
        for row in (
            json.loads(line) for line in args.plan_records.read_text().splitlines()
            if line.strip()
        )
    }
    rows = []
    for sidecar in sorted(args.optimized_dir.glob('G*.json')):
        metadata = json.loads(sidecar.read_text(encoding='utf-8'))
        name = metadata['augmented_name']
        source = metadata['source_name']
        record = records[name]
        timing = np.load(record['timing_file'], allow_pickle=False).astype(np.float32)
        timing = torch.as_tensor(
            timing[record['crop_start']:record['crop_stop']],
            dtype=torch.float32, device=args.device,
        )
        with open(args.contact_dir / f'{source}.pkl', 'rb') as handle:
            contacts = pickle.load(handle)
        initial = [
            tensor_params(
                args.aligned_dir / 'random_aligned' / f'{name}_p{person}.npz',
                args.device,
            )
            for person in (1, 2)
        ]
        final = [
            tensor_params(
                args.aligned_dir / 'optimized_before_restore'
                / f'{name}_p{person}.npz',
                args.device,
            )
            for person in (1, 2)
        ]
        loss = DyadicInteractionLoss(device=args.device).to(args.device)
        with torch.no_grad():
            initial_fixed = semantic(
                loss, initial[0], initial[1], contacts, timing,
            )
            initial_swapped = semantic(
                loss, initial[1], initial[0], contacts, timing,
            )
            swapped = bool(initial_swapped[0] < initial_fixed[0])
            ordered_final = final[::-1] if swapped else final
            final_selected = semantic(
                loss, ordered_final[0], ordered_final[1], contacts, timing,
            )
        initial_selected = initial_swapped if swapped else initial_fixed
        rows.append({
            'source_name': source,
            'augmented_name': name,
            'structured_contact_rows': len(contacts),
            'original_mmm_role': 'swapped' if swapped else 'fixed',
            'initial_attraction': float(initial_selected[0]),
            'final_attraction_dynamic_regions': float(final_selected[0]),
            'attraction_delta': float(final_selected[0] - initial_selected[0]),
            'initial_alignment': float(initial_selected[1]),
            'final_alignment_dynamic_regions': float(final_selected[1]),
            'alignment_delta': float(final_selected[1] - initial_selected[1]),
        })

    active = [row for row in rows if row['structured_contact_rows'] > 0]
    aggregate = {}
    for metric in ('attraction', 'alignment'):
        deltas = np.asarray(
            [row[f'{metric}_delta'] for row in active], dtype=np.float64,
        )
        aggregate[metric] = {
            'active_samples': len(active),
            'mean_delta': float(deltas.mean()),
            'median_delta': float(np.median(deltas)),
            'improved_samples': int(np.sum(deltas < -1e-8)),
            'degraded_samples': int(np.sum(deltas > 1e-8)),
        }
    result = {
        'schema': 'hhc_original_mmm_semantic_objective_diagnostic_v1',
        'status': 'diagnostic_not_production',
        'scope': (
            'original-MMM dynamic semantic-region objective in temporary '
            'ground-aligned coordinates; the role chosen at initialization is '
            'held fixed for the final measurement'
        ),
        'samples': len(rows),
        'active_contact_samples': len(active),
        'aggregate': aggregate,
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps(result['aggregate'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
