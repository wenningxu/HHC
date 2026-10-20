#!/usr/bin/env python
"""Record the role choice made by the unmodified original-MMM contact loss."""

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
    data = npz_to_amass_dict(dict(np.load(path, allow_pickle=True)))
    gender = str(np.asarray(data['gender']).item())
    result = {
        key: torch.as_tensor(data[key], dtype=torch.float32, device=device)
        for key in (
            'root_orient', 'pose_body', 'pose_lhand', 'pose_rhand', 'trans', 'betas'
        )
    }
    result['gender'] = gender
    return result


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
    loss = DyadicInteractionLoss(device=args.device).to(args.device)
    rows = []
    for sidecar in sorted(args.optimized_dir.glob('G*.json')):
        metadata = json.loads(sidecar.read_text())
        name = metadata['augmented_name']
        source = metadata['source_name']
        record = records[name]
        timing = np.load(record['timing_file'], allow_pickle=False).astype(np.float32)
        timing = torch.as_tensor(
            timing[record['crop_start']:record['crop_stop']],
            dtype=torch.float32, device=args.device,
        )
        paths = [
            args.aligned_dir / 'random_aligned' / f'{name}_p{person}.npz'
            for person in (1, 2)
        ]
        params = [tensor_params(path, args.device) for path in paths]
        geometry = [loss._compute_geometry(item) for item in params]
        with open(args.contact_dir / f'{source}.pkl', 'rb') as handle:
            contacts = pickle.load(handle)
        with torch.no_grad():
            fixed = loss._calc_semantic_contact_loss(
                geometry[0][0], geometry[1][0], geometry[0][2], geometry[1][2],
                contacts, timing,
            )[0]
            swapped = loss._calc_semantic_contact_loss(
                geometry[1][0], geometry[0][0], geometry[1][2], geometry[0][2],
                contacts, timing,
            )[0]
        rows.append({
            'source_name': source,
            'augmented_name': name,
            'fixed_role_attraction': float(fixed),
            'swapped_role_attraction': float(swapped),
            'original_mmm_choice': 'swapped' if swapped < fixed else 'fixed',
        })
    result = {
        'schema': 'hhc_original_mmm_role_assignment_diagnostic_v1',
        'status': 'diagnostic_not_production',
        'rows': rows,
        'samples': len(rows),
        'swapped_samples': sum(row['original_mmm_choice'] == 'swapped' for row in rows),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
