#!/usr/bin/env python3
"""Cheap paired foot-sliding preview for a partially completed shape pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.calc_pene_from_npz import npz_to_amass_dict, to_torch_params
from optimization.human_body_prior.body_model.body_model import BodyModel
from tools.score_volumetric_smpl_npz import discover_npz_pairs, foot_sliding


def body_joints(models, path: Path, device: torch.device) -> torch.Tensor:
    data = npz_to_amass_dict(dict(np.load(path, allow_pickle=True)))
    params = to_torch_params(data, device)
    frames = min(int(params['trans'].shape[0]), 150)
    gender = str(params['gender'])
    if gender not in models:
        raise ValueError(f'unsupported Inter-X gender: {gender}')
    pose_body = params['pose_body'][:frames].reshape(frames, 63)
    pose_hand = torch.cat((
        params['pose_lhand'][:frames], params['pose_rhand'][:frames]
    ), dim=1).reshape(frames, 90)
    betas = params['betas'].reshape(1, 10).expand(frames, -1)
    with torch.no_grad():
        output = models[gender](
            pose_body=pose_body,
            pose_hand=pose_hand,
            root_orient=params['root_orient'][:frames].reshape(frames, 3),
            betas=betas,
            trans=params['trans'][:frames].reshape(frames, 3),
        )
    return output.Jtr


def pair_metric(models, p1: Path, p2: Path, device: torch.device) -> float:
    records = [
        foot_sliding(body_joints(models, path, device))
        for path in (p1, p2)
    ]
    observations = sum(row['contact_observations'] for row in records)
    if not observations:
        return 0.0
    return sum(
        row['contact_observations']
        * row['mean_horizontal_speed_m_per_frame']
        for row in records
    ) / observations


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    models = {
        gender: BodyModel(
            bm_fname=f'body_models/smplx/SMPLX_{gender.upper()}.npz',
            num_betas=10,
        ).to(device).eval()
        for gender in ('male', 'female')
    }
    before = {row[0]: row for row in discover_npz_pairs(args.before)}
    after = {row[0]: row for row in discover_npz_pairs(args.after)}
    keys = sorted(before.keys() & after.keys())
    if not keys:
        raise ValueError('before/after have no completed paired samples')
    records = []
    for key in keys:
        before_value = pair_metric(models, before[key][3], before[key][4], device)
        after_value = pair_metric(models, after[key][3], after[key][4], device)
        records.append({
            'sample_key': key,
            'before_mean_foot_sliding_m_per_frame': before_value,
            'after_mean_foot_sliding_m_per_frame': after_value,
            'delta_after_minus_before': after_value - before_value,
        })
    payload = {
        'schema': 'hhc_partial_shape_foot_sliding_diagnostic_v1',
        'status': 'diagnostic_only_incomplete_pilot',
        'sample_count': len(records),
        'mean_before': sum(row['before_mean_foot_sliding_m_per_frame'] for row in records) / len(records),
        'mean_after': sum(row['after_mean_foot_sliding_m_per_frame'] for row in records) / len(records),
        'mean_delta_after_minus_before': sum(row['delta_after_minus_before'] for row in records) / len(records),
        'records': records,
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
