#!/usr/bin/env python3
"""Diagnose whether the released dynamic skate mask drops initial foot plants."""

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
from tools.score_volumetric_smpl_npz import discover_npz_pairs


FOOT_INDICES = (7, 10, 8, 11)
HEIGHT_THRESHOLDS_M = (0.12, 0.05, 0.12, 0.05)
OPTIMIZER_SQUARED_SPEED_THRESHOLD = 0.001


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


def foot_contact_drift(before: torch.Tensor, after: torch.Tensor) -> dict:
    """Compare final feet on the exact contact mask detected initially.

    The released optimizer recomputes its mask from current velocity and
    height at every iteration.  ``reference_contact`` reproduces that mask on
    the unoptimized motion and holds it fixed only for this diagnostic.
    """
    frames = min(int(before.shape[0]), int(after.shape[0]))
    if frames < 2:
        raise ValueError('foot-contact drift requires at least two frames')
    before_feet = before[:frames, FOOT_INDICES]
    after_feet = after[:frames, FOOT_INDICES]
    thresholds = before.new_tensor(HEIGHT_THRESHOLDS_M)

    before_velocity = before_feet[1:] - before_feet[:-1]
    after_velocity = after_feet[1:] - after_feet[:-1]
    before_grounded = before_feet[:-1, :, 1] < thresholds
    after_grounded = after_feet[:-1, :, 1] < thresholds
    before_slow = torch.sum(before_velocity ** 2, dim=-1) < (
        OPTIMIZER_SQUARED_SPEED_THRESHOLD
    )
    after_slow = torch.sum(after_velocity ** 2, dim=-1) < (
        OPTIMIZER_SQUARED_SPEED_THRESHOLD
    )
    reference_contact = before_grounded & before_slow
    dynamic_final_contact = after_grounded & after_slow
    observations = int(reference_contact.sum().item())
    dynamic_observations = int(dynamic_final_contact.sum().item())

    before_horizontal = torch.linalg.vector_norm(
        before_velocity[..., (0, 2)], dim=-1
    )
    after_horizontal = torch.linalg.vector_norm(
        after_velocity[..., (0, 2)], dim=-1
    )

    def masked_sum(values, mask) -> float:
        if not bool(mask.any()):
            return 0.0
        return float(values[mask].double().sum().item())

    retained_height = reference_contact & after_grounded
    retained_slow = reference_contact & after_slow
    retained_dynamic = reference_contact & dynamic_final_contact
    return {
        'reference_contact_observations': observations,
        'before_reference_horizontal_speed_sum_m_per_frame': masked_sum(
            before_horizontal, reference_contact
        ),
        'after_reference_horizontal_speed_sum_m_per_frame': masked_sum(
            after_horizontal, reference_contact
        ),
        'reference_contacts_still_below_height_threshold': int(
            retained_height.sum().item()
        ),
        'reference_contacts_still_below_speed_threshold': int(
            retained_slow.sum().item()
        ),
        'reference_contacts_retained_by_dynamic_mask': int(
            retained_dynamic.sum().item()
        ),
        'final_dynamic_contact_observations': dynamic_observations,
        'final_dynamic_horizontal_speed_sum_m_per_frame': masked_sum(
            after_horizontal, dynamic_final_contact
        ),
    }


def add_counts(target: dict, source: dict) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def summarize(totals: dict) -> dict:
    observations = totals['reference_contact_observations']
    dynamic_observations = totals['final_dynamic_contact_observations']
    return {
        **totals,
        'before_reference_mean_horizontal_speed_m_per_frame': ratio(
            totals['before_reference_horizontal_speed_sum_m_per_frame'],
            observations,
        ),
        'after_reference_mean_horizontal_speed_m_per_frame': ratio(
            totals['after_reference_horizontal_speed_sum_m_per_frame'],
            observations,
        ),
        'reference_height_retention_fraction': ratio(
            totals['reference_contacts_still_below_height_threshold'],
            observations,
        ),
        'reference_speed_retention_fraction': ratio(
            totals['reference_contacts_still_below_speed_threshold'],
            observations,
        ),
        'reference_dynamic_mask_retention_fraction': ratio(
            totals['reference_contacts_retained_by_dynamic_mask'],
            observations,
        ),
        'final_dynamic_mean_horizontal_speed_m_per_frame': ratio(
            totals['final_dynamic_horizontal_speed_sum_m_per_frame'],
            dynamic_observations,
        ),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
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

    totals = {}
    records = []
    for key in keys:
        pair_totals = {}
        for before_path, after_path in zip(
            before[key][3:5], after[key][3:5]
        ):
            add_counts(pair_totals, foot_contact_drift(
                body_joints(models, before_path, device),
                body_joints(models, after_path, device),
            ))
        add_counts(totals, pair_totals)
        records.append({'sample_key': key, **summarize(pair_totals)})

    aggregate = summarize(totals)
    before_mean = aggregate[
        'before_reference_mean_horizontal_speed_m_per_frame'
    ]
    after_mean = aggregate[
        'after_reference_mean_horizontal_speed_m_per_frame'
    ]
    aggregate['reference_mean_speed_delta_after_minus_before'] = (
        None if before_mean is None or after_mean is None
        else after_mean - before_mean
    )
    payload = {
        'schema': 'hhc_shape_foot_contact_drift_diagnostic_v1',
        'status': 'diagnostic_only_incomplete_pilot',
        'contract': {
            'foot_joint_indices': list(FOOT_INDICES),
            'height_thresholds_m': list(HEIGHT_THRESHOLDS_M),
            'optimizer_squared_speed_threshold': (
                OPTIMIZER_SQUARED_SPEED_THRESHOLD
            ),
            'reference_mask': (
                'released optimizer contact predicate evaluated once on the '
                'unoptimized random-shape motion'
            ),
            'dynamic_mask': (
                'same released predicate evaluated on the optimized motion'
            ),
        },
        'sample_count': len(records),
        'aggregate': aggregate,
        'records': records,
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
