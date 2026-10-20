#!/usr/bin/env python
"""Score frozen Inter-X generations with deterministic, LLM-free joint metrics.

The contact target is the frame-level annotation frozen in the request file.
All positions come from the official subject-specific Inter-X skeleton and the
generated 6D rotations/root translations.  No text-to-region model is used.
"""

import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch

from tools.score_fair_interx_generations import sha256_file, validate_generation_file
from utils.fk import FK_Pair_Layer_Batch
from utils.paramUtil import INTERX_SKELETONS, SMPLX_TREE


GROUP_PATTERN = re.compile(r'^G(\d{3})')
FOOT_JOINTS = (7, 10, 8, 11)


def groups_from_names(names, device):
    rows = []
    for name in names:
        match = GROUP_PATTERN.match(str(name))
        if match is None:
            raise ValueError(f'cannot recover Inter-X subject group from sample name {name!r}')
        group_index = int(match.group(1)) - 1
        if not 0 <= group_index < len(INTERX_SKELETONS):
            raise ValueError(f'Inter-X group index out of range in {name!r}')
        rows.append((0, group_index, 1, group_index))
    return torch.tensor(rows, dtype=torch.long, device=device)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def foot_floor_height(feet, valid, policy='valid_frames'):
    heights = feet[..., 1]
    if policy == 'valid_frames':
        if valid.shape != heights.shape[:2] or not valid.any(dim=1).all():
            raise ValueError('each motion needs at least one valid frame')
        heights = heights.masked_fill(~valid.unsqueeze(-1), float('inf'))
    elif policy != 'legacy_all_frames':
        raise ValueError(f'unknown floor policy: {policy}')
    return heights.amin(dim=(1, 2), keepdim=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--generations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None, help='Optional prefix for smoke tests')
    parser.add_argument('--contact-threshold-m', type=float, default=0.10)
    parser.add_argument(
        '--contact-thresholds-m', type=float, nargs='+', default=[0.05, 0.10, 0.15],
        help='Threshold sweep reported in addition to the primary threshold',
    )
    parser.add_argument('--ground-height-threshold-m', type=float, default=0.05)
    parser.add_argument(
        '--floor-policy', choices=('legacy_all_frames', 'valid_frames'),
        default='legacy_all_frames',
        help='Use valid_frames for new experiments; legacy default preserves the audited v1 protocol.',
    )
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--save-per-sample', action='store_true',
                        help='Save primary contact and foot counts for paired prompt-level bootstrap.')
    args = parser.parse_args()

    request = np.load(args.requests, allow_pickle=False)
    names = request['names'].astype(str)
    lengths = request['lengths'].astype(np.int64)
    target_contact = request['contact'].astype(bool)
    if args.limit is not None:
        names = names[:args.limit]
        lengths = lengths[:args.limit]
        target_contact = target_contact[:args.limit]
    device = torch.device(args.device)

    # Subject-specific absolute T-pose joints; FK owns its offset conversion.
    skeletons = torch.from_numpy(np.asarray(INTERX_SKELETONS).copy()).float()
    fk = FK_Pair_Layer_Batch(
        SMPLX_TREE, skeletons[:, 0], skeletons[:, 1], device=device
    ).to(device).eval()

    threshold_values = sorted(set(args.contact_thresholds_m + [args.contact_threshold_m]))
    threshold_totals = {
        threshold: {
            'predicted_contact_frames': 0,
            'true_positive_frames': 0,
            'false_positive_frames': 0,
            'false_negative_frames': 0,
        }
        for threshold in threshold_values
    }
    totals = {
        'valid_frames': 0,
        'target_contact_frames': 0,
        'target_noncontact_frames': 0,
        'min_distance_sum_m': 0.0,
        'contact_distance_sum_m': 0.0,
        'noncontact_distance_sum_m': 0.0,
        'grounded_foot_transitions': 0,
        'grounded_foot_speed_sum_mps': 0.0,
    }
    sample_counts = []

    with h5py.File(args.generations, 'r') as source:
        request_sha256 = sha256_file(args.requests)
        recorded_request_sha256 = source.attrs.get('requests_sha256')
        if (recorded_request_sha256 is not None and
                recorded_request_sha256 != request_sha256):
            raise ValueError(
                'generation request hash does not match the supplied frozen request'
            )
        repeats = int(source['motions'].shape[1])
        if not 0 <= args.repeat < repeats:
            raise ValueError(f'repeat {args.repeat} is outside [0, {repeats})')
        # This scorer permits either a one-repeat main file or a 30-repeat MM
        # file, while enforcing the complete frozen name/length contract.
        validate_generation_file(source, names, lengths, repeats, 'physical')

        for start in range(0, len(names), args.batch_size):
            end = min(start + args.batch_size, len(names))
            batch_motion = torch.from_numpy(
                source['motions'][start:end, args.repeat]
            ).float().to(device).reshape(end - start, 150, 56, 12)
            if not torch.isfinite(batch_motion).all():
                raise ValueError(f'non-finite motion values in batch {start}:{end}')
            batch_groups = groups_from_names(names[start:end], device)
            joints1, joints2 = fk(batch_motion, batch_groups)

            frame_index = torch.arange(150, device=device).view(1, -1)
            valid = frame_index < torch.from_numpy(lengths[start:end]).to(device).view(-1, 1)
            target = torch.from_numpy(target_contact[start:end]).to(device) & valid
            non_target = (~torch.from_numpy(target_contact[start:end]).to(device)) & valid
            if args.save_per_sample:
                per_sample = {'valid_frames': valid.sum(1),
                              'grounded_foot_transitions': torch.zeros(end-start, device=device, dtype=torch.long),
                              'grounded_foot_speed_sum_mps': torch.zeros(end-start, device=device, dtype=torch.float64)}

            minimum_distance = torch.cdist(joints1, joints2).amin(dim=(-1, -2))
            totals['valid_frames'] += int(valid.sum())
            totals['target_contact_frames'] += int(target.sum())
            totals['target_noncontact_frames'] += int(non_target.sum())
            for threshold in threshold_values:
                predicted = (minimum_distance <= threshold) & valid
                threshold_totals[threshold]['predicted_contact_frames'] += int(predicted.sum())
                threshold_totals[threshold]['true_positive_frames'] += int((predicted & target).sum())
                threshold_totals[threshold]['false_positive_frames'] += int((predicted & non_target).sum())
                threshold_totals[threshold]['false_negative_frames'] += int(((~predicted) & target).sum())
                if args.save_per_sample and threshold == args.contact_threshold_m:
                    per_sample.update(true_positive_frames=(predicted & target).sum(1),
                                      false_positive_frames=(predicted & non_target).sum(1),
                                      false_negative_frames=((~predicted) & target).sum(1))
            totals['min_distance_sum_m'] += float(minimum_distance[valid].double().sum())
            totals['contact_distance_sum_m'] += float(minimum_distance[target].double().sum())
            totals['noncontact_distance_sum_m'] += float(minimum_distance[non_target].double().sum())

            transition_valid = frame_index[:, :-1] < (
                torch.from_numpy(lengths[start:end]).to(device).view(-1, 1) - 1
            )
            for joints in (joints1, joints2):
                feet = joints[:, :, FOOT_JOINTS]
                horizontal_speed = torch.linalg.vector_norm(
                    feet[:, 1:, :, (0, 2)] - feet[:, :-1, :, (0, 2)], dim=-1
                ) * args.fps
                floor = foot_floor_height(feet, valid, args.floor_policy)
                grounded = feet[:, :-1, :, 1] <= (
                    floor + args.ground_height_threshold_m
                )
                grounded &= transition_valid.unsqueeze(-1)
                totals['grounded_foot_transitions'] += int(grounded.sum())
                totals['grounded_foot_speed_sum_mps'] += float(
                    horizontal_speed[grounded].double().sum()
                )
                if args.save_per_sample:
                    per_sample['grounded_foot_transitions'] += grounded.sum((1, 2))
                    per_sample['grounded_foot_speed_sum_mps'] += horizontal_speed.double().masked_fill(~grounded, 0).sum((1, 2))

            if args.save_per_sample:
                sample_counts.extend(dict(name=str(names[start+i]),
                                          **{key: value[i].item() for key, value in per_sample.items()})
                                     for i in range(end-start))

            print({'scored': end, 'total': len(names)}, flush=True)

    threshold_metrics = {}
    for threshold in threshold_values:
        counts = threshold_totals[threshold]
        tp = counts['true_positive_frames']
        fp = counts['false_positive_frames']
        fn = counts['false_negative_frames']
        threshold_metrics[f'{threshold:g}'] = {
            'contact_recall': ratio(tp, tp + fn),
            'contact_precision': ratio(tp, tp + fp),
            'contact_f1': ratio(2 * tp, 2 * tp + fp + fn),
            'contact_temporal_iou': ratio(tp, tp + fp + fn),
            'false_contact_rate': ratio(fp, totals['target_noncontact_frames']),
            'predicted_contact_rate': ratio(
                counts['predicted_contact_frames'], totals['valid_frames']
            ),
            'counts': counts,
        }
    primary_metrics = threshold_metrics[f'{args.contact_threshold_m:g}']
    primary_counts = threshold_totals[args.contact_threshold_m]
    all_counts = dict(totals)
    all_counts.update(primary_counts)
    if args.save_per_sample:
        # Independently accumulated sample counts must reproduce unchanged headline totals.
        for key in sample_counts[0]:
            if key != 'name' and not math.isclose(sum(row[key] for row in sample_counts),
                                                 all_counts[key], rel_tol=1e-12, abs_tol=1e-8):
                raise AssertionError(f'per-sample aggregation mismatch: {key}')
    result = {
        'protocol': ('interx_official_v1_physical_joint_v2'
                     if args.floor_policy == 'valid_frames'
                     else 'interx_official_v1_physical_joint_v1'),
        'floor_height_policy': args.floor_policy,
        'samples': int(len(names)),
        'repeat': args.repeat,
        'contact_threshold_m': args.contact_threshold_m,
        'ground_height_threshold_m': args.ground_height_threshold_m,
        'contact_distance_m': ratio(
            totals['contact_distance_sum_m'], totals['target_contact_frames']
        ),
        'noncontact_distance_m': ratio(
            totals['noncontact_distance_sum_m'], totals['target_noncontact_frames']
        ),
        'minimum_interperson_joint_distance_m': ratio(
            totals['min_distance_sum_m'], totals['valid_frames']
        ),
        'contact_recall': primary_metrics['contact_recall'],
        'contact_precision': primary_metrics['contact_precision'],
        'contact_f1': primary_metrics['contact_f1'],
        'contact_temporal_iou': primary_metrics['contact_temporal_iou'],
        'false_contact_rate': primary_metrics['false_contact_rate'],
        'predicted_contact_rate': primary_metrics['predicted_contact_rate'],
        'target_contact_rate': ratio(
            totals['target_contact_frames'], totals['valid_frames']
        ),
        'foot_sliding_mps': ratio(
            totals['grounded_foot_speed_sum_mps'], totals['grounded_foot_transitions']
        ),
        'threshold_sweep': threshold_metrics,
        'counts': all_counts,
        'requests': os.path.abspath(args.requests),
        'requests_sha256': request_sha256,
        'generation_request_hash_verified': recorded_request_sha256 is not None,
        'generations': os.path.abspath(args.generations),
        'generations_sha256': sha256_file(args.generations),
        'definition': (
            'LLM-free joint metrics using the frozen binary contact schedule and '
            'subject-specific Inter-X skeletons; penetration remains a separate mesh metric.'
        ),
    }
    if args.save_per_sample:
        result['per_sample_counts'] = sample_counts
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key != 'per_sample_counts'},
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
