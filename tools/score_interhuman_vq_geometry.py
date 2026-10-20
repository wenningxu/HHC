#!/usr/bin/env python3
"""Supplementary coordinate errors, not InterCLIP or joint-embedding FID."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from dataset.interhuman_generated_h5 import InterHumanGeneratedH5Dataset, sha256_file
from tools.score_interhuman_vq_validation import atomic_json
from utils.interhuman_artifacts import load_hhc_interhuman_vq
from utils.interhuman_vq_evaluation import reconstruct_padded_interhuman_pair
from utils.interhuman_vq_loss import FOOT_JOINTS, FOOT_HEIGHT_THRESHOLDS


def error_sums(target, prediction, lengths, mean, std):
    """Normalized [B,T,262] to additive float64 sums; padding excluded."""
    if target.shape != prediction.shape or target.ndim != 3 or target.shape[-1] != 262:
        raise ValueError('expected matching [B,T,262] tensors')
    lengths = torch.as_tensor(lengths, device=target.device)
    if (lengths.shape != (target.shape[0],) or lengths.is_floating_point()
            or torch.any(lengths <= 0) or torch.any(lengths > target.shape[1])):
        raise ValueError('invalid integer sequence lengths')
    mean = torch.as_tensor(mean, device=target.device, dtype=torch.float64)
    std = torch.as_tensor(std, device=target.device, dtype=torch.float64)
    if (mean.shape != (262,) or std.shape != (262,) or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or torch.any(std <= 0)):
        raise ValueError('invalid normalization statistics')
    mask = torch.arange(target.shape[1], device=target.device)[None] < lengths[:, None]
    target, prediction = target[mask].double(), prediction[mask].double()
    if not torch.isfinite(target).all() or not torch.isfinite(prediction).all():
        raise ValueError('nonfinite valid-frame features')
    # The common mean cancels; std restores official metres without alignment.
    delta = (prediction - target) * std
    xyz = delta[:, :66].reshape(-1, 22, 3)
    distance = torch.linalg.vector_norm(xyz, dim=-1)
    return {
        'person_frames': int(mask.sum()),
        'position_squared_sum': float(xyz.square().sum()),
        'full_distance_sum': float(distance.sum()),
        'body_distance_sum': float(distance[:, 1:].sum()),
        'root_distance_sum': float(distance[:, 0].sum()),
        'normalized_feature_squared_sum': float((prediction[:, :258] - target[:, :258]).square().sum()),
    }


def finalize(sums):
    frames = sums['person_frames']
    if frames <= 0:
        raise ValueError('no valid person frames')
    return {
        'valid_person_frames': frames,
        'joint_coordinate_rmse_m': math.sqrt(sums['position_squared_sum'] / (frames * 22 * 3)),
        'normalized_feature_rmse_0_258': math.sqrt(sums['normalized_feature_squared_sum'] / (frames * 258)),
        'full_mpjpe_m': sums['full_distance_sum'] / (frames * 22),
        'body_nonroot_mpjpe_m': sums['body_distance_sum'] / (frames * 21),
        'root_position_error_m': sums['root_distance_sum'] / frames,
    }


def physical_sums(p1, p2, q1, q2, lengths, mean, std, contact_threshold):
    """Geometry-only contact proxy; GT-grounded horizontal foot displacement."""
    if not math.isfinite(contact_threshold) or contact_threshold <= 0:
        raise ValueError('contact threshold must be positive and finite')
    mean, std = torch.as_tensor(mean), torch.as_tensor(std)
    joints = [(x.double() * std + mean)[..., :66].reshape(*x.shape[:2], 22, 3)
              for x in (p1, p2, q1, q2)]
    gt1, gt2, pred1, pred2 = joints
    valid = torch.arange(p1.shape[1])[None] < lengths[:, None]
    contact = torch.cdist(gt1[valid], gt2[valid]).amin((-1, -2)) <= contact_threshold
    predicted = torch.cdist(pred1[valid], pred2[valid]).amin((-1, -2)) <= contact_threshold
    out = {'contact_tp': int((contact & predicted).sum()),
           'contact_fp': int((~contact & predicted).sum()),
           'contact_fn': int((contact & ~predicted).sum()),
           'pair_frames': int(valid.sum()), 'gt_contact_foot_transitions': 0,
           'gt_foot_displacement_sum_m': 0., 'pred_foot_displacement_sum_m': 0.}
    for gt, pred in ((gt1, pred1), (gt2, pred2)):
        feet, predicted_feet = gt[:, :, FOOT_JOINTS], pred[:, :, FOOT_JOINTS]
        delta = feet[:, 1:] - feet[:, :-1]
        predicted_delta = predicted_feet[:, 1:] - predicted_feet[:, :-1]
        # Official InterHuman contact criteria, evaluated on GT only. A model
        # cannot reduce the denominator by lifting its predicted feet.
        grounded = ((delta.square().sum(-1) < .001)
                    & (feet[:, :-1, :, 1] < torch.tensor(FOOT_HEIGHT_THRESHOLDS))
                    & valid[:, 1:, None])
        out['gt_contact_foot_transitions'] += int(grounded.sum())
        out['gt_foot_displacement_sum_m'] += float(torch.linalg.vector_norm(delta[..., (0, 2)], dim=-1)[grounded].sum())
        out['pred_foot_displacement_sum_m'] += float(torch.linalg.vector_norm(predicted_delta[..., (0, 2)], dim=-1)[grounded].sum())
    return out


def physical_metrics(sums):
    n = sums['gt_contact_foot_transitions']
    denominator = 2 * sums['contact_tp'] + sums['contact_fp'] + sums['contact_fn']
    return {
        'geometry_contact_f1': 2 * sums['contact_tp'] / denominator if denominator else None,
        'gt_contact_foot_sliding_mm_per_frame': 1000 * sums['pred_foot_displacement_sum_m'] / n if n else None,
        'gt_reference_foot_sliding_mm_per_frame': 1000 * sums['gt_foot_displacement_sum_m'] / n if n else None,
        'counts_and_sums': sums,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-manifest', default='output/interhuman_validation_requests/manifest.json')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--generations')
    source.add_argument('--checkpoint')
    parser.add_argument('--training-manifest')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--contact-threshold-m', type=float, default=.1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.training_manifest):
        parser.error('--training-manifest is required only with --checkpoint')
    manifest_path = Path(args.validation_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    dataset = InterHumanValidationRequestDataset(manifest_path)
    if len(dataset) != 480:
        raise ValueError('expected frozen 480-request validation set')
    loader = DataLoader(dataset, batch_size=96, shuffle=False)
    generated = None
    if args.generations:
        entry = manifest['request']
        generated = InterHumanGeneratedH5Dataset(args.generations, Path(entry['path']).resolve(), request_entry=entry, formal=False)
        if len(generated) != 480 or generated.metadata.get('split') != 'val' or generated.metadata.get('status') != 'complete':
            raise ValueError('expected complete 480-request validation reconstructions')
        predictions = iter(DataLoader(generated, batch_size=96, shuffle=False))
        provenance = {'generations': str(Path(args.generations).resolve()), 'sha256': sha256_file(Path(args.generations)), 'metadata': generated.metadata}
    else:
        model, provenance = load_hhc_interhuman_vq(args.checkpoint, args.training_manifest, device=args.device)
        model.eval()
    totals, physics = {}, {}
    try:
        for names, texts, p1, p2, lengths in loader:
            if generated is not None:
                pn, pt, q1, q2, pl = next(predictions)
                if list(names) != list(pn) or list(texts) != list(pt) or not torch.equal(lengths, pl):
                    raise ValueError('GT/reconstruction request mismatch')
            else:
                q1, q2 = reconstruct_padded_interhuman_pair(model, p1.to(args.device), p2.to(args.device), lengths)
                q1, q2 = q1.cpu(), q2.cpu()
            for target, prediction in ((p1, q1), (p2, q2)):
                sums = error_sums(target, prediction, lengths, dataset.adapter.mean, dataset.adapter.std)
                for key, value in sums.items():
                    totals[key] = totals.get(key, 0) + value
            sums = physical_sums(p1, p2, q1, q2, lengths, dataset.adapter.mean,
                                 dataset.adapter.std, args.contact_threshold_m)
            for key, value in sums.items():
                physics[key] = physics.get(key, 0) + value
    finally:
        if generated is not None:
            generated.close()
    result = {
        'schema': 'interhuman_vq_geometry_diagnostics_v2', 'status': 'complete',
        'split': 'val', 'samples': 480, 'metrics': finalize(totals),
        'physical_diagnostics': physical_metrics(physics),
        'physical_protocol': {
            'contact_threshold_m': args.contact_threshold_m,
            'contact': 'frame-level min distance among 22x22 cross-person joints, GT and prediction thresholded identically; geometry pseudo-label, NOT semantic/mesh contact',
            'foot_sliding': 'horizontal xz displacement on GT-contact foot transitions; GT speed squared < .001 m2/frame2 and y < [.12,.05,.12,.05] m for joints [7,10,8,11]; no predicted floor realignment; null if denominator zero',
        },
        'protocol': 'valid-person-frame pooled; official coordinates, no rigid/root alignment; full joints 0:22, body 1:22, root 0; coordinate RMSE over xyz; feature RMSE over normalized 0:258',
        'not_measured': ['Full/Body joint FID', 'semantic/mesh contact F1'],
        'not_applicable': ['Hands MPJPE', 'Hands joint FID (no finger joints)'],
        'source': provenance, 'validation_manifest_sha256': sha256_file(manifest_path),
        'request_file_sha256': manifest['request']['sha256'],
        'implementation_sha256': sha256_file(Path(__file__)),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
