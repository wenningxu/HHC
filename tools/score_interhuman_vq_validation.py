#!/usr/bin/env python3
"""Score a VQ reconstruction on the frozen 480-sample InterHuman val set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from dataset.interhuman_generated_h5 import InterHumanGeneratedH5Dataset, sha256_file
from models.interhuman_evaluator_wrapper import (
    InterHumanEvaluatorWrapper,
    evaluator_provenance,
    load_official_metrics_module,
)
from tools.score_fair_interhuman_main import main_metrics_from_embeddings
from utils.interhuman_vq_evaluation import reconstruction_errors


def atomic_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--validation-manifest',
        default='output/interhuman_validation_requests/manifest.json',
    )
    parser.add_argument('--generations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    manifest_path = Path(args.validation_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    entry = manifest['request']
    request_path = Path(entry['path']).resolve()
    gt_dataset = InterHumanValidationRequestDataset(manifest_path)
    generated_dataset = InterHumanGeneratedH5Dataset(
        args.generations, request_path, request_entry=entry, formal=False,
    )
    if len(gt_dataset) != 480 or len(generated_dataset) != 480:
        raise ValueError('InterHuman VQ validation requires exactly 480 requests')
    if generated_dataset.metadata.get('split') != 'val':
        raise ValueError('VQ validation generations are not marked as val split')
    if generated_dataset.metadata.get('status') != 'complete':
        raise ValueError('VQ validation generations are incomplete')

    gt_loader = DataLoader(gt_dataset, batch_size=96, shuffle=False, drop_last=True)
    generated_loader = DataLoader(
        generated_dataset, batch_size=96, shuffle=False, drop_last=True
    )
    evaluator = InterHumanEvaluatorWrapper.from_official_checkpoint(args.device)
    text_embeddings, prediction_embeddings, gt_embeddings = [], [], []
    weighted_mpjpe = 0.0
    weighted_mae = 0.0
    person_frames = 0
    mean = gt_dataset.adapter.mean
    std = gt_dataset.adapter.std
    for gt_batch, prediction_batch in zip(gt_loader, generated_loader):
        if (
            list(gt_batch[0]) != list(prediction_batch[0])
            or list(gt_batch[1]) != list(prediction_batch[1])
            or not torch.equal(gt_batch[4], prediction_batch[4])
        ):
            raise ValueError('VQ validation GT and reconstruction batches differ')
        text, prediction = evaluator.get_co_embeddings(prediction_batch)
        gt = evaluator.get_motion_embeddings(gt_batch)
        text_embeddings.append(text.cpu().numpy())
        prediction_embeddings.append(prediction.cpu().numpy())
        gt_embeddings.append(gt.cpu().numpy())
        errors = reconstruction_errors(
            gt_batch[2], gt_batch[3], prediction_batch[2], prediction_batch[3],
            gt_batch[4], mean, std,
        )
        frames = int(errors['valid_person_frames'])
        weighted_mpjpe += errors['mpjpe_m'] * frames
        weighted_mae += errors['normalized_feature_mae'] * frames
        person_frames += frames
    metrics = main_metrics_from_embeddings(
        np.concatenate(text_embeddings), np.concatenate(prediction_embeddings),
        np.concatenate(gt_embeddings), load_official_metrics_module(),
        batch_size=96, random_seed=20260130,
    )
    metrics.update({
        'mpjpe_m': weighted_mpjpe / person_frames,
        'normalized_feature_mae': weighted_mae / person_frames,
        'valid_person_frames': person_frames,
    })
    result = {
        'schema': 'interhuman_vq_official_validation_score_v1',
        'status': 'complete',
        'split': 'val',
        **metrics,
        'validation_manifest': str(manifest_path),
        'validation_manifest_sha256': sha256_file(manifest_path),
        'request_file_sha256': entry['sha256'],
        'request_content_sha256': entry['request_sha256'],
        'generations': str(Path(args.generations).resolve()),
        'generations_sha256': sha256_file(Path(args.generations).resolve()),
        'generation_metadata': generated_dataset.metadata,
        'evaluator_provenance': evaluator_provenance(include_checkpoint_sha256=True),
        'implementation_sha256': sha256_file(Path(__file__).resolve()),
    }
    generated_dataset.close()
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
