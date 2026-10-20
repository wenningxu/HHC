#!/usr/bin/env python3
"""Score a frozen InterHuman main generation with released InterCLIP metrics."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.interhuman_fair_requests import InterHumanFrozenRequestDataset
from dataset.interhuman_generated_h5 import InterHumanGeneratedH5Dataset, sha256_file
from models.interhuman_evaluator_wrapper import (
    InterHumanEvaluatorWrapper,
    evaluator_provenance,
    load_official_metrics_module,
)


@contextmanager
def fixed_numpy_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def main_metrics_from_embeddings(
    text_embeddings,
    generated_embeddings,
    gt_embeddings,
    metrics,
    *,
    batch_size=96,
    random_seed=1234,
):
    text_embeddings = np.asarray(text_embeddings)
    generated_embeddings = np.asarray(generated_embeddings)
    gt_embeddings = np.asarray(gt_embeddings)
    if not (
        text_embeddings.shape == generated_embeddings.shape == gt_embeddings.shape
        and text_embeddings.ndim == 2
        and text_embeddings.shape[0] % batch_size == 0
        and text_embeddings.shape[0] > 300
    ):
        raise ValueError(
            f'invalid InterHuman evaluator embeddings: {text_embeddings.shape}, '
            f'{generated_embeddings.shape}, {gt_embeddings.shape}'
        )
    top_k_count = np.zeros(3, dtype=np.int64)
    matching_sum = 0.0
    for start in range(0, len(text_embeddings), batch_size):
        end = start + batch_size
        distance = metrics.euclidean_distance_matrix(
            text_embeddings[start:end], generated_embeddings[start:end]
        )
        matching_sum += float(np.trace(distance))
        ordering = np.argsort(distance, axis=1)
        top_k_count += metrics.calculate_top_k(ordering, top_k=3).sum(axis=0)
    generated_mu, generated_cov = metrics.calculate_activation_statistics(
        generated_embeddings, 6
    )
    gt_mu, gt_cov = metrics.calculate_activation_statistics(gt_embeddings, 6)
    fid = metrics.calculate_frechet_distance(gt_mu, gt_cov, generated_mu, generated_cov)
    with fixed_numpy_seed(random_seed):
        diversity = metrics.calculate_diversity(generated_embeddings, 300, 6, 2)
        diversity_gt = metrics.calculate_diversity(gt_embeddings, 300, 6, 2)
    count = len(text_embeddings)
    return {
        'samples': count,
        'fid': float(fid),
        'matching': matching_sum / count,
        'top1': float(top_k_count[0] / count),
        'top2': float(top_k_count[1] / count),
        'top3': float(top_k_count[2] / count),
        'diversity': float(diversity),
        'diversity_gt': float(diversity_gt),
    }


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='output/interhuman_fair_requests_v3/manifest.json')
    parser.add_argument('--replication', type=int, required=True)
    parser.add_argument('--main-generations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--require-hhc-common50', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size != 96:
        raise ValueError('released InterHuman main protocol requires batch_size=96')
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    entries = {int(row['replication']): row for row in manifest['replications']}
    if args.replication not in entries:
        raise ValueError(f'replication {args.replication} absent from manifest')
    entry = entries[args.replication]
    request_path = Path(entry['path']).resolve()
    gt_dataset = InterHumanFrozenRequestDataset(
        args.replication, manifest_path=manifest_path
    )
    generated_dataset = InterHumanGeneratedH5Dataset(
        args.main_generations, request_path, request_entry=entry, formal=True
    )
    if args.require_hhc_common50:
        required = {
            'sampler': 'sample_intra_cas', 'sampling_steps': 50,
            'temperature': 0.5, 'cfg_scale': 3.5,
        }
        for key, expected in required.items():
            if generated_dataset.metadata.get(key) != expected:
                raise ValueError(
                    f'HHC InterHuman generation {key} mismatch: '
                    f'{generated_dataset.metadata.get(key)!r} != {expected!r}'
                )

    gt_loader = DataLoader(gt_dataset, batch_size=96, shuffle=False, drop_last=True)
    generated_loader = DataLoader(
        generated_dataset, batch_size=96, shuffle=False, drop_last=True
    )
    wrapper = InterHumanEvaluatorWrapper.from_official_checkpoint(args.device)
    text_embeddings = []
    generated_embeddings = []
    gt_embeddings = []
    for gt_batch, generated_batch in zip(gt_loader, generated_loader):
        if (
            list(gt_batch[0]) != list(generated_batch[0])
            or list(gt_batch[1]) != list(generated_batch[1])
            or not torch.equal(gt_batch[4], generated_batch[4])
        ):
            raise ValueError('GT and generated evaluator batches are not request-identical')
        text, generated = wrapper.get_co_embeddings(generated_batch)
        gt = wrapper.get_motion_embeddings(gt_batch)
        text_embeddings.append(text.cpu().numpy())
        generated_embeddings.append(generated.cpu().numpy())
        gt_embeddings.append(gt.cpu().numpy())
    metrics_module = load_official_metrics_module()
    result = main_metrics_from_embeddings(
        np.concatenate(text_embeddings),
        np.concatenate(generated_embeddings),
        np.concatenate(gt_embeddings),
        metrics_module,
        batch_size=96,
        random_seed=1234 + args.replication,
    )
    result.update({
        'schema': 'interhuman_fair_main_score_v1',
        'status': 'complete',
        'replication': args.replication,
        'request_manifest': str(manifest_path),
        'request_manifest_sha256': sha256_file(manifest_path),
        'request_file': str(request_path),
        'request_file_sha256': entry['sha256'],
        'request_content_sha256': entry['request_sha256'],
        'main_generations': str(Path(args.main_generations).resolve()),
        'main_generations_sha256': sha256_file(Path(args.main_generations).resolve()),
        'generation_metadata': generated_dataset.metadata,
        'evaluator_provenance': evaluator_provenance(include_checkpoint_sha256=True),
        'metric_contract': {
            'embedding_scale': 6,
            'distance_divisor': 2,
            'diversity_pairs': 300,
            'retrieval_scope': 'within each released batch of 96',
            'drop_last': True,
        },
    })
    atomic_json(Path(args.output).resolve(), result)
    generated_dataset.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

