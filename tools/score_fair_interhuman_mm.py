#!/usr/bin/env python3
"""Score frozen InterHuman multimodality generations with released InterCLIP."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from dataset.interhuman_generated_h5 import (
    InterHumanMMGeneratedH5Dataset,
    sha256_file,
)
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


def multimodality_from_embeddings(embeddings, metrics, *, pairs=10, random_seed=1234):
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 3 or embeddings.shape[:2] != (100, 30):
        raise ValueError(f'expected InterHuman MM embeddings [100,30,D], got {embeddings.shape}')
    if not np.isfinite(embeddings).all():
        raise FloatingPointError('InterHuman MM embeddings contain NaN or Inf')
    with fixed_numpy_seed(random_seed):
        value = metrics.calculate_multimodality(embeddings, pairs, 6, 2)
    return float(value)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='output/interhuman_fair_mm_requests/manifest.json')
    parser.add_argument('--replication', type=int, required=True)
    parser.add_argument('--mm-generations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--require-hhc-common50', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    entries = {int(row['replication']): row for row in manifest['replications']}
    if args.replication not in entries:
        raise ValueError(f'replication {args.replication} absent from MM manifest')
    entry = entries[args.replication]
    request_path = Path(entry['path']).resolve()
    generated_dataset = InterHumanMMGeneratedH5Dataset(
        args.mm_generations, request_path, request_entry=entry, formal=True
    )
    if args.require_hhc_common50:
        required = {
            'sampler': 'sample_intra_cas',
            'sampling_steps': 50,
            'temperature': 0.5,
            'cfg_scale': 3.5,
        }
        for key, expected in required.items():
            if generated_dataset.metadata.get(key) != expected:
                raise ValueError(
                    f'HHC InterHuman MM generation {key} mismatch: '
                    f'{generated_dataset.metadata.get(key)!r} != {expected!r}'
                )

    wrapper = InterHumanEvaluatorWrapper.from_official_checkpoint(args.device)
    prompt_embeddings = []
    for prompt in range(len(generated_dataset)):
        batch = generated_dataset[prompt]
        prompt_embeddings.append(wrapper.get_motion_embeddings(batch).cpu().numpy())
    embeddings = np.stack(prompt_embeddings, axis=0)
    metrics = load_official_metrics_module()
    value = multimodality_from_embeddings(
        embeddings, metrics, pairs=10, random_seed=1234 + args.replication
    )
    result = {
        'schema': 'interhuman_fair_mm_score_v1',
        'status': 'complete',
        'replication': args.replication,
        'multimodality': value,
        'request_manifest': str(manifest_path),
        'request_manifest_sha256': sha256_file(manifest_path),
        'request_file': str(request_path),
        'request_file_sha256': entry['sha256'],
        'request_content_sha256': entry['request_sha256'],
        'mm_generations': str(Path(args.mm_generations).resolve()),
        'mm_generations_sha256': sha256_file(Path(args.mm_generations).resolve()),
        'generation_metadata': generated_dataset.metadata,
        'evaluator_provenance': evaluator_provenance(include_checkpoint_sha256=True),
        'metric_contract': {
            'prompts': 100,
            'repeats': 30,
            'pairs': 10,
            'embedding_scale': 6,
            'distance_divisor': 2,
            'pair_sampling_seed': 1234 + args.replication,
        },
    }
    atomic_json(Path(args.output).resolve(), result)
    generated_dataset.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
