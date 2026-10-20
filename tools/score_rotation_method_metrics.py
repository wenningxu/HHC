#!/usr/bin/env python3
"""Score cached InterHuman rotation-method embeddings on frozen val requests."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.interhuman_fair_requests import (
    DEFAULT_MANIFEST, DEFAULT_VALIDATION_MANIFEST,
    InterHumanFrozenRequestDataset, InterHumanValidationRequestDataset,
    sha256_file,
)
from models.interhuman_evaluator_wrapper import (
    evaluator_provenance, load_official_interclip, load_official_metrics_module,
)
from tools.score_fair_interhuman_main import main_metrics_from_embeddings

DEFAULT_CACHE = ROOT / 'output/rotation_methods_20261007/text_embeddings.npz'


def text_cache(path, dataset, request_sha, device):
    expected_shape = (len(dataset), 512)
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            if str(cache['request_sha256'].item()) != request_sha:
                raise ValueError(f'text embedding cache request hash mismatch: {path}')
            values = cache['text_embeddings']
        if values.shape != expected_shape:
            raise ValueError(f'invalid text embedding cache shape {values.shape}')
        return values

    loader = DataLoader(dataset, batch_size=96, shuffle=False, drop_last=True)
    model, _ = load_official_interclip(device)
    outputs = []
    with torch.inference_mode():
        for names, texts, *_ in loader:
            outputs.append(model.encode_text({'text': list(texts)})['text_emb'].cpu().numpy())
    values = np.concatenate(outputs)
    if values.shape != expected_shape:
        raise ValueError(f'official text encoder returned {values.shape}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, text_embeddings=values,
                            request_sha256=np.asarray(request_sha))
    os.replace(temporary, path)
    return values


def output_path(source):
    name = source.name
    if name.endswith('.embeddings.npz'):
        name = name[:-len('.embeddings.npz')]
    elif name.endswith('_embeddings.npz'):
        name = name[:-len('_embeddings.npz')]
    else:
        name = source.stem
    return source.parent / f'{name}_metrics.json'


def prior_fid(source, key):
    name = source.name
    if name.endswith('.embeddings.npz'):
        base = name[:-len('.embeddings.npz')]
    elif name.endswith('_embeddings.npz'):
        base = name[:-len('_embeddings.npz')]
    else:
        base = source.stem
    path = source.with_name(base + '.json')
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if key == 'raw':
        return value.get('raw_fid_recomputed')
    if key.startswith('alpha_'):
        alpha = float(key.removeprefix('alpha_'))
        return next((fid for name, fid in value.get('alpha', {}).items()
                     if float(name) == alpha), None)
    if key == 'release_wrist_identity':
        return value.get('release_wrist_identity', {}).get('gen_fid')
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', nargs='+', required=True, type=Path)
    parser.add_argument('--replication', type=int)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--validation-manifest', type=Path,
                        default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument('--text-cache', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--random-seed', type=int, default=20261006)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.replication is None:
        manifest_path = args.validation_manifest
        dataset = InterHumanValidationRequestDataset(manifest_path)
        split = 'validation'
        cache_path = args.text_cache or DEFAULT_CACHE
    else:
        manifest_path = args.manifest.resolve()
        dataset = InterHumanFrozenRequestDataset(args.replication, manifest_path)
        split = 'test'
        default_test_cache = ROOT / f'output/rotation_methods_20261007/test_replication_{args.replication:02d}_text_embeddings.npz'
        if args.text_cache is not None and args.text_cache.resolve() == DEFAULT_CACHE.resolve():
            raise ValueError('test scoring requires a test-specific --text-cache, not the validation cache')
        cache_path = args.text_cache or default_test_cache
    request_sha = dataset.payload['request_sha256']
    text = text_cache(cache_path, dataset, request_sha, args.device)
    metrics = load_official_metrics_module()
    provenance = evaluator_provenance()
    results = []
    for source in args.embeddings:
        source = source.resolve()
        with np.load(source, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        gt = values.get('GT', values.get('gt'))
        if gt is None or gt.shape != (len(dataset), 512):
            raise ValueError(f'{source}: expected GT/gt embedding with shape ({len(dataset)},512)')
        candidate_keys = [key for key in values if key not in {'GT', 'gt'}]
        if not candidate_keys:
            raise ValueError(f'{source}: no generated embedding arms')
        scored = {}
        for key in candidate_keys:
            generated = values[key]
            if generated.shape != (len(dataset), 512):
                raise ValueError(f'{source}:{key}: invalid shape {generated.shape}')
            metric = main_metrics_from_embeddings(
                text, generated, gt, metrics, batch_size=96,
                random_seed=args.random_seed,
            )
            expected_fid = prior_fid(source, key)
            fid_check = ({'reference': expected_fid,
                          'absolute_error': abs(metric['fid'] - expected_fid),
                          'within_1e-4': abs(metric['fid'] - expected_fid) < 1e-4}
                         if expected_fid is not None else None)
            if fid_check is not None and not fid_check['within_1e-4']:
                raise AssertionError(f'{source}:{key} FID differs from projection result: {fid_check}')
            scored[key] = {
                'FID': metric['fid'], 'Matching': metric['matching'],
                'R1': metric['top1'], 'R2': metric['top2'], 'R3': metric['top3'],
                'Div': metric['diversity'], 'GTDiv': metric['diversity_gt'],
                'MM': '待评（当前没有重复生成 cache）',
                'fid_reference_check': fid_check,
            }
        result = {
            'status': 'complete', 'split': split, 'samples': len(dataset),
            'replication': args.replication,
            'source_embeddings': str(source), 'source_embeddings_sha256': sha256_file(source),
            'request_sha256': request_sha, 'manifest': str(manifest_path.resolve()),
            'text_embedding_cache': str(cache_path.resolve()),
            'batch_size': 96, 'random_seed': args.random_seed,
            'metric_contract': 'score_fair_interhuman_main.main_metrics_from_embeddings; official metrics; retrieval within each batch of 96; diversity uses 300 pairs and scale 6/divisor 2.',
            'evaluator_provenance': provenance,
            'arms': scored,
        }
        destination = output_path(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f'.tmp.{os.getpid()}')
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        os.replace(temporary, destination)
        print(json.dumps({'output': str(destination), 'arms': scored}, sort_keys=True), flush=True)
        results.append(str(destination))
    print(json.dumps({'status': 'complete', 'outputs': results,
                      'pid': os.getpid(), 'text_cache': str(cache_path.resolve())}), flush=True)


if __name__ == '__main__':
    main()
