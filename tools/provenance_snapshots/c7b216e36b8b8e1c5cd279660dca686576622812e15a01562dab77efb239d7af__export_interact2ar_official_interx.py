#!/usr/bin/env python
"""Generate Interact2AR outputs for frozen official Inter-X evaluator requests."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


DEFAULT_REPOSITORY = '/mnt/8T-HDD/wenning/Interact2Ar'


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resume_attributes(source, expected):
    mismatches = {
        key: (source.attrs.get(key), value)
        for key, value in expected.items()
        if key not in source.attrs or source.attrs.get(key) != value
    }
    if mismatches:
        raise ValueError(f'resume provenance mismatch: {mismatches}')


def configure_reproducibility(seed):
    import lightning as pl

    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_official_checkpoint(repository, checkpoint):
    manifest_path = repository / 'assets' / 'checkpoints.json'
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    matches = [
        entry for entry in manifest['models']
        if entry['name'] == 'interact2ar_mixed_memory'
    ]
    if len(matches) != 1:
        raise ValueError('official manifest must contain one main Interact2AR model')
    record = matches[0]
    actual_size = checkpoint.stat().st_size
    actual_hash = sha256_file(checkpoint)
    if actual_size != int(record['size_bytes']) or actual_hash != record['sha256']:
        raise ValueError(
            'Interact2AR checkpoint does not match the official asset manifest: '
            f'size={actual_size}, sha256={actual_hash}'
        )
    return manifest_path, actual_hash


def load_model(repository, checkpoint, cfg_weight, device):
    sys.path.insert(0, str(repository / 'src'))
    from models.wrapper import LitInteract2ArModel

    model = LitInteract2ArModel.load_from_checkpoint(
        str(checkpoint), map_location=device
    )
    model.opt_diffusion['CFG_WEIGHT'] = cfg_weight
    model.model.opt.CFG_WEIGHT = cfg_weight
    return model.to(device).eval()


def build_batch(captions, length):
    return (
        None,
        None,
        list(captions),
        None,
        None,
        [int(length)] * len(captions),
        None,
        None,
        None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repository-root', default=DEFAULT_REPOSITORY)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--indices-key', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cfg-weight', type=float, default=2.5)
    args = parser.parse_args()

    repository = Path(args.repository_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    requests = Path(args.requests).resolve()
    output = Path(args.output).resolve()
    if args.batch_size < 1 or args.repeats < 1:
        raise ValueError('batch size and repeats must be positive')
    manifest_path, checkpoint_hash = verify_official_checkpoint(
        repository, checkpoint
    )
    revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repository, text=True
    ).strip()

    request = np.load(requests, allow_pickle=False)
    captions = request['captions'].astype(str)
    lengths = request['lengths'].astype(np.int64)
    names = request['names'].astype(str)
    if args.indices_key is not None:
        indices = request[args.indices_key].astype(np.int64)
        captions, lengths, names = captions[indices], lengths[indices], names[indices]
    if args.limit is not None:
        captions = captions[:args.limit]
        lengths = lengths[:args.limit]
        names = names[:args.limit]
    if not len(captions) == len(lengths) == len(names):
        raise ValueError('request arrays have inconsistent lengths')
    if len(captions) == 0 or np.any(lengths < 1) or np.any(lengths > 150):
        raise ValueError('requests must contain lengths in [1, 150]')

    expected_attributes = {
        'author_checkpoint_manifest': str(manifest_path),
        'author_checkpoint_manifest_sha256': sha256_file(manifest_path),
        'batch_size': args.batch_size,
        'cfg_weight': args.cfg_weight,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': checkpoint_hash,
        'first_frame_conditioning': 'none',
        'generation_length_policy': 'exact_request_length_grouped',
        'repository_revision': revision,
        'requests': str(requests),
        'requests_sha256': sha256_file(requests),
        'sampling_process': 'author_autoregressive_ancestral_10_steps',
        'sampling_steps': 10,
        'seed': args.seed,
        'seed_strategy': 'base_seed + repeat_index',
        'strategy': 'author_autoregressive_ancestral',
    }
    expected_shape = (len(captions), args.repeats, 150, 56 * 12)
    if args.resume and output.is_file():
        with h5py.File(output, 'r') as existing:
            validate_resume_attributes(existing, expected_attributes)
            complete = (
                existing['motions'].shape == expected_shape
                and np.array_equal(existing['lengths'][:], lengths)
                and np.array_equal(existing['names'][:].astype(str), names)
                and 'completed_repeats' in existing
                and existing['completed_repeats'][:].astype(bool).all()
            )
        if complete:
            print({'output': str(output), 'status': 'all repeats already complete'})
            return

    device = torch.device(f'cuda:{args.gpu}')
    configure_reproducibility(args.seed)
    # The official model resolves SMPL-X assets relative to its repository.
    # Keep all user-supplied paths absolute above, then preserve the author's
    # runtime path semantics for checkpoint construction and sampling.
    os.chdir(repository)
    model = load_model(repository, checkpoint, args.cfg_weight, device)

    output.parent.mkdir(parents=True, exist_ok=True)
    resume_existing = args.resume and output.is_file()
    target = h5py.File(output, 'r+' if resume_existing else 'w')
    try:
        if resume_existing:
            validate_resume_attributes(target, expected_attributes)
            if target['motions'].shape != expected_shape:
                raise ValueError(
                    f'resume shape mismatch: {target["motions"].shape} != {expected_shape}'
                )
            if not np.array_equal(target['lengths'][:], lengths):
                raise ValueError('resume lengths mismatch')
            if not np.array_equal(target['names'][:].astype(str), names):
                raise ValueError('resume names mismatch')
            if 'completed_repeats' not in target:
                raise ValueError('resume file predates repeat-completion tracking')
            motions = target['motions']
            completed_repeats = target['completed_repeats']
        else:
            motions = target.create_dataset(
                'motions',
                shape=expected_shape,
                dtype=np.float32,
                chunks=(1, 1, 150, 56 * 12),
                compression='lzf',
            )
            target.create_dataset('lengths', data=lengths)
            target.create_dataset('names', data=np.asarray(names, dtype='S'))
            completed_repeats = target.create_dataset(
                'completed_repeats',
                data=np.zeros(args.repeats, dtype=np.bool_),
            )
            for key, value in expected_attributes.items():
                target.attrs[key] = value

        for repeat in range(args.repeats):
            if bool(completed_repeats[repeat]):
                print({'repeat': repeat, 'status': 'already complete'}, flush=True)
                continue
            configure_reproducibility(args.seed + repeat)
            generated_count = 0
            for valid_length in sorted(np.unique(lengths).tolist()):
                length_indices = np.flatnonzero(lengths == valid_length)
                for start in range(0, len(length_indices), args.batch_size):
                    batch_indices = length_indices[start:start + args.batch_size]
                    batch_captions = captions[batch_indices].tolist()
                    with torch.inference_mode():
                        generated = model(
                            build_batch(batch_captions, valid_length)
                        )
                    if tuple(generated.shape) != (
                        len(batch_indices), int(valid_length), 56 * 12
                    ):
                        raise RuntimeError(
                            'unexpected Interact2AR generation shape: '
                            f'{tuple(generated.shape)}'
                        )
                    if not torch.isfinite(generated).all():
                        raise RuntimeError('Interact2AR generated NaN or Inf')
                    official = np.zeros(
                        (len(batch_indices), 150, 56 * 12), dtype=np.float32
                    )
                    official[:, :int(valid_length)] = (
                        generated.detach().float().cpu().numpy()
                    )
                    motions[batch_indices, repeat] = official
                    generated_count += len(batch_indices)
                    print({
                        'repeat': repeat,
                        'generated': generated_count,
                        'total': len(captions),
                        'request_length': int(valid_length),
                    }, flush=True)
            target.flush()
            completed_repeats[repeat] = True
            target.flush()
    finally:
        target.close()


if __name__ == '__main__':
    main()
