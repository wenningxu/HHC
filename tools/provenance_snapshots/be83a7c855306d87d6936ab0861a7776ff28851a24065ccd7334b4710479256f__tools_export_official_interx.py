#!/usr/bin/env python
"""Generate InterMask outputs for frozen official Inter-X evaluator requests."""

import argparse
import hashlib
import os

import h5py
import numpy as np
import torch

from models.mask_transformer.transformer import MaskTransformer
from models.vq.model import RVQVAE
from utils.get_opt import get_opt
from utils.utils import fixseed


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


def validate_optional_resume_attribute(source, key, expected, legacy_default):
    """Accept pre-contract files only when the requested value is the old default."""
    if key in source.attrs:
        if source.attrs.get(key) != expected:
            raise ValueError(
                f'resume provenance mismatch: {key}='
                f'{source.attrs.get(key)!r}, expected {expected!r}'
            )
    elif not legacy_default:
        raise ValueError(
            f'resume file predates {key!r} provenance and the requested value '
            'is not the legacy default'
        )


def load_models(device, checkpoint_root, transformer_name, vq_name):
    trans_opt = get_opt(
        os.path.join(checkpoint_root, 'interx', transformer_name, 'opt.txt'), device
    )
    vq_opt = get_opt(
        os.path.join(checkpoint_root, 'interx', vq_name, 'opt.txt'), device
    )
    trans_opt.num_tokens = vq_opt.nb_code
    trans_opt.code_dim = vq_opt.code_dim

    vq = RVQVAE(
        vq_opt, 6, vq_opt.nb_code, vq_opt.code_dim, vq_opt.code_dim,
        vq_opt.down_t, vq_opt.stride_t, vq_opt.width, vq_opt.depth,
        vq_opt.dilation_growth_rate, vq_opt.vq_act, vq_opt.vq_norm,
    )
    vq_checkpoint = torch.load(
        os.path.join(checkpoint_root, 'interx', vq_name, 'model', 'best_fid.tar'),
        map_location='cpu',
    )
    vq_state_key = 'vq_model' if 'vq_model' in vq_checkpoint else 'net'
    missing, unexpected = vq.load_state_dict(vq_checkpoint[vq_state_key], strict=False)
    if unexpected or not all(
        key.startswith('decoder.conv') or key.startswith('decoder.resnets') for key in missing
    ):
        raise RuntimeError(f'VQ checkpoint mismatch: missing={missing}, unexpected={unexpected}')

    transformer = MaskTransformer(
        code_dim=trans_opt.code_dim,
        cond_mode='text',
        latent_dim=trans_opt.latent_dim,
        ff_size=trans_opt.ff_size,
        num_layers=trans_opt.n_layers,
        num_heads=trans_opt.n_heads,
        dropout=trans_opt.dropout,
        clip_dim=768,
        cond_drop_prob=trans_opt.cond_drop_prob,
        clip_version='ViT-L/14@336px',
        opt=trans_opt,
    )
    trans_path = os.path.join(
        checkpoint_root, 'interx', transformer_name, 'model', 'best_fid.tar'
    )
    trans_checkpoint = torch.load(trans_path, map_location='cpu')
    trans_state_key = 't2m_transformer' if 't2m_transformer' in trans_checkpoint else 'trans'
    missing, unexpected = transformer.load_state_dict(
        trans_checkpoint[trans_state_key], strict=False
    )
    if unexpected or not all(key.startswith('clip_') for key in missing):
        raise RuntimeError(
            f'transformer checkpoint mismatch: missing={missing}, unexpected={unexpected}'
        )
    return vq.to(device).eval(), transformer.to(device).eval(), trans_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint-root', default='checkpoints')
    parser.add_argument('--transformer-name', default='trans_default')
    parser.add_argument('--vq-name', default='vq_default')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--resume', action='store_true', help='Resume completed repeat blocks')
    parser.add_argument('--indices-key', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--time-steps', type=int, default=20)
    parser.add_argument('--cond-scale', type=float, default=2.0)
    parser.add_argument('--topk-filter-thres', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument(
        '--length-policy', choices=('exact', 'author-floor'), default='exact',
        help=(
            'exact uses ceil(length/4) and crops to the frozen requested length; '
            'author-floor reproduces the original InterMask length//4 behavior'
        ),
    )
    args = parser.parse_args()

    fixseed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    request = np.load(args.requests, allow_pickle=False)
    captions = request['captions'].astype(str)
    lengths = request['lengths'].astype(np.int64)
    names = request['names'].astype(str)
    if args.indices_key is not None:
        indices = request[args.indices_key].astype(np.int64)
        captions, lengths, names = captions[indices], lengths[indices], names[indices]
    if args.limit is not None:
        captions, lengths, names = captions[:args.limit], lengths[:args.limit], names[:args.limit]
    expected_checkpoint = os.path.abspath(os.path.join(
        args.checkpoint_root, 'interx', args.transformer_name,
        'model', 'best_fid.tar',
    ))
    expected_vq_checkpoint = os.path.abspath(os.path.join(
        args.checkpoint_root, 'interx', args.vq_name,
        'model', 'best_fid.tar',
    ))
    expected_checkpoint_sha256 = sha256_file(expected_checkpoint)
    expected_vq_checkpoint_sha256 = sha256_file(expected_vq_checkpoint)
    expected_attributes = {
        'checkpoint': expected_checkpoint,
        'requests': os.path.abspath(args.requests),
        'seed': args.seed,
        'seed_strategy': 'base_seed + repeat_index',
        'time_steps': args.time_steps,
        'cond_scale': args.cond_scale,
        'topk_filter_thres': args.topk_filter_thres,
        'temperature': args.temperature,
        'length_policy': args.length_policy,
    }
    expected_request_sha256 = sha256_file(args.requests)
    if args.resume and os.path.isfile(args.output):
        with h5py.File(args.output, 'r') as existing:
            validate_resume_attributes(existing, expected_attributes)
            validate_optional_resume_attribute(
                existing, 'requests_sha256', expected_request_sha256,
                legacy_default=True,
            )
            validate_optional_resume_attribute(
                existing, 'checkpoint_sha256', expected_checkpoint_sha256,
                legacy_default=True,
            )
            validate_optional_resume_attribute(
                existing, 'vq_checkpoint_sha256', expected_vq_checkpoint_sha256,
                legacy_default=True,
            )
            validate_optional_resume_attribute(
                existing, 'vq_checkpoint', expected_vq_checkpoint,
                legacy_default=(args.vq_name == 'vq_default'),
            )
            validate_optional_resume_attribute(
                existing, 'batch_size', args.batch_size,
                legacy_default=(args.batch_size == 32),
            )
            expected_shape = (len(captions), args.repeats, 150, 56 * 12)
            has_native_lengths = (
                'generated_lengths' in existing
                and np.array_equal(existing['generated_lengths'][:], (lengths // 4) * 4)
            )
            complete = (
                existing['motions'].shape == expected_shape
                and np.array_equal(existing['lengths'][:], lengths)
                and np.array_equal(existing['names'][:].astype(str), names)
                and 'completed_repeats' in existing
                and existing['completed_repeats'][:].astype(bool).all()
                and (args.length_policy == 'exact' or has_native_lengths)
            )
        if complete:
            print({'output': args.output, 'status': 'all repeats already complete'}, flush=True)
            return
    vq, transformer, checkpoint_path = load_models(
        device, args.checkpoint_root, args.transformer_name, args.vq_name
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    resume_existing = args.resume and os.path.isfile(args.output)
    target = h5py.File(args.output, 'r+' if resume_existing else 'w')
    try:
        if resume_existing:
            validate_resume_attributes(target, expected_attributes)
            validate_optional_resume_attribute(
                target, 'requests_sha256', expected_request_sha256,
                legacy_default=True,
            )
            validate_optional_resume_attribute(
                target, 'checkpoint_sha256', expected_checkpoint_sha256,
                legacy_default=True,
            )
            validate_optional_resume_attribute(
                target, 'vq_checkpoint_sha256', expected_vq_checkpoint_sha256,
                legacy_default=True,
            )
            validate_optional_resume_attribute(
                target, 'vq_checkpoint', expected_vq_checkpoint,
                legacy_default=(args.vq_name == 'vq_default'),
            )
            validate_optional_resume_attribute(
                target, 'batch_size', args.batch_size,
                legacy_default=(args.batch_size == 32),
            )
            expected_shape = (len(captions), args.repeats, 150, 56 * 12)
            if target['motions'].shape != expected_shape:
                raise ValueError(f'resume shape mismatch: {target["motions"].shape} != {expected_shape}')
            if not np.array_equal(target['lengths'][:], lengths):
                raise ValueError('resume lengths mismatch')
            if not np.array_equal(target['names'][:].astype(str), names):
                raise ValueError('resume names mismatch')
            if 'completed_repeats' not in target:
                raise ValueError('resume file predates repeat-completion tracking')
            motions = target['motions']
            completed_repeats = target['completed_repeats']
            if 'generated_lengths' in target:
                generated_lengths = target['generated_lengths']
            else:
                generated_lengths = None
        else:
            motions = target.create_dataset(
                'motions', shape=(len(captions), args.repeats, 150, 56 * 12),
                dtype=np.float32, chunks=(1, 1, 150, 56 * 12), compression='lzf',
            )
            target.create_dataset('lengths', data=lengths)
            target.create_dataset('names', data=np.asarray(names, dtype='S'))
            if args.length_policy == 'exact':
                effective_lengths = lengths.copy()
            else:
                effective_lengths = (lengths // 4) * 4
            generated_lengths = target.create_dataset(
                'generated_lengths', data=effective_lengths.astype(np.int64)
            )
            completed_repeats = target.create_dataset(
                'completed_repeats', data=np.zeros(args.repeats, dtype=np.bool_)
            )
            if os.path.abspath(checkpoint_path) != expected_checkpoint:
                raise RuntimeError('resolved transformer checkpoint changed after preflight')
            for key, value in expected_attributes.items():
                target.attrs[key] = value
            target.attrs['requests_sha256'] = expected_request_sha256
            target.attrs['vq_checkpoint'] = expected_vq_checkpoint
            target.attrs['checkpoint_sha256'] = expected_checkpoint_sha256
            target.attrs['vq_checkpoint_sha256'] = expected_vq_checkpoint_sha256
            target.attrs['batch_size'] = args.batch_size
        for repeat in range(args.repeats):
            if bool(completed_repeats[repeat]):
                print({'repeat': repeat, 'status': 'already complete'}, flush=True)
                continue
            fixseed(args.seed + repeat)
            for start in range(0, len(captions), args.batch_size):
                end = min(start + args.batch_size, len(captions))
                batch_lengths = torch.from_numpy(lengths[start:end]).long().to(device)
                if args.length_policy == 'exact':
                    token_lengths = torch.ceil(batch_lengths.float() / 4).long()
                else:
                    token_lengths = batch_lengths // 4
                with torch.inference_mode():
                    motion_ids = transformer.generate(
                        captions[start:end].tolist(), token_lengths,
                        args.time_steps, args.cond_scale,
                        temperature=args.temperature,
                        topk_filter_thres=args.topk_filter_thres,
                    )
                    halfway = motion_ids.shape[1] // 2
                    person1 = vq.forward_decoder(motion_ids[:, :halfway].unsqueeze(-1))
                    person2 = vq.forward_decoder(motion_ids[:, halfway:].unsqueeze(-1))
                    generated = torch.cat([person1, person2], dim=-1)
                    generated = generated.reshape(generated.shape[0], generated.shape[1], 56 * 12)
                    official = torch.zeros(
                        generated.shape[0], 150, 56 * 12,
                        device=generated.device, dtype=generated.dtype,
                    )
                    copy_length = min(150, generated.shape[1])
                    official[:, :copy_length] = generated[:, :copy_length]
                    for local_index, valid_length in enumerate(batch_lengths.tolist()):
                        official[local_index, int(valid_length):] = 0.0
                if not torch.isfinite(official).all():
                    raise RuntimeError(f'non-finite generation in batch {start}:{end}')
                motions[start:end, repeat] = official.cpu().numpy()
                print({'repeat': repeat, 'generated': end, 'total': len(captions)}, flush=True)
            if generated_lengths is None and args.length_policy == 'author-floor':
                raise ValueError(
                    'author-floor resume file predates generated-length tracking; '
                    'use a fresh output path'
                )
            target.flush()
            completed_repeats[repeat] = True
            target.flush()
    finally:
        target.close()


if __name__ == '__main__':
    main()
