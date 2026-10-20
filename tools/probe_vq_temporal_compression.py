#!/usr/bin/env python3
"""Fixed-decoder token thinning/neighbor diagnostics, not generation metrics."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_vq_interx_reconstruction import model_from_checkpoint, sha256_file
from utils.motion_process import MotionNormalizerTorchInterX


def thin_tokens(tokens, mode):
    if tokens.ndim < 1 or tokens.shape[0] == 0 or mode not in ('even', 'odd'):
        raise ValueError('nonempty tokens and even/odd mode required')
    indices = torch.arange(0, len(tokens), 2, device=tokens.device)
    if mode == 'odd':
        indices = (indices + 1).clamp(max=len(tokens) - 1)
    return tokens[indices]


def nearest_token_ids(tokens, codebooks):
    if tokens.ndim != 2 or tokens.shape[1] != len(codebooks):
        raise ValueError('tokens must be [time, parts] matching codebooks')
    output = tokens.clone()
    for channel, book in enumerate(codebooks):
        if book.ndim != 2 or len(book) < 2 or not torch.isfinite(book).all():
            raise ValueError('finite codebook with at least two codes required')
        unique, inverse = tokens[:, channel].unique(return_inverse=True)
        if torch.any(unique < 0) or torch.any(unique >= len(book)):
            raise ValueError('special/out-of-range token is not a VQ code')
        # Double precision avoids cancellation when distinct codes are close.
        distances = torch.cdist(book[unique].double(), book.double())
        distances.scatter_(1, unique[:, None], float('inf'))
        output[:, channel] = distances.argmin(dim=1)[inverse]
    return output


def part_neighbor_alternatives(tokens, nearest):
    if tokens.ndim != 2 or tokens.shape[1] != 3 or nearest.shape != tokens.shape:
        raise ValueError('matching [time,3] tokens/nearest required')
    alternatives = {}
    for channel, part in enumerate(('body', 'hands', 'root')):
        changed = tokens.clone()
        changed[:, channel] = nearest[:, channel]
        alternatives[f'nearest_{part}'] = changed
    return alternatives


def part_thinning(tokens, phase):
    if phase not in ('even', 'odd'):
        raise ValueError('part thinning phase must be even or odd')
    thinned = thin_tokens(tokens, phase).repeat_interleave(2, dim=0)[:len(tokens)]
    alternatives = part_neighbor_alternatives(tokens, thinned)
    return {f'{phase}_{part}': alternatives[f'nearest_{part}']
            for part in ('body', 'hands', 'root')}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='checkpoints/vq_precleanup_raw12288/net_175000.pth')
    p.add_argument('--requests', default='output/interx_official_val_requests/validation.npz')
    p.add_argument('--ground-truth', default='output/interx_validation_raw_ground_truth.h5')
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--seed', type=int, default=90123)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--intervention', choices=('temporal_thinning', 'code_neighbors', 'part_neighbors', 'part_thinning'),
                   default='temporal_thinning')
    a = p.parse_args()
    torch.set_num_threads(4)
    req = np.load(a.requests, allow_pickle=False)
    names, lengths = req['names'].astype(str), req['lengths'].astype(np.int64)
    if not 0 < a.samples <= len(names):
        p.error('--samples must fit the nonempty request')
    indices = np.sort(np.random.default_rng(a.seed).choice(len(names), a.samples, replace=False))
    request_hash = sha256_file(a.requests)
    model, _ = model_from_checkpoint(
        torch.load(a.checkpoint, map_location='cpu', weights_only=False), torch.device(a.device)
    )
    norm = MotionNormalizerTorchInterX('raw', stats_dir='dataset/interx', stats_prefix='Interx')
    masks = {'full': torch.ones(672, dtype=torch.bool, device=a.device),
             'body': model.upper_mask, 'hands': model.lower_mask, 'root': model.root_mask}
    modes = {
        'temporal_thinning': ('baseline', 'even', 'odd'),
        'code_neighbors': ('baseline', 'nearest_all', 'nearest_10pct', 'random_10pct'),
        'part_neighbors': ('baseline', 'nearest_body', 'nearest_hands', 'nearest_root'),
        'part_thinning': ('baseline', 'even_body', 'odd_body', 'even_hands', 'odd_hands', 'even_root', 'odd_root'),
    }[a.intervention]
    sums = {mode: {part: {'squared_error_to_gt': 0., 'squared_delta_to_baseline': 0., 'values': 0}
                   for part in masks} for mode in modes}
    changed = {mode: 0 for mode in modes[1:]}
    rng = torch.Generator(device=a.device).manual_seed(a.seed)
    books = [q.codebook for q in (model.quantizer_upper, model.quantizer_lower, model.quantizer_root)]
    token_values = retained = original = 0
    retained_by_mode = {mode: 0 for mode in modes}
    with h5py.File(a.ground_truth, 'r') as gt:
        if (not np.array_equal(gt['names'][:].astype(str), names)
                or not np.array_equal(gt['lengths'][:], lengths)
                or gt.attrs.get('requests_sha256') != request_hash
                or gt['motions'].shape != (len(names), 1, 150, 672)):
            raise ValueError('ground truth is not bound to frozen requests')
        for index in indices:
            length = int(lengths[index])
            raw = torch.from_numpy(gt['motions'][index, 0, :length]).to(a.device)
            normalized = norm.forward(raw.unsqueeze(0))
            factor = model.temporal_stride
            padded = torch.zeros(1, ((length + factor - 1) // factor) * factor, 672, device=a.device)
            padded[:, :length] = normalized
            beta = torch.from_numpy(req['betas'][index:index + 1]).float().to(a.device)
            gender = torch.from_numpy(req['gender'][index:index + 1]).long().to(a.device)
            ids = model(padded, beta, gender, type='encode')
            baseline = model(ids, beta, gender, type='decode')[:, :length]
            original += ids.shape[1]
            retained += ((ids.shape[1] + 1) // 2 if a.intervention == 'temporal_thinning'
                         else ids.shape[1])
            token_values += ids.numel()
            alternatives = {}
            if a.intervention == 'part_neighbors':
                alternatives = part_neighbor_alternatives(ids[0], nearest_token_ids(ids[0], books))
            if a.intervention == 'part_thinning':
                for phase in ('even', 'odd'):
                    alternatives.update(part_thinning(ids[0], phase))
            if a.intervention == 'code_neighbors':
                nearest = nearest_token_ids(ids[0], books)
                replace = torch.rand(ids[0].shape, generator=rng, device=a.device) < 0.1
                # Matched positions and guaranteed different random codes.
                random = torch.stack([
                    (ids[0, :, c] + torch.randint(1, len(book), ids[0, :, c].shape,
                                                  generator=rng, device=a.device)) % len(book)
                    for c, book in enumerate(books)], dim=-1)
                alternatives = {'nearest_all': nearest,
                                'nearest_10pct': torch.where(replace, nearest, ids[0]),
                                'random_10pct': torch.where(replace, random, ids[0])}
            for mode in sums:
                out = baseline
                if mode != 'baseline':
                    expanded = (alternatives[mode] if alternatives else
                                thin_tokens(ids[0], mode).repeat_interleave(2, dim=0)[:ids.shape[1]])
                    if a.intervention == 'part_thinning':
                        retained_by_mode[mode] += 2 * ids.shape[1] + (ids.shape[1] + 1) // 2
                    changed[mode] += int((expanded != ids[0]).sum())
                    out = model(expanded.unsqueeze(0), beta, gender, type='decode')[:, :length]
                if not torch.isfinite(out).all():
                    raise FloatingPointError('non-finite reconstruction')
                for part, mask in masks.items():
                    error = (out - normalized)[..., mask].double()
                    delta = (out - baseline)[..., mask].double()
                    sums[mode][part]['squared_error_to_gt'] += float(error.square().sum())
                    sums[mode][part]['squared_delta_to_baseline'] += float(delta.square().sum())
                    sums[mode][part]['values'] += error.numel()
    metrics = {mode: {part: {
        'normalized_recon_mse_to_gt': s['squared_error_to_gt'] / s['values'],
        'normalized_delta_mse_to_baseline': s['squared_delta_to_baseline'] / s['values'],
        'feature_values': s['values'],
    } for part, s in parts.items()} for mode, parts in sums.items()}
    result = {
        'schema': {
            'temporal_thinning': 'hhc_vq_temporal_compression_probe_v1',
            'code_neighbors': 'hhc_vq_code_neighbor_probe_v1',
            'part_neighbors': 'hhc_vq_part_neighbor_probe_v1',
            'part_thinning': 'hhc_vq_part_thinning_probe_v1',
        }[a.intervention], 'status': 'complete',
        'diagnostic_only': True,
        'intervention': a.intervention,
        'temporal_stride': model.temporal_stride,
        'warning': 'fixed_decoder_token_intervention; NOT trained tokenizer or generation quality/MM success',
        'seed': a.seed, 'names': names[indices].tolist(), 'samples': a.samples,
        'retained_token_ratio': ({mode: retained_by_mode[mode] / (3 * original)
                                  for mode in modes[1:]}
                                 if a.intervention == 'part_thinning' else retained / original),
        'fraction_tokenids_changed': {mode: n / token_values for mode, n in changed.items()},
        'metrics': metrics, 'requests_sha256': request_hash,
        'ground_truth_sha256': sha256_file(a.ground_truth), 'vq_sha256': sha256_file(a.checkpoint),
        'normalization_sha256': [sha256_file(norm.mean_path), sha256_file(norm.std_path)],
    }
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
