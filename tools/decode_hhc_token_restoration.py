#!/usr/bin/env python
"""Posthoc B=1 VQ token restoration decode; not sampling or official MM."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import torch

from tools.eval_vq_interx_reconstruction import model_from_checkpoint, sha256_file
from utils.motion_process import MotionNormalizerTorchInterX


def restore(candidate, control, arm):
    out = candidate.copy()
    if arm == 'restore_hands':
        out[..., 1] = control[..., 1]
    elif arm == 'restore_body':
        out[..., 0] = control[..., 0]
    elif arm != 'fresh_candidate':
        raise ValueError(f'unknown arm: {arm}')
    return out


def self_check():
    candidate = np.arange(24, dtype=np.int32).reshape(2, 4, 3)
    control = candidate + 100
    original = candidate.copy()
    original_control = control.copy()
    hands = restore(candidate, control, 'restore_hands')
    body = restore(candidate, control, 'restore_body')
    assert np.array_equal(candidate, original)
    assert np.array_equal(control, original_control)
    fresh = restore(candidate, control, 'fresh_candidate')
    assert np.array_equal(fresh, candidate) and not np.shares_memory(fresh, candidate)
    assert np.array_equal(hands[..., 1], control[..., 1])
    assert np.array_equal(hands[..., (0, 2)], candidate[..., (0, 2)])
    assert np.array_equal(body[..., 0], control[..., 0])
    assert np.array_equal(body[..., (1, 2)], candidate[..., (1, 2)])
    assert np.array_equal(hands[..., 2], body[..., 2])
    print('self-check: PASS')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=False)
    p.add_argument('--requests', required=False)
    p.add_argument('--checkpoint', required=False)
    p.add_argument('--output-dir', required=False)
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--self-check', action='store_true')
    a = p.parse_args()
    if a.self_check:
        self_check()
        return
    for name in ('source', 'requests', 'checkpoint', 'output_dir'):
        if not getattr(a, name):
            p.error(f'--{name.replace("_", "-")} is required unless --self-check')
    source, requests, outdir = map(Path, (a.source, a.requests, a.output_dir))
    if not source.is_file() or not requests.is_file():
        raise FileNotFoundError(source if not source.is_file() else requests)
    outdir.mkdir(parents=True, exist_ok=True)
    arms = ('fresh_control', 'fresh_candidate', 'restore_hands', 'restore_body')
    outputs = [outdir / f'{arm}.h5' for arm in arms]
    if any(x.exists() for x in outputs):
        raise FileExistsError('refusing to overwrite existing arm output')
    req = np.load(requests, allow_pickle=False)
    names, lengths = req['names'].astype(str), req['lengths'].astype(np.int64)
    betas = req['betas'].astype(np.float32)
    gender = req['gender'].astype(np.int64)
    if len(betas) != len(names) or len(gender) != len(names) or not np.isfinite(betas).all():
        raise ValueError('invalid requested shape/gender conditions')
    provenance = {
        'source': str(source.resolve()), 'source_sha256': sha256_file(source),
        'requests': str(requests.resolve()), 'requests_sha256': sha256_file(requests),
        'vq_checkpoint': str(Path(a.checkpoint).resolve()),
        'vq_checkpoint_sha256': sha256_file(a.checkpoint),
        'decoder_tool_sha256': sha256_file(__file__),
        'decode_device': a.device, 'torch_version': str(torch.__version__),
    }
    with h5py.File(source, 'r') as src:
        required = {'motions', 'names', 'lengths', 'completed_repeats', 'token_ids', 'root_replay_control_token_ids'}
        if not required.issubset(src):
            raise ValueError(f'source missing fields: {required - set(src)}')
        if not bool(np.asarray(src['completed_repeats'][:]).all()):
            raise ValueError('source is incomplete')
        if not bool(src.attrs.get('root_replay_control', False)):
            raise ValueError('source is not a root-replay control archive')
        if src.attrs.get('vq_beta_policy', src.attrs.get('beta_policy', 'requested')) != 'requested':
            raise ValueError('source VQ beta policy is not requested')
        if src.attrs.get('requests_sha256') != sha256_file(requests):
            raise ValueError('source/request SHA mismatch')
        if src.attrs.get('vq_checkpoint_sha256') != sha256_file(a.checkpoint):
            raise ValueError('source/checkpoint SHA mismatch')
        sn, sl = src['names'][:].astype(str), src['lengths'][:]
        candidate, control = src['token_ids'][:], src['root_replay_control_token_ids'][:]
        if candidate.ndim != 4 or src['motions'].shape != (len(sn), candidate.shape[1], 150, 672):
            raise ValueError('invalid source token/motion dimensions')
        for kind in ('mean', 'std'):
            path = Path(f'dataset/interx/Interx_{kind.capitalize()}.npy')
            digest = sha256_file(path)
            if src.attrs.get(f'normalization_{kind}_sha256') != digest:
                raise ValueError(f'source normalization {kind} mismatch')
            provenance[f'normalization_{kind}_sha256'] = digest
        if (candidate.shape[:2] != src['motions'].shape[:2] or candidate.shape[0] != len(sn)
                or src['motions'].shape[0] != len(sn) or len(src['completed_repeats']) != candidate.shape[1]):
            raise ValueError('source sample/repeat shape mismatch')
        if not np.array_equal(sn, names) or not np.array_equal(sl, lengths):
            raise ValueError('source/request names or lengths mismatch')
        if candidate.shape != control.shape or candidate.ndim != 4 or candidate.shape[-1] != 3:
            raise ValueError('token stream shape mismatch')
        if not np.issubdtype(candidate.dtype, np.integer) or not np.issubdtype(control.dtype, np.integer):
            raise ValueError('token streams must be integer')
        if not np.array_equal(candidate[..., 2], control[..., 2]):
            raise ValueError('root token streams differ')
        if not np.isfinite(src['motions'][:]).all():
            raise ValueError('source motions are non-finite')
        candidate, control = candidate.astype(np.int64), control.astype(np.int64)
    if a.limit is not None:
        if a.limit <= 0:
            raise ValueError('--limit must be positive')
        n = min(int(a.limit), len(names)); names, lengths, betas, gender = names[:n], lengths[:n], betas[:n], gender[:n]
        candidate, control = candidate[:n], control[:n]
    if np.any((lengths < 1) | (lengths > 150)):
        raise ValueError('lengths must be in 1..150')
    if np.any((lengths + 3) // 4 > candidate.shape[2]):
        raise ValueError('valid token length exceeds archive')
    device = torch.device(a.device)
    model, _ = model_from_checkpoint(torch.load(a.checkpoint, map_location='cpu', weights_only=False), device)
    if model.temporal_stride != 4:
        raise ValueError(f'expected VQ temporal stride 4, got {model.temporal_stride}')
    for ids in (candidate, control):
        for i, length in enumerate(lengths):
            t = (int(length) + 3) // 4
            if np.any(ids[i, :, :t] < 0) or np.any(ids[i, :, :t] >= model.num_code):
                raise ValueError('valid token prefix outside VQ vocabulary')
    normalizer = MotionNormalizerTorchInterX('raw', stats_dir='dataset/interx', stats_prefix='Interx')
    arm_ids = {
        'fresh_control': control,
        'fresh_candidate': candidate,
        'restore_hands': restore(candidate, control, 'restore_hands'),
        'restore_body': restore(candidate, control, 'restore_body'),
    }
    decoded = {}
    for arm in arms:
        decoded[arm] = np.zeros((len(names), candidate.shape[1], 150, 672), dtype=np.float32)
    with torch.inference_mode():
        for i, length in enumerate(lengths.tolist()):
            token_length = (int(length) + 3) // 4
            for arm, ids in arm_ids.items():
                for repeat in range(candidate.shape[1]):
                    x = torch.from_numpy(ids[i:i + 1, repeat, :token_length]).to(device=device, dtype=torch.long)
                    raw = normalizer.backward(model(x, torch.from_numpy(betas[i:i + 1]).to(device),
                                                     torch.from_numpy(gender[i:i + 1]).to(device), type='decode'))
                    decoded[arm][i, repeat, :int(length)] = raw[0, :int(length)].detach().cpu().numpy()
    for arm, path in zip(arms, outputs):
        with h5py.File(path, 'x') as dst:
            dst.create_dataset('motions', data=decoded[arm], compression='lzf')
            dst.create_dataset('lengths', data=lengths)
            dst.create_dataset('names', data=np.asarray(names, dtype='S'))
            dst.create_dataset('completed_repeats', data=np.zeros(candidate.shape[1], dtype=np.bool_))
            dst.create_dataset('token_ids', data=arm_ids[arm].astype(np.int32), compression='lzf')
            dst.attrs.update({'diagnostic_only': True, 'intervention': arm,
                              'description': 'posthoc decoder token intervention; not new sampling or official MM',
                              **provenance, 'vq_beta_policy': 'requested',
                              'decode_batch_size': 1, 'temporal_stride': 4})
            if not np.isfinite(decoded[arm]).all():
                raise ValueError(f'non-finite decoded output for {arm}')
            dst['completed_repeats'][:] = True
    print(json.dumps({'status': 'complete', 'arms': [str(x) for x in outputs], 'samples': len(names)}, indent=2))


if __name__ == '__main__':
    main()
