#!/usr/bin/env python
"""Fail-fast training-state audit for an HHC three-part VQ checkpoint."""

import argparse
import hashlib
import json
import os

import torch


PART_DIMENSIONS = {
    'quantizer_upper': 96,
    'quantizer_lower': 96,
    'quantizer_root': 64,
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensors_finite(value):
    tensor_count = 0
    floating_count = 0
    finite = True

    def visit(item):
        nonlocal tensor_count, floating_count, finite
        if torch.is_tensor(item):
            tensor_count += 1
            if item.is_floating_point():
                floating_count += 1
                finite = finite and bool(torch.isfinite(item).all())
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return {
        'all_floating_tensors_finite': finite,
        'floating_tensor_count': floating_count,
        'tensor_count': tensor_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-iteration', type=int, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    iteration = int(checkpoint.get('iter', -1))
    if args.expected_iteration is not None and iteration != args.expected_iteration:
        raise ValueError(
            f'expected iteration {args.expected_iteration}, got {iteration}'
        )
    scheduler_iteration = int(checkpoint.get('scheduler', {}).get('last_epoch', -1))
    if scheduler_iteration != iteration:
        raise ValueError(
            f'scheduler iteration {scheduler_iteration} != checkpoint iteration {iteration}'
        )
    optimizer = checkpoint.get('optimizer')
    if not isinstance(optimizer, dict) or not optimizer.get('state'):
        raise ValueError('checkpoint has no non-empty optimizer state')
    net_audit = tensors_finite(checkpoint.get('net'))
    optimizer_audit = tensors_finite(optimizer)
    if not net_audit['all_floating_tensors_finite']:
        raise ValueError('VQ model state contains NaN or Inf')
    if not optimizer_audit['all_floating_tensors_finite']:
        raise ValueError('VQ optimizer state contains NaN or Inf')

    codebook_size = int(checkpoint.get('args', {}).get('nb_code', 8096))
    quantizer_state = checkpoint.get('quantizer_training_state')
    if not isinstance(quantizer_state, dict):
        raise ValueError('checkpoint lacks explicit EMA quantizer training state')
    quantizer_audit = {}
    for name, dimension in PART_DIMENSIONS.items():
        state = quantizer_state.get(name)
        if not isinstance(state, dict):
            raise ValueError(f'checkpoint lacks {name} training state')
        expected_sum_shape = (codebook_size, dimension)
        if tuple(state['code_sum'].shape) != expected_sum_shape:
            raise ValueError(
                f'{name} code_sum shape {tuple(state["code_sum"].shape)} '
                f'!= {expected_sum_shape}'
            )
        if tuple(state['code_count'].shape) != (codebook_size,):
            raise ValueError(f'{name} code_count shape is invalid')
        state_audit = tensors_finite(state)
        if not state_audit['all_floating_tensors_finite']:
            raise ValueError(f'{name} EMA state contains NaN or Inf')
        quantizer_audit[name] = {
            **state_audit,
            'code_count_shape': list(state['code_count'].shape),
            'code_sum_shape': list(state['code_sum'].shape),
            'initialized': bool(state['init']),
            'reset_count': int(state['reset_count']),
        }

    result = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_bytes': os.path.getsize(args.checkpoint),
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'iteration': iteration,
        'scheduler_last_epoch': scheduler_iteration,
        'optimizer_parameter_state_entries': len(optimizer['state']),
        'optimizer_lr': float(optimizer['param_groups'][0]['lr']),
        'model_state': net_audit,
        'optimizer_state': optimizer_audit,
        'quantizer_training_state': quantizer_audit,
        'status': 'pass',
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
