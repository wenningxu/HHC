#!/usr/bin/env python
"""Fail-fast structural and numerical audit for a Lightning checkpoint."""

import argparse
import hashlib
import json
import os

import torch


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_audit(value):
    tensors = []

    def visit(item):
        if torch.is_tensor(item):
            tensors.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    floating = [tensor for tensor in tensors if tensor.is_floating_point()]
    return {
        'all_floating_tensors_finite': all(
            bool(torch.isfinite(tensor).all()) for tensor in floating
        ),
        'floating_tensor_count': len(floating),
        'tensor_count': len(tensors),
        'tensor_values': int(sum(tensor.numel() for tensor in tensors)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-step', type=int, default=None)
    parser.add_argument('--expected-lr', type=float, default=None)
    parser.add_argument('--expected-weight-decay', type=float, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    global_step = int(checkpoint.get('global_step', -1))
    if args.expected_step is not None and global_step != args.expected_step:
        raise ValueError(f'expected global step {args.expected_step}, got {global_step}')
    state_dict = checkpoint.get('state_dict')
    optimizer_states = checkpoint.get('optimizer_states')
    lr_schedulers = checkpoint.get('lr_schedulers')
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError('checkpoint has no non-empty state_dict')
    if not isinstance(optimizer_states, list) or not optimizer_states:
        raise ValueError('checkpoint has no optimizer states')
    if not isinstance(lr_schedulers, list) or not lr_schedulers:
        raise ValueError('checkpoint has no LR scheduler states')

    model_audit = tensor_audit(state_dict)
    optimizer_audit = tensor_audit(optimizer_states)
    if not model_audit['all_floating_tensors_finite']:
        raise ValueError('model state contains NaN or Inf')
    if not optimizer_audit['all_floating_tensors_finite']:
        raise ValueError('optimizer state contains NaN or Inf')

    optimizer_parameter_groups = []
    for optimizer_index, optimizer in enumerate(optimizer_states):
        for group_index, group in enumerate(optimizer.get('param_groups', [])):
            summary = {
                'optimizer_index': optimizer_index,
                'group_index': group_index,
                'parameter_count': len(group.get('params', [])),
            }
            for key in ('lr', 'initial_lr', 'weight_decay', 'eps', 'amsgrad'):
                if key in group:
                    summary[key] = group[key]
            if 'betas' in group:
                summary['betas'] = list(group['betas'])
            optimizer_parameter_groups.append(summary)

    if args.expected_lr is not None:
        unexpected = [
            group for group in optimizer_parameter_groups
            if abs(float(group.get('lr', float('nan'))) - args.expected_lr) > 1e-12
        ]
        if unexpected:
            raise ValueError(
                f'expected optimizer LR {args.expected_lr}, got groups {unexpected}'
            )
    if args.expected_weight_decay is not None:
        unexpected = [
            group for group in optimizer_parameter_groups
            if abs(
                float(group.get('weight_decay', float('nan')))
                - args.expected_weight_decay
            ) > 1e-12
        ]
        if unexpected:
            raise ValueError(
                'expected optimizer weight decay '
                f'{args.expected_weight_decay}, got groups {unexpected}'
            )

    result = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_bytes': os.path.getsize(args.checkpoint),
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'epoch': int(checkpoint.get('epoch', -1)),
        'global_step': global_step,
        'model_state': model_audit,
        'optimizer_state': optimizer_audit,
        'optimizer_count': len(optimizer_states),
        'optimizer_parameter_state_entries': int(sum(
            len(optimizer.get('state', {})) for optimizer in optimizer_states
        )),
        'optimizer_parameter_groups': optimizer_parameter_groups,
        'lr_scheduler_count': len(lr_schedulers),
        'lr_scheduler_progress': [
            {
                key: scheduler[key] for key in ('last_epoch', '_step_count')
                if key in scheduler
            }
            for scheduler in lr_schedulers
        ],
        'callback_keys': sorted(checkpoint.get('callbacks', {})),
        'has_loop_state': bool(checkpoint.get('loops')),
        'has_rng_state': any('rng' in str(key).lower() for key in checkpoint),
        'coshmdm_runtime_contract': checkpoint.get(
            'coshmdm_runtime_contract'
        ),
        'status': 'pass',
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
