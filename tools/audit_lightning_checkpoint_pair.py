#!/usr/bin/env python
"""Prove that two same-step Lightning checkpoints share training state."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def recursive_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left) and torch.is_tensor(right)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
            and left.dtype == right.dtype and np.array_equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict) and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(recursive_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(recursive_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def differing_tensor_keys(left, right):
    keys = sorted(set(left) | set(right))
    differences = []
    for key in keys:
        if key not in left or key not in right:
            differences.append(key)
            continue
        if not recursive_equal(left[key], right[key]):
            differences.append(key)
    return differences


def atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def validation_checkpoint_state(checkpoint):
    matches = [
        value for value in checkpoint.get('callbacks', {}).values()
        if value.get('monitor') == 'val/FID'
    ]
    if len(matches) != 1:
        raise ValueError(f'expected one val/FID callback, got {len(matches)}')
    state = matches[0]
    scalar = lambda value: None if value is None else float(value)
    return {
        'current_fid': scalar(state.get('current_score')),
        'best_fid': scalar(state.get('best_model_score')),
        'best_model_path': str(state.get('best_model_path', '')),
        'retained_models': len(state.get('best_k_models', {})),
    }


def fixed_validation_pair_state(checkpoint):
    matches = [
        value for key, value in checkpoint.get('callbacks', {}).items()
        if 'FixedStepValidationPairCheckpoint' in str(key)
    ]
    if len(matches) != 1:
        raise ValueError(
            'expected one FixedStepValidationPairCheckpoint state, got '
            f'{len(matches)}'
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--expected-before-fid', type=float)
    parser.add_argument('--expected-after-fid', type=float)
    parser.add_argument('--expected-best-fid', type=float)
    parser.add_argument('--expected-best-step', type=int)
    parser.add_argument('--require-fixed-pair-marker', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    before_path = Path(args.before).resolve()
    after_path = Path(args.after).resolve()
    if args.require_fixed_pair_marker:
        expected_names = (
            f'prevalidation-{args.expected_step:09d}.ckpt',
            f'postvalidation-{args.expected_step:09d}.ckpt',
        )
        actual_names = (before_path.name, after_path.name)
        if actual_names != expected_names:
            raise ValueError(
                'fixed validation-pair checkpoint names are invalid: '
                f'expected={expected_names}, actual={actual_names}'
            )
    before = torch.load(before_path, map_location='cpu', weights_only=False)
    after = torch.load(after_path, map_location='cpu', weights_only=False)
    steps = [int(before.get('global_step', -1)), int(after.get('global_step', -1))]
    if steps != [args.expected_step, args.expected_step]:
        raise ValueError(
            f'expected both global steps {args.expected_step}, got {steps}'
        )

    state_differences = differing_tensor_keys(
        before.get('state_dict', {}), after.get('state_dict', {})
    )
    exact = {
        'model_state': not state_differences,
        'optimizer_state': recursive_equal(
            before.get('optimizer_states'), after.get('optimizer_states')
        ),
        'lr_scheduler_state': recursive_equal(
            before.get('lr_schedulers'), after.get('lr_schedulers')
        ),
        'hyper_parameters': recursive_equal(
            before.get('hyper_parameters'), after.get('hyper_parameters')
        ),
    }
    if not all(exact.values()):
        raise ValueError(
            'same-step validation changed training state: '
            f'exact={exact}, differing_model_keys={state_differences[:20]}'
        )

    callback_keys = {
        'before': sorted(before.get('callbacks', {})),
        'after': sorted(after.get('callbacks', {})),
    }
    validation_callback = {
        'before': validation_checkpoint_state(before),
        'after': validation_checkpoint_state(after),
    }
    fixed_pair_callback = None
    if args.require_fixed_pair_marker:
        fixed_pair_callback = {
            'before': fixed_validation_pair_state(before),
            'after': fixed_validation_pair_state(after),
        }
        fixed_pair_valid = (
            fixed_pair_callback['before'].get('step') == args.expected_step
            and fixed_pair_callback['before'].get('post_saved') is False
            and fixed_pair_callback['after'].get('step') == args.expected_step
            and fixed_pair_callback['after'].get('pre_saved') is True
            and fixed_pair_callback['after'].get('post_saved') is True
        )
        if not fixed_pair_valid:
            raise ValueError(
                'fixed validation-pair callback markers are invalid: '
                f'{fixed_pair_callback}'
            )
    callback_expectations = {
        'before_fid': (args.expected_before_fid,
                       validation_callback['before']['current_fid']),
        'after_fid': (args.expected_after_fid,
                      validation_callback['after']['current_fid']),
        'best_fid_before': (args.expected_best_fid,
                            validation_callback['before']['best_fid']),
        'best_fid_after': (args.expected_best_fid,
                           validation_callback['after']['best_fid']),
    }
    mismatches = {
        key: {'expected': expected, 'actual': actual}
        for key, (expected, actual) in callback_expectations.items()
        if expected is not None and actual != expected
    }
    if args.expected_best_step is not None:
        marker = f'model-{args.expected_best_step:09d}-'
        for stage in ('before', 'after'):
            path = validation_callback[stage]['best_model_path']
            if marker not in os.path.basename(path):
                mismatches[f'best_step_{stage}'] = {
                    'expected': args.expected_best_step, 'actual_path': path,
                }
    if mismatches:
        raise ValueError(f'validation callback metric mismatch: {mismatches}')
    result = {
        'schema': 'hhc_lightning_same_step_pair_audit_v1',
        'status': 'pass',
        'interpretation': (
            'the post-validation checkpoint is bitwise identical to the '
            'pre-validation checkpoint for model, optimizer, scheduler '
            'and hyperparameters; callback/loop metadata may record validation'
        ),
        'expected_step': args.expected_step,
        'before': {
            'path': str(before_path), 'sha256': sha256_file(before_path),
            'bytes': before_path.stat().st_size,
        },
        'after': {
            'path': str(after_path), 'sha256': sha256_file(after_path),
            'bytes': after_path.stat().st_size,
        },
        'exact': exact,
        'model_tensor_count': len(before.get('state_dict', {})),
        'differing_model_keys': state_differences,
        'loop_state_exact': recursive_equal(
            before.get('loops'), after.get('loops')
        ),
        'callback_state_exact': recursive_equal(
            before.get('callbacks'), after.get('callbacks')
        ),
        'callback_keys': callback_keys,
        'validation_callback': validation_callback,
        'fixed_pair_callback': fixed_pair_callback,
        'implementation': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
