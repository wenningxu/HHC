#!/usr/bin/env python
"""Compare two VQ checkpoints and localize training-state differences."""

import argparse
import hashlib
import json
import os

import torch


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def recursive_equal(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return (
            left.keys() == right.keys()
            and all(recursive_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            recursive_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def tensor_difference(left, right):
    record = {
        'shape_left': list(left.shape),
        'shape_right': list(right.shape),
        'dtype_left': str(left.dtype),
        'dtype_right': str(right.dtype),
    }
    if left.shape == right.shape and left.dtype == right.dtype:
        record['unequal_values'] = int(torch.count_nonzero(left != right))
        if left.is_floating_point():
            record['max_abs_difference'] = float(
                (left.double() - right.double()).abs().max()
            )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    left = torch.load(args.left, map_location='cpu')
    right = torch.load(args.right, map_location='cpu')
    left_net, right_net = left['net'], right['net']
    all_keys = sorted(set(left_net) | set(right_net))
    net_differences = {}
    for key in all_keys:
        if key not in left_net or key not in right_net:
            net_differences[key] = {'missing_from': 'left' if key not in left_net else 'right'}
        elif not torch.equal(left_net[key], right_net[key]):
            net_differences[key] = tensor_difference(left_net[key], right_net[key])

    quantizer_differences = {
        key: value for key, value in net_differences.items()
        if 'quantizer_' in key
    }
    non_quantizer_differences = {
        key: value for key, value in net_differences.items()
        if 'quantizer_' not in key
    }
    result = {
        'left': os.path.abspath(args.left),
        'right': os.path.abspath(args.right),
        'left_sha256': sha256_file(args.left),
        'right_sha256': sha256_file(args.right),
        'iteration_left': int(left.get('iter', -1)),
        'iteration_right': int(right.get('iter', -1)),
        'net_tensor_count': len(all_keys),
        'differing_net_tensors': len(net_differences),
        'differing_quantizer_tensors': quantizer_differences,
        'differing_non_quantizer_tensors': non_quantizer_differences,
        'optimizer_exact': recursive_equal(left.get('optimizer'), right.get('optimizer')),
        'scheduler_exact': recursive_equal(left.get('scheduler'), right.get('scheduler')),
        'explicit_quantizer_state_exact': recursive_equal(
            left.get('quantizer_training_state'), right.get('quantizer_training_state')
        ),
    }
    result['difference_localized_to_quantizer'] = bool(
        quantizer_differences
        and not non_quantizer_differences
        and result['optimizer_exact']
        and result['scheduler_exact']
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
