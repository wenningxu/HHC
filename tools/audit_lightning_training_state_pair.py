#!/usr/bin/env python
"""Prove that two same-step Lightning files carry identical training state."""

import argparse
import json
from pathlib import Path

import torch

from audit_lightning_checkpoint_pair import (
    atomic_json,
    differing_tensor_keys,
    recursive_equal,
    sha256_file,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    paths = [Path(args.left).resolve(), Path(args.right).resolve()]
    checkpoints = [
        torch.load(path, map_location='cpu', weights_only=False)
        for path in paths
    ]
    steps = [int(checkpoint.get('global_step', -1)) for checkpoint in checkpoints]
    if steps != [args.expected_step, args.expected_step]:
        raise ValueError(
            f'expected both global steps {args.expected_step}, got {steps}'
        )

    left, right = checkpoints
    model_differences = differing_tensor_keys(
        left.get('state_dict', {}), right.get('state_dict', {})
    )
    exact = {
        'model_state': not model_differences,
        'optimizer_state': recursive_equal(
            left.get('optimizer_states'), right.get('optimizer_states')
        ),
        'lr_scheduler_state': recursive_equal(
            left.get('lr_schedulers'), right.get('lr_schedulers')
        ),
        'loop_state': recursive_equal(left.get('loops'), right.get('loops')),
        'hyper_parameters': recursive_equal(
            left.get('hyper_parameters'), right.get('hyper_parameters')
        ),
        'runtime_contract': recursive_equal(
            left.get('coshmdm_runtime_contract'),
            right.get('coshmdm_runtime_contract'),
        ),
    }
    if not all(exact.values()):
        raise ValueError(
            'same-step Lightning training state differs: '
            f'exact={exact}, differing_model_keys={model_differences[:20]}'
        )

    result = {
        'schema': 'hhc_lightning_training_state_pair_audit_v1',
        'status': 'pass',
        'expected_step': args.expected_step,
        'left': {
            'path': str(paths[0]),
            'bytes': paths[0].stat().st_size,
            'sha256': sha256_file(paths[0]),
        },
        'right': {
            'path': str(paths[1]),
            'bytes': paths[1].stat().st_size,
            'sha256': sha256_file(paths[1]),
        },
        'exact': exact,
        'model_tensor_count': len(left.get('state_dict', {})),
        'differing_model_keys': model_differences,
        'callback_state_exact': recursive_equal(
            left.get('callbacks'), right.get('callbacks')
        ),
        'runtime_contract': left.get('coshmdm_runtime_contract'),
        'implementation': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
