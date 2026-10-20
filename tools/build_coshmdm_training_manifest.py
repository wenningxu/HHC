#!/usr/bin/env python
"""Freeze the corrected CoShMDM Inter-X retraining launch contract."""

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path


FILES = (
    'configs/datasets.yaml',
    'configs/model.yaml',
    'configs/train.yaml',
    'data/interx_global_mean.npy',
    'data/interx_global_std.npy',
    'data/interx_processed/test.txt',
    'data/interx_processed/train.txt',
    'data/interx_processed/val.txt',
    'models/losses.py',
    'models/nets.py',
    'tools/train.py',
    'utils/utils.py',
)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repository, *arguments, binary=False):
    result = subprocess.run(
        ['git', '-C', str(repository), *arguments], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout if binary else result.stdout.decode().strip()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--repository', type=Path,
        default=Path('/mnt/8T-HDD/wenning/coshmdm-code'),
    )
    parser.add_argument(
        '--experiment',
        default='CoShMDM-InterX-fair-statsfix-seed123-bs32x2',
    )
    parser.add_argument(
        '--stats-manifest', type=Path,
        default=Path('/mnt/8T-HDD/wenning/coshmdm-code/output/'
                     'interx_stats_recompute_audit/'
                     'interx_global_stats_manifest.json'),
    )
    args = parser.parse_args()
    repository = args.repository.resolve()
    stats_manifest = args.stats_manifest.resolve()
    output = repository / 'checkpoints' / args.experiment / 'meta' / (
        'training_manifest.json'
    )
    hashes = {
        path: sha256_file(repository / path)
        for path in FILES
    }
    hashes['normalization_recompute_manifest'] = sha256_file(stats_manifest)
    diff = git_output(repository, 'diff', '--binary', binary=True)
    payload = {
        'schema': 'coshmdm_interx_fair_training_manifest_v2',
        'artifact_role': 'CoShMDM Inter-X fair retraining provenance',
        'command': [
            'python', '-u', 'tools/train.py', '--dataset', 'interx',
            '--gpu', '0', '--seed', '123', '--batch-size', '32',
            '--accumulate-grad-batches', '2', '--no-progress-bar',
            '--stats-prefix', 'interx_global', '--exp-name', args.experiment,
        ],
        'cwd': str(repository),
        'launch_time': dt.datetime.now().astimezone().isoformat(),
        'runtime_contract': {
            'checkpoint_embeds_runtime_contract': True,
            'interx_requires_explicit_stats_prefix': True,
            'stats_prefix': 'interx_global',
            'normalization_mean_sha256': hashes['data/interx_global_mean.npy'],
            'normalization_std_sha256': hashes['data/interx_global_std.npy'],
        },
        'environment': {'conda_environment': 'New_MMM'},
        'repository': {
            'revision': git_output(repository, 'rev-parse', 'HEAD'),
            'tracked_diff_sha256': hashlib.sha256(diff).hexdigest(),
        },
        'training': {
            'accumulation_steps': 2,
            'effective_batch_size': 64,
            'epochs': 2000,
            'expected_optimizer_updates': 568000,
            'micro_batches_per_epoch': 569,
            'optimizer_updates_per_epoch': 284,
            'physical_batch_equivalent_entries_per_epoch': 18176,
            'dropped_entries_per_epoch': 44,
            'incomplete_accumulation_policy': 'drop at every epoch boundary',
            'physical_batch_emulation': (
                'exact sample exposure, optimizer grouping, epoch schedule '
                'and update count; two-worker stochastic augmentation '
                'assignment is distributionally but not bitwise equivalent'
            ),
            'learning_rate': 0.0001,
            'max_steps_safety_cap': 1000000,
            'micro_batch_size': 32,
            'optimizer': 'AdamW',
            'precision': 'fp32',
            'float32_matmul_precision': (
                'medium (TF32-eligible on RTX 4090)'
            ),
            'resume_semantics': (
                'Lightning optimization state and the embedded normalization '
                'contract must both match; worker augmentation streams are '
                'not bitwise checkpointed'
            ),
            'seed': 123,
            'weight_decay': 0.00002,
        },
        'epoch_boundary_verification': {
            'completed_optimizer_updates_in_epoch_0': 284,
            'dropped_incomplete_accumulation_micro_batches_at_epoch_0': 1,
            'tensorboard_scalar': (
                'dropped_incomplete_accumulation_micro_batches'
            ),
            'result': 'pass',
        },
        'model': {
            'cfg_weight': 3.5,
            'diffusion_steps': 1000,
            'input_dim_per_person': 262,
            'latent_dim': 1024,
            'num_heads': 8,
            'num_layers': 8,
            'sampling_strategy': 'ddim50',
        },
        'data': {
            'dataset': 'Inter-X',
            'augmented_training_entries': 18220,
            'split_counts': {'test': 1708, 'train': 9110, 'val': 570},
            'split_overlap': 0,
            'statistics_scope': (
                'official_interx_train_with_mirror_and_person_order'
            ),
            'statistics_frames': 12938776,
            'statistics_recompute_exact': True,
        },
        'supersedes': {
            'path': str(repository / 'checkpoints' /
                        'CoShMDM-InterX-fair-seed123-bs32x2'),
            'reason': (
                'the accepted 300k restart omitted COSHMDM_STATS_PREFIX and '
                'therefore continued with the released InterHuman '
                'global_mean/std; no exact 300k checkpoint remained after '
                'callback rotation, so the mixed-stat run is excluded and '
                'the corrected arm restarts from initialization'
            ),
        },
        'sha256': hashes,
    }
    atomic_json(output, payload)
    print(json.dumps({'status': 'pass', 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
