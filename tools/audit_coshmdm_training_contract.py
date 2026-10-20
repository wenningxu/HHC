#!/usr/bin/env python
"""Independently audit the frozen CoShMDM Inter-X training contract."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np


DEFAULT_REPOSITORY = Path('/mnt/8T-HDD/wenning/coshmdm-code')
DEFAULT_MANIFEST = DEFAULT_REPOSITORY / (
    'checkpoints/CoShMDM-InterX-fair-statsfix-seed123-bs32x2/'
    'meta/training_manifest.json'
)
DEFAULT_STATS_MANIFEST = DEFAULT_REPOSITORY / (
    'output/interx_stats_recompute_audit/interx_global_stats_manifest.json'
)
EXPECTED_UPSTREAM_PRETRAIN_DOWNLOAD_SHA256 = (
    '1fa25981b86a93f3957a7a91854be0ea9a28d89066ee54626729a80d7a38eaac'
)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git_output(repository, *arguments, binary=False):
    result = subprocess.run(
        ['git', '-C', str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout if binary else result.stdout.decode().strip()


def nonempty_lines(path):
    return sum(
        bool(line.strip())
        for line in Path(path).read_text(encoding='utf-8').splitlines()
    )


def audit_released_checkpoint_compatibility(repository, revision):
    download_script = git_output(
        repository, 'show',
        f'{revision}:prepare/download_pretrain_model.sh', binary=True,
    )
    download_sha256 = hashlib.sha256(download_script).hexdigest()
    require(
        download_sha256 == EXPECTED_UPSTREAM_PRETRAIN_DOWNLOAD_SHA256,
        'public CoShMDM pretrained-model download script drifted',
    )
    download_text = download_script.decode('utf-8')
    require('# InterHuman' in download_text,
            'public CoShMDM checkpoint is no longer labelled InterHuman')
    require(download_text.count('gdown ') == 1,
            'public CoShMDM checkpoint-link inventory changed')
    tracked_paths = git_output(
        repository, 'ls-tree', '-r', '--name-only', revision
    ).splitlines()
    checkpoint_paths = [
        path for path in tracked_paths
        if Path(path).suffix.lower() in {'.ckpt', '.pth', '.pt', '.tar'}
    ]
    require(not checkpoint_paths,
            'public CoShMDM Git tree unexpectedly embeds checkpoints')
    interx_paths = [
        path for path in tracked_paths
        if 'interx' in path.lower() or 'inter-x' in path.lower()
    ]
    return {
        'public_revision': revision,
        'upstream_pretrain_download_sha256': download_sha256,
        'upstream_checkpoint_download_label': 'InterHuman',
        'documented_pretrained_model_links': 1,
        'tracked_checkpoint_paths': checkpoint_paths,
        'tracked_interx_source_paths': interx_paths,
        'documented_interx_checkpoint_available': False,
        'fairness_decision': (
            'the only documented author checkpoint is InterHuman; train the '
            'author architecture on the frozen official Inter-X train split '
            'rather than relabel the released checkpoint'
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--stats-manifest', type=Path,
                        default=DEFAULT_STATS_MANIFEST)
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/coshmdm_training_contract_audit.json'),
    )
    args = parser.parse_args()

    repository = args.repository.resolve()
    manifest_path = args.manifest.resolve()
    stats_manifest_path = args.stats_manifest.resolve()
    require(repository.is_dir(), f'missing CoShMDM repository: {repository}')
    require(manifest_path.is_file(), f'missing training manifest: {manifest_path}')
    require(stats_manifest_path.is_file(),
            f'missing statistics manifest: {stats_manifest_path}')

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    stats = json.loads(stats_manifest_path.read_text(encoding='utf-8'))
    expected_command = [
        'python', '-u', 'tools/train.py', '--dataset', 'interx', '--gpu', '0',
        '--seed', '123', '--batch-size', '32', '--accumulate-grad-batches', '2',
        '--no-progress-bar', '--stats-prefix', 'interx_global', '--exp-name',
        'CoShMDM-InterX-fair-statsfix-seed123-bs32x2',
    ]
    require(manifest.get('artifact_role')
            == 'CoShMDM Inter-X fair retraining provenance',
            'unexpected artifact role')
    require(manifest.get('command') == expected_command,
            'formal launch command changed')
    require(Path(manifest.get('cwd', '')).resolve() == repository,
            'formal launch cwd differs from repository')
    require(manifest.get('environment', {}).get('conda_environment')
            == 'New_MMM', 'conda environment changed')
    runtime_contract = manifest.get('runtime_contract', {})
    require(
        runtime_contract.get('checkpoint_embeds_runtime_contract') is True
        and runtime_contract.get('interx_requires_explicit_stats_prefix') is True
        and runtime_contract.get('stats_prefix') == 'interx_global',
        'explicit Inter-X normalization runtime contract changed',
    )

    training = manifest.get('training', {})
    expected_training = {
        'accumulation_steps': 2,
        'effective_batch_size': 64,
        'epochs': 2000,
        'expected_optimizer_updates': 568000,
        'micro_batches_per_epoch': 569,
        'optimizer_updates_per_epoch': 284,
        'physical_batch_equivalent_entries_per_epoch': 18176,
        'dropped_entries_per_epoch': 44,
        'incomplete_accumulation_policy': 'drop at every epoch boundary',
        'micro_batch_size': 32,
        'optimizer': 'AdamW',
        'precision': 'fp32',
        'seed': 123,
    }
    for key, expected in expected_training.items():
        require(training.get(key) == expected,
                f'training contract changed for {key}: {training.get(key)!r}')
    require(training['optimizer_updates_per_epoch'] * training['epochs']
            == training['expected_optimizer_updates'],
            'epoch/update arithmetic is inconsistent')
    require(training['micro_batch_size'] * training['accumulation_steps']
            == training['effective_batch_size'],
            'effective batch arithmetic is inconsistent')
    require(math.isclose(training.get('learning_rate', math.nan), 1e-4),
            'learning rate changed')
    require(math.isclose(training.get('weight_decay', math.nan), 2e-5),
            'weight decay changed')

    boundary = manifest.get('epoch_boundary_verification', {})
    require(boundary.get('result') == 'pass'
            and boundary.get('completed_optimizer_updates_in_epoch_0') == 284
            and boundary.get(
                'dropped_incomplete_accumulation_micro_batches_at_epoch_0') == 1,
            'epoch-boundary accumulation proof is missing or changed')
    model = manifest.get('model', {})
    require(model == {
        'cfg_weight': 3.5,
        'diffusion_steps': 1000,
        'input_dim_per_person': 262,
        'latent_dim': 1024,
        'num_heads': 8,
        'num_layers': 8,
        'sampling_strategy': 'ddim50',
    }, 'model/sampling contract changed')

    data = manifest.get('data', {})
    require(data.get('dataset') == 'Inter-X'
            and data.get('augmented_training_entries') == 18220
            and data.get('split_counts')
            == {'test': 1708, 'train': 9110, 'val': 570}
            and data.get('split_overlap') == 0
            and data.get('statistics_scope')
            == 'official_interx_train_with_mirror_and_person_order'
            and data.get('statistics_frames') == 12938776
            and data.get('statistics_recompute_exact') is True,
            'Inter-X data/statistics contract changed')

    special_paths = {
        'normalization_recompute_manifest': stats_manifest_path,
    }
    rehashed = {}
    for label, expected_hash in manifest.get('sha256', {}).items():
        path = special_paths.get(label, repository / label).resolve()
        require(path.is_file(), f'missing frozen input/source: {label}: {path}')
        current_hash = sha256_file(path)
        require(current_hash == expected_hash,
                f'frozen input/source changed: {label}')
        rehashed[label] = {
            'path': str(path),
            'sha256': current_hash,
            'bytes': path.stat().st_size,
        }
    require(len(rehashed) == 13, 'expected exactly 13 frozen file records')

    split_counts = {
        split: nonempty_lines(repository / f'data/interx_processed/{split}.txt')
        for split in ('train', 'val', 'test')
    }
    require(split_counts == data['split_counts'], 'live split counts changed')
    split_sets = {
        split: {
            line.strip() for line in
            (repository / f'data/interx_processed/{split}.txt').read_text(
                encoding='utf-8'
            ).splitlines() if line.strip()
        }
        for split in ('train', 'val', 'test')
    }
    overlap = {
        'train_val': len(split_sets['train'] & split_sets['val']),
        'train_test': len(split_sets['train'] & split_sets['test']),
        'val_test': len(split_sets['val'] & split_sets['test']),
    }
    require(not any(overlap.values()), f'live split overlap detected: {overlap}')

    require(stats.get('configurations_per_sample') == 4
            and stats.get('feature_dim') == 262
            and stats.get('frames') == 12938776
            and stats.get('processed_samples') == 9110
            and stats.get('requested_samples') == 9110
            and stats.get('statistics_scope')
            == 'official_interx_train_with_mirror_and_person_order'
            and stats.get('train_split_sha256')
            == manifest['sha256']['data/interx_processed/train.txt'],
            'statistics recomputation manifest changed')
    mean_path = repository / 'data/interx_global_mean.npy'
    std_path = repository / 'data/interx_global_std.npy'
    require(stats.get('mean_sha256') == sha256_file(mean_path)
            and stats.get('std_sha256') == sha256_file(std_path),
            'statistics manifest does not bind the training mean/std')
    require(
        runtime_contract.get('normalization_mean_sha256')
        == sha256_file(mean_path)
        and runtime_contract.get('normalization_std_sha256')
        == sha256_file(std_path),
        'training runtime contract does not bind the Inter-X mean/std',
    )
    train_source = (repository / 'tools/train.py').read_text(encoding='utf-8')
    for fragment in (
        "parser.error('--stats-prefix is required when --dataset interx')",
        "parser.error('fair Inter-X training requires --stats-prefix interx_global')",
        "checkpoint['coshmdm_runtime_contract'] = self.runtime_contract",
        "if saved_contract != self.runtime_contract:",
    ):
        require(fragment in train_source,
                f'training entrypoint lost runtime guard: {fragment}')
    mean = np.load(mean_path)
    std = np.load(std_path)
    require(mean.shape == (262,) and std.shape == (262,),
            'CoShMDM mean/std must both have shape (262,)')
    require(np.isfinite(mean).all() and np.isfinite(std).all()
            and bool((std > 0).all()),
            'CoShMDM mean/std contain non-finite or non-positive values')
    require(math.isclose(float(std.min()), stats.get('min_std', math.nan),
                         rel_tol=0.0, abs_tol=2e-12),
            'minimum standard deviation differs from recomputation manifest')

    repository_record = manifest.get('repository', {})
    live_revision = git_output(repository, 'rev-parse', 'HEAD')
    live_diff_hash = hashlib.sha256(
        git_output(repository, 'diff', '--binary', binary=True)
    ).hexdigest()
    require(live_revision == repository_record.get('revision'),
            'live repository revision differs from training launch')
    require(live_diff_hash == repository_record.get('tracked_diff_sha256'),
            'live tracked diff differs from training launch')
    released_checkpoint_compatibility = audit_released_checkpoint_compatibility(
        repository, live_revision
    )

    result = {
        'schema': 'hhc_coshmdm_interx_training_contract_v1',
        'status': 'pass',
        'training_manifest': {
            'path': str(manifest_path),
            'sha256': sha256_file(manifest_path),
        },
        'repository': {
            'path': str(repository),
            'revision': live_revision,
            'tracked_diff_sha256': live_diff_hash,
            'matches_training_launch': True,
        },
        'released_checkpoint_compatibility': released_checkpoint_compatibility,
        'training': training,
        'runtime_contract': runtime_contract,
        'model': model,
        'dataset': {
            **data,
            'independent_split_counts': split_counts,
            'independent_split_overlap': overlap,
        },
        'statistics': {
            'manifest': {
                'path': str(stats_manifest_path),
                'sha256': sha256_file(stats_manifest_path),
            },
            'feature_dim': int(mean.shape[0]),
            'mean_finite': bool(np.isfinite(mean).all()),
            'std_finite': bool(np.isfinite(std).all()),
            'std_strictly_positive': bool((std > 0).all()),
            'minimum_std': float(std.min()),
        },
        'frozen_files_independently_rehashed': True,
        'frozen_files': rehashed,
        'epoch_boundary_verification': boundary,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
