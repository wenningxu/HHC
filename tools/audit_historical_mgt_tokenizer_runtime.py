#!/usr/bin/env python
"""Bind the original MMM token cache to its actual February 2026 runtime.

This audit deliberately distinguishes the representation used to train/evaluate
the VQ from the tokenizer path that produced the frozen MGT targets.  The latter
is the contract that a train-only shape overlay must reproduce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MMM = Path('/home/wenning/projects/MMM')
DEFAULT_COMMIT = '75814e57e0702789176da6713056a15ed4a192e2'
DEFAULT_RUN = (
    DEFAULT_MMM / 'wandb/run-20260201_191158-vq_mask_gen_4/files'
)
DEFAULT_RECOVERED_TRAIN = (
    ROOT / 'output/historical_mmm_local_history_tokenizer_train/'
    '1770759672056_set-417659_change-417658_ContentChange.py'
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def git_source(repo: Path, revision: str, relative: str) -> bytes:
    return subprocess.check_output(
        ['git', 'show', f'{revision}:{relative}'], cwd=repo,
    )


def named_tree(directory: Path) -> dict:
    files = sorted(directory.glob('*.npy'), key=lambda path: path.name)
    digest = hashlib.sha256()
    minimum_ns = None
    maximum_ns = None
    names = []
    for path in files:
        stat = path.stat()
        minimum_ns = stat.st_mtime_ns if minimum_ns is None else min(
            minimum_ns, stat.st_mtime_ns
        )
        maximum_ns = stat.st_mtime_ns if maximum_ns is None else max(
            maximum_ns, stat.st_mtime_ns
        )
        names.append(path.name)
        digest.update(path.name.encode('utf-8'))
        digest.update(b'\0')
        digest.update(bytes.fromhex(sha256_file(path)))
    return {
        'directory': str(directory.resolve()),
        'files': len(files),
        'names': names,
        'tree_sha256': digest.hexdigest(),
        'minimum_mtime_ns': minimum_ns,
        'maximum_mtime_ns': maximum_ns,
        'minimum_mtime_utc': datetime.fromtimestamp(
            minimum_ns / 1e9, tz=timezone.utc
        ).isoformat() if minimum_ns is not None else None,
        'maximum_mtime_utc': datetime.fromtimestamp(
            maximum_ns / 1e9, tz=timezone.utc
        ).isoformat() if maximum_ns is not None else None,
    }


def recovery_record(source: Path) -> dict:
    manifest_path = source.parent / 'recovery_manifest.tsv'
    lines = manifest_path.read_text(encoding='utf-8').splitlines()
    header = lines[0].split('\t')
    rows = [dict(zip(header, line.split('\t'))) for line in lines[1:] if line]
    matches = [row for row in rows if row['output'] == source.name]
    if len(matches) != 1:
        raise ValueError(f'expected one recovery row for {source}, got {len(matches)}')
    row = matches[0]
    verified = bool(
        int(row['bytes']) == source.stat().st_size
        and row['sha256'] == sha256_file(source)
    )
    return {
        'manifest': artifact(manifest_path),
        'timestamp_ms': int(row['timestamp_ms']),
        'timestamp_utc': datetime.fromtimestamp(
            int(row['timestamp_ms']) / 1000, tz=timezone.utc
        ).isoformat(),
        'set_id': int(row['set_id']),
        'change_id': int(row['change_id']),
        'content_id': int(row['content_id']),
        'verified': verified,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mmm-repo', type=Path, default=DEFAULT_MMM)
    parser.add_argument('--precleanup-commit', default=DEFAULT_COMMIT)
    parser.add_argument('--run-files', type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        '--recovered-train-source', type=Path, default=DEFAULT_RECOVERED_TRAIN,
    )
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'output/historical_mgt_tokenizer_runtime_audit.json',
    )
    args = parser.parse_args()

    repo = args.mmm_repo.resolve()
    run_files = args.run_files.resolve()
    recovered_train = args.recovered_train_source.resolve()
    metadata_path = run_files / 'wandb-metadata.json'
    output_log_path = run_files / 'output.log'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    started = datetime.fromisoformat(metadata['startedAt'].replace('Z', '+00:00'))
    started_ns = int(started.timestamp() * 1e9)

    code_tree = named_tree(repo / 'checkpoints/vq/codebook')
    contact_tree = named_tree(repo / 'checkpoints/vq/contact_token_label')
    names_equal = code_tree.pop('names') == contact_tree.pop('names')

    train_text = recovered_train.read_text(encoding='utf-8')
    tokenizer_bytes = git_source(
        repo, args.precleanup_commit, 'dataset/dataset_interx_tokenize.py'
    )
    motion_process_bytes = git_source(
        repo, args.precleanup_commit, 'utils/motion_process.py'
    )
    options_bytes = git_source(
        repo, args.precleanup_commit, 'options/option_transformer.py'
    )
    tokenizer_text = tokenizer_bytes.decode('utf-8')
    motion_process_text = motion_process_bytes.decode('utf-8')
    options_text = options_bytes.decode('utf-8')

    source_checks = {
        'global_rng_seeded_before_cache': (
            train_text.index('pl.seed_everything(args.seed)')
            < train_text.index('if len(os.listdir(codebook_dir)) == 0:')
        ),
        'cache_uses_interx_tokenizer_loader': (
            'dataset_interx_tokenize.DATALoader(' in train_text
        ),
        'cache_encodes_vq_tokens': (
            "target = net(pose, betas, gender, type='encode')" in train_text
        ),
        'cache_saves_code_and_contact_pairs': all(marker in train_text for marker in (
            "np.save(pjoin('checkpoints/vq/contact_token_label'",
            'np.save(pjoin(codebook_dir',
        )),
        'tokenizer_normalizer_uses_default_argument': (
            'self.normalizer = MotionNormalizerTorchInterX()' in tokenizer_text
        ),
        'tokenizer_converts_axis_angle_to_rot6d': (
            'motion = self.to_rot_6d(motion).float()' in tokenizer_text
        ),
        'tokenizer_normalizes_after_rot6d': (
            'motion = self.normalizer.forward(motion)' in tokenizer_text
        ),
        'tokenizer_does_not_prepare_velocity_features': (
            'prepare_interx' not in tokenizer_text
        ),
        'normalizer_default_is_vel_label': (
            "def __init__(self, motion_rep='vel')" in motion_process_text
        ),
        'default_branch_loads_root_statistics': all(
            marker in motion_process_text for marker in (
                'Interx_Mean_root.npy', 'Interx_Std_root.npy'
            )
        ),
        'historical_default_seed_is_114514': (
            "parser.add_argument('--seed', default=114514" in options_text
        ),
    }
    timeline_checks = {
        'code_and_contact_name_sets_equal': names_equal,
        'code_files_complete': code_tree['files'] == 11370,
        'contact_files_complete': contact_tree['files'] == 11370,
        'code_cache_completed_before_wandb_start': (
            code_tree['maximum_mtime_ns'] < started_ns
        ),
        'contact_cache_completed_before_wandb_start': (
            contact_tree['maximum_mtime_ns'] < started_ns
        ),
        'paired_cache_has_identical_time_window': (
            code_tree['minimum_mtime_ns'] == contact_tree['minimum_mtime_ns']
            or abs(code_tree['minimum_mtime_ns'] - contact_tree['minimum_mtime_ns'])
            < 2_000_000
        ) and abs(
            code_tree['maximum_mtime_ns'] - contact_tree['maximum_mtime_ns']
        ) < 2_000_000,
    }
    metadata_checks = {
        'program_is_two_person_mgt': metadata.get('program') == str(
            repo / 'train_t2i_trans.py'
        ),
        'arguments_match_cache_creation_run': metadata.get('args') == [
            '--dataname', 'hhi', '--exp-name', 'vq_mask_gen_4'
        ],
        'upstream_git_base_matches': metadata.get('git', {}).get('commit')
        == '8c3e7ad107ed717b878779724958cf6eff00938e',
    }
    recovery = recovery_record(recovered_train)
    recovery_valid = bool(
        recovery['verified']
        and recovery['timestamp_ms'] > int(started.timestamp() * 1000)
    )
    checks = {
        **source_checks, **timeline_checks, **metadata_checks,
        'recovered_source_vfs_verified': recovery_valid,
    }
    status = 'pass' if all(checks.values()) else 'fail'
    report = {
        'schema': 'hhc_historical_mgt_tokenizer_runtime_audit_v1',
        'status': status,
        'interpretation': {
            'frozen_baseline_policy': (
                'import the byte-exact historical train+validation cache'
            ),
            'shape_overlay_rebuild_contract': (
                'to_rot_6d_then_Interx_Mean_root_Std_root_without_prepare_interx; '
                'retain source single/single/double crop distribution with a '
                'sample-keyed deterministic RNG'
            ),
            'representation_label_warning': (
                'raw/vel labels elsewhere do not override this observed cache '
                'creation path'
            ),
        },
        'run': {
            'metadata': artifact(metadata_path),
            'output_log': artifact(output_log_path),
            'started_at_utc': started.isoformat(),
            'started_at_ns': started_ns,
        },
        'cache': {'code': code_tree, 'contact': contact_tree},
        'sources': {
            'recovered_train': {
                **artifact(recovered_train),
                'recovery': recovery,
                'role': 'first pre-edit Local History state after cache creation',
            },
            'tokenizer_git_blob': {
                'revision': args.precleanup_commit,
                'path': 'dataset/dataset_interx_tokenize.py',
                'bytes': len(tokenizer_bytes),
                'sha256': sha256_bytes(tokenizer_bytes),
            },
            'motion_process_git_blob': {
                'revision': args.precleanup_commit,
                'path': 'utils/motion_process.py',
                'bytes': len(motion_process_bytes),
                'sha256': sha256_bytes(motion_process_bytes),
            },
            'options_git_blob': {
                'revision': args.precleanup_commit,
                'path': 'options/option_transformer.py',
                'bytes': len(options_bytes),
                'sha256': sha256_bytes(options_bytes),
            },
            'root_mean': artifact(repo / 'dataset/interx/Interx_Mean_root.npy'),
            'root_std': artifact(repo / 'dataset/interx/Interx_Std_root.npy'),
        },
        'checks': checks,
        'implementation': artifact(Path(__file__)),
    }
    atomic_json(args.output, report)
    print(json.dumps({
        'status': status,
        'cache_files': code_tree['files'],
        'cache_window_utc': [
            code_tree['minimum_mtime_utc'], code_tree['maximum_mtime_utc']
        ],
        'wandb_started_at_utc': started.isoformat(),
        'output': str(args.output.resolve()),
    }, indent=2))
    if status != 'pass':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
