#!/usr/bin/env python
"""Recover the exact VQ launch sources from the local append-only task journal.

The formal VQ run recorded SHA-256 for every training source, but its legacy
``source/`` directory was created without copying files.  Later, independent
generalization work changed four of those files.  This utility starts from the
current files, reverses only FileChange records after the formal launch, checks
the recovered bytes against the frozen training manifest, and emits an
``apply_patch`` patch that can materialize the verified archive.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = Path(
    'output/vq-interx-sep/'
    '2026-08-26-00-09-40_paper_vq_8096_150k_seed123_exact_rawstats_ema_safe'
)
DEFAULT_ROLLOUT = Path(
    '/home/wenning/.codex/sessions/2026/08/23/'
    'rollout-2026-08-23T14-18-21-01a02ec5-abdc-7583-8567-f0cfd4e75a89.jsonl'
)
LAUNCH_TIMESTAMP_UTC = '2026-08-25T23:09:38.000Z'
SOURCE_PATHS = (
    'dataset/dataset_interx.py',
    'models/model_tools.py',
    'models/vqvae_sep3_shape.py',
    'options/option_vq.py',
    'train/train_vq_interx.py',
    'utils/losses.py',
    'utils/motion_process.py',
    'utils/quantizer_state.py',
    'utils/runtime.py',
)


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def post_launch_changes(rollout):
    wanted = {str((ROOT / path).resolve()): path for path in SOURCE_PATHS}
    changes = []
    with rollout.open(encoding='utf-8', errors='replace') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # One legacy compaction boundary is not JSON.  It contains no
                # FileChange item and is retained in the journal as-is.
                continue
            timestamp = record.get('timestamp', '')
            if timestamp <= LAUNCH_TIMESTAMP_UTC:
                continue
            item = record.get('payload', {}).get('item', {})
            if item.get('type') != 'FileChange':
                continue
            for absolute, change in item.get('changes', {}).items():
                if absolute not in wanted:
                    continue
                if change.get('type') != 'update' or change.get('move_path'):
                    raise ValueError(
                        f'unsupported post-launch source change at line {line_number}'
                    )
                changes.append({
                    'timestamp': timestamp,
                    'ordinal': record.get('ordinal'),
                    'line_number': line_number,
                    'path': wanted[absolute],
                    'unified_diff': change['unified_diff'],
                })
    return changes


def reverse_change(root, change):
    relative = change['path']
    patch = (
        f'--- a/{relative}\n'
        f'+++ b/{relative}\n'
        f"{change['unified_diff'].lstrip()}"
    )
    result = subprocess.run(
        ['patch', '--reverse', '--batch', '--silent', '-p1', '-d', str(root)],
        input=patch, text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"cannot reverse ordinal {change['ordinal']} for {relative}: "
            f'{result.stdout}{result.stderr}'
        )


def recover(rollout, training_manifest):
    manifest = json.loads(training_manifest.read_text(encoding='utf-8'))
    expected = manifest['sha256']
    changes = post_launch_changes(rollout)
    with tempfile.TemporaryDirectory(prefix='hhc-vq-launch-source-') as temporary:
        temporary = Path(temporary)
        for relative in SOURCE_PATHS:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        for change in reversed(changes):
            reverse_change(temporary, change)
        recovered = {
            relative: (temporary / relative).read_bytes()
            for relative in SOURCE_PATHS
        }
    mismatches = {
        relative: {
            'expected': expected[relative],
            'recovered': sha256_bytes(payload),
        }
        for relative, payload in recovered.items()
        if sha256_bytes(payload) != expected[relative]
    }
    if mismatches:
        raise ValueError(f'recovered launch sources do not match manifest: {mismatches}')
    return recovered, changes, manifest


def add_file_patch(path, payload):
    text = payload.decode('utf-8')
    if not text.endswith('\n'):
        raise ValueError(f'apply_patch archive requires final newline: {path}')
    lines = ''.join(f'+{line}' for line in text.splitlines(keepends=True))
    return f'*** Add File: {path}\n{lines}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument(
        '--training-manifest', type=Path,
        default=Path('checkpoints/vq_paper/training_manifest.json'),
    )
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--emit-apply-patch', action='store_true')
    parser.add_argument('--verify-archive', action='store_true')
    args = parser.parse_args()
    if args.emit_apply_patch and args.verify_archive:
        parser.error('--emit-apply-patch and --verify-archive are mutually exclusive')
    recovered, changes, training_manifest = recover(
        args.rollout.resolve(), args.training_manifest.resolve()
    )
    archive_root = (args.run_dir.resolve() / 'source')
    snapshot = {
        'schema': 'hhc_vq_launch_source_snapshot_v2',
        'status': 'pass',
        'launch_timestamp_utc': LAUNCH_TIMESTAMP_UTC,
        'training_manifest': str(args.training_manifest.resolve()),
        'training_manifest_sha256': sha256_bytes(
            args.training_manifest.resolve().read_bytes()
        ),
        'recovery_evidence': {
            'append_only_rollout': str(args.rollout.resolve()),
            'append_only_rollout_sha256': sha256_bytes(
                args.rollout.resolve().read_bytes()
            ),
            'reversed_post_launch_changes': [
                {
                    key: change[key]
                    for key in ('timestamp', 'ordinal', 'line_number', 'path')
                }
                for change in changes
            ],
        },
        'sources': {
            relative: {
                'path': str(archive_root / relative),
                'bytes': len(payload),
                'sha256': sha256_bytes(payload),
                'training_manifest_sha256': training_manifest['sha256'][relative],
            }
            for relative, payload in recovered.items()
        },
    }
    snapshot_path = archive_root / 'complete_source_snapshot_manifest.json'
    if args.verify_archive:
        if not snapshot_path.is_file():
            raise FileNotFoundError(snapshot_path)
        archived_snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
        for relative, payload in recovered.items():
            archived = archive_root / relative
            if not archived.is_file() or archived.read_bytes() != payload:
                raise ValueError(f'archived launch source differs: {archived}')
            record = archived_snapshot.get('sources', {}).get(relative, {})
            if record.get('sha256') != sha256_bytes(payload):
                raise ValueError(f'archive manifest hash differs: {relative}')
        if (
            archived_snapshot.get('schema') != snapshot['schema']
            or archived_snapshot.get('launch_timestamp_utc')
            != snapshot['launch_timestamp_utc']
            or set(archived_snapshot.get('sources', {})) != set(recovered)
        ):
            raise ValueError('archive manifest contract differs from reconstruction')
        print(json.dumps({
            'schema': snapshot['schema'], 'status': 'pass',
            'archive_manifest': str(snapshot_path),
            'archive_manifest_sha256': sha256_bytes(snapshot_path.read_bytes()),
            'verified_sources': len(recovered),
        }, indent=2, sort_keys=True))
        return
    if not args.emit_apply_patch:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        return
    parts = ['*** Begin Patch\n']
    for relative, payload in recovered.items():
        destination = archive_root / relative
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ValueError(f'existing archived source differs: {destination}')
            continue
        parts.append(add_file_patch(destination, payload))
    if snapshot_path.exists():
        raise FileExistsError(snapshot_path)
    snapshot_payload = (
        json.dumps(snapshot, indent=2, sort_keys=True) + '\n'
    ).encode('utf-8')
    parts.append(add_file_patch(snapshot_path, snapshot_payload))
    parts.append('*** End Patch\n')
    print(''.join(parts), end='')


if __name__ == '__main__':
    main()
