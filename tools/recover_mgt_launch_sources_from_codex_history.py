#!/usr/bin/env python
"""Recover and freeze exact MGT launch sources from recorded Codex patches.

The V7 process hashes every training source at launch. Two files were improved
after the process had already imported them. This tool reconstructs their
launch bytes in a temporary directory by reversing the exact, timestamped
``apply_patch`` calls from the local Codex session. A reconstruction is accepted
only when every byte hash equals the immutable training manifest.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / (
    'output/t2i_trans/'
    '2026-08-29-16-44-04_'
    'mgt_known_good_historical_contract_v7_trainmode_runtime_loss_sampler_seed114514'
)
DEFAULT_SESSION = Path(
    '/home/wenning/.codex/sessions/2026/08/23/'
    'rollout-2026-08-23T14-18-21-01a02ec5-abdc-7583-8567-f0cfd4e75a89.jsonl'
)
PATCH_LINES = (68502, 68508, 68514, 68572, 68683, 68689, 68854)
CHANGED_SOURCES = {
    'options/option_transformer.py',
    'train/train_t2i_trans.py',
}
SCHEMA = 'hhc_mgt_launch_source_recovery_v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def artifact(path):
    path = Path(path).resolve()
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def session_records(path):
    wanted = set(PATCH_LINES)
    records = {}
    with Path(path).open(encoding='utf-8', errors='strict') as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number not in wanted:
                continue
            row = json.loads(line)
            payload = row.get('payload', {})
            source = payload.get('input', '')
            match = re.fullmatch(
                r'const patch = ("(?:[^"\\]|\\.)*");\n'
                r'text\(await tools\.apply_patch\(patch\)\);\n?',
                source,
                flags=re.DOTALL,
            )
            if match is None:
                raise ValueError(
                    f'Codex session line {line_number} is not the expected '
                    'apply_patch call'
                )
            patch = json.loads(match.group(1))
            records[line_number] = {
                'line': line_number,
                'timestamp': row['timestamp'],
                'ordinal': row['ordinal'],
                'patch': patch,
                'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
                'session_line_sha256': hashlib.sha256(line.encode()).hexdigest(),
                'raw_session_line': line,
            }
    missing = sorted(wanted - set(records))
    if missing:
        raise ValueError(f'Codex patch records are missing: {missing}')
    return records


def reverse_patch(patch, temporary_root):
    if not patch.startswith('*** Begin Patch\n') or not patch.endswith(
            '\n*** End Patch'):
        raise ValueError('unexpected apply_patch envelope')
    rewritten = []
    touched = set()
    for line in patch.splitlines():
        if line.startswith('*** Update File: '):
            original = Path(line.removeprefix('*** Update File: ')).resolve()
            try:
                relative = original.relative_to(ROOT)
            except ValueError as error:
                raise ValueError(f'patch target is outside HHC: {original}') from error
            relative_text = relative.as_posix()
            if relative_text not in CHANGED_SOURCES:
                raise ValueError(f'unexpected source in recovery patch: {relative_text}')
            touched.add(relative_text)
            line = f'*** Update File: {(temporary_root / relative).resolve()}'
        elif line.startswith('+') and not line.startswith('+++'):
            line = '-' + line[1:]
        elif line.startswith('-') and not line.startswith('---'):
            line = '+' + line[1:]
        rewritten.append(line)
    if not touched:
        raise ValueError('recovery patch does not touch a bound source')
    return '\n'.join(rewritten) + '\n', sorted(touched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--training-manifest', type=Path,
        default=DEFAULT_RUN / 'training_manifest.json',
    )
    parser.add_argument('--session', type=Path, default=DEFAULT_SESSION)
    parser.add_argument(
        '--snapshot-root', type=Path,
        default=DEFAULT_RUN / 'source',
    )
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'output/historical_mgt_v7_launch_source_recovery.json',
    )
    args = parser.parse_args()

    manifest_path = args.training_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {
        name: digest for name, digest in manifest.get('sha256', {}).items()
        if name.endswith('.py') and '/' in name
    }
    if len(expected) != 14 or not CHANGED_SOURCES <= set(expected):
        raise ValueError(f'unexpected MGT source manifest: {sorted(expected)}')

    records = session_records(args.session.resolve())
    snapshot_root = args.snapshot_root.resolve()
    snapshot_root.mkdir(parents=True, exist_ok=True)
    selected_records_path = snapshot_root / 'codex_selected_patch_records.jsonl'
    selected_records_text = ''.join(
        records[line]['raw_session_line'] for line in sorted(records)
    )
    temporary_records = selected_records_path.with_suffix(
        selected_records_path.suffix + f'.tmp.{os.getpid()}'
    )
    temporary_records.write_text(selected_records_text, encoding='utf-8')
    os.replace(temporary_records, selected_records_path)
    apply_patch_executable = shutil.which('apply_patch')
    if apply_patch_executable is None:
        raise FileNotFoundError('apply_patch executable is unavailable')
    destinations = {
        relative: snapshot_root / (
            f'{expected_hash}__{relative.replace("/", "__")}'
        )
        for relative, expected_hash in expected.items()
    }
    existing_snapshots = {
        relative for relative, destination in destinations.items()
        if destination.is_file()
        and sha256_file(destination) == expected[relative]
    }
    corrupt_snapshots = [
        relative for relative, destination in destinations.items()
        if destination.is_file() and relative not in existing_snapshots
    ]
    if corrupt_snapshots:
        raise ValueError(
            f'corrupt existing launch snapshots: {sorted(corrupt_snapshots)}'
        )

    # Once a content-addressed snapshot exists, the immutable launch manifest
    # is the authority.  Do not require the evolving worktree to keep the same
    # bytes.  Reconstruction is needed only to materialize a missing snapshot.
    missing_snapshots = set(expected) - existing_snapshots
    reconstruct_changed = bool(missing_snapshots & CHANGED_SOURCES)
    recovery_rows = []
    snapshots = {}
    materialization = {}
    with tempfile.TemporaryDirectory(prefix='hhc_mgt_launch_source_recovery_') as temp:
        temporary_root = Path(temp)
        if reconstruct_changed:
            for relative in CHANGED_SOURCES:
                source = ROOT / relative
                destination = temporary_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)

        for line_number in sorted(records, reverse=True):
            record = records[line_number]
            reversed_patch, touched = reverse_patch(
                record['patch'], temporary_root
            )
            if reconstruct_changed:
                completed = subprocess.run(
                    [apply_patch_executable], input=reversed_patch, text=True,
                    cwd=ROOT, capture_output=True,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f'reverse patch failed at session line {line_number}: '
                        f'{completed.stdout}\n{completed.stderr}'
                    )
            recovery_rows.append({
                key: record[key] for key in (
                    'line', 'timestamp', 'ordinal', 'patch_sha256',
                    'session_line_sha256',
                )
            } | {
                'touched_sources': touched,
                'applied_during_this_run': reconstruct_changed,
            })

        for relative, expected_hash in sorted(expected.items()):
            destination = destinations[relative]
            if relative in existing_snapshots:
                materialization[relative] = 'validated_existing_snapshot'
            else:
                source = (
                    temporary_root / relative
                    if relative in CHANGED_SOURCES
                    else ROOT / relative
                )
                actual_hash = sha256_file(source)
                if actual_hash != expected_hash:
                    raise ValueError(
                        f'launch source recovery hash mismatch for {relative}: '
                        f'{actual_hash} != {expected_hash}'
                    )
                shutil.copyfile(source, destination)
                materialization[relative] = (
                    'reconstructed_from_recorded_patches'
                    if relative in CHANGED_SOURCES
                    else 'copied_after_launch_manifest_hash_check'
                )
            snapshots[relative] = artifact(destination)

    payload = {
        'schema': SCHEMA,
        'status': 'pass',
        'training_manifest': artifact(manifest_path),
        'source_session_path_at_recovery': str(args.session.resolve()),
        'selected_session_records': artifact(selected_records_path),
        'reversed_patch_records': recovery_rows,
        'snapshot_materialization': materialization,
        'snapshots_existing_before_this_run': len(existing_snapshots),
        'snapshots_materialized_during_this_run': len(missing_snapshots),
        'changed_sources_recovered_from_history': sorted(CHANGED_SOURCES),
        'unchanged_sources_copied_after_manifest_hash_check': sorted(
            set(expected) - CHANGED_SOURCES
        ),
        'snapshots': snapshots,
        'all_snapshot_hashes_match_launch_manifest': True,
        'implementation': artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps({
        'status': payload['status'],
        'sources': len(snapshots),
        'reversed_patches': len(recovery_rows),
        'snapshot_root': str(snapshot_root),
        'output': str(args.output.resolve()),
    }, indent=2))


if __name__ == '__main__':
    main()
