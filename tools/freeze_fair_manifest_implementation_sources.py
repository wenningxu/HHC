#!/usr/bin/env python
"""Freeze implementation files already hash-bound by fair-run manifests.

This does not bless changed source: every live source must still match the hash
recorded when generation ran.  It only moves that exact byte stream to a
content-addressed immutable location and updates the manifest path.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


SCHEMA = 'fair_interx_content_addressed_implementation_snapshots_v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def freeze_manifest(manifest_path, snapshot_root):
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    provenance = manifest.get('provenance', {})
    implementation = provenance.get('implementation', {})
    if not implementation:
        raise ValueError(f'manifest has no implementation records: {manifest_path}')
    snapshot_root.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for name, record in implementation.items():
        source = Path(record['path']).resolve()
        expected_hash = record.get('sha256')
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(
                f'{name}: source is absent or differs from the generation hash: '
                f'{source}'
            )
        already_content_addressed = (
            record.get('content_addressed_snapshot') is True
            and source.name.startswith(expected_hash + '__')
        )
        destination = (
            source if already_content_addressed
            else (snapshot_root / f'{expected_hash}__{source.name}').resolve()
        )
        if destination.is_file():
            if sha256_file(destination) != expected_hash:
                raise ValueError(f'corrupt existing source snapshot: {destination}')
        else:
            shutil.copyfile(source, destination)
        frozen[name] = {
            **record,
            'path': str(destination),
            'source_path_at_execution': record.get(
                'source_path_at_execution', str(source)
            ),
            'content_addressed_snapshot': True,
        }
    provenance['implementation'] = frozen
    provenance['source_validation_contract'] = SCHEMA
    provenance['repository_records_role'] = (
        'historical context and training-generation equality; immutable '
        'implementation bytes are validated through snapshots'
    )
    provenance['implementation_snapshot_freeze_mode'] = (
        'posthoc_exact_recorded_hash_only'
    )
    manifest['provenance'] = provenance
    atomic_json(manifest_path, manifest)
    return {
        'manifest': str(manifest_path),
        'implementation_files': len(frozen),
        'snapshot_root': str(snapshot_root.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifests', nargs='+', type=Path)
    parser.add_argument(
        '--snapshot-root', type=Path,
        default=Path('tools/provenance_snapshots'),
    )
    args = parser.parse_args()
    reports = [
        freeze_manifest(path, args.snapshot_root.resolve())
        for path in args.manifests
    ]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
