#!/usr/bin/env python
"""Validate frozen Inter-X request files and write a checksum manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


EXPECTED_SHAPES = {
    'names': (1696,), 'captions': (1696,), 'lengths': (1696,),
    'text_embeddings': (1696, 512), 'gt_embeddings': (1696, 512),
    'gt_body_embeddings': (1696, 512), 'betas': (1696, 20),
    'gender': (1696, 2), 'contact': (1696, 150), 'mm_indices': (100,),
    'seed': (),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path):
    absolute = os.path.abspath(path)
    return {
        'path': absolute,
        'bytes': os.path.getsize(absolute),
        'sha256': sha256(absolute),
    }


def resolve_frozen_artifact(record, snapshot_root):
    """Resolve recorded bytes without silently blessing a changed live file."""
    source = Path(record['path']).resolve()
    expected_hash = record.get('sha256')
    expected_bytes = record.get('bytes')
    if (source.is_file() and source.stat().st_size == expected_bytes
            and sha256(source) == expected_hash):
        return source, 'live_path'
    snapshot = (
        snapshot_root / f'{expected_hash}__{source.name}'
    ).resolve()
    if (snapshot.is_file() and snapshot.stat().st_size == expected_bytes
            and sha256(snapshot) == expected_hash):
        return snapshot, 'content_addressed_snapshot'
    raise ValueError(
        f'frozen provenance artifact is absent or changed: {source}; '
        f'expected snapshot {snapshot}'
    )


def verify_frozen_manifest(frozen, current, snapshot_root):
    immutable_fields = (
        'protocol', 'request_contract_version', 'replications',
        'samples_per_replication', 'multimodality', 'records', 'status',
    )
    for field in immutable_fields:
        if frozen.get(field) != current.get(field):
            raise ValueError(f'frozen request manifest field changed: {field}')
    frozen_provenance = frozen.get('frozen_input_provenance', {})
    current_provenance = current.get('frozen_input_provenance', {})
    if set(frozen_provenance) != set(current_provenance):
        raise ValueError('frozen request provenance keys changed')
    resolutions = {}
    for label, record in frozen_provenance.items():
        resolved, mode = resolve_frozen_artifact(record, snapshot_root)
        live = current_provenance[label]
        resolutions[label] = {
            'recorded': record,
            'resolved_path': str(resolved),
            'resolution': mode,
            'live_path_now': live,
            'live_matches_recorded': (
                live.get('bytes') == record.get('bytes')
                and live.get('sha256') == record.get('sha256')
            ),
        }
    return {
        'schema': 'fair_interx_request_manifest_verification_v1',
        'status': 'pass',
        'canonical_manifest_unchanged': True,
        'request_npz_records_match': True,
        'replications': frozen['replications'],
        'samples_per_replication': frozen['samples_per_replication'],
        'provenance_resolution': resolutions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-dir', default='output/fair_interx_requests')
    parser.add_argument('--replications', type=int, default=20)
    parser.add_argument('--base-seed', type=int, default=1234)
    parser.add_argument('--output', default='output/fair_interx_requests/manifest.json')
    parser.add_argument(
        '--refresh-provenance', action='store_true',
        help=(
            'Explicitly replace the provenance manifest with current source '
            'records. By default an existing frozen manifest is checked '
            'without modification.'
        ),
    )
    parser.add_argument(
        '--snapshot-root', type=Path,
        default=Path('tools/provenance_snapshots'),
    )
    args = parser.parse_args()

    records = []
    for replication in range(args.replications):
        path = os.path.join(args.request_dir, f'replication_{replication:02d}.npz')
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        request = np.load(path, allow_pickle=False)
        if set(request.files) != set(EXPECTED_SHAPES):
            raise ValueError(f'{path}: fields {request.files}')
        for key, shape in EXPECTED_SHAPES.items():
            if request[key].shape != shape:
                raise ValueError(f'{path}: {key} expected {shape}, got {request[key].shape}')
        if int(request['seed']) != args.base_seed + replication:
            raise ValueError(f'{path}: seed mismatch')
        if len(np.unique(request['names'])) != 1696:
            raise ValueError(f'{path}: sample names are not unique')
        if len(np.unique(request['mm_indices'])) != 100:
            raise ValueError(f'{path}: MM indices are not unique')
        if request['mm_indices'].min() < 0 or request['mm_indices'].max() >= 1696:
            raise ValueError(f'{path}: MM index out of bounds')
        if request['lengths'].min() < 24 or request['lengths'].max() > 150:
            raise ValueError(f'{path}: invalid motion lengths')
        if not np.isin(request['contact'], (0, 1)).all():
            raise ValueError(f'{path}: contact is not binary')
        for key in ('text_embeddings', 'gt_embeddings', 'gt_body_embeddings', 'betas'):
            if not np.isfinite(request[key]).all():
                raise ValueError(f'{path}: {key} contains non-finite values')
        records.append({
            'replication': replication,
            'seed': int(request['seed']),
            'path': os.path.abspath(path),
            'sha256': sha256(path),
            'bytes': os.path.getsize(path),
        })

    manifest = {
        'protocol': 'interx_official_v1',
        'request_contract_version': 2,
        'replications': args.replications,
        'samples_per_replication': 1696,
        'multimodality': {'prompts': 100, 'repeats': 30, 'pairs': 10},
        'records': records,
        'frozen_input_provenance': {
            label: artifact_record(path)
            for label, path in {
                'protocol': 'configs/fair_interx_protocol.yaml',
                'request_preparer': 'tools/prepare_fair_eval_requests.py',
                'request_verifier': 'tools/verify_fair_eval_requests.py',
                'dataset_loader': 'dataset/dataset_interx_TM.py',
                'normalizer': 'utils/motion_process.py',
                'evaluator_wrapper': 'models/evaluator_wrapper.py',
                'evaluator_options': 'checkpoints/hhi/Comp_v6_KLD01/opt.txt',
                'evaluator_weights': 'checkpoints/hhi/text_mot_match/model/finest.tar',
                'test_split': 'dataset/interx/splits/test.txt',
                'test_motions': 'dataset/interx/motions/test.h5',
                'raw_mean': 'dataset/interx/Interx_Mean.npy',
                'raw_std': 'dataset/interx/Interx_Std.npy',
            }.items()
        },
        'status': 'pass',
    }
    output = Path(args.output).resolve()
    if output.is_file() and not args.refresh_provenance:
        frozen = json.loads(output.read_text(encoding='utf-8'))
        report = verify_frozen_manifest(
            frozen, manifest, args.snapshot_root.resolve()
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = str(output) + f'.tmp.{os.getpid()}'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(temporary, output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
