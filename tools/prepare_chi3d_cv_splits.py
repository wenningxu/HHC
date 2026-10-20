#!/usr/bin/env python
"""Create frozen, leakage-checked leave-one-group-out CHI3D CV splits."""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


GROUPS = ('G2', 'G3', 'G4')
PROTOCOL_VERSION = 'chi3d_leave_one_group_out_v1'


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_id(sample_id):
    fields = sample_id.split('_')
    if len(fields) < 3 or fields[0] not in GROUPS:
        raise ValueError(f'invalid CHI3D sample ID: {sample_id!r}')
    return fields[0], '_'.join(fields[1:-1])


def summarize(ids):
    groups = collections.Counter()
    actions = collections.Counter()
    for sample_id in ids:
        group, action = parse_id(sample_id)
        groups[group] += 1
        actions[action] += 1
    return {
        'actions': dict(sorted(actions.items())),
        'groups': dict(sorted(groups.items())),
        'samples': len(ids),
    }


def write_split(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(f'{sample_id}\n' for sample_id in ids)
    path.write_text(payload, encoding='utf-8')
    return {
        'path': str(path.resolve()),
        'sha256': sha256_bytes(payload.encode('utf-8')),
        **summarize(ids),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-split',
        default='/home/wenning/projects/MMM/dataset/chi3d/splits/train.txt',
    )
    parser.add_argument('--output-dir', default='configs/chi3d_cv')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError('--validation-fraction must be in (0, 0.5)')

    source = Path(args.source_split).resolve()
    ids = sorted(
        line.strip()
        for line in source.read_text(encoding='utf-8').splitlines()
        if line.strip()
    )
    if len(ids) != len(set(ids)):
        raise ValueError('source split contains duplicate IDs')
    if {parse_id(sample_id)[0] for sample_id in ids} != set(GROUPS):
        raise ValueError(f'source split must contain exactly {GROUPS}')

    output_dir = Path(args.output_dir).resolve()
    manifest = {
        'folds': {},
        'inner_validation': {
            'fraction': args.validation_fraction,
            'method': (
                'within each training group/action stratum, select ceil(N*fraction) '
                'lowest SHA256 ranks keyed by protocol version and held-out group'
            ),
        },
        'protocol_version': PROTOCOL_VERSION,
        'source': {
            'path': str(source),
            'sha256': sha256_file(source),
            **summarize(ids),
        },
    }

    all_ids = set(ids)
    for held_out in GROUPS:
        test_ids = sorted(sample_id for sample_id in ids if parse_id(sample_id)[0] == held_out)
        candidates = [sample_id for sample_id in ids if parse_id(sample_id)[0] != held_out]
        strata = collections.defaultdict(list)
        for sample_id in candidates:
            strata[parse_id(sample_id)].append(sample_id)

        validation_ids = []
        for stratum, stratum_ids in sorted(strata.items()):
            count = max(1, math.ceil(len(stratum_ids) * args.validation_fraction))
            ranked = sorted(
                stratum_ids,
                key=lambda sample_id: sha256_bytes(
                    f'{PROTOCOL_VERSION}|{held_out}|{sample_id}'.encode('utf-8')
                ),
            )
            validation_ids.extend(ranked[:count])
        validation_ids = sorted(validation_ids)
        validation_set = set(validation_ids)
        train_ids = sorted(sample_id for sample_id in candidates if sample_id not in validation_set)

        train_set, test_set = set(train_ids), set(test_ids)
        if train_set & validation_set or train_set & test_set or validation_set & test_set:
            raise AssertionError(f'overlapping identities in fold {held_out}')
        if train_set | validation_set | test_set != all_ids:
            raise AssertionError(f'fold {held_out} does not partition the source split')
        if {parse_id(sample_id)[0] for sample_id in test_ids} != {held_out}:
            raise AssertionError(f'fold {held_out} test set is not group-pure')

        fold_dir = output_dir / f'heldout_{held_out.lower()}'
        split_records = {
            'train': write_split(fold_dir / 'train.txt', train_ids),
            'validation': write_split(fold_dir / 'validation.txt', validation_ids),
            'test': write_split(fold_dir / 'test.txt', test_ids),
        }
        manifest['folds'][held_out] = {
            'held_out_group': held_out,
            'no_identity_overlap': True,
            'partitions_source_exactly': True,
            'splits': split_records,
        }

    manifest_path = output_dir / 'manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps({**manifest, 'manifest_path': str(manifest_path)}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
