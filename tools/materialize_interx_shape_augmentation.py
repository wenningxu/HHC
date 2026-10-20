#!/usr/bin/env python
"""Materialize optimized shape samples as an Inter-X training overlay."""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.interx_shape_augmentation_plan import (
    validate_interx_shape_augmentation_plan,
)
from utils.interx_shape_augmentation import validated_quality_selection


SCHEMA = 'hhc_interx_shape_augmentation_materialization_v1'
HYBRID_MODE = 'optimized_positive_plus_random_zero'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(value, encoding='utf-8')
    os.replace(temporary, path)


def atomic_npz(path, **payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'wb') as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def atomic_npy(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'wb') as handle:
        np.save(handle, payload)
    os.replace(temporary, path)


def read_names(path):
    names = [
        line.strip() for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(names) != len(set(names)):
        raise ValueError(f'duplicate names in {path}')
    return names


def read_plan(path):
    manifest, _, rows, _, _ = validate_interx_shape_augmentation_plan(path)
    return manifest, {row['augmented_name']: row for row in rows}


def resolve_augmentation_mode(completions):
    """Resolve one formal mode and reject ambiguous mixed optimizer shards."""
    modes = {completion.get('optimizer', {}).get('mode')
             for _, completion in completions}
    if not modes <= {'optimized', 'random_shape_only'}:
        raise ValueError(f'unsupported optimization completion modes: {modes}')
    if len(modes) == 1:
        return next(iter(modes))
    if modes != {'optimized', 'random_shape_only'}:
        raise ValueError('optimization shards use incompatible augmentation modes')
    for path, completion in completions:
        selection = completion.get('selection', {})
        mode = completion['optimizer']['mode']
        if mode == 'optimized':
            valid = (
                selection.get('positive_timing_only') is True
                and selection.get('zero_timing_only') is not True
            )
        else:
            valid = (
                selection.get('zero_timing_only') is True
                and selection.get('positive_timing_only') is not True
            )
        if not valid:
            raise ValueError(
                f'{path}: mixed formal materialization requires optimized '
                'positive-timing shards and random-only zero-timing shards'
            )
    return HYBRID_MODE


def output_raw(p1_path, p2_path):
    with np.load(p1_path, allow_pickle=True) as p1, np.load(
        p2_path, allow_pickle=True
    ) as p2:
        poses1 = np.asarray(p1['poses'], dtype=np.float32).reshape(-1, 55, 3)
        poses2 = np.asarray(p2['poses'], dtype=np.float32).reshape(-1, 55, 3)
        trans1 = np.asarray(p1['trans'], dtype=np.float32).reshape(-1, 3)
        trans2 = np.asarray(p2['trans'], dtype=np.float32).reshape(-1, 3)
        if not (len(poses1) == len(poses2) == len(trans1) == len(trans2)):
            raise ValueError('optimized person outputs have different lengths')
        raw = np.zeros((len(poses1), 56, 6), dtype=np.float32)
        raw[:, :55, :3] = poses1
        raw[:, :55, 3:] = poses2
        raw[:, 55, :3] = trans1
        raw[:, 55, 3:] = trans2
        betas = np.stack([
            np.asarray(p1['betas'], dtype=np.float32).reshape(-1)[:10],
            np.asarray(p2['betas'], dtype=np.float32).reshape(-1)[:10],
        ])
        genders = np.asarray([
            str(np.asarray(p1['gender']).reshape(-1)[0]),
            str(np.asarray(p2['gender']).reshape(-1)[0]),
        ])
    return raw, betas, genders


def timing_crop(record):
    if record.get('timing_file') is None:
        raise ValueError(
            f'{record["source_name"]}: v2 augmentation requires an explicit '
            'manual timing file'
        )
    path = Path(record['timing_file']).resolve()
    if sha256_file(path) != record['timing_sha256']:
        raise ValueError(f'{record["source_name"]}: timing hash changed')
    full = np.load(path, allow_pickle=False).astype(np.uint8).reshape(-1)
    crop = full[record['crop_start']:record['crop_stop']]
    if len(crop) != record['crop_frames']:
        raise ValueError(f'{record["source_name"]}: invalid timing crop')
    return crop


def build_link_h5(output_path, source_path, source_names, augmented):
    temporary = output_path.with_suffix(output_path.suffix + '.building')
    with h5py.File(source_path, 'r') as source, h5py.File(temporary, 'w') as target:
        missing = sorted(set(source_names) - set(source.keys()))
        if missing:
            raise ValueError(f'{source_path}: split names absent: {missing[:5]}')
        for name in source_names:
            target[name] = h5py.ExternalLink(str(source_path.resolve()), f'/{name}')
        for name, raw in augmented.items():
            if name in target:
                raise ValueError(f'duplicate original/augmented motion key {name}')
            target.create_dataset(
                name, data=raw, compression='gzip', compression_opts=4,
                shuffle=True,
            )
        target.attrs['original_motion_h5'] = str(source_path.resolve())
        target.attrs['original_motion_h5_sha256'] = sha256_file(source_path)
        target.attrs['augmented_samples'] = len(augmented)
    os.replace(temporary, output_path)


def prepare_materialization_root(output_root, identity):
    """Create or strictly resume an incomplete immutable overlay build."""
    output_root = output_root.resolve()
    status_path = output_root / '_materialization_status.json'
    manifest_path = output_root / 'materialization_manifest.json'
    identity_sha256 = canonical_sha256(identity)
    resumed = False
    if output_root.exists():
        if manifest_path.exists():
            raise FileExistsError(
                f'completed immutable output root already exists: {output_root}'
            )
        if not status_path.is_file():
            raise FileExistsError(
                f'output root exists without a resumable status: {output_root}'
            )
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if (
            status.get('schema') != SCHEMA
            or status.get('status') != 'running'
            or status.get('identity_sha256') != identity_sha256
        ):
            raise ValueError(
                'existing materialization directory has another or '
                'non-running contract'
            )
        resumed = True
        created_at_utc = status['created_at_utc']
    else:
        output_root.mkdir(parents=True)
        created_at_utc = datetime.now(timezone.utc).isoformat()
    for child in ('motions', 'splits', 'betas', 'contact', 'texts_processed'):
        (output_root / child).mkdir(exist_ok=True)
    atomic_json(status_path, {
        'schema': SCHEMA,
        'status': 'running',
        'created_at_utc': created_at_utc,
        'identity': identity,
        'identity_sha256': identity_sha256,
        'resumed': resumed,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
    })
    return created_at_utc, resumed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--optimization-dir', required=True, type=Path, action='append',
        help=(
            'Completed optimizer directory. Repeat for deterministic '
            'multi-GPU/process shards.'
        ),
    )
    parser.add_argument(
        '--plan', type=Path,
        default=Path(
            'configs/interx_shape_augmentation/timing_covered_seed123_v2/'
            'plan_manifest.json'
        ),
    )
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--quality-gate', type=Path, action='append',
                        help='One completed two-estimator gate per shard; only its accepted samples are packed.')
    parser.add_argument('--semantic-exclusions', type=Path,
                        help='Hash-bound known-risk exclusions after physical gates; preserves full denominator.')
    parser.add_argument('--train-split', type=Path, default=Path('dataset/interx/splits/train.txt'))
    parser.add_argument('--val-split', type=Path, default=Path('dataset/interx/splits/val.txt'))
    parser.add_argument('--all-split', type=Path, default=Path('dataset/interx/splits/all.txt'))
    parser.add_argument('--train-motion', type=Path, default=Path('dataset/interx/motions/train.h5'))
    parser.add_argument('--all-motion', type=Path, default=Path('dataset/interx/motions/all.h5'))
    parser.add_argument('--text-dir', type=Path, default=Path('dataset/interx/texts_processed'))
    parser.add_argument('--stats-dir', type=Path, default=Path('dataset/interx'))
    parser.add_argument('--stats-prefix', default='Interx')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.semantic_exclusions and not args.quality_gate:
        raise ValueError('Semantic exclusions require physical quality gates')
    optimization_dirs = [path.resolve() for path in args.optimization_dir]
    output_root = args.output_root.resolve()
    completions = []
    sample_sources = {}
    selection_positions = {}
    for optimization_dir in optimization_dirs:
        completion_path = optimization_dir / '_completion_manifest.json'
        completion = json.loads(completion_path.read_text(encoding='utf-8'))
        if not completion.get('complete'):
            raise ValueError(
                f'optimization completion manifest is incomplete: '
                f'{completion_path}'
            )
        augmentation_mode = completion.get('optimizer', {}).get('mode')
        if augmentation_mode not in ('optimized', 'random_shape_only'):
            raise ValueError(
                f'completion manifest has no supported augmentation mode: '
                f'{completion_path}'
            )
        names = completion['completed_names']
        positions = completion.get('selection', {}).get(
            'full_selection_positions'
        )
        if positions is None:
            if len(optimization_dirs) != 1:
                raise ValueError(
                    'multi-shard materialization requires full selection '
                    'positions in every completion manifest'
                )
            positions = list(range(len(names)))
        if len(names) != len(positions):
            raise ValueError(f'{completion_path}: names/positions differ')
        for name, position in zip(names, positions):
            if name in sample_sources:
                raise ValueError(f'duplicate optimization sample {name}')
            position = int(position)
            if position in selection_positions:
                raise ValueError(f'duplicate full selection position {position}')
            sample_sources[name] = optimization_dir
            selection_positions[position] = name
        completions.append((completion_path, completion))
    augmentation_mode = resolve_augmentation_mode(completions)
    plan_manifest, plan_records = read_plan(args.plan.resolve())
    expected_positions = list(range(len(selection_positions)))
    if any(position < 0 for position in selection_positions):
        raise ValueError('Negative full selection position')
    if not args.quality_gate and sorted(selection_positions) != expected_positions:
        raise ValueError(
            'optimization shards do not form one contiguous full selection'
        )
    selected_names = [selection_positions[index] for index in sorted(selection_positions)]
    if len(selected_names) != len(set(selected_names)):
        raise ValueError('optimization completion contains duplicate names')
    if set(selected_names) - set(plan_records):
        raise ValueError('optimization outputs contain names outside the frozen plan')
    for _, completion in completions:
        if (Path(completion['plan']).resolve() != args.plan.resolve()
                or completion['plan_sha256'] != sha256_file(args.plan)):
            raise ValueError('Optimization completion and materialization plans differ')
    quality = None
    if args.quality_gate:
        entries = [{'path': str(p.resolve()), 'sha256': sha256_file(p)} for p in args.quality_gate]
        review = ({'path': str(args.semantic_exclusions.resolve()),
                   'sha256': sha256_file(args.semantic_exclusions)}
                  if args.semantic_exclusions else None)
        quality = validated_quality_selection(entries, completions, selected_names, review)
        selected_names = quality['accepted_names']
    materialization_identity = {
        'output_root': str(output_root),
        'optimization_completions': [
            {
                'path': str(path),
                'sha256': sha256_file(path),
            }
            for path, _ in completions
        ],
        'plan': str(args.plan.resolve()),
        'plan_sha256': sha256_file(args.plan),
        'train_split': str(args.train_split.resolve()),
        'val_split': str(args.val_split.resolve()),
        'all_split': str(args.all_split.resolve()),
        'train_motion': str(args.train_motion.resolve()),
        'all_motion': str(args.all_motion.resolve()),
        'text_dir': str(args.text_dir.resolve()),
        'stats_dir': str(args.stats_dir.resolve()),
        'stats_prefix': args.stats_prefix,
        'requested_augmented_samples': len(selected_names),
    }
    if quality is not None:
        materialization_identity['requested_augmented_samples'] = quality['requested_count']
        materialization_identity['quality_selection'] = quality
    materialization_created_at, materialization_resumed = (
        prepare_materialization_root(output_root, materialization_identity)
    )

    augmented = {}
    asset_records = []
    for name in selected_names:
        optimization_dir = sample_sources[name]
        record = plan_records[name]
        p1_path = optimization_dir / f'{name}_p1.npz'
        p2_path = optimization_dir / f'{name}_p2.npz'
        sidecar_path = optimization_dir / f'{name}.json'
        sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
        if sidecar.get('output_p1_sha256') != sha256_file(p1_path):
            raise ValueError(f'{name}: p1 output hash mismatch')
        if sidecar.get('output_p2_sha256') != sha256_file(p2_path):
            raise ValueError(f'{name}: p2 output hash mismatch')
        if sidecar.get('plan_record_sha256') != canonical_sha256(record):
            raise ValueError(f'{name}: plan record hash mismatch')
        raw, betas, genders = output_raw(p1_path, p2_path)
        if raw.shape != (record['crop_frames'], 56, 6):
            raise ValueError(f'{name}: optimized raw shape differs from plan')
        timing = timing_crop(record)
        if len(timing) != len(raw):
            raise ValueError(f'{name}: materialized timing/motion mismatch')
        augmented[name] = raw
        beta_path = output_root / 'betas' / f'{name}.npz'
        contact_path = output_root / 'contact' / f'{name}.npy'
        text_path = output_root / 'texts_processed' / f'{name}.txt'
        atomic_npz(beta_path, betas=betas, gender=genders)
        atomic_npy(contact_path, timing)
        shutil.copy2(record['text_file'], text_path)
        asset_records.append({
            'augmented_name': name,
            'source_name': record['source_name'],
            'motion_frames': len(raw),
            'motion_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
            'betas': betas.tolist(),
            'genders': genders.tolist(),
            'beta_file_sha256': sha256_file(beta_path),
            'contact_file_sha256': sha256_file(contact_path),
            'text_file_sha256': sha256_file(text_path),
            'optimization_sidecar_sha256': sha256_file(sidecar_path),
        })

    train_names = read_names(args.train_split.resolve())
    val_names = read_names(args.val_split.resolve())
    all_names = read_names(args.all_split.resolve())
    if set(train_names) & set(selected_names) or set(all_names) & set(selected_names):
        raise ValueError('augmented names collide with official split names')
    atomic_text(
        output_root / 'splits' / 'train.txt',
        ''.join(f'{name}\n' for name in train_names + selected_names),
    )
    atomic_text(
        output_root / 'splits' / 'val.txt',
        ''.join(f'{name}\n' for name in val_names),
    )
    atomic_text(
        output_root / 'splits' / 'all.txt',
        ''.join(f'{name}\n' for name in all_names + selected_names),
    )
    build_link_h5(
        output_root / 'motions' / 'train.h5', args.train_motion.resolve(),
        train_names, augmented,
    )
    build_link_h5(
        output_root / 'motions' / 'all.h5', args.all_motion.resolve(),
        all_names, augmented,
    )

    mean_source = args.stats_dir / f'{args.stats_prefix}_Mean.npy'
    std_source = args.stats_dir / f'{args.stats_prefix}_Std.npy'
    root_mean_source = args.stats_dir / f'{args.stats_prefix}_Mean_root.npy'
    root_std_source = args.stats_dir / f'{args.stats_prefix}_Std_root.npy'
    mean_output = output_root / f'{args.stats_prefix}_Mean.npy'
    std_output = output_root / f'{args.stats_prefix}_Std.npy'
    root_mean_output = output_root / f'{args.stats_prefix}_Mean_root.npy'
    root_std_output = output_root / f'{args.stats_prefix}_Std_root.npy'
    shutil.copy2(mean_source, mean_output)
    shutil.copy2(std_source, std_output)
    shutil.copy2(root_mean_source, root_mean_output)
    shutil.copy2(root_std_source, root_std_output)
    mean_source_hash = sha256_file(mean_source)
    std_source_hash = sha256_file(std_source)
    root_mean_source_hash = sha256_file(root_mean_source)
    root_std_source_hash = sha256_file(root_std_source)
    mean_output_hash = sha256_file(mean_output)
    std_output_hash = sha256_file(std_output)
    root_mean_output_hash = sha256_file(root_mean_output)
    root_std_output_hash = sha256_file(root_std_output)
    if (
        mean_source_hash != mean_output_hash
        or std_source_hash != std_output_hash
        or root_mean_source_hash != root_mean_output_hash
        or root_std_source_hash != root_std_output_hash
    ):
        raise RuntimeError('normalization statistics were not copied byte-identically')
    manifest = {
        'schema': SCHEMA,
        'status': 'complete',
        'created_at_utc': materialization_created_at,
        'resumed_from_incomplete_build': materialization_resumed,
        'materialization_identity_sha256': canonical_sha256(
            materialization_identity
        ),
        'output_root': str(output_root),
        'optimization_completions': [
            {
                'path': str(path),
                'sha256': sha256_file(path),
            }
            for path, _ in completions
        ],
        'plan': str(args.plan.resolve()),
        'plan_sha256': sha256_file(args.plan),
        'original': {
            'train_split': str(args.train_split.resolve()),
            'train_split_sha256': sha256_file(args.train_split),
            'val_split': str(args.val_split.resolve()),
            'val_split_sha256': sha256_file(args.val_split),
            'all_split': str(args.all_split.resolve()),
            'all_split_sha256': sha256_file(args.all_split),
            'train_motion': str(args.train_motion.resolve()),
            'train_motion_sha256': sha256_file(args.train_motion),
            'all_motion': str(args.all_motion.resolve()),
            'all_motion_sha256': sha256_file(args.all_motion),
        },
        'counts': {
            'official_train': len(train_names),
            'augmented': len(selected_names),
            'combined_train': len(train_names) + len(selected_names),
            'official_all': len(all_names),
            'combined_all': len(all_names) + len(selected_names),
        },
        'augmentation_mode': augmentation_mode,
        'splits': {
            label: {
                'path': str(output_root / 'splits' / f'{label}.txt'),
                'sha256': sha256_file(output_root / 'splits' / f'{label}.txt'),
            }
            for label in ('train', 'val', 'all')
        },
        'normalization': {
            'policy': 'reuse_original_raw_and_root_interx_stats_no_recompute',
            'generation_and_decode_representation': 'raw',
            'mgt_tokenizer_contract': 'deterministic_historical_root_v1',
            'source_mean': str(mean_source.resolve()),
            'source_mean_sha256': mean_source_hash,
            'source_std': str(std_source.resolve()),
            'source_std_sha256': std_source_hash,
            'mean': str(mean_output),
            'mean_sha256': mean_output_hash,
            'std': str(std_output),
            'std_sha256': std_output_hash,
            'source_root_mean': str(root_mean_source.resolve()),
            'source_root_mean_sha256': root_mean_source_hash,
            'source_root_std': str(root_std_source.resolve()),
            'source_root_std_sha256': root_std_source_hash,
            'root_mean': str(root_mean_output),
            'root_mean_sha256': root_mean_output_hash,
            'root_std': str(root_std_output),
            'root_std_sha256': root_std_output_hash,
            'copy_is_byte_identical': True,
        },
        'motions': {
            'train': str(output_root / 'motions' / 'train.h5'),
            'train_sha256': sha256_file(output_root / 'motions' / 'train.h5'),
            'all': str(output_root / 'motions' / 'all.h5'),
            'all_sha256': sha256_file(output_root / 'motions' / 'all.h5'),
            'official_samples_are_hdf5_external_links': True,
        },
        'asset_records': asset_records,
        'implementation': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(__file__),
        },
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    if len(completions) == 1:
        # Retain the released single-directory fields for existing consumers;
        # the list above is authoritative for new sharded materializations.
        manifest['optimization_completion'] = str(completions[0][0])
        manifest['optimization_completion_sha256'] = sha256_file(
            completions[0][0]
        )
    if quality is not None:
        manifest['quality_selection'] = quality
    manifest_path = output_root / 'materialization_manifest.json'
    atomic_json(manifest_path, manifest)
    atomic_json(output_root / '_materialization_status.json', {
        'schema': SCHEMA,
        'status': 'complete',
        'manifest': str(manifest_path),
        'manifest_sha256': sha256_file(manifest_path),
    })
    print(json.dumps({
        'output_root': str(output_root),
        'official_train': len(train_names),
        'augmented': len(selected_names),
        'combined_train': len(train_names) + len(selected_names),
        'augmentation_mode': augmentation_mode,
        'normalization_policy': manifest['normalization']['policy'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
