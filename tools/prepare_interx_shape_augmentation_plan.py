#!/usr/bin/env python
"""Freeze deterministic train-only random-shape augmentation assignments.

The plan is deliberately separate from optimization.  It binds every source
motion, manual contact-timing mask, caption, and randomly sampled train shape
before any expensive GPU work starts.
"""

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


SCHEMA = 'hhc_interx_shape_augmentation_plan_v2'
RECORD_SCHEMA = 'hhc_interx_shape_augmentation_record_v2'
EXTERNAL_DONOR_SCHEMA = 'hhc_humanml3d_amass_smplx_shape_donor_pool_v1'


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


def atomic_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    os.replace(temporary, path)


def first_caption(path):
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            return line.split('#', 1)[0].strip()
    raise ValueError(f'empty text file: {path}')


def gender_string(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    value = str(value)
    if value not in ('male', 'female'):
        raise ValueError(f'unsupported gender {value!r}')
    return value


def load_external_shape_pool(manifest_path):
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema') not in (EXTERNAL_DONOR_SCHEMA, 'hhc_ssp3d_smplx_shape_donor_pool_v1')
            or manifest.get('status') != 'complete'):
        raise ValueError('external donor manifest is not complete/supported')
    donor_artifact = manifest.get('donor_pool', {})
    donor_path = Path(donor_artifact.get('path', '')).resolve()
    if (not donor_path.is_file()
            or donor_path.stat().st_size != donor_artifact.get('bytes')
            or sha256_file(donor_path) != donor_artifact.get('sha256')):
        raise ValueError('external donor NPZ artifact is missing or stale')
    with np.load(donor_path, allow_pickle=False) as payload:
        expected = {'shape_id', 'gender', 'betas', 'transfer_vertex_rmse_m'}
        if set(payload.files) != expected:
            raise ValueError(f'external donor fields {set(payload.files)} != {expected}')
        shape_ids = np.asarray(payload['shape_id']).astype(str)
        genders = np.asarray(payload['gender']).astype(str)
        betas = np.asarray(payload['betas'], dtype=np.float32)
        residuals = np.asarray(
            payload['transfer_vertex_rmse_m'], dtype=np.float64,
        )
    count = len(shape_ids)
    if (betas.shape != (count, 10) or genders.shape != (count,)
            or residuals.shape != (count,) or len(set(shape_ids)) != count
            or set(genders) - {'male', 'female'}
            or not np.isfinite(betas).all() or not np.isfinite(residuals).all()
            or count < 3):
        raise ValueError('external donor arrays violate the shape-pool contract')
    record_by_id = {record['shape_id']: record for record in manifest['records']}
    if (manifest['schema'] == 'hhc_ssp3d_smplx_shape_donor_pool_v1'
            and np.any((residuals < 0) | (residuals > .01))):
        raise ValueError('SSP transfer residual must be within 0..0.01 m')
    if set(record_by_id) != set(shape_ids):
        raise ValueError('external donor manifest records differ from NPZ shape ids')
    shape_pool = []
    for index, (shape_id, gender, beta, residual) in enumerate(
            zip(shape_ids, genders, betas, residuals)):
        recomputed = canonical_sha256({
            'gender': gender, 'betas': [float(value) for value in beta],
        })
        if recomputed != shape_id:
            raise ValueError(f'external target SMPL-X shape id is invalid: {shape_id}')
        record = record_by_id[shape_id]
        if (record['gender'] != gender
                or not np.isclose(record['transfer_vertex_rmse_m'], residual)):
            raise ValueError(f'external donor record differs for {shape_id}')
        shape_pool.append({
            'shape_id': shape_id,
            'gender': gender,
            'betas': beta.tolist(),
            'donor_pool_kind': ('ssp3d_transferred_smplx'
                                if manifest['schema'] == 'hhc_ssp3d_smplx_shape_donor_pool_v1'
                                else 'humanml3d_train_amass_transferred_smplx'),
            'donor_index': index,
            'donor_file': str(donor_path),
            'donor_file_sha256': donor_artifact['sha256'],
            'source_identity_ids': record['source_identity_ids'],
            'transfer_vertex_rmse_m': float(residual),
        })
    return sorted(shape_pool, key=lambda row: row['shape_id']), manifest, manifest_path


def action_id(name):
    match = re.search(r'A(\d{3})', name)
    if not match:
        raise ValueError(f'cannot parse action id from {name}')
    return match.group(1)


def select_crop(mask, length, max_frames, rng):
    if length <= max_frames:
        return 0, length, 'full_sequence'
    mask = np.asarray(mask, dtype=np.int64).reshape(-1)
    if np.any(mask):
        counts = np.convolve(mask, np.ones(max_frames, dtype=np.int64), 'valid')
        best = np.flatnonzero(counts == counts.max())
        # Break equivalent-density ties deterministically without always
        # biasing toward the beginning of long recordings.
        start = int(best[int(rng.integers(len(best)))])
        return start, max_frames, 'max_contact_density'
    start = int(rng.integers(length - max_frames + 1))
    return start, max_frames, 'seeded_uniform_no_contact'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, default=Path('dataset/interx/splits/train.txt'))
    parser.add_argument('--motion-file', type=Path, default=Path('dataset/interx/motions/train.h5'))
    parser.add_argument('--beta-dir', type=Path, default=Path('dataset/interx/betas'))
    parser.add_argument('--external-donor-manifest', type=Path)
    parser.add_argument('--contact-dir', type=Path, default=Path('dataset/interx/contact'))
    parser.add_argument('--text-dir', type=Path, default=Path('dataset/interx/texts_processed'))
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('configs/interx_shape_augmentation/timing_covered_seed123_v2'),
    )
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--min-frames', type=int, default=30)
    parser.add_argument('--max-source-frames', type=int, default=999)
    parser.add_argument('--max-output-frames', type=int, default=150)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.min_frames <= args.max_output_frames:
        raise ValueError('require 1 <= min-frames <= max-output-frames')
    if args.max_source_frames < args.max_output_frames:
        raise ValueError('max-source-frames must be >= max-output-frames')

    split = args.split.resolve()
    motion_file = args.motion_file.resolve()
    beta_dir = args.beta_dir.resolve()
    contact_dir = args.contact_dir.resolve()
    text_dir = args.text_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        line.strip() for line in split.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(names) != len(set(names)):
        raise ValueError('training split contains duplicate sample names')

    shape_by_name = {}
    unique_shapes = {}
    for name in names:
        beta_path = beta_dir / f'{name}.npz'
        if not beta_path.is_file():
            raise FileNotFoundError(beta_path)
        with np.load(beta_path, allow_pickle=True) as payload:
            betas = np.asarray(payload['betas'], dtype=np.float32).reshape(2, 10)
            genders = [gender_string(value) for value in payload['gender']]
        shape_by_name[name] = (betas, genders, beta_path)
        for person in range(2):
            identity = canonical_sha256({
                'gender': genders[person],
                'betas': [float(value) for value in betas[person]],
            })
            unique_shapes.setdefault(identity, {
                'shape_id': identity,
                'source_name': name,
                'source_person': person,
                'gender': genders[person],
                'betas': betas[person].tolist(),
                'beta_file': str(beta_path),
                'beta_file_sha256': sha256_file(beta_path),
            })
    donor_manifest = None
    donor_manifest_path = None
    if args.external_donor_manifest is not None:
        shape_pool, donor_manifest, donor_manifest_path = load_external_shape_pool(
            args.external_donor_manifest
        )
        donor_pool_kind = shape_pool[0]['donor_pool_kind']
        shape_sampling = ('uniform_over_unique_same_gender_ssp3d_smplx_shapes'
                          if donor_pool_kind == 'ssp3d_transferred_smplx'
                          else 'uniform_over_unique_humanml3d_train_amass_smplx_shapes')
    else:
        shape_pool = sorted(unique_shapes.values(), key=lambda row: row['shape_id'])
        donor_pool_kind = 'interx_train_native'
        shape_sampling = 'uniform_over_unique_train_person_shapes'
    if len(shape_pool) < 3:
        raise ValueError('need at least three unique train shapes')

    action_file_counts = defaultdict(lambda: [0, 0])
    for name in names:
        counts = action_file_counts[action_id(name)]
        counts[0] += 1
        counts[1] += int((contact_dir / f'{name}.npy').is_file())
    rng = np.random.default_rng(args.seed)
    records = []
    excluded = []
    with h5py.File(motion_file, 'r') as motions:
        for source_index, name in enumerate(names):
            if name not in motions:
                raise KeyError(f'{name} is absent from {motion_file}')
            dataset = motions[name]
            if dataset.ndim != 3 or dataset.shape[1:] != (56, 6):
                raise ValueError(f'{name}: unexpected motion shape {dataset.shape}')
            length = int(dataset.shape[0])
            if length < args.min_frames or length > args.max_source_frames:
                excluded.append({
                    'sample_name': name,
                    'frames': length,
                    'reason': 'outside_training_eligibility',
                })
                continue

            contact_path = contact_dir / f'{name}.npy'
            if not contact_path.is_file():
                # Whole action classes being absent does not mean no contact:
                # A006/A013/A017/A020/A024/A029/A037/A038 contain explicit
                # inter-human contact descriptions.  Missing labels therefore
                # cannot be materialized as negative timing supervision.
                excluded.append({
                    'sample_name': name,
                    'action_id': action_id(name),
                    'frames': length,
                    'reason': 'missing_manual_contact_timing',
                })
                continue
            timing = np.load(contact_path, allow_pickle=False).reshape(-1)
            if len(timing) != length:
                raise ValueError(
                    f'{name}: contact timing length {len(timing)} != {length}'
                )
            unique_values = set(np.unique(timing).tolist())
            if not unique_values <= {0, 1}:
                raise ValueError(f'{name}: non-binary contact timing {unique_values}')
            timing = timing.astype(np.uint8)
            timing_source = 'manual_frame_annotation_file'
            timing_sha256 = sha256_file(contact_path)

            crop_start, crop_length, crop_policy = select_crop(
                timing, length, args.max_output_frames, rng,
            )
            crop_stop = crop_start + crop_length
            source_betas, source_genders, beta_path = shape_by_name[name]
            donors = []
            used_shape_ids = set()
            for person in range(2):
                source_identity = canonical_sha256({
                    'gender': source_genders[person],
                    'betas': [float(value) for value in source_betas[person]],
                })
                choices = [
                    item for item in shape_pool
                    if item['shape_id'] != source_identity
                    and item['shape_id'] not in used_shape_ids
                    and (donor_pool_kind != 'ssp3d_transferred_smplx'
                         or item['gender'] == source_genders[person])
                ]
                if not choices:
                    raise ValueError(f'{name}: no distinct eligible donor for person {person}')
                donor = choices[int(rng.integers(len(choices)))]
                used_shape_ids.add(donor['shape_id'])
                donors.append({
                    **donor,
                    'target_person': person,
                    'beta_l2_from_source': float(np.linalg.norm(
                        np.asarray(donor['betas']) - source_betas[person]
                    )),
                })

            text_path = text_dir / f'{name}.txt'
            caption = first_caption(text_path)
            augmented_name = f'{name}__shapeaug_s{args.seed:08d}'
            records.append({
                'schema': RECORD_SCHEMA,
                'source_index': source_index,
                'source_name': name,
                'augmented_name': augmented_name,
                'action_id': action_id(name),
                'source_frames': length,
                'crop_start': crop_start,
                'crop_stop': crop_stop,
                'crop_frames': crop_length,
                'crop_policy': crop_policy,
                'caption': caption,
                'caption_sha256': hashlib.sha256(caption.encode('utf-8')).hexdigest(),
                'text_file': str(text_path),
                'text_file_sha256': sha256_file(text_path),
                'source_beta_file': str(beta_path),
                'source_beta_file_sha256': sha256_file(beta_path),
                'source_betas': source_betas.tolist(),
                'source_genders': source_genders,
                'shape_donors': donors,
                'timing_source': timing_source,
                'timing_file': str(contact_path),
                'timing_sha256': timing_sha256,
                'timing_positive_frames_full': int(timing.sum()),
                'timing_positive_frames_crop': int(timing[crop_start:crop_stop].sum()),
            })

    records_path = output_dir / 'augmentation_records.jsonl'
    atomic_jsonl(records_path, records)
    split_path = output_dir / 'augmented_names.txt'
    temporary = split_path.with_suffix(split_path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        ''.join(f'{record["augmented_name"]}\n' for record in records),
        encoding='utf-8',
    )
    os.replace(temporary, split_path)
    manifest = {
        'schema': SCHEMA,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'seed': args.seed,
        'policy': {
            'shape_sampling': shape_sampling,
            'two_different_donor_shapes_per_augmented_pair': True,
            'source_shape_excluded': True,
            'crop': 'max_contact_density_else_seeded_uniform',
            'timing': 'require_explicit_manual_binary_annotation_file',
            'caption': 'first_processed_caption',
        },
        'inputs': {
            'split': str(split),
            'split_sha256': sha256_file(split),
            'motion_file': str(motion_file),
            'motion_file_sha256': sha256_file(motion_file),
            'beta_dir': str(beta_dir),
            'contact_dir': str(contact_dir),
            'text_dir': str(text_dir),
            'external_donor_manifest': (
                str(donor_manifest_path) if donor_manifest_path else None
            ),
            'external_donor_manifest_sha256': (
                sha256_file(donor_manifest_path) if donor_manifest_path else None
            ),
            'external_donor_npz': (
                donor_manifest['donor_pool'] if donor_manifest else None
            ),
        },
        'thresholds': {
            'min_frames': args.min_frames,
            'max_source_frames': args.max_source_frames,
            'max_output_frames': args.max_output_frames,
        },
        'official_train_samples': len(names),
        'eligible_samples': len(records),
        'excluded_samples': len(excluded),
        'excluded_missing_manual_timing': sum(
            row['reason'] == 'missing_manual_contact_timing' for row in excluded
        ),
        'excluded_outside_training_eligibility': sum(
            row['reason'] == 'outside_training_eligibility' for row in excluded
        ),
        'excluded': excluded,
        'unique_train_person_shapes': len(shape_pool),
        'unique_donor_shapes': len(shape_pool),
        'donor_pool_kind': donor_pool_kind,
        'shape_pool': shape_pool,
        'manual_timing_coverage_by_action': {
            action: {
                'split_samples': counts[0],
                'manual_timing_files': counts[1],
            }
            for action, counts in sorted(action_file_counts.items())
        },
        'timing_source_counts': {
            key: sum(record['timing_source'] == key for record in records)
            for key in sorted({record['timing_source'] for record in records})
        },
        'positive_timing_samples': sum(
            record['timing_positive_frames_crop'] > 0 for record in records
        ),
        'records': str(records_path),
        'records_sha256': sha256_file(records_path),
        'augmented_split': str(split_path),
        'augmented_split_sha256': sha256_file(split_path),
        'status': 'complete',
    }
    atomic_json(output_dir / 'plan_manifest.json', manifest)
    print(json.dumps({
        'output_dir': str(output_dir),
        'eligible_samples': len(records),
        'excluded_samples': len(excluded),
        'unique_shapes': len(shape_pool),
        'positive_timing_samples': manifest['positive_timing_samples'],
        'excluded_missing_manual_timing': manifest['excluded_missing_manual_timing'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
