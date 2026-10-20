"""Strict provenance validation for frozen Inter-X shape-augmentation plans."""

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np


PLAN_SCHEMA = 'hhc_interx_shape_augmentation_plan_v2'
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


def _names(path):
    names = [
        line.strip() for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(names) != len(set(names)):
        raise ValueError(f'duplicate names in {path}')
    return names


def _gender(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    value = str(value)
    if value not in ('male', 'female'):
        raise ValueError(f'unsupported gender {value!r}')
    return value


def _first_caption(path):
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            return line.split('#', 1)[0].strip()
    raise ValueError(f'empty caption file: {path}')


def _action_id(name):
    match = re.search(r'A(\d{3})', name)
    if not match:
        raise ValueError(f'cannot parse action id from {name}')
    return match.group(1)


def validate_interx_shape_augmentation_plan(plan_path):
    """Validate a v2 plan and every source it will feed to optimization."""
    plan_path = Path(plan_path).resolve()
    manifest = json.loads(plan_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != PLAN_SCHEMA or manifest.get('status') != 'complete':
        raise ValueError('augmentation plan is not a complete supported v2 plan')
    donor_pool_kind = manifest.get('donor_pool_kind', 'interx_train_native')
    shape_sampling = {
        'interx_train_native': 'uniform_over_unique_train_person_shapes',
        'humanml3d_train_amass_transferred_smplx': (
            'uniform_over_unique_humanml3d_train_amass_smplx_shapes'
        ),
        'ssp3d_transferred_smplx': 'uniform_over_unique_same_gender_ssp3d_smplx_shapes',
    }.get(donor_pool_kind)
    if shape_sampling is None:
        raise ValueError(f'unsupported donor pool kind: {donor_pool_kind!r}')
    expected_policy = {
        'shape_sampling': shape_sampling,
        'two_different_donor_shapes_per_augmented_pair': True,
        'source_shape_excluded': True,
        'crop': 'max_contact_density_else_seeded_uniform',
        'timing': 'require_explicit_manual_binary_annotation_file',
        'caption': 'first_processed_caption',
    }
    if manifest.get('policy') != expected_policy:
        raise ValueError('augmentation plan policy differs from frozen v2 policy')

    hash_cache = {}

    def require_file(path, expected_hash, label):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'{label} is missing: {path}')
        key = str(path)
        if key not in hash_cache:
            hash_cache[key] = sha256_file(path)
        actual = hash_cache[key]
        if actual != expected_hash:
            raise ValueError(f'{label} hash mismatch: {actual} != {expected_hash}')
        return path

    inputs = manifest['inputs']
    split = require_file(inputs['split'], inputs['split_sha256'], 'train split')
    motion_file = require_file(
        inputs['motion_file'], inputs['motion_file_sha256'], 'train motion HDF5',
    )
    records_path = require_file(
        manifest['records'], manifest['records_sha256'], 'augmentation records',
    )
    augmented_split = require_file(
        manifest['augmented_split'], manifest['augmented_split_sha256'],
        'augmentation-name split',
    )
    official_names = _names(split)
    official_index = {name: index for index, name in enumerate(official_names)}
    if manifest.get('official_train_samples') != len(official_names):
        raise ValueError('official train count differs from frozen split')

    records = []
    for line_number, line in enumerate(
        records_path.read_text(encoding='utf-8').splitlines(), start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('schema') != RECORD_SCHEMA:
            raise ValueError(f'{records_path}:{line_number}: unsupported record schema')
        records.append(row)
    augmented_names = [row.get('augmented_name') for row in records]
    source_names = [row.get('source_name') for row in records]
    if len(augmented_names) != len(set(augmented_names)):
        raise ValueError('duplicate augmented names in plan')
    if len(source_names) != len(set(source_names)):
        raise ValueError('duplicate source names in plan')
    if augmented_names != _names(augmented_split):
        raise ValueError('augmentation-name split differs from record order')
    if manifest.get('eligible_samples') != len(records):
        raise ValueError('eligible sample count differs from plan records')

    excluded = manifest.get('excluded')
    if not isinstance(excluded, list):
        raise ValueError('plan excluded records are missing')
    excluded_names = [row.get('sample_name') for row in excluded]
    if len(excluded_names) != len(set(excluded_names)):
        raise ValueError('duplicate excluded source names in plan')
    if set(source_names) & set(excluded_names):
        raise ValueError('a source is both eligible and excluded')
    if set(source_names) | set(excluded_names) != set(official_names):
        raise ValueError('eligible and excluded sources do not partition train split')
    if manifest.get('excluded_samples') != len(excluded):
        raise ValueError('excluded sample count differs from records')
    reason_counts = {
        reason: sum(row.get('reason') == reason for row in excluded)
        for reason in (
            'missing_manual_contact_timing', 'outside_training_eligibility',
        )
    }
    if manifest.get('excluded_missing_manual_timing') != reason_counts[
            'missing_manual_contact_timing']:
        raise ValueError('missing-timing exclusion count differs')
    if manifest.get('excluded_outside_training_eligibility') != reason_counts[
            'outside_training_eligibility']:
        raise ValueError('length exclusion count differs')
    if any(row.get('reason') not in reason_counts for row in excluded):
        raise ValueError('plan contains an unsupported exclusion reason')

    beta_cache = {}

    def beta_payload(path, expected_hash, label):
        path = require_file(path, expected_hash, label)
        key = str(path)
        if key not in beta_cache:
            with np.load(path, allow_pickle=True) as payload:
                beta_cache[key] = (
                    np.asarray(payload['betas'], dtype=np.float32).reshape(2, 10),
                    [_gender(value) for value in payload['gender']],
                )
        return beta_cache[key]

    shape_pool = manifest.get('shape_pool')
    if not isinstance(shape_pool, list):
        raise ValueError('shape pool is missing')
    pool_by_id = {}
    if donor_pool_kind == 'interx_train_native':
        for shape in shape_pool:
            shape_id = shape.get('shape_id')
            if shape_id in pool_by_id:
                raise ValueError(f'duplicate shape id in pool: {shape_id}')
            betas, genders = beta_payload(
                shape['beta_file'], shape['beta_file_sha256'],
                f'shape-pool source {shape.get("source_name")}',
            )
            person = int(shape['source_person'])
            if person not in (0, 1):
                raise ValueError(f'invalid shape-pool source person: {person}')
            recorded_betas = np.asarray(shape['betas'], dtype=np.float32)
            if not np.array_equal(recorded_betas, betas[person]):
                raise ValueError(f'{shape_id}: shape-pool beta values changed')
            if shape['gender'] != genders[person]:
                raise ValueError(f'{shape_id}: shape-pool gender changed')
            recomputed_id = canonical_sha256({
                'gender': shape['gender'],
                'betas': [float(value) for value in recorded_betas],
            })
            if recomputed_id != shape_id:
                raise ValueError(f'{shape_id}: shape-pool identity is invalid')
            pool_by_id[shape_id] = shape
    else:
        donor_manifest_path = require_file(
            inputs['external_donor_manifest'],
            inputs['external_donor_manifest_sha256'], 'external donor manifest',
        )
        donor_manifest = json.loads(donor_manifest_path.read_text(encoding='utf-8'))
        expected_donor_schema = ('hhc_ssp3d_smplx_shape_donor_pool_v1'
                                 if donor_pool_kind == 'ssp3d_transferred_smplx'
                                 else EXTERNAL_DONOR_SCHEMA)
        if (donor_manifest.get('schema') != expected_donor_schema
                or donor_manifest.get('status') != 'complete'):
            raise ValueError('external donor manifest is invalid')
        donor_artifact = inputs.get('external_donor_npz')
        if donor_artifact != donor_manifest.get('donor_pool'):
            raise ValueError('plan and external donor manifest bind different NPZ')
        donor_path = require_file(
            donor_artifact['path'], donor_artifact['sha256'], 'external donor NPZ',
        )
        if donor_path.stat().st_size != donor_artifact['bytes']:
            raise ValueError('external donor NPZ byte size differs')
        with np.load(donor_path, allow_pickle=False) as donor_npz:
            shape_ids = np.asarray(donor_npz['shape_id']).astype(str)
            genders = np.asarray(donor_npz['gender']).astype(str)
            donor_betas = np.asarray(donor_npz['betas'], dtype=np.float32)
            residuals = np.asarray(
                donor_npz['transfer_vertex_rmse_m'], dtype=np.float64,
            )
        if len(shape_pool) != len(shape_ids):
            raise ValueError('external shape pool size differs from donor NPZ')
        if donor_pool_kind == 'ssp3d_transferred_smplx' and (
                donor_betas.shape != (len(shape_ids), 10)
                or not np.isfinite(donor_betas).all()
                or not np.isfinite(residuals).all()
                or np.any((residuals < 0) | (residuals > .01))):
            raise ValueError('SSP pool requires finite 10D betas and transfer RMSE <= 0.01 m')
        for shape in shape_pool:
            shape_id = shape.get('shape_id')
            if shape_id in pool_by_id:
                raise ValueError(f'duplicate shape id in pool: {shape_id}')
            index = int(shape.get('donor_index', -1))
            if not 0 <= index < len(shape_ids) or shape_ids[index] != shape_id:
                raise ValueError(f'{shape_id}: donor index/id differs from NPZ')
            recorded_betas = np.asarray(shape['betas'], dtype=np.float32)
            if (shape.get('donor_pool_kind') != donor_pool_kind
                    or shape.get('donor_file') != str(donor_path)
                    or shape.get('donor_file_sha256') != donor_artifact['sha256']
                    or not np.array_equal(recorded_betas, donor_betas[index])
                    or shape.get('gender') != genders[index]
                    or not np.isclose(
                        shape.get('transfer_vertex_rmse_m'), residuals[index])):
                raise ValueError(f'{shape_id}: external donor values changed')
            recomputed_id = canonical_sha256({
                'gender': shape['gender'],
                'betas': [float(value) for value in recorded_betas],
            })
            if recomputed_id != shape_id:
                raise ValueError(f'{shape_id}: external target identity is invalid')
            pool_by_id[shape_id] = shape
    if manifest.get('unique_train_person_shapes') != len(pool_by_id):
        raise ValueError('unique train shape count differs from shape pool')
    if manifest.get('unique_donor_shapes', len(pool_by_id)) != len(pool_by_id):
        raise ValueError('unique donor shape count differs from shape pool')

    thresholds = manifest['thresholds']
    positive_timing_samples = 0
    with h5py.File(motion_file, 'r') as motions:
        for row in records:
            name = row['source_name']
            if name not in official_index or row.get('source_index') != official_index[name]:
                raise ValueError(f'{name}: source index differs from train split')
            expected_augmented = f'{name}__shapeaug_s{int(manifest["seed"]):08d}'
            if row['augmented_name'] != expected_augmented:
                raise ValueError(f'{name}: augmented name differs from plan seed')
            if row.get('action_id') != _action_id(name):
                raise ValueError(f'{name}: action id is invalid')
            if name not in motions:
                raise ValueError(f'{name}: source motion is missing')
            frames = int(motions[name].shape[0])
            if motions[name].shape[1:] != (56, 6) or row['source_frames'] != frames:
                raise ValueError(f'{name}: source motion shape differs from record')
            if not thresholds['min_frames'] <= frames <= thresholds['max_source_frames']:
                raise ValueError(f'{name}: eligible record violates frame thresholds')
            start, stop = int(row['crop_start']), int(row['crop_stop'])
            if not (0 <= start < stop <= frames):
                raise ValueError(f'{name}: crop bounds are invalid')
            if stop - start != row['crop_frames'] or row['crop_frames'] > thresholds[
                    'max_output_frames']:
                raise ValueError(f'{name}: crop length is invalid')

            text_path = require_file(
                row['text_file'], row['text_file_sha256'], f'{name} caption file',
            )
            if row['caption'] != _first_caption(text_path):
                raise ValueError(f'{name}: caption differs from first processed caption')
            if row['caption_sha256'] != hashlib.sha256(
                    row['caption'].encode('utf-8')).hexdigest():
                raise ValueError(f'{name}: caption hash is invalid')

            source_betas, source_genders = beta_payload(
                row['source_beta_file'], row['source_beta_file_sha256'],
                f'{name} source beta',
            )
            if not np.array_equal(
                    source_betas, np.asarray(row['source_betas'], dtype=np.float32)):
                raise ValueError(f'{name}: source beta values changed')
            if source_genders != row['source_genders']:
                raise ValueError(f'{name}: source genders changed')

            timing_path = row.get('timing_file')
            if row.get('timing_source') != 'manual_frame_annotation_file' or not timing_path:
                raise ValueError(f'{name}: explicit manual timing is required')
            timing_path = require_file(
                timing_path, row['timing_sha256'], f'{name} contact timing',
            )
            timing = np.load(timing_path, allow_pickle=False).reshape(-1)
            if len(timing) != frames or not set(np.unique(timing).tolist()) <= {0, 1}:
                raise ValueError(f'{name}: timing is non-binary or wrong length')
            if int(timing.sum()) != row['timing_positive_frames_full']:
                raise ValueError(f'{name}: full timing-positive count differs')
            cropped_positive = int(timing[start:stop].sum())
            if cropped_positive != row['timing_positive_frames_crop']:
                raise ValueError(f'{name}: cropped timing-positive count differs')
            positive_timing_samples += int(cropped_positive > 0)

            donors = row.get('shape_donors')
            if not isinstance(donors, list) or len(donors) != 2:
                raise ValueError(f'{name}: expected exactly two shape donors')
            if sorted(donor.get('target_person') for donor in donors) != [0, 1]:
                raise ValueError(f'{name}: donor target people are invalid')
            donor_ids = [donor.get('shape_id') for donor in donors]
            if len(set(donor_ids)) != 2 or any(
                    shape_id not in pool_by_id for shape_id in donor_ids):
                raise ValueError(f'{name}: donor shapes are duplicated/outside pool')
            for donor in donors:
                pool = pool_by_id[donor['shape_id']]
                donor_keys = (
                    ('source_name', 'source_person', 'gender', 'betas',
                     'beta_file', 'beta_file_sha256')
                    if donor_pool_kind == 'interx_train_native' else
                    ('gender', 'betas', 'donor_pool_kind', 'donor_index',
                     'donor_file', 'donor_file_sha256', 'source_identity_ids',
                     'transfer_vertex_rmse_m')
                )
                if any(donor.get(key) != pool.get(key) for key in donor_keys):
                    raise ValueError(f'{name}: donor differs from frozen shape pool')
                target = int(donor['target_person'])
                if (donor_pool_kind == 'ssp3d_transferred_smplx'
                        and donor['gender'] != source_genders[target]):
                    raise ValueError(f'{name}: SSP donor must preserve source gender')
                source_id = canonical_sha256({
                    'gender': source_genders[target],
                    'betas': [float(value) for value in source_betas[target]],
                })
                if donor['shape_id'] == source_id:
                    raise ValueError(f'{name}: donor repeats target source shape')
                expected_distance = float(np.linalg.norm(
                    np.asarray(donor['betas'], dtype=np.float64)
                    - source_betas[target].astype(np.float64)
                ))
                if not np.isclose(
                        donor['beta_l2_from_source'], expected_distance,
                        rtol=1e-6, atol=1e-7):
                    raise ValueError(f'{name}: donor beta distance is invalid')

    if manifest.get('positive_timing_samples') != positive_timing_samples:
        raise ValueError('positive-timing sample count differs from records')
    return manifest, records_path, records, motion_file, split
