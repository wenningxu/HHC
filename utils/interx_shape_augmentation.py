"""Strict validation for materialized Inter-X shape-augmentation overlays."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


SCHEMA = 'hhc_interx_shape_augmentation_materialization_v1'
PLAN_SCHEMA = 'hhc_interx_shape_augmentation_plan_v2'
PLAN_RECORD_SCHEMA = 'hhc_interx_shape_augmentation_record_v2'
SAMPLE_SCHEMA = 'hhc_interx_optimized_shape_augmentation_sample_v1'
HYBRID_MODE = 'optimized_positive_plus_random_zero'
ALLOWED_MODES = {'optimized', 'random_shape_only', HYBRID_MODE}


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


def _resolved(path):
    return str(Path(path).resolve())


def _require_file(path, expected_hash, label):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{label} is missing: {path}')
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f'{label} hash mismatch: {actual_hash} != {expected_hash}'
        )
    return path


def _names(path):
    values = [
        line.strip() for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(values) != len(set(values)):
        raise ValueError(f'duplicate sample names in {path}')
    return values


def _gender(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return str(value)


def validated_quality_selection(entries, completions, requested_names, semantic_exclusions=None):
    """Rebuild bound two-estimator gates; never relabel filtered inputs as requests."""
    from tools.build_shape_volume_quality_gate import build_volume_gate

    expected = {str(path.resolve()): completion for path, completion in completions}
    if not entries or len(entries) != len(expected):
        raise ValueError('One quality gate is required per optimization completion')
    seen, accepted, policies = set(), set(), set()
    for entry in entries:
        path = _require_file(entry['path'], entry['sha256'], 'physical quality gate')
        gate = json.loads(path.read_text())
        completion_path = _resolved(gate['completion'])
        if completion_path not in expected or completion_path in seen:
            raise ValueError('Quality gate completion is duplicate or outside this build')
        seen.add(completion_path)
        _require_file(gate['report'], gate['report_sha256'], 'physical report')
        if gate['schema'] == 'hhc_physical_mesh_ground_quality_gate_v1':
            from tools.build_physical_shape_quality_gate import build_physical_gate
            rebuilt = build_physical_gate(gate['report'], gate['completion'],
                                          gate['candidate_dir'], gate['floor_report'])
            policies.add('physical_original_gt_volumetric_no_tolerance')
        else:
            rebuilt = build_volume_gate(gate['report'], gate['condition'], gate['references'],
                                        gate['gt_dir'], gate['candidate_dir'])
            policies.add('original_gt_two_estimator_intersection_no_tolerance')
        if gate != rebuilt:
            raise ValueError('Quality gate no longer matches its evidence/decisions')
        accepted.update(gate['accepted_names'])
    if len(policies) != 1:
        raise ValueError('Cannot mix old-coordinate and physical-coordinate quality policies')
    completion_names = [name for c in expected.values() for name in c['completed_names']]
    if (len(requested_names) != len(set(requested_names))
            or len(completion_names) != len(requested_names)
            or set(completion_names) != set(requested_names)):
        raise ValueError('Quality selection must preserve the full requested denominator')
    review_metadata = {}
    if semantic_exclusions is not None:
        path = _require_file(semantic_exclusions['path'], semantic_exclusions['sha256'],
                             'semantic exclusions')
        review = json.loads(path.read_text())
        if review.get('status') != 'withhold_pending_semantic_review':
            raise ValueError('Unsupported semantic exclusion status')
        rows = review['entries']
        if not rows or any(not isinstance(r.get('reason'), str) or not r['reason'].strip()
                           for r in rows):
            raise ValueError('Semantic exclusions require nonempty reasons')
        excluded = [r['augmented_name'] for r in rows]
        if len(excluded) != len(set(excluded)) or set(excluded) - set(requested_names):
            raise ValueError('Semantic exclusions are duplicate or outside requested samples')
        review_metadata = {
            'semantic_exclusions': semantic_exclusions,
            'semantic_scope': 'Known-risk exclusions only; remaining samples are not certified semantically correct.',
            'physical_accepted_names': [n for n in requested_names if n in accepted],
            'semantic_excluded_names': excluded,
        }
        accepted.difference_update(excluded)
    return {
        **review_metadata,
        'policy': next(iter(policies)),
        'gates': entries,
        'requested_names': requested_names,
        'requested_count': len(requested_names),
        'accepted_names': [n for n in requested_names if n in accepted],
        'rejected_names': [n for n in requested_names if n not in accepted],
        'accepted_count': len(accepted),
        'pass_fraction': len(accepted) / len(requested_names),
    }


def validate_materialized_interx_shape_augmentation(
    output_root,
    manifest_path,
    verify_augmented_motion_content=True,
):
    """Validate all training-facing assets and return the parsed manifest."""
    output_root = Path(output_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete':
        raise ValueError('augmentation manifest is not complete/supported')
    if _resolved(manifest.get('output_root', '')) != str(output_root):
        raise ValueError('augmentation manifest output_root differs from requested root')
    mode = manifest.get('augmentation_mode')
    if mode not in ALLOWED_MODES:
        raise ValueError(f'unsupported augmentation mode: {mode!r}')

    plan_path = _require_file(
        manifest['plan'], manifest['plan_sha256'], 'augmentation plan manifest',
    )
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('schema') != PLAN_SCHEMA or plan.get('status') != 'complete':
        raise ValueError('augmentation plan is not complete/supported')
    plan_records_path = _require_file(
        plan['records'], plan['records_sha256'], 'augmentation plan records',
    )
    plan_records = {}
    for line_number, line in enumerate(
        plan_records_path.read_text(encoding='utf-8').splitlines(), start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('schema') != PLAN_RECORD_SCHEMA:
            raise ValueError(
                f'{plan_records_path}:{line_number}: unsupported plan record'
            )
        name = row.get('augmented_name')
        if name in plan_records:
            raise ValueError(f'duplicate augmented name in plan records: {name}')
        plan_records[name] = row

    completion_entries = manifest.get('optimization_completions')
    if completion_entries is None:
        completion_entries = [{
            'path': manifest['optimization_completion'],
            'sha256': manifest['optimization_completion_sha256'],
        }]
    if not isinstance(completion_entries, list) or not completion_entries:
        raise ValueError('materialization has no optimization completions')
    completions = []
    completion_modes = set()
    completion_by_name = {}
    positioned_names = []
    unpositioned_names = []
    for entry in completion_entries:
        completion_path = _require_file(
            entry['path'], entry['sha256'], 'optimization completion manifest',
        )
        completion = json.loads(completion_path.read_text(encoding='utf-8'))
        if not completion.get('complete'):
            raise ValueError('optimization completion is no longer complete')
        completion_mode = completion.get('optimizer', {}).get('mode')
        completion_modes.add(completion_mode)
        if mode == HYBRID_MODE:
            selection = completion.get('selection', {})
            if completion_mode == 'optimized':
                valid_component = (
                    selection.get('positive_timing_only') is True
                    and selection.get('zero_timing_only') is not True
                )
            elif completion_mode == 'random_shape_only':
                valid_component = (
                    selection.get('zero_timing_only') is True
                    and selection.get('positive_timing_only') is not True
                )
            else:
                valid_component = False
            if not valid_component:
                raise ValueError(
                    'hybrid materialization contains an invalid timing/mode component'
                )
        elif completion_mode != mode:
            raise ValueError('materialization and optimization modes differ')
        if _resolved(completion.get('plan', '')) != str(plan_path):
            raise ValueError(
                'optimization completion and materialization use different plans'
            )
        if completion.get('plan_sha256') != manifest['plan_sha256']:
            raise ValueError(
                'optimization completion plan hash differs from materialization'
            )
        if _resolved(completion.get('plan_records', '')) != str(plan_records_path):
            raise ValueError(
                'optimization completion references different plan records'
            )
        if completion.get('plan_records_sha256') != plan['records_sha256']:
            raise ValueError('optimization completion plan-record hash differs')
        names = completion.get('completed_names')
        positions = completion.get('selection', {}).get(
            'full_selection_positions'
        )
        if positions is not None and len(positions) != len(names):
            raise ValueError('optimization completion names/positions differ')
        for index, name in enumerate(names):
            if name in completion_by_name:
                raise ValueError(f'duplicate optimized sample across shards: {name}')
            completion_by_name[name] = (completion_path, completion)
            if positions is None:
                unpositioned_names.append(name)
            else:
                positioned_names.append((int(positions[index]), name))
        completions.append((completion_path, completion))
    if mode == HYBRID_MODE and completion_modes != {
            'optimized', 'random_shape_only'}:
        raise ValueError('hybrid materialization must contain both component modes')
    if positioned_names and unpositioned_names:
        raise ValueError('optimization shards mix positioned and legacy manifests')
    if positioned_names:
        positions = [position for position, _ in positioned_names]
        if len(set(positions)) != len(positions) or min(positions) < 0:
            raise ValueError('Optimization positions must be unique and nonnegative')
        if not manifest.get('quality_selection') and sorted(positions) != list(range(len(positions))):
            raise ValueError('optimization shard positions are not contiguous')
        completed_names = [
            name for _, name in sorted(positioned_names)
        ]
    else:
        completed_names = unpositioned_names
    if manifest.get('quality_selection'):
        quality = validated_quality_selection(manifest['quality_selection']['gates'],
                                               completions, completed_names,
                                               manifest['quality_selection'].get('semantic_exclusions'))
        if quality != manifest['quality_selection']:
            raise ValueError('Materialized quality selection differs from bound gates')
        completed_names = quality['accepted_names']

    original = manifest['original']
    original_paths = {}
    for label in ('train_split', 'val_split', 'all_split', 'train_motion', 'all_motion'):
        original_paths[label] = _require_file(
            original[label], original[f'{label}_sha256'], f'original {label}'
        )

    split_paths = {}
    split_names = {}
    for label in ('train', 'val', 'all'):
        entry = manifest['splits'][label]
        expected_path = output_root / 'splits' / f'{label}.txt'
        if _resolved(entry['path']) != str(expected_path):
            raise ValueError(f'{label} split is outside the materialization root')
        split_paths[label] = _require_file(
            expected_path, entry['sha256'], f'materialized {label} split'
        )
        split_names[label] = _names(split_paths[label])

    asset_records = manifest.get('asset_records')
    if not isinstance(asset_records, list):
        raise ValueError('asset_records is not a list')
    augmented_names = [row.get('augmented_name') for row in asset_records]
    if any(not isinstance(name, str) or not name for name in augmented_names):
        raise ValueError('asset record has an invalid augmented name')
    if len(augmented_names) != len(set(augmented_names)):
        raise ValueError('duplicate augmented names in asset records')
    augmented_set = set(augmented_names)
    if completed_names != augmented_names:
        raise ValueError(
            'asset record order/names differ from optimization completions'
        )
    if any(name not in plan_records for name in augmented_names):
        raise ValueError('materialized sample is absent from the frozen plan')

    original_train = _names(original_paths['train_split'])
    original_val = _names(original_paths['val_split'])
    original_all = _names(original_paths['all_split'])
    if split_names['train'] != original_train + augmented_names:
        raise ValueError('combined train split is not official train plus augmentation')
    if split_names['val'] != original_val:
        raise ValueError('validation split changed under training augmentation')
    if split_names['all'] != original_all + augmented_names:
        raise ValueError('combined all split is not official all plus augmentation')

    counts = manifest['counts']
    expected_counts = {
        'official_train': len(original_train),
        'augmented': len(augmented_names),
        'combined_train': len(split_names['train']),
        'official_all': len(original_all),
        'combined_all': len(split_names['all']),
    }
    mismatched_counts = {
        key: (counts.get(key), value) for key, value in expected_counts.items()
        if counts.get(key) != value
    }
    if mismatched_counts:
        raise ValueError(f'materialization counts differ: {mismatched_counts}')

    normalization = manifest['normalization']
    if (
        normalization.get('policy')
        != 'reuse_original_raw_and_root_interx_stats_no_recompute'
        or normalization.get('generation_and_decode_representation') != 'raw'
        or normalization.get('mgt_tokenizer_contract')
        != 'deterministic_historical_root_v1'
    ):
        raise ValueError('augmentation changed the frozen Inter-X representation policy')
    for label, filename in (
        ('mean', 'Interx_Mean.npy'),
        ('std', 'Interx_Std.npy'),
        ('root_mean', 'Interx_Mean_root.npy'),
        ('root_std', 'Interx_Std_root.npy'),
    ):
        expected_path = output_root / filename
        if _resolved(normalization[label]) != str(expected_path):
            raise ValueError(f'{label} statistics are outside the materialization root')
        _require_file(expected_path, normalization[f'{label}_sha256'], label)

    motion_paths = {}
    for label in ('train', 'all'):
        entry = manifest['motions'][label]
        expected_path = output_root / 'motions' / f'{label}.h5'
        if _resolved(entry) != str(expected_path):
            raise ValueError(f'{label} motion HDF5 is outside the materialization root')
        motion_paths[label] = _require_file(
            expected_path, manifest['motions'][f'{label}_sha256'],
            f'materialized {label} motion HDF5',
        )

    records = {row['augmented_name']: row for row in asset_records}
    for name, record in records.items():
        completion_path, completion = completion_by_name[name]
        sidecar_path = _require_file(
            completion_path.parent / f'{name}.json',
            record['optimization_sidecar_sha256'], f'{name} optimization sidecar',
        )
        sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
        expected_sidecar = {
            'schema': SAMPLE_SCHEMA,
            'source_name': record['source_name'],
            'augmented_name': name,
            'run_contract_sha256': completion.get('contract_sha256'),
            'plan_record_sha256': canonical_sha256(plan_records[name]),
        }
        mismatched_sidecar = {
            key: (sidecar.get(key), value)
            for key, value in expected_sidecar.items()
            if sidecar.get(key) != value
        }
        if mismatched_sidecar:
            raise ValueError(
                f'{name}: optimization sidecar mismatch {mismatched_sidecar}'
            )
        _require_file(
            completion_path.parent / f'{name}_p1.npz',
            sidecar.get('output_p1_sha256'), f'{name} optimizer person 1 output',
        )
        _require_file(
            completion_path.parent / f'{name}_p2.npz',
            sidecar.get('output_p2_sha256'), f'{name} optimizer person 2 output',
        )
        beta_path = _require_file(
            output_root / 'betas' / f'{name}.npz',
            record['beta_file_sha256'], f'{name} beta',
        )
        contact_path = _require_file(
            output_root / 'contact' / f'{name}.npy',
            record['contact_file_sha256'], f'{name} contact timing',
        )
        _require_file(
            output_root / 'texts_processed' / f'{name}.txt',
            record['text_file_sha256'], f'{name} text',
        )
        with np.load(beta_path, allow_pickle=True) as payload:
            betas = np.asarray(payload['betas'], dtype=np.float32).reshape(2, 10)
            genders = [_gender(value) for value in payload['gender']]
        if not np.array_equal(betas, np.asarray(record['betas'], dtype=np.float32)):
            raise ValueError(f'{name}: beta payload differs from asset record')
        if genders != record['genders']:
            raise ValueError(f'{name}: gender payload differs from asset record')
        timing = np.load(contact_path, allow_pickle=False).reshape(-1)
        if len(timing) != record['motion_frames']:
            raise ValueError(f'{name}: timing and motion lengths differ')
        if not set(np.unique(timing).tolist()) <= {0, 1}:
            raise ValueError(f'{name}: contact timing is not binary')

    official_sets = {'train': set(original_train), 'all': set(original_all)}
    expected_sets = {
        'train': set(split_names['train']), 'all': set(split_names['all'])
    }
    for label, path in motion_paths.items():
        with h5py.File(path, 'r') as motions:
            if set(motions.keys()) != expected_sets[label]:
                raise ValueError(f'{label} HDF5 keys differ from its split')
            for name in official_sets[label]:
                if not isinstance(motions.get(name, getlink=True), h5py.ExternalLink):
                    raise ValueError(f'{label}/{name}: official motion is not an external link')
            for name in augmented_set:
                if isinstance(motions.get(name, getlink=True), h5py.ExternalLink):
                    raise ValueError(f'{label}/{name}: augmented motion is an external link')
                dataset = motions[name]
                record = records[name]
                if dataset.shape != (record['motion_frames'], 56, 6):
                    raise ValueError(f'{label}/{name}: unexpected augmented motion shape')
                if verify_augmented_motion_content:
                    raw = dataset[:].astype(np.float32, copy=False)
                    actual = hashlib.sha256(raw.tobytes()).hexdigest()
                    if actual != record['motion_sha256']:
                        raise ValueError(f'{label}/{name}: augmented motion hash mismatch')

    return manifest
