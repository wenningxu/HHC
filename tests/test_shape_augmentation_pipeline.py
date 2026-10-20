import tempfile
import sys
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.dataset_interx_TM import Text2MotionDatasetV2HHI_Token
from dataset.dataset_interx_tokenize_historical_root import Text2MotionDatasetV2HHI
from utils.interx_shape_augmentation import (
    canonical_sha256,
    sha256_file,
    validate_materialized_interx_shape_augmentation,
)
from tools.score_volumetric_smpl_npz import discover_npz_pairs
from tools.run_interx_shape_augmentation import (
    sample_output_state,
    select_record_shard,
    validate_caption_binding,
)
from tools.run_paper_contact_optimization import (
    sample_output_state as formal_optimizer_output_state,
)
from tools.materialize_interx_shape_augmentation import (
    prepare_materialization_root,
    resolve_augmentation_mode,
)
from tools.aggregate_interx_shape_augmentation_shards import aggregate_shards
from tools.assess_shape_augmentation_pilot import assess as assess_shape_pilot
from tools.audit_shape_augmentation_disk_capacity import (
    project_training_bytes_per_arm,
)
from tools.audit_hhc_shape_aug_training_launch_contract import (
    EXPECTED_TRAINING_OPTIONS,
    EXPECTED_TRAINING_SWITCHES,
    training_option_map,
)
import tools.build_shape_augmentation_results as shape_results


ORIGINAL = 'G001T000A000R000'
AUGMENTED = ORIGINAL + '__shapeaug_s00000123'


def _write_beta(path, value):
    np.savez_compressed(
        path,
        betas=np.full((2, 10), value, dtype=np.float32),
        gender=np.asarray(['male', 'female']),
    )


def _write_text(path, caption):
    path.write_text(
        f'{caption}#person/OTHER interaction/OTHER#0.0#0.0\n',
        encoding='utf-8',
    )


def test_shape_optimizer_partial_output_is_restartable():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = [root / 'p1.npz', root / 'p2.npz', root / 'sample.json']
        assert sample_output_state(*paths) == 'missing'
        paths[0].write_bytes(b'atomic-p1')
        assert sample_output_state(*paths) == 'partial_recompute'
        paths[1].write_bytes(b'atomic-p2')
        assert sample_output_state(*paths) == 'partial_recompute'
        paths[2].write_text('{}', encoding='utf-8')
        assert sample_output_state(*paths) == 'complete'


def test_shape_optimizer_seed_position_is_invariant_to_sharding():
    records = [{'name': f'sample-{index}'} for index in range(8)]
    full, full_positions = select_record_shard(
        records, selection_seed=20260828,
    )
    shard, shard_positions = select_record_shard(
        records, selection_seed=20260828, offset=3, limit=2,
    )
    assert shard == full[3:5]
    assert shard_positions == full_positions[3:5] == [3, 4]
    base_seed = 1234
    assert [base_seed + position * 100 for position in shard_positions] == [
        base_seed + full_positions[index] * 100 for index in range(3, 5)
    ]


def test_disjoint_timing_groups_can_form_one_contiguous_selection():
    positive = [{'name': f'positive-{index}'} for index in range(4)]
    zero = [{'name': f'zero-{index}'} for index in range(2)]
    selected_positive, positive_positions = select_record_shard(
        positive, selection_seed=20260828, selection_position_base=0,
    )
    selected_zero, zero_positions = select_record_shard(
        zero, selection_seed=20260828,
        selection_position_base=len(selected_positive),
    )
    assert len(selected_zero) == 2
    assert positive_positions + zero_positions == list(range(6))


def test_materialization_accepts_only_explicit_positive_optimized_zero_random_mix():
    completions = [
        (Path('/fixture/positive'), {
            'optimizer': {'mode': 'optimized'},
            'selection': {
                'positive_timing_only': True,
                'zero_timing_only': False,
            },
        }),
        (Path('/fixture/zero'), {
            'optimizer': {'mode': 'random_shape_only'},
            'selection': {
                'positive_timing_only': False,
                'zero_timing_only': True,
            },
        }),
    ]
    assert resolve_augmentation_mode(completions) == (
        'optimized_positive_plus_random_zero'
    )
    completions[1][1]['selection']['zero_timing_only'] = False
    try:
        resolve_augmentation_mode(completions)
    except ValueError as error:
        assert 'random-only zero-timing shards' in str(error)
    else:
        raise AssertionError('ambiguous mixed augmentation was accepted')


def test_formal_optimizer_partial_output_is_restartable():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = [root / 'p1.npz', root / 'p2.npz', root / 'sample.json']
        assert formal_optimizer_output_state(*paths) == 'missing'
        paths[1].write_bytes(b'atomic-p2')
        assert formal_optimizer_output_state(*paths) == 'partial_recompute'
        paths[0].write_bytes(b'atomic-p1')
        paths[2].write_text('{}', encoding='utf-8')
        assert formal_optimizer_output_state(*paths) == 'complete'


def test_materialization_root_strictly_resumes_only_matching_incomplete_build():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / 'overlay'
        identity = {'completion_sha256': 'fixture', 'samples': 2}
        created, resumed = prepare_materialization_root(root, identity)
        assert resumed is False
        created_again, resumed = prepare_materialization_root(root, identity)
        assert resumed is True
        assert created_again == created
        try:
            prepare_materialization_root(root, {**identity, 'samples': 3})
        except ValueError as error:
            assert 'another or non-running contract' in str(error)
        else:
            raise AssertionError('mismatched partial materialization resumed')
        (root / 'materialization_manifest.json').write_text(
            '{}', encoding='utf-8'
        )
        try:
            prepare_materialization_root(root, identity)
        except FileExistsError as error:
            assert 'completed immutable output root' in str(error)
        else:
            raise AssertionError('completed immutable materialization resumed')


def test_shape_shard_aggregate_binds_exact_positions_and_output_hashes():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        shard_dirs = []
        for shard_index, positions in enumerate(((0, 1), (2, 3))):
            shard = root / f'shard{shard_index}'
            shard.mkdir()
            names = [f'sample-{position}' for position in positions]
            for name, position in zip(names, positions):
                p1 = shard / f'{name}_p1.npz'
                p2 = shard / f'{name}_p2.npz'
                np.savez_compressed(p1, value=np.asarray([position]))
                np.savez_compressed(p2, value=np.asarray([-position]))
                (shard / f'{name}.json').write_text(json.dumps({
                    'selection_position': position,
                    'output_p1_sha256': sha256_file(p1),
                    'output_p2_sha256': sha256_file(p2),
                }), encoding='utf-8')
            manifest = {
                'schema': 'hhc_interx_shape_augmentation_run_v1',
                'complete': True,
                'requested_samples': len(names),
                'completed_samples': len(names),
                'completed_names': names,
                'selection': {
                    'selected_names': names,
                    'full_selection_positions': list(positions),
                    'selection_seed': 20260828,
                    'positive_timing_only': False,
                },
                'caption_binding_policy': 'fixture',
                'contact_manifest': '/fixture/contact.json',
                'contact_manifest_sha256': 'contact-sha',
                'contract_sha256': f'contract-{shard_index}',
                'implementation': {'runner': 'same-sha'},
                'optimizer': {
                    'mode': 'optimized', 'query_chunk': 10475,
                    'seed_policy': (
                        'base_seed_plus_100x_full_selection_position_before_slicing'
                    ),
                },
                'plan': '/fixture/plan.json',
                'plan_records': '/fixture/records.jsonl',
                'plan_records_sha256': 'records-sha',
                'plan_sha256': 'plan-sha',
                'source_motion': '/fixture/train.h5',
                'source_motion_sha256': 'motion-sha',
            }
            (shard / '_completion_manifest.json').write_text(
                json.dumps(manifest), encoding='utf-8',
            )
            shard_dirs.append(shard)
        aggregate = aggregate_shards(
            list(reversed(shard_dirs)), expected_samples=4,
            expected_shards=2,
        )
        assert aggregate['schema'] == (
            'hhc_interx_shape_augmentation_shard_aggregate_v1'
        )
        assert aggregate['completed_names'] == [
            'sample-0', 'sample-1', 'sample-2', 'sample-3',
        ]
        assert [record['first_position'] for record in aggregate['shards']] == [
            0, 2,
        ]
        (shard_dirs[1] / 'sample-2_p1.npz').write_bytes(b'tampered')
        try:
            aggregate_shards(
                shard_dirs, expected_samples=4, expected_shards=2,
            )
        except ValueError as error:
            assert 'p1 hash mismatch' in str(error)
        else:
            raise AssertionError('tampered shard output was accepted')


def _make_overlay(root):
    primary_beta = root / 'overlay' / 'betas'
    primary_contact = root / 'overlay' / 'contact'
    primary_text = root / 'overlay' / 'texts_processed'
    fallback_beta = root / 'official' / 'betas'
    fallback_contact = root / 'official' / 'contact'
    fallback_text = root / 'official' / 'texts_processed'
    codebook = root / 'cache' / 'codebook'
    contact_tokens = root / 'cache' / 'contact_token_label'
    for directory in (
        primary_beta, primary_contact, primary_text,
        fallback_beta, fallback_contact, fallback_text,
        codebook, contact_tokens,
    ):
        directory.mkdir(parents=True)

    motion_file = root / 'all.h5'
    with h5py.File(motion_file, 'w') as motions:
        for name in (ORIGINAL, AUGMENTED):
            motions.create_dataset(name, data=np.zeros((40, 56, 6), np.float32))
    split = root / 'train.txt'
    split.write_text(f'{ORIGINAL}\n{AUGMENTED}\n', encoding='utf-8')

    _write_beta(fallback_beta / f'{ORIGINAL}.npz', 3.0)
    _write_beta(primary_beta / f'{AUGMENTED}.npz', 7.0)
    np.save(fallback_contact / f'{ORIGINAL}.npy', np.zeros(40, np.uint8))
    np.save(primary_contact / f'{AUGMENTED}.npy', np.ones(40, np.uint8))
    _write_text(fallback_text / f'{ORIGINAL}.txt', 'official caption')
    _write_text(primary_text / f'{AUGMENTED}.txt', 'augmented caption')
    for name in (ORIGINAL, AUGMENTED):
        np.save(codebook / f'{name}.npy', np.zeros((24, 3), np.int64))
        np.save(contact_tokens / f'{name}.npy', np.zeros(24, np.uint8))
    return {
        'motion_file': motion_file,
        'split': split,
        'primary_beta': primary_beta,
        'primary_contact': primary_contact,
        'primary_text': primary_text,
        'fallback_beta': fallback_beta,
        'fallback_contact': fallback_contact,
        'fallback_text': fallback_text,
        'codebook': codebook,
        'contact_tokens': contact_tokens,
    }


def test_tokenizer_cache_loader_uses_augmented_assets_and_official_fallback():
    with tempfile.TemporaryDirectory() as temporary:
        paths = _make_overlay(Path(temporary))
        dataset = Text2MotionDatasetV2HHI(
            paths['split'], None, paths['motion_file'], unit_length=4,
            cache_seed=123,
            beta_dir=paths['primary_beta'],
            contact_dir=paths['primary_contact'],
            fallback_beta_dir=paths['fallback_beta'],
            fallback_contact_dir=paths['fallback_contact'],
            strict_assets=True,
        )
        assert set(dataset.data_dict) == {ORIGINAL, AUGMENTED}
        assert np.all(dataset.data_dict[ORIGINAL]['betas'] == 3.0)
        assert np.all(dataset.data_dict[AUGMENTED]['betas'] == 7.0)

        by_name = {
            dataset[index][2]: dataset[index]
            for index in range(len(dataset))
        }
        assert by_name[ORIGINAL][6].size > 0
        assert not np.any(by_name[ORIGINAL][6])
        assert np.all(by_name[AUGMENTED][6] == 1)


def test_tokenizer_cache_loader_fails_if_augmented_timing_is_missing():
    with tempfile.TemporaryDirectory() as temporary:
        paths = _make_overlay(Path(temporary))
        (paths['primary_contact'] / f'{AUGMENTED}.npy').unlink()
        augmented_split = Path(temporary) / 'augmented.txt'
        augmented_split.write_text(f'{AUGMENTED}\n', encoding='utf-8')
        try:
            Text2MotionDatasetV2HHI(
                augmented_split, None, paths['motion_file'], unit_length=4,
                cache_seed=123,
                beta_dir=paths['primary_beta'],
                contact_dir=paths['primary_contact'],
                fallback_beta_dir=paths['fallback_beta'],
                fallback_contact_dir=paths['fallback_contact'],
                strict_assets=True,
            )
        except RuntimeError as error:
            assert 'lacks contact timing' in str(error)
        else:
            raise AssertionError('missing augmented contact timing was accepted')


def test_mgt_loader_uses_augmented_assets_and_official_fallback():
    with tempfile.TemporaryDirectory() as temporary:
        paths = _make_overlay(Path(temporary))
        dataset = Text2MotionDatasetV2HHI_Token(
            paths['split'], paths['codebook'], unit_length=4,
            codebook_size=12288,
            text_dir=paths['primary_text'],
            max_motion_length=50,
            contact_dir=paths['contact_tokens'],
            beta_dir=paths['primary_beta'],
            fallback_beta_dir=paths['fallback_beta'],
            fallback_text_dir=paths['fallback_text'],
            strict_assets=True,
        )
        assert set(dataset.data_dict) == {ORIGINAL, AUGMENTED}
        assert np.all(dataset.data_dict[ORIGINAL]['betas'] == 3.0)
        assert np.all(dataset.data_dict[AUGMENTED]['betas'] == 7.0)
        assert dataset.data_dict[ORIGINAL]['text'][0]['caption'] == 'official caption'
        assert dataset.data_dict[AUGMENTED]['text'][0]['caption'] == 'augmented caption'


def _make_materialized_fixture(root):
    official = root / 'official'
    materialized = root / 'materialized'
    for directory in (
        official, materialized / 'splits', materialized / 'motions',
        materialized / 'betas', materialized / 'contact',
        materialized / 'texts_processed',
    ):
        directory.mkdir(parents=True, exist_ok=True)
    original_train = official / 'train.txt'
    original_val = official / 'val.txt'
    original_all = official / 'all.txt'
    original_train.write_text(f'{ORIGINAL}\n', encoding='utf-8')
    original_val.write_text('', encoding='utf-8')
    original_all.write_text(f'{ORIGINAL}\n', encoding='utf-8')
    original_train_h5 = official / 'train.h5'
    original_all_h5 = official / 'all.h5'
    raw_original = np.zeros((40, 56, 6), np.float32)
    raw_augmented = np.ones((40, 56, 6), np.float32)
    for path in (original_train_h5, original_all_h5):
        with h5py.File(path, 'w') as motions:
            motions.create_dataset(ORIGINAL, data=raw_original)
    for label, names in (
        ('train', [ORIGINAL, AUGMENTED]),
        ('val', []),
        ('all', [ORIGINAL, AUGMENTED]),
    ):
        (materialized / 'splits' / f'{label}.txt').write_text(
            ''.join(f'{name}\n' for name in names), encoding='utf-8'
        )
    for label, source in (
        ('train', original_train_h5), ('all', original_all_h5)
    ):
        with h5py.File(materialized / 'motions' / f'{label}.h5', 'w') as motions:
            motions[ORIGINAL] = h5py.ExternalLink(str(source.resolve()), f'/{ORIGINAL}')
            motions.create_dataset(AUGMENTED, data=raw_augmented)
    beta_path = materialized / 'betas' / f'{AUGMENTED}.npz'
    contact_path = materialized / 'contact' / f'{AUGMENTED}.npy'
    text_path = materialized / 'texts_processed' / f'{AUGMENTED}.txt'
    _write_beta(beta_path, 7.0)
    np.save(contact_path, np.ones(40, np.uint8))
    _write_text(text_path, 'augmented caption')
    source_mean = official / 'Interx_Mean.npy'
    source_std = official / 'Interx_Std.npy'
    source_root_mean = official / 'Interx_Mean_root.npy'
    source_root_std = official / 'Interx_Std_root.npy'
    np.save(source_mean, np.zeros(4, np.float32))
    np.save(source_std, np.ones(4, np.float32))
    np.save(source_root_mean, np.zeros(4, np.float32))
    np.save(source_root_std, np.full(4, 2.0, np.float32))
    shutil.copy2(source_mean, materialized / 'Interx_Mean.npy')
    shutil.copy2(source_std, materialized / 'Interx_Std.npy')
    shutil.copy2(source_root_mean, materialized / 'Interx_Mean_root.npy')
    shutil.copy2(source_root_std, materialized / 'Interx_Std_root.npy')
    plan_record = {
        'schema': 'hhc_interx_shape_augmentation_record_v2',
        'source_name': ORIGINAL,
        'augmented_name': AUGMENTED,
    }
    plan_records_path = root / 'augmentation_records.jsonl'
    plan_records_path.write_text(json.dumps(plan_record) + '\n', encoding='utf-8')
    plan_path = root / 'plan_manifest.json'
    plan_path.write_text(json.dumps({
        'schema': 'hhc_interx_shape_augmentation_plan_v2',
        'status': 'complete',
        'records': str(plan_records_path.resolve()),
        'records_sha256': sha256_file(plan_records_path),
    }), encoding='utf-8')
    optimizer_dir = root / 'optimizer'
    optimizer_dir.mkdir()
    p1_path = optimizer_dir / f'{AUGMENTED}_p1.npz'
    p2_path = optimizer_dir / f'{AUGMENTED}_p2.npz'
    np.savez(p1_path, poses=np.zeros((40, 55, 3), np.float32))
    np.savez(p2_path, poses=np.zeros((40, 55, 3), np.float32))
    contract_sha256 = 'fixture-contract-sha256'
    sidecar_path = optimizer_dir / f'{AUGMENTED}.json'
    sidecar_path.write_text(json.dumps({
        'schema': 'hhc_interx_optimized_shape_augmentation_sample_v1',
        'source_name': ORIGINAL,
        'augmented_name': AUGMENTED,
        'run_contract_sha256': contract_sha256,
        'plan_record_sha256': canonical_sha256(plan_record),
        'output_p1_sha256': sha256_file(p1_path),
        'output_p2_sha256': sha256_file(p2_path),
    }), encoding='utf-8')
    completion_path = optimizer_dir / '_completion_manifest.json'
    completion_path.write_text(json.dumps({
        'complete': True,
        'completed_names': [AUGMENTED],
        'optimizer': {'mode': 'random_shape_only'},
        'plan': str(plan_path.resolve()),
        'plan_sha256': sha256_file(plan_path),
        'plan_records': str(plan_records_path.resolve()),
        'plan_records_sha256': sha256_file(plan_records_path),
        'contract_sha256': contract_sha256,
    }), encoding='utf-8')
    asset_record = {
        'augmented_name': AUGMENTED,
        'source_name': ORIGINAL,
        'motion_frames': 40,
        'motion_sha256': hashlib.sha256(raw_augmented.tobytes()).hexdigest(),
        'betas': np.full((2, 10), 7.0, np.float32).tolist(),
        'genders': ['male', 'female'],
        'beta_file_sha256': sha256_file(beta_path),
        'contact_file_sha256': sha256_file(contact_path),
        'text_file_sha256': sha256_file(text_path),
        'optimization_sidecar_sha256': sha256_file(sidecar_path),
    }
    manifest = {
        'schema': 'hhc_interx_shape_augmentation_materialization_v1',
        'status': 'complete',
        'output_root': str(materialized.resolve()),
        'optimization_completion': str(completion_path.resolve()),
        'optimization_completion_sha256': sha256_file(completion_path),
        'plan': str(plan_path.resolve()),
        'plan_sha256': sha256_file(plan_path),
        'augmentation_mode': 'random_shape_only',
        'original': {
            'train_split': str(original_train.resolve()),
            'train_split_sha256': sha256_file(original_train),
            'val_split': str(original_val.resolve()),
            'val_split_sha256': sha256_file(original_val),
            'all_split': str(original_all.resolve()),
            'all_split_sha256': sha256_file(original_all),
            'train_motion': str(original_train_h5.resolve()),
            'train_motion_sha256': sha256_file(original_train_h5),
            'all_motion': str(original_all_h5.resolve()),
            'all_motion_sha256': sha256_file(original_all_h5),
        },
        'counts': {
            'official_train': 1, 'augmented': 1, 'combined_train': 2,
            'official_all': 1, 'combined_all': 2,
        },
        'splits': {
            label: {
                'path': str((materialized / 'splits' / f'{label}.txt').resolve()),
                'sha256': sha256_file(materialized / 'splits' / f'{label}.txt'),
            }
            for label in ('train', 'val', 'all')
        },
        'normalization': {
            'policy': 'reuse_original_raw_and_root_interx_stats_no_recompute',
            'generation_and_decode_representation': 'raw',
            'mgt_tokenizer_contract': 'deterministic_historical_root_v1',
            'source_mean': str(source_mean.resolve()),
            'source_mean_sha256': sha256_file(source_mean),
            'source_std': str(source_std.resolve()),
            'source_std_sha256': sha256_file(source_std),
            'mean': str((materialized / 'Interx_Mean.npy').resolve()),
            'mean_sha256': sha256_file(materialized / 'Interx_Mean.npy'),
            'std': str((materialized / 'Interx_Std.npy').resolve()),
            'std_sha256': sha256_file(materialized / 'Interx_Std.npy'),
            'source_root_mean': str(source_root_mean.resolve()),
            'source_root_mean_sha256': sha256_file(source_root_mean),
            'source_root_std': str(source_root_std.resolve()),
            'source_root_std_sha256': sha256_file(source_root_std),
            'root_mean': str((materialized / 'Interx_Mean_root.npy').resolve()),
            'root_mean_sha256': sha256_file(materialized / 'Interx_Mean_root.npy'),
            'root_std': str((materialized / 'Interx_Std_root.npy').resolve()),
            'root_std_sha256': sha256_file(materialized / 'Interx_Std_root.npy'),
            'copy_is_byte_identical': True,
        },
        'motions': {
            label: str((materialized / 'motions' / f'{label}.h5').resolve())
            for label in ('train', 'all')
        },
        'asset_records': [asset_record],
    }
    manifest['motions'].update({
        f'{label}_sha256': sha256_file(materialized / 'motions' / f'{label}.h5')
        for label in ('train', 'all')
    })
    manifest_path = materialized / 'materialization_manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    return {
        'root': materialized,
        'manifest': manifest_path,
        'contact': contact_path,
        'plan': plan_path,
        'sidecar': sidecar_path,
        'source_mean': source_mean,
    }


def test_materialized_overlay_validator_binds_all_training_assets():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _make_materialized_fixture(Path(temporary))
        validated = validate_materialized_interx_shape_augmentation(
            fixture['root'], fixture['manifest'],
        )
        assert validated['augmentation_mode'] == 'random_shape_only'
        np.save(fixture['contact'], np.zeros(40, np.uint8))
        try:
            validate_materialized_interx_shape_augmentation(
                fixture['root'], fixture['manifest'],
            )
        except ValueError as error:
            assert 'contact timing hash mismatch' in str(error)
        else:
            raise AssertionError('tampered augmented timing was accepted')


def test_materialized_overlay_validator_accepts_positioned_shard_manifest():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _make_materialized_fixture(Path(temporary))
        manifest = json.loads(fixture['manifest'].read_text(encoding='utf-8'))
        completion_path = Path(manifest.pop('optimization_completion'))
        manifest.pop('optimization_completion_sha256')
        completion = json.loads(completion_path.read_text(encoding='utf-8'))
        completion['selection'] = {'full_selection_positions': [0]}
        completion_path.write_text(json.dumps(completion), encoding='utf-8')
        manifest['optimization_completions'] = [{
            'path': str(completion_path.resolve()),
            'sha256': sha256_file(completion_path),
        }]
        fixture['manifest'].write_text(json.dumps(manifest), encoding='utf-8')
        validated = validate_materialized_interx_shape_augmentation(
            fixture['root'], fixture['manifest'],
        )
        assert validated['optimization_completions'][0]['path'] == str(
            completion_path.resolve()
        )


def test_materialized_overlay_validator_binds_plan_and_optimizer_sidecar():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _make_materialized_fixture(Path(temporary))
        fixture['sidecar'].write_text('{}', encoding='utf-8')
        try:
            validate_materialized_interx_shape_augmentation(
                fixture['root'], fixture['manifest'],
            )
        except ValueError as error:
            assert 'optimization sidecar hash mismatch' in str(error)
        else:
            raise AssertionError('tampered optimization sidecar was accepted')

    with tempfile.TemporaryDirectory() as temporary:
        fixture = _make_materialized_fixture(Path(temporary))
        fixture['plan'].write_text('{}', encoding='utf-8')
        try:
            validate_materialized_interx_shape_augmentation(
                fixture['root'], fixture['manifest'],
            )
        except ValueError as error:
            assert 'augmentation plan manifest hash mismatch' in str(error)
        else:
            raise AssertionError('tampered augmentation plan was accepted')


def test_physical_scorer_accepts_shape_augmentation_npz_names():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for suffix in ('_p1.npz', '_p2.npz'):
            np.savez(root / f'{AUGMENTED}{suffix}', poses=np.zeros((1, 55, 3)))
        pairs = discover_npz_pairs(root)
        assert len(pairs) == 1
        assert pairs[0][0] == AUGMENTED
        assert pairs[0][-1] == 'shape_augmentation'


def test_contact_caption_binding_allows_only_boundary_whitespace():
    metadata_caption = 'Two people hold hands. '
    metadata = {
        'caption': metadata_caption,
        'caption_sha256': hashlib.sha256(
            metadata_caption.encode('utf-8')
        ).hexdigest(),
    }
    binding = validate_caption_binding(
        metadata, 'Two people hold hands.', 'sample',
    )
    assert binding['match_mode'] == 'boundary_whitespace_normalized'
    try:
        validate_caption_binding(
            metadata, 'Two people shake hands.', 'sample',
        )
    except ValueError as error:
        assert 'differ beyond boundary whitespace' in str(error)
    else:
        raise AssertionError('caption content mismatch was accepted')


def test_shape_result_method_gate_binds_raw_stats_and_contact_contract():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        label = 'shape_method_fixture'
        directory = root / 'output' / 'fair_eval' / label
        directory.mkdir(parents=True)
        overlay = root / 'overlay'
        overlay.mkdir()
        selection_path = root / 'selection.json'
        contact_map_path = root / 'contact.npy'
        cache_path = root / 'cache.json'
        mean_path = overlay / 'Interx_Mean.npy'
        std_path = overlay / 'Interx_Std.npy'
        selection_path.write_text('{}', encoding='utf-8')
        cache_path.write_text('{}', encoding='utf-8')
        np.save(contact_map_path, np.zeros((3, 2), np.float32))
        np.save(mean_path, np.zeros(2, np.float32))
        np.save(std_path, np.ones(2, np.float32))

        def record(path):
            return {'path': str(path.resolve()), 'sha256': sha256_file(path)}

        selection = record(selection_path)
        contact_map = record(contact_map_path)
        cache = record(cache_path)
        materialization = {
            'output_root': str(overlay.resolve()),
            'normalization': {
                'mean_sha256': sha256_file(mean_path),
                'std_sha256': sha256_file(std_path),
            },
        }
        run = {
            'method': 'hhc', 'label': label,
            'replication_range': [0, 20], 'samples_per_replication': 1696,
            'sampling_steps': 50, 'temperature': 0.5, 'cfg_scale': 3.5,
            'contact_guidance_scale': 1.0,
            'hhc_vq_contract': 'precleanup_raw12288',
            'hhc_mgt_contract': 'current',
            'hhc_contact_map_contract': 'paper_formula_train_only',
            'hhc_stats_dir': str(overlay.resolve()),
            'hhc_stats_prefix': 'Interx',
            'selection_manifest': selection['path'],
            'hhc_contact_map': contact_map['path'],
            'contact_guidance': True,
            'contact_timing_source': 'request_oracle',
            'contact_timing_source_requested': 'request_oracle',
            'provenance': {'artifacts': {
                'normalization_mean': record(mean_path),
                'normalization_std': record(std_path),
                'mgt_selection_manifest': selection,
                'contact_map': contact_map,
                'contact_map_manifest': cache,
            }},
        }
        (directory / 'run_manifest.json').write_text(
            json.dumps(run), encoding='utf-8'
        )
        sampler_identity = {
            'sampler': 'sample_intra_cas',
            'sampler_semantics': 'paper_high_confidence_tokens_fixed',
            'cascade_order': 'root,body,hands', 'sampling_steps': 50,
            'sampling_steps_unit': 'outer_maskgit_iterations',
            'channel_updates_per_outer_step': 3,
            'model_forward_calls_per_outer_step_with_cfg': 6,
            'temperature': 0.5, 'cfg_scale': 3.5,
            'contact_guidance': True, 'contact_guidance_scale': 1.0,
            'contact_map_contract': 'paper_formula_train_only',
            'contact_timing_source': 'request_oracle',
            'generation_length_policy': 'exact_request_length',
        }
        for name, payload in (
            ('summary.json', {
                'replications': 20, 'samples_per_replication': 1696,
                'provenance': {
                    'legacy_provenance_allowed': False,
                    'generation_files_sha256_verified': True,
                    'unique_requests_verified': True,
                    'sampler_identity': sampler_identity,
                },
            }),
            ('multimodality_audit_summary.json', {'replications': 20}),
            ('replication_00_physical.json', {'samples': 1696}),
            ('generation_timing_summary.json', {'replications': 20}),
        ):
            (directory / name).write_text(json.dumps(payload), encoding='utf-8')

        previous_root = shape_results.ROOT
        shape_results.ROOT = root
        try:
            shape_results.validate_method(
                label, selection, contact_map, materialization, cache, True,
            )
            run['hhc_contact_map_contract'] = 'mmm_legacy_full_axis_bug'
            (directory / 'run_manifest.json').write_text(
                json.dumps(run), encoding='utf-8'
            )
            try:
                shape_results.validate_method(
                    label, selection, contact_map, materialization, cache, True,
                )
            except ValueError as error:
                assert 'generation contract' in str(error)
            else:
                raise AssertionError('legacy map entered shape main result')
        finally:
            shape_results.ROOT = previous_root


def test_shape_disk_capacity_gate_is_non_destructive_and_precedes_full_assets():
    gate = (ROOT / 'tools/audit_shape_augmentation_disk_capacity.py').read_text(
        encoding='utf-8'
    )
    launcher = (
        ROOT / 'tools/run_interx_shape_augmentation_nearest_last_stage.sh'
    ).read_text(encoding='utf-8')
    for destructive in ('shutil.rmtree(', '.unlink(', 'os.remove('):
        assert destructive not in gate
    assert 'insufficient_capacity' in gate
    assert 'free_space_after_queue_gib' in gate
    assert 'validation_checkpoints_per_arm' in gate
    assert 'rolling_full_checkpoints_per_arm' in gate
    assert 'measured_full_checkpoint_bytes' in gate
    assert 'save_weights_only=True' in (
        ROOT / 'train/train_t2i_trans.py'
    ).read_text(encoding='utf-8')
    assert "['artifacts']['training_manifest']['path']" in launcher
    assert launcher.index(
        'tools/audit_shape_augmentation_disk_capacity.py'
    ) < launcher.index(
        '--output-dir "${full_random}"'
    )


def test_shape_disk_projection_counts_all_save_all_candidates():
    projection = project_training_bytes_per_arm(
        training_bytes=3_000,
        checkpoint_sizes=[1_000, 1_100],
        validation_checkpoints=30,
        rolling_checkpoints=2,
    )
    assert projection['planned_checkpoint_count'] == 32
    assert projection['noncheckpoint_training_bytes'] == 900
    assert projection['count_based_bytes'] == 1_100 * 32 + 900
    assert projection['projected_bytes'] == projection['count_based_bytes']


def test_shape_disk_projection_separates_weights_only_from_recovery_checkpoints():
    projection = project_training_bytes_per_arm(
        training_bytes=100_000,
        checkpoint_sizes=[4_000, 4_100],
        validation_checkpoints=30,
        rolling_checkpoints=2,
        validation_checkpoint_bytes=2_100,
    )
    assert projection['planned_checkpoint_count'] == 32
    assert projection['validation_checkpoint_storage'] == 'weights_only'
    assert projection['measured_validation_checkpoint_bytes'] == 2_100
    assert projection['measured_full_checkpoint_bytes'] == 4_100
    assert projection['noncheckpoint_training_bytes'] == 91_900
    assert projection['count_based_bytes'] == 2_100 * 30 + 4_100 * 2 + 91_900
    # The full-state reference tree belongs to a deliberately different
    # storage contract and must not erase the measured weights-only saving.
    assert projection['projected_bytes'] == projection['count_based_bytes']


def test_each_shape_training_arm_has_its_own_capacity_gate():
    launcher = (
        ROOT / 'tools/run_hhc_shape_aug_historical_paper_mask_arm.sh'
    ).read_text(encoding='utf-8')
    gate = 'tools/audit_shape_augmentation_disk_capacity.py'
    train = 'train/train_t2i_trans.py --cache-only'
    assert gate in launcher
    assert '--arms 1 --labels-per-arm 2' in launcher
    assert "['artifacts']['training_manifest']['path']" in launcher
    assert launcher.index(gate) < launcher.index(train)


def test_shape_training_preserves_primary_validation_rng_but_final_eval_is_common50():
    launcher = (
        ROOT / 'tools/run_hhc_shape_aug_historical_paper_mask_arm.sh'
    ).read_text(encoding='utf-8')
    training_start = launcher.index('train/train_t2i_trans.py \\\n')
    selection_start = launcher.index(
        'tools/select_hhc_checkpoint_on_official_interx_val.py'
    )
    training = launcher[training_start:selection_start]
    final_eval = launcher[selection_start:]
    assert '--training-validation-contract historical_test_side_effect' in training
    assert '--sampler sample_intra_cas --temperature 0.0 --sampling-steps 10' in training
    assert '--sampling-steps 50 --temperature 0.5 --cfg-scale 3.5' in final_eval


def test_shape_training_command_is_parsed_without_silent_option_overrides():
    launcher = (
        ROOT / 'tools/run_hhc_shape_aug_historical_paper_mask_arm.sh'
    ).read_text(encoding='utf-8')
    options, duplicates = training_option_map(launcher)
    assert duplicates == []
    assert all(options.get(key) == value
               for key, value in EXPECTED_TRAINING_OPTIONS.items())
    assert all(options.get(key) is True for key in EXPECTED_TRAINING_SWITCHES)

    fixture = '''
if [[ "${training_status}" != complete ]]; then
  train/train_t2i_trans.py --lr 0.1 --lr 0.2 --shape
fi
'''
    options, duplicates = training_option_map(fixture)
    assert options['--lr'] == '0.2'
    assert duplicates == ['--lr']


def test_shape_pilot_physical_gate_precedes_full_assets_and_fails_closed():
    gate = json.loads((
        ROOT / 'configs/interx_shape_augmentation/pilot_physical_gate_v1.json'
    ).read_text(encoding='utf-8'))
    launcher = (
        ROOT / 'tools/run_interx_shape_augmentation_nearest_last_stage.sh'
    ).read_text(encoding='utf-8')
    assert launcher.index(
        'tools/assess_shape_augmentation_pilot.py'
    ) < launcher.index(
        '--output-dir "${full_random}"'
    )
    comparisons = {}
    for metric in gate['criteria']:
        comparisons[metric] = {
            'before_mean': 2.0,
            'after_mean': 1.0,
            'paired_delta_after_minus_before_mean': -1.0,
            'paired_delta_bootstrap_95ci': [-1.2, -0.8],
            'wilcoxon_two_sided_p': 0.01,
        }
    comparison = {
        'schema': gate['comparison_schema'],
        'sample_count': 25,
        'sample_keys': [f'sample_{index:02d}' for index in range(25)],
        'comparisons': comparisons,
    }
    result, passed = assess_shape_pilot(comparison, gate)
    assert passed and result['full_augmentation_authorized']
    comparisons['mean_foot_sliding_m_per_frame'][
        'paired_delta_after_minus_before_mean'
    ] = 0.01
    result, passed = assess_shape_pilot(comparison, gate)
    assert not passed and not result['full_augmentation_authorized']


def test_five_sample_shape_gate_only_authorizes_pilot25():
    gate = json.loads((
        ROOT / 'configs/interx_shape_augmentation/'
        'grounded_chain_pilot5_physical_gate_v1.json'
    ).read_text(encoding='utf-8'))
    comparisons = {
        metric: {
            'before_mean': 2.0,
            'after_mean': 1.0,
            'paired_delta_after_minus_before_mean': -1.0,
            'paired_delta_bootstrap_95ci': [-1.2, -0.8],
            'wilcoxon_two_sided_p': 0.01,
        }
        for metric in gate['criteria']
    }
    comparison = {
        'schema': gate['comparison_schema'],
        'sample_count': gate['expected_sample_count'],
        'sample_keys': [
            f'sample_{index:02d}'
            for index in range(gate['expected_sample_count'])
        ],
        'comparisons': comparisons,
    }
    result, passed = assess_shape_pilot(comparison, gate)
    assert passed
    assert result['authorized_action'] == 'pilot25'
    assert result['pilot25_authorized'] is True
    assert result['full_augmentation_authorized'] is False


def test_shape_results_require_full_current_wandb_history_audit():
    source = (
        ROOT / 'tools/build_shape_augmentation_results.py'
    ).read_text(encoding='utf-8')
    for marker in (
        "'all_training_metrics_reach_expected_step'",
        "'all_validation_metrics_follow_final_training_event'",
        "'nonfinite_counts_zero'",
        "== 'train/optimizer_step'",
        "'tools/audit_wandb_online_metric_history.py'",
        "complete_wandb_norm_telemetry(wandb_audit, 300000)",
    ):
        assert marker in source


def test_shape_results_reject_truncated_gradient_or_weight_telemetry():
    payload = {'metrics': {}}
    for metric in (
        'train/gradient_l2_norm', 'train/gradient_max_abs',
        'train/weight_l2_norm', 'train/weight_max_abs',
        'train/nonfinite_gradients', 'train/nonfinite_weights',
    ):
        payload['metrics'][metric] = {
            'count': 6000,
            'last_step': 300000,
            'minimum': 0.0 if 'nonfinite' in metric else 1.0,
            'maximum': 0.0 if 'nonfinite' in metric else 2.0,
        }
    assert shape_results.complete_wandb_norm_telemetry(payload, 300000)
    payload['metrics']['train/weight_l2_norm']['last_step'] = 299950
    assert not shape_results.complete_wandb_norm_telemetry(payload, 300000)
    payload['metrics']['train/weight_l2_norm']['last_step'] = 300000
    payload['metrics']['train/nonfinite_gradients']['maximum'] = 1.0
    assert not shape_results.complete_wandb_norm_telemetry(payload, 300000)


def test_shape_results_bind_exact_multi_shard_materialization_provenance():
    completion = {
        'schema': 'hhc_interx_shape_augmentation_shard_aggregate_v1',
        'shards': [
            {'completion_manifest': {'path': '/tmp/shard0.json', 'sha256': 'a'}},
            {'completion_manifest': {'path': '/tmp/shard1.json', 'sha256': 'b'}},
        ],
    }
    materialization = {'optimization_completions': [
        {'path': '/tmp/shard0.json', 'sha256': 'a'},
        {'path': '/tmp/shard1.json', 'sha256': 'b'},
    ]}
    assert shape_results.materialization_binds_completion(
        materialization, completion, {'sha256': 'aggregate'},
    )
    materialization['optimization_completions'][1]['sha256'] = 'tampered'
    assert not shape_results.materialization_binds_completion(
        materialization, completion, {'sha256': 'aggregate'},
    )


def test_shape_queue_freezes_runtime_evidence_before_physical_gate_and_full_run():
    launcher = (ROOT / 'tools/queue_interx_shape_augmentation_gpu1.sh').read_text(
        encoding='utf-8'
    )
    runtime = 'tools/analyze_shape_augmentation_runtime.py'
    physical = 'tools/score_volumetric_smpl_npz.py'
    full_run = (
        'full_optimized_aggregate="output/interx_shape_aug_full_'
        'batch8_query10475_seed20260828_aggregate"'
    )
    volumetric_gate = 'tools/require_interx_volumetric_smpl_contract.py'
    assert runtime in launcher
    assert launcher.index(volumetric_gate) < launcher.index(runtime)
    assert launcher.index(runtime) < launcher.index(physical)
    assert launcher.index(runtime) < launcher.index(full_run)


if __name__ == '__main__':
    tests = (
        test_shape_optimizer_partial_output_is_restartable,
        test_shape_optimizer_seed_position_is_invariant_to_sharding,
        test_formal_optimizer_partial_output_is_restartable,
        test_materialization_root_strictly_resumes_only_matching_incomplete_build,
        test_shape_shard_aggregate_binds_exact_positions_and_output_hashes,
        test_tokenizer_cache_loader_uses_augmented_assets_and_official_fallback,
        test_tokenizer_cache_loader_fails_if_augmented_timing_is_missing,
        test_mgt_loader_uses_augmented_assets_and_official_fallback,
        test_materialized_overlay_validator_binds_all_training_assets,
        test_materialized_overlay_validator_accepts_positioned_shard_manifest,
        test_materialized_overlay_validator_binds_plan_and_optimizer_sidecar,
        test_physical_scorer_accepts_shape_augmentation_npz_names,
        test_contact_caption_binding_allows_only_boundary_whitespace,
        test_shape_result_method_gate_binds_raw_stats_and_contact_contract,
        test_shape_disk_capacity_gate_is_non_destructive_and_precedes_full_assets,
        test_shape_disk_projection_counts_all_save_all_candidates,
        test_shape_disk_projection_separates_weights_only_from_recovery_checkpoints,
        test_each_shape_training_arm_has_its_own_capacity_gate,
        test_shape_training_preserves_primary_validation_rng_but_final_eval_is_common50,
        test_shape_training_command_is_parsed_without_silent_option_overrides,
        test_shape_pilot_physical_gate_precedes_full_assets_and_fails_closed,
        test_five_sample_shape_gate_only_authorizes_pilot25,
        test_shape_results_require_full_current_wandb_history_audit,
        test_shape_results_reject_truncated_gradient_or_weight_telemetry,
        test_shape_results_bind_exact_multi_shard_materialization_provenance,
        test_shape_queue_freezes_runtime_evidence_before_physical_gate_and_full_run,
    )
    for test in tests:
        test()
        print({'test': test.__name__, 'status': 'pass'})
