#!/usr/bin/env python
"""Fail-fast audit for the official Inter-X/InterMask evaluation protocol."""

import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import yaml


def contains(path, snippets):
    with open(path, encoding='utf-8') as handle:
        source = handle.read()
    return all(snippet in source for snippet in snippets)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def method_defaults_from_source(source, source_name, class_name, method_name):
    """Return literal positional defaults without importing a training repo."""
    tree = ast.parse(source, filename=source_name)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if child.name != method_name:
                continue
            names = [argument.arg for argument in child.args.args]
            defaults = [ast.literal_eval(value) for value in child.args.defaults]
            return dict(zip(names[-len(defaults):], defaults))
    raise ValueError(f'{class_name}.{method_name} not found in {source_name}')


def git_source(repository, revision, relative_path):
    return subprocess.check_output(
        ['git', '-C', str(repository), 'show', f'{revision}:{relative_path}'],
        text=True,
    )


def numpy_asset(path):
    array = np.load(path, allow_pickle=False)
    return {
        'shape': list(array.shape),
        'dtype': str(array.dtype),
        'finite': bool(np.isfinite(array).all()),
        'sha256': sha256_file(path),
    }


def h5_motion_asset(path):
    with h5py.File(path, 'r') as handle:
        shapes = {tuple(dataset.shape[1:]) for dataset in handle.values()}
        dtypes = {str(dataset.dtype) for dataset in handle.values()}
        return {
            'samples': len(handle),
            'feature_shapes': [list(shape) for shape in sorted(shapes)],
            'dtypes': sorted(dtypes),
            'sha256': sha256_file(path),
        }


def git_revision(path):
    return subprocess.check_output(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True
    ).strip()


def git_resolve(path, revision):
    return subprocess.check_output(
        ['git', '-C', str(path), 'rev-parse', revision], text=True
    ).strip()


def matching_line_records(path, needle):
    """Record exact runtime-log evidence without treating it as source code."""
    records = []
    for line_number, line in enumerate(
            path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
        if needle in line:
            records.append({
                'line': line_number,
                'text': line.strip(),
                'sha256': hashlib.sha256(line.encode('utf-8')).hexdigest(),
            })
    return records


def protocol_alias(mapping, preferred, legacy):
    """Read a renamed protocol field without silently accepting conflicts."""
    preferred_present = preferred in mapping
    legacy_present = legacy in mapping
    if preferred_present and legacy_present:
        if mapping[preferred] != mapping[legacy]:
            raise ValueError(
                f'conflicting protocol aliases {preferred!r} and {legacy!r}: '
                f'{mapping[preferred]!r} != {mapping[legacy]!r}'
            )
        return mapping[preferred]
    if preferred_present:
        return mapping[preferred]
    if legacy_present:
        return mapping[legacy]
    raise KeyError(f'missing protocol field {preferred!r} (legacy: {legacy!r})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='dataset/interx')
    parser.add_argument('--interx-repo', default='/mnt/8T-HDD/wenning/Inter-X')
    parser.add_argument('--intermask-repo', default='/mnt/8T-HDD/wenning/InterMask')
    parser.add_argument('--mmm-repo', default='/home/wenning/projects/MMM')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    split_path = os.path.join(args.data_root, 'splits', 'test.txt')
    motion_path = os.path.join(args.data_root, 'motions', 'test.h5')
    with open(split_path) as handle:
        identifiers = [line.strip() for line in handle if line.strip()]
    with h5py.File(motion_path, 'r') as motions:
        valid = [name for name in identifiers if name in motions and 24 <= len(motions[name]) < 1000]

    val_split_path = os.path.join(args.data_root, 'splits', 'val.txt')
    val_motion_path = os.path.join(args.data_root, 'motions', 'val.h5')
    with open(val_split_path) as handle:
        val_identifiers = [line.strip() for line in handle if line.strip()]
    with h5py.File(val_motion_path, 'r') as motions:
        val_valid = [
            name for name in val_identifiers
            if name in motions and 24 <= len(motions[name]) < 1000
        ]
        val_truncated_at_150 = sum(
            len(motions[name]) > 150 for name in val_valid
        )
        val_excluded = [
            name for name in val_identifiers
            if name not in motions or not (24 <= len(motions[name]) < 1000)
        ]

    batch_size = 32
    evaluator_weights = 'checkpoints/hhi/text_mot_match/model/finest.tar'
    protocol_path = 'configs/fair_interx_protocol.yaml'
    with open(protocol_path, encoding='utf-8') as handle:
        protocol = yaml.safe_load(handle)
    protocol_sampling = protocol['sampling']
    hhc_sampling = protocol_sampling['hhc']
    maximum_active_channel_stages = protocol_alias(
        hhc_sampling,
        'maximum_active_channel_stages_per_outer_iteration',
        'channel_updates_per_outer_step',
    )
    maximum_model_forward_calls = protocol_alias(
        hhc_sampling,
        'maximum_model_forward_calls_per_outer_iteration_with_cfg',
        'model_forward_calls_per_outer_step_with_cfg',
    )
    maximum_common50_active_channel_stages = protocol_alias(
        hhc_sampling,
        'maximum_common50_active_channel_stages',
        'common50_channel_updates',
    )
    maximum_common50_model_forward_calls = protocol_alias(
        hhc_sampling,
        'maximum_common50_model_forward_calls_with_cfg',
        'common50_model_forward_calls_with_cfg',
    )
    resolved_channels_skip_forward = hhc_sampling.get(
        'resolved_channels_skip_forward_but_not_outer_iterations',
        hhc_sampling.get('active_sampler_semantics')
        == 'paper_high_confidence_tokens_fixed',
    )
    mmm_transformer = os.path.join(args.mmm_repo, 'models/two_person_transformer.py')
    mmm_history = protocol_sampling['hhc']
    pre_cleanup_revision = mmm_history['mmm_pre_cleanup_baseline_commit']
    cleanup_revision = mmm_history['mmm_cleanup_first_commit']
    cleanup_parent_revision = git_resolve(args.mmm_repo, f'{cleanup_revision}^')
    pre_cleanup_paths = (
        'models/t2i_trans.py', 'utils/eval_trans.py',
        'train_t2i_trans.py', 'generate_mm_samples.py',
        'options/option_transformer.py',
    )
    pre_cleanup_sources = {
        path: git_source(args.mmm_repo, pre_cleanup_revision, path)
        for path in pre_cleanup_paths
    }
    pre_cleanup_source_records = {
        path: {
            'git_blob': git_resolve(
                args.mmm_repo, f'{pre_cleanup_revision}:{path}'
            ),
            'sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
            'bytes': len(source.encode('utf-8')),
        }
        for path, source in pre_cleanup_sources.items()
    }
    pre_cleanup_transformer_path = 'models/t2i_trans.py'
    pre_cleanup_transformer = pre_cleanup_sources[pre_cleanup_transformer_path]
    pre_cleanup_parallel_defaults = method_defaults_from_source(
        pre_cleanup_transformer,
        f'{pre_cleanup_revision}:{pre_cleanup_transformer_path}',
        'Text2Motion_Transformer', 'sample',
    )
    pre_cleanup_cascade_defaults = method_defaults_from_source(
        pre_cleanup_transformer,
        f'{pre_cleanup_revision}:{pre_cleanup_transformer_path}',
        'Text2Motion_Transformer', 'sample_intra_cas',
    )
    historical_runtime_logs = [
        Path(args.mmm_repo) / 'wandb/run-20260217_033057-vq_mask_gen_4_ablation_adalnshape_1_cfg/files/output.log',
        Path(args.mmm_repo) / 'wandb/run-20260305_101848-vq_mask_gen_4_ablation_adalnshape_1_cfg_hbr_wocas/files/output.log',
    ]
    historical_runtime_metadata = [
        path.with_name('wandb-metadata.json')
        for path in historical_runtime_logs
    ]
    historical_runtime_sampler_matches = all(
        path.is_file() and 'type="sample_intra_cas"' in path.read_text(
            encoding='utf-8', errors='replace'
        )
        for path in historical_runtime_logs
    )
    historical_runtime_call_records = {
        str(path): matching_line_records(path, 'type="sample_intra_cas"')
        for path in historical_runtime_logs
    }
    historical_runtime_metadata_records = {}
    for path in historical_runtime_metadata:
        metadata = json.loads(path.read_text(encoding='utf-8'))
        historical_runtime_metadata_records[str(path)] = {
            'sha256': sha256_file(path),
            'git_remote': metadata.get('git', {}).get('remote'),
            'git_commit': metadata.get('git', {}).get('commit'),
            'program': metadata.get('program'),
            'code_path': metadata.get('codePath'),
        }
    historical_local_history_audit_path = Path(
        mmm_history['mmm_historical_runtime_local_history_audit']
    )
    historical_local_history_audit = json.loads(
        historical_local_history_audit_path.read_text(encoding='utf-8')
    )
    stats_names = (
        'Interx_Mean.npy', 'Interx_Std.npy',
        'Interx_Mean_root.npy', 'Interx_Std_root.npy',
        'Mean.npy', 'Std.npy',
    )
    active_stats = {
        name: numpy_asset(os.path.join(args.data_root, name))
        for name in stats_names
    }
    mmm_stats = {
        name: numpy_asset(os.path.join(args.mmm_repo, 'dataset/interx', name))
        for name in stats_names
    }
    torch_checkpoints = Path.home() / '.cache/torch/hub/checkpoints'
    mmm_data_root = Path(args.mmm_repo) / 'dataset/interx'
    link_names = ('motions', 'splits', 'betas', 'skeleton', 'texts_processed')
    report = {
        'source_protocol': {
            'mmm_repository': str(Path(args.mmm_repo).resolve()),
            'mmm_revision': git_revision(args.mmm_repo),
            'mmm_two_person_transformer_sha256': sha256_file(mmm_transformer),
            'mmm_eval_trans_sha256': sha256_file(
                os.path.join(args.mmm_repo, 'utils/eval_trans.py')
            ),
            'mmm_train_t2i_trans_sha256': sha256_file(
                os.path.join(args.mmm_repo, 'train_t2i_trans.py')
            ),
            'mmm_pre_cleanup_baseline_commit': pre_cleanup_revision,
            'mmm_cleanup_first_commit': cleanup_revision,
            'mmm_cleanup_parent_commit': cleanup_parent_revision,
            'mmm_pre_cleanup_is_immediate_parent': (
                cleanup_parent_revision == pre_cleanup_revision
            ),
            'mmm_pre_cleanup_tree': git_resolve(
                args.mmm_repo, f'{pre_cleanup_revision}^{{tree}}'
            ),
            'mmm_cleanup_tree': git_resolve(
                args.mmm_repo, f'{cleanup_revision}^{{tree}}'
            ),
            'mmm_pre_cleanup_source_records': pre_cleanup_source_records,
            'mmm_pre_cleanup_transformer_path': pre_cleanup_transformer_path,
            'mmm_pre_cleanup_transformer_sha256': hashlib.sha256(
                pre_cleanup_transformer.encode('utf-8')
            ).hexdigest(),
            'mmm_historical_runtime_log_sha256': {
                str(path): sha256_file(path) for path in historical_runtime_logs
            },
            'historical_evidence_priority': [
                'historical_runtime_logs',
                'local_history_time_bracketed_source',
                'pre_cleanup_git_tree',
                'cleaned_main',
            ],
            'mmm_historical_runtime_call_records': (
                historical_runtime_call_records
            ),
            'mmm_historical_runtime_metadata_records': (
                historical_runtime_metadata_records
            ),
            'mmm_historical_runtime_source_state': (
                'uncommitted_two_person_code_on_upstream_git_base'
            ),
            'mmm_historical_runtime_local_history_audit': {
                'path': str(historical_local_history_audit_path),
                'sha256': sha256_file(historical_local_history_audit_path),
                'schema': historical_local_history_audit.get('schema'),
                'status': historical_local_history_audit.get('status'),
                'scope': historical_local_history_audit.get('scope'),
            },
        },
        'split_entries': len(identifiers),
        'valid_entries': len(valid),
        'batch_size': batch_size,
        'drop_last': True,
        'evaluated_entries': len(valid) // batch_size * batch_size,
        'max_motion_length': 150,
        'checkpoint_selection_validation': {
            'split': str(Path(val_split_path).resolve()),
            'split_entries': len(val_identifiers),
            'eligible_entries': len(val_valid),
            'batch_size': 32,
            'drop_last': True,
            'evaluated_entries': len(val_valid) // 32 * 32,
            'excluded_entries': val_excluded,
            'clips_truncated_at_150': val_truncated_at_150,
            'active_min_motion_frames': 24,
            'active_max_motion_frames': 150,
            'intermask_contract_matches': (
                contains(
                    os.path.join(args.intermask_repo, 'data/interx.py'),
                    ('min_motion_len = 24',)
                )
                and contains(
                    os.path.join(args.intermask_repo, 'eval.py'),
                    ('main_opt.max_motion_length = 150',)
                )
            ),
            'official_interx_contract_matches': (
                contains(
                    os.path.join(
                        args.interx_repo,
                        'evaluation/text2motion/data/dataset.py',
                    ),
                    ("else 24",)
                )
                and contains(
                    os.path.join(
                        args.interx_repo,
                        'evaluation/text2motion/utils/get_opt.py',
                    ),
                    ('opt.max_motion_length = 150',)
                )
            ),
            'mmm_development_native_min_motion_frames': 30,
            'mmm_development_native_max_motion_frames': 152,
            'mmm_development_contract_matches': contains(
                os.path.join(args.mmm_repo, 'dataset/dataset_interx_TM.py'),
                ('self.max_motion_length = 152', 'min_motion_len = 30'),
            ),
            'split_sha256': sha256_file(val_split_path),
            'motion_h5_sha256': sha256_file(val_motion_path),
            'assets_byte_identical_to_mmm': (
                sha256_file(val_split_path) == sha256_file(
                    os.path.join(args.mmm_repo, 'dataset/interx/splits/val.txt')
                )
                and sha256_file(val_motion_path) == sha256_file(
                    os.path.join(args.mmm_repo, 'dataset/interx/motions/val.h5')
                )
            ),
        },
        'official_evaluator_options': os.path.isfile('checkpoints/hhi/Comp_v6_KLD01/opt.txt'),
        'official_evaluator_weights': os.path.isfile(evaluator_weights),
        'official_evaluator_weights_sha256': (
            sha256_file(evaluator_weights) if os.path.isfile(evaluator_weights) else None
        ),
        'official_interx_source_matches': contains(
            os.path.join(args.interx_repo, 'evaluation/text2motion/final_evaluations.py'),
            ('mm_num_samples = 100', 'mm_num_repeats = 30',
             'replication_times = 20', 'batch_size = 32'),
        ),
        'intermask_source_matches': contains(
            os.path.join(args.intermask_repo, 'models/evaluator/evaluator_interx.py'),
            ("'splits/test.txt'", 'drop_last=True'),
        ),
        'sampling_protocol': {
            'hhc_common_steps': protocol_sampling['hhc']['maskgit_steps'],
            'hhc_sampling_steps_unit': (
                protocol_sampling['hhc']['sampling_steps_unit']
            ),
            'hhc_maximum_active_channel_stages_per_outer_iteration': (
                maximum_active_channel_stages
            ),
            'hhc_maximum_model_forward_calls_per_outer_iteration_with_cfg': (
                maximum_model_forward_calls
            ),
            'hhc_maximum_common50_active_channel_stages': (
                maximum_common50_active_channel_stages
            ),
            'hhc_maximum_common50_model_forward_calls_with_cfg': (
                maximum_common50_model_forward_calls
            ),
            'hhc_resolved_channels_skip_forward_but_not_outer_iterations': (
                resolved_channels_skip_forward
            ),
            'hhc_active_sampler_semantics': (
                protocol_sampling['hhc']['active_sampler_semantics']
            ),
            'intermask_common_steps': protocol_sampling['intermask']['sampling_steps'],
            'hhc_temperature': protocol_sampling['hhc']['temperature'],
            'hhc_cfg_scale': protocol_sampling['hhc']['cfg_scale'],
            'hhc_contact_guidance_scale': (
                protocol_sampling['hhc']['contact_guidance_scale']
            ),
            'mmm_pre_cleanup_parallel_default_steps': (
                pre_cleanup_parallel_defaults['max_steps']
            ),
            'mmm_pre_cleanup_cascade_default_outer_steps': (
                pre_cleanup_cascade_defaults['max_steps']
            ),
            'mmm_pre_cleanup_cascade_default_temperature': (
                pre_cleanup_cascade_defaults['temperature']
            ),
            'mmm_pre_cleanup_cascade_forward_passes_without_cfg': (
                protocol_sampling['hhc'][
                    'mmm_pre_cleanup_snapshot_cascade_forward_passes_without_cfg'
                ]
            ),
            'mmm_pre_cleanup_parameter_audit_scope': (
                protocol_sampling['hhc'][
                    'mmm_pre_cleanup_snapshot_audit_scope'
                ]
            ),
            'mmm_pre_cleanup_source_contact_guidance': (
                protocol_sampling['hhc'][
                    'mmm_pre_cleanup_source_contact_guidance'
                ]
            ),
            'mmm_pre_cleanup_source_token_update': (
                protocol_sampling['hhc']['mmm_pre_cleanup_source_token_update']
            ),
            'mmm_cleaned_main_contains_parallel_validation_call': contains(
                os.path.join(args.mmm_repo, 'utils/eval_trans.py'),
                ('type="sample"', 'rand_pos=rand_pos'),
            ),
            'mmm_historical_runtime_sampler': mmm_history[
                'mmm_historical_runtime_sampler'
            ],
            'mmm_historical_runtime_sampler_log_matches': (
                historical_runtime_sampler_matches
            ),
            'mmm_historical_runtime_minimum_sampling_updates_user_recorded': (
                protocol_sampling['hhc'][
                    'mmm_historical_runtime_minimum_sampling_updates_user_recorded'
                ]
            ),
            'mmm_historical_runtime_exact_step_count_status': (
                protocol_sampling['hhc'][
                    'mmm_historical_runtime_exact_step_count_status'
                ]
            ),
            'mmm_historical_runtime_outer_steps': (
                historical_local_history_audit['interpretation'][
                    'exact_historical_outer_steps'
                ]
            ),
            'mmm_historical_runtime_channel_updates': (
                historical_local_history_audit['interpretation'][
                    'exact_historical_channel_updates'
                ]
            ),
            'mmm_historical_runtime_local_history_audit_passes': (
                historical_local_history_audit.get('status') == 'pass'
                and all(
                    run.get('status') == 'pass'
                    for run in historical_local_history_audit.get(
                        'runs', {}
                    ).values()
                )
            ),
            'mmm_historical_runtime_vfs_content_ids_verified': all(
                all(
                    run[source_name]['recovery_manifest']['verified']
                    for source_name in ('model', 'evaluation', 'train')
                )
                for run in historical_local_history_audit['runs'].values()
            ),
            'intermask_author_default_steps': (
                protocol_sampling['intermask']['author_native_steps_audit']
            ),
            'intermask_release_default_source_matches': contains(
                os.path.join(args.intermask_repo, 'options/eval_option.py'),
                ('--time_steps', 'default=[20]'),
            ),
            'common_runner_passes_same_step_argument': contains(
                'tools/run_fair_interx_method.py',
                ("'--sampling-steps', str(args.sampling_steps)",
                 "'--time-steps', str(args.sampling_steps)"),
            ),
            'hhc_export_records_step_unit_and_bounded_compute_semantics': contains(
                'tools/export_hhc_official_interx.py',
                ("'sampling_steps_unit': 'outer_maskgit_iterations'",
                 "'maximum_active_channel_stages_per_outer_iteration': 3",
                 "'maximum_model_forward_calls_per_outer_iteration_with_cfg': 6",
                 "'resolved_channels_skip_forward_but_not_outer_iterations': True"),
            ),
        },
        'asset_protocol': {
            'mmm_data_links_match': all(
                (Path(args.data_root) / name).resolve()
                == (mmm_data_root / name).resolve()
                for name in link_names
            ),
            'mmm_evaluator_link_matches': (
                Path('checkpoints/hhi').resolve()
                == (Path(args.mmm_repo) / 'checkpoints/hhi').resolve()
            ),
            'mmm_smplx_link_matches': (
                Path('body_models/smplx').resolve()
                == (Path(args.mmm_repo) / 'body_models/smplx').resolve()
            ),
            'statistics': active_stats,
            'statistics_byte_identical_to_mmm': active_stats == mmm_stats,
            'motions': {
                split: h5_motion_asset(
                    os.path.join(args.data_root, 'motions', f'{split}.h5')
                )
                for split in ('train', 'val', 'test')
            },
            'smplx_model_sha256': {
                'male': sha256_file('body_models/smplx/SMPLX_MALE.npz'),
                'female': sha256_file('body_models/smplx/SMPLX_FEMALE.npz'),
            },
            'volumetric_smpl_checkpoint_sha256': {
                gender: sha256_file(
                    torch_checkpoints / f'VolumetricSMPL_smplx_{gender}.ckpt'
                )
                for gender in ('male', 'female')
            },
            'evaluator_options_sha256': sha256_file(
                'checkpoints/hhi/Comp_v6_KLD01/opt.txt'
            ),
        },
    }
    expected = {
        'source_protocol': {
            'mmm_repository': str(Path(args.mmm_repo).resolve()),
            'mmm_revision': 'a01d5accca54676694402b962de60ef1a23d54a7',
            'mmm_two_person_transformer_sha256': (
                'e29bef565bcde379fb273b1a265c6a58b8bb85fa951eb2e41eca014f03bfa7f7'
            ),
            'mmm_eval_trans_sha256': (
                'a9c67b14c43036f9b6bcd1cdf7496ee3c765337ec0df7206f40b8ff5fc9c6e1c'
            ),
            'mmm_train_t2i_trans_sha256': (
                '2c4551a627229c8ca70db7c59272e11aea1f200bd4a7ff73245070c8c5fd5b4f'
            ),
            'mmm_pre_cleanup_baseline_commit': (
                '75814e57e0702789176da6713056a15ed4a192e2'
            ),
            'mmm_cleanup_first_commit': (
                '0f58ffbb9db95a5d9ddc7f2e8a40b8602c17a325'
            ),
            'mmm_cleanup_parent_commit': (
                '75814e57e0702789176da6713056a15ed4a192e2'
            ),
            'mmm_pre_cleanup_is_immediate_parent': True,
            'mmm_pre_cleanup_tree': (
                'd6911e7f8e63ce6333b1707fd587c556be450cf6'
            ),
            'mmm_cleanup_tree': (
                'c1d3e14ecc49671516a0e0c38e6e8ec5feaf24ba'
            ),
            'mmm_pre_cleanup_source_records': {
                'models/t2i_trans.py': {
                    'git_blob': '123d7240deb0a42639dea7fe83a2b569f771db8d',
                    'sha256': (
                        '13069e9292c9f39e1271657def16b5e7cfcf7d6eeb827df3d145b96b00e36ce1'
                    ),
                    'bytes': len(pre_cleanup_sources['models/t2i_trans.py'].encode('utf-8')),
                },
                'utils/eval_trans.py': {
                    'git_blob': '64d902cc2fe09c79823c521f29676dc797b19017',
                    'sha256': (
                        '9113bc2d6d77af5ab0bb984df814f41965b837024c1c492d8916dd72a04cd139'
                    ),
                    'bytes': len(pre_cleanup_sources['utils/eval_trans.py'].encode('utf-8')),
                },
                'train_t2i_trans.py': {
                    'git_blob': 'ae6bf74f15376376c24ee64a4d5cdc8913b4fdb8',
                    'sha256': (
                        '36a1d00adfe0c1a7919334029fbce134dd99e3ebfa1d9b95e1a34c3ac7eae318'
                    ),
                    'bytes': len(pre_cleanup_sources['train_t2i_trans.py'].encode('utf-8')),
                },
                'generate_mm_samples.py': {
                    'git_blob': '43b0544d714f3fc96a51324290d97b81a0519e7a',
                    'sha256': (
                        '47552dfff5fcf1f9c2ece33b290a1a4e1072cc5dcf352c7f29efafa1326b960c'
                    ),
                    'bytes': len(pre_cleanup_sources['generate_mm_samples.py'].encode('utf-8')),
                },
                'options/option_transformer.py': {
                    'git_blob': 'a1378dcbcb37a376987b587bad3bedc0d5236922',
                    'sha256': (
                        '8168bc8a06df26e41da2bf4c385e1dabe58d1b0af08b6e4f30316b6ae59670d3'
                    ),
                    'bytes': len(pre_cleanup_sources['options/option_transformer.py'].encode('utf-8')),
                },
            },
            'mmm_pre_cleanup_transformer_path': 'models/t2i_trans.py',
            'mmm_pre_cleanup_transformer_sha256': (
                '13069e9292c9f39e1271657def16b5e7cfcf7d6eeb827df3d145b96b00e36ce1'
            ),
            'mmm_historical_runtime_log_sha256': {
                str(Path(args.mmm_repo) / 'wandb/run-20260217_033057-vq_mask_gen_4_ablation_adalnshape_1_cfg/files/output.log'): (
                    '5915fb53ecaa7b467c5ad9151738f6c65f05a401317a84560bcdd70a0941f631'
                ),
                str(Path(args.mmm_repo) / 'wandb/run-20260305_101848-vq_mask_gen_4_ablation_adalnshape_1_cfg_hbr_wocas/files/output.log'): (
                    '4a1108f6029ef42eb9fd394b0a8933fbe8f92f718b065b1440609e5725dbb73a'
                ),
            },
            'historical_evidence_priority': [
                'historical_runtime_logs',
                'local_history_time_bracketed_source',
                'pre_cleanup_git_tree',
                'cleaned_main',
            ],
            'mmm_historical_runtime_call_records': {
                str(Path(args.mmm_repo) / 'wandb/run-20260217_033057-vq_mask_gen_4_ablation_adalnshape_1_cfg/files/output.log'): [
                    {
                        'line': 107,
                        'text': 'index_motion = trans(feat_clip_text, word_emb, type="sample_intra_cas", m_length=pred_len, rand_pos=rand_pos, betas=betas, gender=gender, CFG=CFG)',
                        'sha256': '8fc37de2db1042e18a472fb2130df3e2a658c08d81ec62e325eafa002f3d96a7',
                    },
                ],
                str(Path(args.mmm_repo) / 'wandb/run-20260305_101848-vq_mask_gen_4_ablation_adalnshape_1_cfg_hbr_wocas/files/output.log'): [
                    {
                        'line': 60,
                        'text': 'index_motion = trans(feat_clip_text, word_emb, type="sample_intra_cas", m_length=pred_len, rand_pos=rand_pos, betas=betas, gender=gender, CFG=CFG)',
                        'sha256': '8fc37de2db1042e18a472fb2130df3e2a658c08d81ec62e325eafa002f3d96a7',
                    },
                ],
            },
            'mmm_historical_runtime_metadata_records': {
                str(Path(args.mmm_repo) / 'wandb/run-20260217_033057-vq_mask_gen_4_ablation_adalnshape_1_cfg/files/wandb-metadata.json'): {
                    'sha256': 'b629c9470952e38fe84c38ca614e9f33fa1b362bfa9cd3ea4e7a908ed429d80a',
                    'git_remote': 'https://github.com/exitudio/MMM.git',
                    'git_commit': '8c3e7ad107ed717b878779724958cf6eff00938e',
                    'program': '/home/wenning/projects/MMM/train_t2i_trans.py',
                    'code_path': 'train_t2i_trans.py',
                },
                str(Path(args.mmm_repo) / 'wandb/run-20260305_101848-vq_mask_gen_4_ablation_adalnshape_1_cfg_hbr_wocas/files/wandb-metadata.json'): {
                    'sha256': '3d8248796317e78f36c33910a08871248fd45e0ea4f22513c4c323c0f24a40b9',
                    'git_remote': 'https://github.com/exitudio/MMM.git',
                    'git_commit': '8c3e7ad107ed717b878779724958cf6eff00938e',
                    'program': '/home/wenning/projects/MMM/train_t2i_trans.py',
                    'code_path': 'train_t2i_trans.py',
                },
            },
            'mmm_historical_runtime_source_state': (
                'uncommitted_two_person_code_on_upstream_git_base'
            ),
            'mmm_historical_runtime_local_history_audit': {
                'path': 'output/historical_mmm_sampler_local_history_audit.json',
                'sha256': (
                    '70624e44a5346cf6586bce3e0c3f9db66854e517e90d0a23d905aded1cb3e953'
                ),
                'schema': 'historical_mmm_sampler_local_history_audit_v1',
                'status': 'pass',
                'scope': (
                    'read_only_recovery_of_uncommitted_runtime_source_not_git_history'
                ),
            },
        },
        'split_entries': 1708,
        'valid_entries': 1706,
        'evaluated_entries': 1696,
        'max_motion_length': 150,
        'checkpoint_selection_validation': {
            'split': str(Path(val_split_path).resolve()),
            'split_entries': 570,
            'eligible_entries': 569,
            'batch_size': 32,
            'drop_last': True,
            'evaluated_entries': 544,
            'excluded_entries': ['G035T005A029R021'],
            'clips_truncated_at_150': 299,
            'active_min_motion_frames': 24,
            'active_max_motion_frames': 150,
            'intermask_contract_matches': True,
            'official_interx_contract_matches': True,
            'mmm_development_native_min_motion_frames': 30,
            'mmm_development_native_max_motion_frames': 152,
            'mmm_development_contract_matches': True,
            'split_sha256': (
                '6c77e85017b14b512f42f2d540da2ce05874d9777d707bebd85782d6221f28a8'
            ),
            'motion_h5_sha256': (
                '91701e2f7687b7fb0985013fa2995aa9d06aad8e55340ad6c7e9881662249ffd'
            ),
            'assets_byte_identical_to_mmm': True,
        },
        'official_evaluator_options': True,
        'official_evaluator_weights': True,
        'official_evaluator_weights_sha256': (
            'b1699f13276c044d8b7ae5f6063eb241b7ef96d7e92c878857abd5081e40701e'
        ),
        'official_interx_source_matches': True,
        'intermask_source_matches': True,
        'sampling_protocol': {
            'hhc_common_steps': 50,
            'hhc_sampling_steps_unit': 'outer_maskgit_iterations',
            'hhc_maximum_active_channel_stages_per_outer_iteration': 3,
            'hhc_maximum_model_forward_calls_per_outer_iteration_with_cfg': 6,
            'hhc_maximum_common50_active_channel_stages': 150,
            'hhc_maximum_common50_model_forward_calls_with_cfg': 300,
            'hhc_resolved_channels_skip_forward_but_not_outer_iterations': True,
            'hhc_active_sampler_semantics': 'paper_high_confidence_tokens_fixed',
            'intermask_common_steps': 50,
            'hhc_temperature': 0.5,
            'hhc_cfg_scale': 3.5,
            'hhc_contact_guidance_scale': 1.0,
            'mmm_pre_cleanup_parallel_default_steps': 10,
            'mmm_pre_cleanup_cascade_default_outer_steps': 3,
            'mmm_pre_cleanup_cascade_default_temperature': None,
            'mmm_pre_cleanup_cascade_forward_passes_without_cfg': 9,
            'mmm_pre_cleanup_parameter_audit_scope': (
                'function_default_parameters_only_not_historical_runtime'
            ),
            'mmm_pre_cleanup_source_contact_guidance': 'linear_decay_with_0.2_gate',
            'mmm_pre_cleanup_source_token_update': 'overwrites_previous_fixed_tokens',
            'mmm_cleaned_main_contains_parallel_validation_call': True,
            'mmm_historical_runtime_sampler': 'sample_intra_cas',
            'mmm_historical_runtime_sampler_log_matches': True,
            'mmm_historical_runtime_minimum_sampling_updates_user_recorded': 30,
            'mmm_historical_runtime_exact_step_count_status': (
                'recovered_and_ast_verified'
            ),
            'mmm_historical_runtime_outer_steps': 10,
            'mmm_historical_runtime_channel_updates': 30,
            'mmm_historical_runtime_local_history_audit_passes': True,
            'mmm_historical_runtime_vfs_content_ids_verified': True,
            'intermask_author_default_steps': 20,
            'intermask_release_default_source_matches': True,
            'common_runner_passes_same_step_argument': True,
            'hhc_export_records_step_unit_and_bounded_compute_semantics': True,
        },
        'asset_protocol': {
            'mmm_data_links_match': True,
            'mmm_evaluator_link_matches': True,
            'mmm_smplx_link_matches': True,
            'statistics': {
                'Interx_Mean.npy': {
                    'shape': [672], 'dtype': 'float32', 'finite': True,
                    'sha256': 'a54dcfd012e07742077050e3a7ee1b4a2918ba2825466ab698d0cf6f9e438446',
                },
                'Interx_Std.npy': {
                    'shape': [672], 'dtype': 'float32', 'finite': True,
                    'sha256': '23a2cf52a19cbaab16334821f2a3183e915c671f59fac702b68a1256742bc376',
                },
                'Interx_Mean_root.npy': {
                    'shape': [672], 'dtype': 'float64', 'finite': True,
                    'sha256': '94a0421c4fdb416884fb74e075d46ce0357bd4301d8002f42ab7564f25648ed3',
                },
                'Interx_Std_root.npy': {
                    'shape': [672], 'dtype': 'float64', 'finite': True,
                    'sha256': 'e708d969a2708250bba8005aa6af16f43859773ab752d2999247267d268d8bc9',
                },
                'Mean.npy': {
                    'shape': [336], 'dtype': 'float32', 'finite': True,
                    'sha256': 'e21e25caf67251f07f9459b29c82a13de0d09042be9cb5a6c0045b473efb70bc',
                },
                'Std.npy': {
                    'shape': [336], 'dtype': 'float32', 'finite': True,
                    'sha256': 'c0cf0df8bdf881ca5a2f411025a29590855f4aeb25d08c40f21b2ed36de136f6',
                },
            },
            'statistics_byte_identical_to_mmm': True,
            'motions': {
                'train': {
                    'samples': 9110, 'feature_shapes': [[56, 6]],
                    'dtypes': ['float32'],
                    'sha256': '42e242fcd267c83b095591bb3445887ffc2578c1e45eb036bbd0188f174a7d2a',
                },
                'val': {
                    'samples': 570, 'feature_shapes': [[56, 6]],
                    'dtypes': ['float32'],
                    'sha256': '91701e2f7687b7fb0985013fa2995aa9d06aad8e55340ad6c7e9881662249ffd',
                },
                'test': {
                    'samples': 1708, 'feature_shapes': [[56, 6]],
                    'dtypes': ['float32'],
                    'sha256': 'd67fd5f24342aff843ff2588314f9ffe6e932fcc5590393e9660a306ce9d6957',
                },
            },
            'smplx_model_sha256': {
                'male': 'ab318e3f37d2bfaae26abf4e6fab445c2a610e1d63714794d60379cc263bc2a5',
                'female': 'b2a3686c9d6d218ff6822fba411c607a3c8125a70af340f384ce68bebecabe0e',
            },
            'volumetric_smpl_checkpoint_sha256': {
                'male': '96b22f6409e67dc950dafdd83bb8dc4779381ecc2b2c752e1c6fa3a205b2e18c',
                'female': '2b582f009c2daeec50c90b5f39689a248bac96a4cc9983548ad4be677b72ccdb',
            },
            'evaluator_options_sha256': (
                'f4ddaa81157259ce6b795f456be98263e50cd788a7800c30aacbfc2361b2d6dd'
            ),
        },
    }
    mismatches = {
        key: {'actual': report[key], 'expected': value}
        for key, value in expected.items() if report[key] != value
    }
    report['status'] = 'pass' if not mismatches else 'fail'
    report['mismatches'] = mismatches
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(
            args.output.suffix + f'.tmp.{os.getpid()}'
        )
        temporary.write_text(
            json.dumps(report, indent=2) + '\n', encoding='utf-8'
        )
        os.replace(temporary, args.output)
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
