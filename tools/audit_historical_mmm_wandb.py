#!/usr/bin/env python
"""Freeze the non-comparable historical MMM W&B reference configuration."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

try:
    from tools.audit_historical_mmm_local_history import (
        eval_record as local_history_eval_record,
        parse_iso,
        sampler_record as local_history_sampler_record,
        validation_train_record,
    )
except ModuleNotFoundError:  # Direct `python tools/...py` execution.
    from audit_historical_mmm_local_history import (
        eval_record as local_history_eval_record,
        parse_iso,
        sampler_record as local_history_sampler_record,
        validation_train_record,
    )


DEFAULT_RUN = Path(
    '/home/wenning/projects/MMM/wandb/'
    'run-20260217_140446-vq_mask_gen_4_ablation_adalnshape_1_cfg/files'
)
ROOT = Path(__file__).resolve().parents[1]
MMM_ROOT = Path('/home/wenning/projects/MMM')
COMPLETED_RUN_LOCAL_HISTORY = {
    'model': ROOT / (
        'output/historical_mmm_local_history/'
        '1771844856924_set-428941_change-428940_ContentChange.py'
    ),
    'train': ROOT / (
        'output/historical_mmm_local_history_train/'
        '1772113378759_set-429082_change-429081_ContentChange.py'
    ),
    'evaluation': ROOT / (
        'output/historical_mmm_local_history_eval/'
        '1772239498876_set-430119_change-430118_ContentChange.py'
    ),
}


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def wandb_value(config, key):
    return config[key]['value']


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scan_offline_wandb(path, include_history=True):
    """Read an offline W&B transaction log without requiring a sync."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(str(path))
    run = None
    latest = {}
    validation_by_step = {}
    last_global_step = None
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.HasField('run') and run is None:
            config = {
                item.key: json.loads(item.value_json)
                for item in record.run.config.update
            }
            run = {
                'run_id': record.run.run_id,
                'project': record.run.project,
                'display_name': record.run.display_name,
                'start_time_unix': record.run.start_time.seconds,
                'config': config,
            }
            if not include_history:
                break
        if include_history and record.HasField('history'):
            row = {}
            for item in record.history.item:
                key = item.key or '/'.join(item.nested_key)
                row[key] = json.loads(item.value_json)
            if 'trainer/global_step' in row:
                last_global_step = int(row['trainer/global_step'])
            if last_global_step is not None:
                row.setdefault('trainer/global_step', last_global_step)
            latest.update(row)
            validation_keys = (
                'val/FID', 'val/Diversity', 'val/TOP1', 'val/TOP2',
                'val/TOP3', 'val/Matching',
            )
            if any(key in row for key in validation_keys):
                if last_global_step is None:
                    raise ValueError('validation W&B row has no global step')
                validation = validation_by_step.setdefault(
                    last_global_step,
                    {
                        'trainer/global_step': last_global_step,
                        'completed_updates': last_global_step + 1,
                    },
                )
                validation.update({
                    key: row[key] for key in validation_keys if key in row
                })
    if run is None:
        raise ValueError(f'W&B transaction log contains no run record: {path}')
    if include_history:
        run['latest_training_metrics'] = {
            key: latest[key]
            for key in (
                'trainer/global_step', 'epoch', 'Train/all',
                'Train/every_token', 'Train/masked', 'Train/no_masked',
                'Train/mask_ratio_root', 'Train/mask_ratio_body',
                'Train/mask_ratio_hand',
            )
            if key in latest
        }
        run['validation_history'] = [
            validation_by_step[step] for step in sorted(validation_by_step)
        ]
    return run


def discover_current_wandb(root, current_out_dir, display_name):
    matches = []
    for path in sorted(Path(root).glob('offline-run-*/run-*.wandb')):
        run = scan_offline_wandb(path, include_history=False)
        configured_out_dir = str(run['config'].get('out_dir', '')).rstrip('/')
        if (run['display_name'] == display_name
                and configured_out_dir == str(current_out_dir).rstrip('/')):
            matches.append(path.resolve())
    if len(matches) != 1:
        raise RuntimeError(
            f'expected one W&B log for {display_name!r} and '
            f'out_dir={current_out_dir!r}, got {matches}'
        )
    return matches[0]


def completed_historical_validation_runtime(historical_run_files, old_fields):
    metadata_path = historical_run_files / 'wandb-metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    started_at = parse_iso(metadata['startedAt'])
    model = local_history_sampler_record(
        COMPLETED_RUN_LOCAL_HISTORY['model'], rand_pos=False, cfg=-1
    )
    train = validation_train_record(COMPLETED_RUN_LOCAL_HISTORY['train'])
    evaluation = local_history_eval_record(
        COMPLETED_RUN_LOCAL_HISTORY['evaluation']
    )
    interval_checks = {}
    for name, record in (
        ('model', model), ('train', train), ('evaluation', evaluation)
    ):
        interval = record['interval']
        lower = interval['state_valid_from_previous_change_epoch_ms'] / 1000
        upper = interval['state_valid_until_snapshot_change_epoch_ms'] / 1000
        interval_checks[name] = lower <= started_at.timestamp() < upper
    checks = {
        'wandb_start_inside_all_source_intervals': all(
            interval_checks.values()
        ),
        'all_sources_bound_to_vfs_content_ids': all(
            record['recovery_manifest']['verified']
            for record in (model, train, evaluation)
        ),
        'upstream_git_base_matches': metadata.get('git', {}).get('commit')
        == '8c3e7ad107ed717b878779724958cf6eff00938e',
        'historical_config_disables_contact': old_fields.get('contact') is False,
        'cascade_order_root_body_hand':
        model['cascade_order_channel_indices'] == [2, 0, 1],
        'model_default_outer_steps_10': model['defaults'].get('max_steps') == 10,
        'thirty_channel_updates': model['channel_updates'] == 30,
        'temperature_zero_for_validation_rand_pos_false':
        model['selected_temperature'] == 0.0,
        'validation_cfg_disabled': train['keyword_literals'].get('CFG') == -1,
        'validation_contact_reads_training_flag':
        train['keyword_expressions'].get('contact') == 'self.contact',
        'validation_rand_pos_uses_false_default': train['rand_pos_omitted']
        and evaluation['function_defaults'].get('rand_pos') is False,
        'validation_omits_explicit_sampler_steps': train['max_steps_omitted']
        and evaluation['all_calls_omit_max_steps'],
        'validation_calls_cascade_sampler':
        evaluation['sample_intra_cas_call_count'] > 0,
        'validation_uses_raw_normalizer': evaluation['raw_normalizer'],
    }
    if not all(checks.values()):
        raise ValueError(
            f'completed historical validation runtime audit failed: {checks}'
        )
    return {
        'status': 'pass',
        'wandb_run': {
            'directory': str(historical_run_files.parent.resolve()),
            'started_at': metadata['startedAt'],
            'metadata_path': str(metadata_path.resolve()),
            'metadata_sha256': sha256_file(metadata_path),
            'git': metadata.get('git'),
        },
        'source_interval_checks': interval_checks,
        'model': model,
        'train': train,
        'evaluation': evaluation,
        'effective_validation_runtime': {
            'sampler': 'sample_intra_cas',
            'outer_steps': 10,
            'channel_updates': 30,
            'cascade_order': 'root_body_hand',
            'rand_pos': False,
            'temperature': 0.0,
            'cfg_scale': -1,
            'contact': False,
            'model_forward_calls_per_generation': 30,
        },
        'checks': checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-run-files', type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        '--historical-remote-history', type=Path,
        default=Path('output/historical_mmm_wandb_remote_history.json'),
    )
    parser.add_argument(
        '--current-manifest', type=Path,
        default=None,
    )
    parser.add_argument(
        '--selection-manifest', type=Path,
        default=Path('output/hhc_formal_raw12288_repaired_mgt_selection.json'),
        help='When supplied, take the completed run manifest bound by selection.',
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/historical_mmm_wandb_comparison.json'),
    )
    parser.add_argument('--current-wandb-file', type=Path)
    parser.add_argument('--current-wandb-root', type=Path, default=Path('wandb'))
    parser.add_argument('--expected-current-final-step', type=int, default=300000)
    args = parser.parse_args()

    config_path = args.historical_run_files / 'config.yaml'
    summary_path = args.historical_run_files / 'wandb-summary.json'
    historical_config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    historical_summary = json.loads(summary_path.read_text(encoding='utf-8'))
    historical_remote_path = args.historical_remote_history.resolve()
    historical_remote = json.loads(
        historical_remote_path.read_text(encoding='utf-8')
    )
    remote_steps = [
        record.get('trainer/global_step')
        for record in historical_remote.get('validation_history', [])
    ]
    if not (
        historical_remote.get('schema')
        == 'hhc_historical_mmm_wandb_remote_history_v1'
        and historical_remote.get('status') == 'pass'
        and historical_remote.get('run_state') == 'finished'
        and historical_remote.get('server_summary_exact_match') is True
        and historical_remote.get('source_code_archive_available') is False
        and historical_remote.get('validation_milestones') == 30
        and remote_steps == list(range(9999, 300000, 10000))
        and historical_remote.get('local_summary', {}).get('sha256')
        == sha256_file(summary_path)
        and historical_remote.get('implementation_sha256')
        == sha256_file('tools/freeze_historical_mmm_wandb_remote_history.py')
    ):
        raise ValueError('frozen historical W&B remote trajectory is stale')
    current_manifest_path = args.current_manifest
    if args.selection_manifest is not None:
        selection = json.loads(
            args.selection_manifest.read_text(encoding='utf-8')
        )
        if selection.get('run_complete') is not True:
            raise ValueError('current MGT selection is not complete')
        current_manifest_path = Path(selection['training_manifest'])
    if current_manifest_path is None:
        raise ValueError(
            'provide --selection-manifest or --current-manifest explicitly'
        )
    current_manifest = json.loads(
        current_manifest_path.read_text(encoding='utf-8')
    )
    current = current_manifest['resolved_arguments']
    current_wandb_file = args.current_wandb_file
    if current_wandb_file is None:
        current_wandb_file = discover_current_wandb(
            args.current_wandb_root, current['out_dir'], current['exp_name']
        )
    current_wandb = scan_offline_wandb(current_wandb_file)
    current_step = int(
        current_wandb['latest_training_metrics'].get('trainer/global_step', -1)
    )
    expected_wandb_step = args.expected_current_final_step - 1
    if current_step != expected_wandb_step:
        raise ValueError(
            f'current W&B run is not complete: expected zero-based W&B '
            f'global step {expected_wandb_step} for '
            f'{args.expected_current_final_step} completed updates, got '
            f'{current_step}'
        )
    current_wandb['latest_training_metrics']['completed_updates'] = (
        args.expected_current_final_step
    )
    if not current_wandb['validation_history']:
        raise ValueError('current W&B run contains no validation metrics')
    required_validation = {
        'trainer/global_step', 'val/FID', 'val/Diversity', 'val/TOP1',
        'val/TOP2', 'val/TOP3', 'val/Matching',
    }
    for record in current_wandb['validation_history']:
        missing = sorted(required_validation - set(record))
        if missing:
            raise ValueError(
                f'incomplete W&B validation record at step '
                f'{record.get("trainer/global_step")}: {missing}'
            )

    old_fields = {
        key: wandb_value(historical_config, key)
        for key in (
            'batch_size', 'total_iter', 'warm_up_iter', 'lr', 'weight_decay',
            'nb_code', 'embed_dim_gpt', 'num_layers', 'num_local_layer',
            'n_head_gpt', 'ff_rate', 'contact', 'shape', 'seed', 'resume_pth',
        )
    }
    current_fields = {
        key: current[key]
        for key in (
            'batch_size', 'total_iter', 'warm_up_iter', 'lr', 'weight_decay',
            'nb_code', 'embed_dim_gpt', 'num_layers', 'num_local_layer',
            'n_head_gpt', 'ff_rate', 'contact', 'shape', 'seed',
            'vq_checkpoint', 'paper_architecture', 'sampler', 'temperature',
            'sampling_steps', 'cfg_scale', 'contact_guidance_scale',
        )
    }
    historical_vq_checkpoint = (
        MMM_ROOT / str(old_fields['resume_pth']).lstrip('./')
    ).resolve()
    current_vq_checkpoint = Path(current['vq_checkpoint'])
    if not current_vq_checkpoint.is_absolute():
        current_vq_checkpoint = (ROOT / current_vq_checkpoint).resolve()
    historical_vq_hash = sha256_file(historical_vq_checkpoint)
    current_vq_hash = sha256_file(current_vq_checkpoint)
    same_vq_checkpoint = historical_vq_hash == current_vq_hash
    if old_fields['nb_code'] != 12288 or current_fields['nb_code'] != 12288:
        raise ValueError('historical/current MGT comparison is not raw12288')
    if not same_vq_checkpoint:
        raise ValueError(
            'historical and current MGT runs do not share the byte-identical VQ'
        )
    historical_validation_runtime = completed_historical_validation_runtime(
        args.historical_run_files, old_fields
    )
    result = {
        'schema': 'hhc_historical_mmm_wandb_comparison_v1',
        'status': 'pass',
        'historical_run': {
            'name': 'vq_mask_gen_4_ablation_adalnshape_1_cfg',
            'config_path': str(config_path.resolve()),
            'summary_path': str(summary_path.resolve()),
            'configuration': old_fields,
            'final_metrics': {
                key: historical_summary[key]
                for key in (
                    'trainer/global_step', 'Train/all', 'Train/every_token',
                    'Train/masked', 'Train/no_masked', 'val/FID',
                    'val/Diversity', 'val/TOP1', 'val/TOP2', 'val/TOP3',
                    'val/Matching',
                )
            },
            'validation_history': historical_remote['validation_history'],
            'remote_history_artifact': {
                'path': str(historical_remote_path),
                'sha256': sha256_file(historical_remote_path),
            },
            'remote_history_provenance': {
                'run_path': historical_remote['run_path'],
                'history_artifact': historical_remote['history_artifact'],
                'server_summary_exact_match': True,
                'source_code_archive_available': False,
            },
            'validation_runtime': historical_validation_runtime,
        },
        'current_run': {
            'manifest': str(current_manifest_path.resolve()),
            'configuration': current_fields,
            'wandb_file': str(current_wandb_file.resolve()),
            'wandb_file_sha256': sha256_file(current_wandb_file),
            'wandb_run_id': current_wandb['run_id'],
            'wandb_project': current_wandb['project'],
            'wandb_display_name': current_wandb['display_name'],
            'latest_training_metrics': current_wandb['latest_training_metrics'],
            'validation_history': current_wandb['validation_history'],
            'metric_status': (
                'recorded for trajectory reference; val/FID is the current '
                'milestone value, while val/Diversity, val/TOP1-3 and '
                'val/Matching are running-best state emitted by the legacy '
                'validation hook; numerical comparison to the historical run '
                'is not fair because configurations differ'
            ),
        },
        'comparability': {
            'same_configuration': False,
            'same_byte_identical_vq_checkpoint': True,
            'vq_checkpoint_sha256': historical_vq_hash,
            'historical_vq_checkpoint': str(historical_vq_checkpoint),
            'current_vq_checkpoint': str(current_vq_checkpoint),
            'reasons': [
                'legacy 1024-d/16-head architecture vs paper 512-d/8-head architecture',
                (
                    'both validation paths use sample_intra_cas; the '
                    'time-bracketed historical run uses 10 outer steps / 30 '
                    'channel updates, temperature 0, CFG disabled and no '
                    'contact, whereas the current run uses 50 / 150, '
                    'temperature 0.5, CFG 3.5 and validation-oracle contact'
                ),
                'legacy validation/export path vs frozen common evaluator protocol',
                'different batch size and random seed',
            ],
            'allowed_use': 'historical reference only; not a fair model comparison',
        },
        'implementation': {
            'audit': {
                'path': str(Path(__file__).resolve()),
                'bytes': Path(__file__).resolve().stat().st_size,
                'sha256': sha256_file(Path(__file__).resolve()),
            },
            'local_history_parser': {
                'path': str((ROOT / 'tools/audit_historical_mmm_local_history.py').resolve()),
                'bytes': (ROOT / 'tools/audit_historical_mmm_local_history.py').stat().st_size,
                'sha256': sha256_file(
                    ROOT / 'tools/audit_historical_mmm_local_history.py'
                ),
            },
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output.resolve(), result)
    print(json.dumps({
        'output': str(args.output.resolve()),
        'historical_val_fid': historical_summary['val/FID'],
        'same_configuration': False,
    }))


if __name__ == '__main__':
    main()
