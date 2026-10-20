#!/usr/bin/env python
"""Bind main-model training wall times and update/sample budgets to artifacts."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        'path': str(path), 'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def parse_time(value):
    if not value:
        raise ValueError('training manifest has no launch timestamp')
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError(f'training timestamp has no timezone: {value!r}')
    return parsed


def checkpoint_end_time(path):
    return datetime.fromtimestamp(
        Path(path).stat().st_mtime, tz=timezone.utc
    )


def observe_gpu_hardware():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,name,uuid',
        '--format=csv,noheader,nounits',
    ], text=True)
    gpus = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        index, name, uuid = (part.strip() for part in line.split(',', 2))
        gpus[int(index)] = {
            'index': int(index), 'name': name, 'uuid': uuid,
        }
    if not gpus:
        raise ValueError('nvidia-smi returned no GPU hardware records')
    return gpus


def build_record(name, manifest_path, checkpoint_path, expected_updates,
                 effective_batch_size, state_audit_path=None,
                 completion_time_key=None, assigned_gpu_index=None,
                 gpu_hardware=None):
    manifest_path = Path(manifest_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    launch = parse_time(
        manifest.get('launch_time_utc') or manifest.get('launch_time')
    )
    if completion_time_key and manifest.get(completion_time_key):
        completed = parse_time(manifest[completion_time_key])
        endpoint = f'training_manifest.{completion_time_key}'
    else:
        completed = checkpoint_end_time(checkpoint_path)
        endpoint = 'final_checkpoint_mtime_utc'
    elapsed = (completed.astimezone(timezone.utc)
               - launch.astimezone(timezone.utc)).total_seconds()
    if elapsed <= 0:
        raise ValueError(f'{name} has non-positive wall time: {elapsed}')

    record = {
        'training_manifest': artifact(manifest_path),
        'final_checkpoint': artifact(checkpoint_path),
        'launch_time': launch.isoformat(),
        'completion_time': completed.isoformat(),
        'completion_endpoint': endpoint,
        'elapsed_seconds': elapsed,
        'elapsed_hours': elapsed / 3600.0,
        'optimizer_updates': int(expected_updates),
        'effective_batch_size': int(effective_batch_size),
        'sample_exposures': int(expected_updates * effective_batch_size),
    }
    if assigned_gpu_index is not None:
        assigned_gpu_index = int(assigned_gpu_index)
        if assigned_gpu_index not in gpu_hardware:
            raise ValueError(
                f'{name} assigned GPU {assigned_gpu_index} was not observed'
            )
        record['assigned_gpu_index'] = assigned_gpu_index
        record['gpu_name_observed_at_audit'] = gpu_hardware[
            assigned_gpu_index
        ]['name']
        record['gpu_uuid_observed_at_audit'] = gpu_hardware[
            assigned_gpu_index
        ]['uuid']
    if state_audit_path is not None:
        audit_path = Path(state_audit_path).resolve()
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        audit_step = audit.get('global_step', audit.get('iteration'))
        if audit.get('status') != 'pass' or int(audit_step) != expected_updates:
            raise ValueError(
                f'{name} final state audit is not pass at {expected_updates}'
            )
        if (Path(audit['checkpoint']).resolve() != checkpoint_path
                or audit.get('checkpoint_sha256')
                != record['final_checkpoint']['sha256']):
            raise ValueError(f'{name} final checkpoint differs from state audit')
        record['final_state_audit'] = artifact(audit_path)
    return record


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--hhc-selection', type=Path,
        default=Path('output/hhc_formal_raw12288_repaired_mgt_selection.json'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/training_wallclock_audit.json'),
    )
    args = parser.parse_args()

    selection_path = args.hhc_selection.resolve()
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if (selection.get('run_complete') is not True
            or selection.get('expected_final_step') != 300000):
        raise ValueError('HHC selection does not prove a complete 300k run')

    gpu_hardware = observe_gpu_hardware()
    cosh_selection_path = Path(
        'output/coshmdm_interx_val_selection.json'
    ).resolve()
    cosh_selection = json.loads(
        cosh_selection_path.read_text(encoding='utf-8')
    )
    if (cosh_selection.get('status') != 'complete'
            or cosh_selection.get('test_metrics_used_for_selection') is not False):
        raise ValueError('CoShMDM capped validation selection is incomplete')
    cosh_selected_step = int(cosh_selection['selected_step'])
    cosh_selected_checkpoint = cosh_selection['selected_checkpoint']

    records = {
        'hhc_mgt': build_record(
            'hhc_mgt', selection['training_manifest'],
            selection['final_latest_checkpoint'], 300000, 96,
            assigned_gpu_index=1, gpu_hardware=gpu_hardware,
        ),
        'coshmdm': build_record(
            'coshmdm',
            '/mnt/8T-HDD/wenning/coshmdm-code/checkpoints/'
            'CoShMDM-InterX-fair-statsfix-seed123-bs32x2/meta/training_manifest.json',
            cosh_selected_checkpoint,
            cosh_selected_step, 64, 'output/coshmdm_final_checkpoint_audit.json',
            assigned_gpu_index=0, gpu_hardware=gpu_hardware,
        ),
        'timotion': build_record(
            'timotion',
            '/mnt/8T-HDD/wenning/TIMotion/checkpoints/'
            'TIMotion-Transformer-LPA-InterX-fair-seed123-bs32x8/'
            'meta/training_manifest.json',
            '/mnt/8T-HDD/wenning/TIMotion/checkpoints/'
            'TIMotion-Transformer-LPA-InterX-fair-seed123-bs32x8/'
            'model/last.ckpt',
            106500, 256,
            'output/timotion_interx_final_checkpoint_audit.json',
            completion_time_key='completed_time_utc',
            assigned_gpu_index=0, gpu_hardware=gpu_hardware,
        ),
    }
    records['hhc_mgt']['selection_manifest'] = artifact(selection_path)
    records['coshmdm']['selection_manifest'] = artifact(cosh_selection_path)
    records['coshmdm']['training_budget_cap'] = artifact(
        'output/coshmdm_training_budget_cap.json'
    )
    records['coshmdm']['original_author_target_optimizer_updates'] = 568000
    records['coshmdm']['training_budget_contract'] = (
        'reported checkpoint selected on held-out validation within the '
        'user-imposed two-day wall-clock window'
    )
    if (records['hhc_mgt']['training_manifest']['sha256']
            != selection.get('training_manifest_sha256')):
        raise ValueError('HHC training manifest hash differs from selection')
    if (records['hhc_mgt']['final_checkpoint']['sha256']
            != selection.get('final_latest_checkpoint_sha256')):
        raise ValueError('HHC final checkpoint hash differs from selection')
    final_state = selection.get('final_state_audit', {})
    if (final_state.get('status') != 'pass'
            or final_state.get('global_step') != 300000
            or final_state.get('model_state', {}).get(
                'all_floating_tensors_finite'
            ) is not True
            or final_state.get('optimizer_state', {}).get(
                'all_floating_tensors_finite'
            ) is not True):
        raise ValueError('HHC final 300k checkpoint state audit is invalid')

    payload = {
        'schema': 'hhc_fair_training_wallclock_v1',
        'status': 'pass',
        'wall_time_scope': (
            'launch-to-final-checkpoint wall time including validation and '
            'checkpoint I/O; HHC MGT uses its final checkpoint and CoShMDM '
            'uses the validation-selected milestone inside the disclosed '
            'two-day cap, '
            'TiMotion records its exact post-fit completion time. The frozen '
            'raw12288 HHC VQ is imported from the pre-cleanup MMM run and is '
            'therefore excluded from current-study retraining wall time.'
        ),
        'hardware_observation': {
            'captured_at_utc': datetime.now(timezone.utc).isoformat(),
            'source': 'nvidia-smi --query-gpu=index,name,uuid',
            'gpus': [gpu_hardware[index] for index in sorted(gpu_hardware)],
        },
        'records': records,
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({
        'output': str(args.output.resolve()),
        'hours': {name: record['elapsed_hours']
                  for name, record in records.items()},
    }, indent=2))


if __name__ == '__main__':
    main()
