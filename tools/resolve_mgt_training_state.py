#!/usr/bin/env python
"""Resolve a unique new, resumable, or completed MGT run without mutation."""

import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path


LATEST = re.compile(r'latest-(\d+)\.ckpt$')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def resolve_training_state(
    run_glob,
    experiment,
    expected_final_step,
    acknowledged_failed_run_ids=(),
    wandb_root='wandb',
):
    acknowledged_failed_run_ids = set(acknowledged_failed_run_ids)
    wandb_root = Path(wandb_root).resolve()
    directories = sorted(
        Path(path).resolve() for path in glob.glob(run_glob)
        if Path(path).is_dir()
    )
    records = []
    abandoned_attempts = []
    for directory in directories:
        if not directory.name.endswith('_' + experiment):
            raise ValueError(
                f'run directory does not match experiment {experiment}: '
                f'{directory}'
            )
        manifest = directory / 'training_manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError(
                f'ambiguous run directory has no training manifest: {manifest}'
            )
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        if payload.get('resolved_arguments', {}).get('exp_name') != experiment:
            raise ValueError(f'training manifest experiment mismatch: {manifest}')
        checkpoints = []
        for checkpoint in directory.glob('latest-*.ckpt'):
            match = LATEST.fullmatch(checkpoint.name)
            if match:
                checkpoints.append((int(match.group(1)), checkpoint.resolve()))
        if not checkpoints:
            run_id = payload.get('observability', {}).get('wandb_run_id')
            if run_id not in acknowledged_failed_run_ids:
                raise RuntimeError(
                    'run directory exists without an atomic latest checkpoint '
                    'and is not explicitly acknowledged as failed: '
                    f'{directory} (wandb_run_id={run_id!r})'
                )
            wandb_runs = sorted(wandb_root.glob(f'run-*-{run_id}'))
            if len(wandb_runs) != 1:
                raise RuntimeError(
                    f'acknowledged failed run {run_id} has {len(wandb_runs)} '
                    f'local W&B directories under {wandb_root}'
                )
            output_log = wandb_runs[0] / 'files/output.log'
            if not output_log.is_file():
                raise FileNotFoundError(
                    f'acknowledged failed run has no W&B output log: {output_log}'
                )
            output_text = output_log.read_text(encoding='utf-8', errors='replace')
            exception_lines = [
                line for line in output_text.splitlines()
                if re.match(r'^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception):', line)
            ]
            if 'Traceback (most recent call last):' not in output_text or not exception_lines:
                raise RuntimeError(
                    f'acknowledged failed run has no terminal traceback: {output_log}'
                )
            abandoned_attempts.append({
                'directory': str(directory),
                'manifest': str(manifest.resolve()),
                'manifest_sha256': sha256_file(manifest),
                'wandb_run_id': run_id,
                'wandb_output_log': str(output_log.resolve()),
                'wandb_output_log_sha256': sha256_file(output_log),
                'terminal_exception': exception_lines[-1],
                'latest_checkpoint': None,
                'latest_step': 0,
                'disposition': 'preserved_failed_precheckpoint_attempt',
            })
            continue
        step, checkpoint = max(checkpoints)
        if step > expected_final_step:
            raise ValueError(
                f'checkpoint exceeds expected final step: {checkpoint} ({step})'
            )
        records.append({
            'directory': str(directory),
            'manifest': str(manifest.resolve()),
            'manifest_sha256': sha256_file(manifest),
            'latest_checkpoint': str(checkpoint),
            'latest_checkpoint_bytes': checkpoint.stat().st_size,
            'latest_step': step,
        })

    completed = [
        record for record in records
        if record['latest_step'] == expected_final_step
    ]
    incomplete = [
        record for record in records
        if record['latest_step'] < expected_final_step
    ]
    if len(completed) > 1:
        raise RuntimeError(f'multiple completed runs are ambiguous: {completed}')
    if completed and incomplete:
        raise RuntimeError(
            'completed and incomplete same-experiment runs coexist; refusing '
            f'to guess lineage: completed={completed}, incomplete={incomplete}'
        )
    if completed:
        status = 'complete'
        active = completed[0]
    elif len(incomplete) == 1:
        status = 'resume'
        active = incomplete[0]
    elif not records:
        status = 'new'
        active = None
    else:
        raise RuntimeError(
            f'multiple incomplete runs are ambiguous; refusing restart: {incomplete}'
        )

    result = {
        'schema': 'hhc_mgt_training_state_resolution_v2',
        'status': status,
        'experiment': experiment,
        'expected_final_step': expected_final_step,
        'matched_runs': records,
        'abandoned_attempts': abandoned_attempts,
        'active_run': active,
        'resume_checkpoint': (
            active['latest_checkpoint'] if status == 'resume' else None
        ),
        'training_required': status != 'complete',
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-glob', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--expected-final-step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--acknowledge-failed-no-checkpoint-run-id', action='append', default=[],
        help=(
            'Explicit W&B run ID for a preserved pre-checkpoint attempt. The '
            'resolver still requires a local W&B output log ending in a traceback.'
        ),
    )
    parser.add_argument('--wandb-root', default='wandb')
    args = parser.parse_args()
    result = resolve_training_state(
        args.run_glob, args.experiment, args.expected_final_step,
        acknowledged_failed_run_ids=(
            args.acknowledge_failed_no_checkpoint_run_id
        ),
        wandb_root=args.wandb_root,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
