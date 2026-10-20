#!/usr/bin/env python
"""Freeze the completed historical MMM W&B trajectory for offline auditing."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import wandb


RUN_PATH = 'edmond_gla/NewInter/vq_mask_gen_4_ablation_adalnshape_1_cfg'
HISTORY_ARTIFACT = (
    'edmond_gla/NewInter/'
    'run-vq_mask_gen_4_ablation_adalnshape_1_cfg-history:v0'
)
METRICS = (
    'trainer/global_step', 'epoch', 'val/FID', 'val/Diversity',
    'val/TOP1', 'val/TOP2', 'val/TOP3', 'val/Matching',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--local-summary', type=Path,
        default=Path(
            '/home/wenning/projects/MMM/wandb/'
            'run-20260217_140446-vq_mask_gen_4_ablation_adalnshape_1_cfg/'
            'files/wandb-summary.json'
        ),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/historical_mmm_wandb_remote_history.json'),
    )
    args = parser.parse_args()

    local_summary = json.loads(args.local_summary.read_text(encoding='utf-8'))
    api = wandb.Api(timeout=60)
    run = api.run(RUN_PATH)
    artifact = api.artifact(HISTORY_ARTIFACT)
    rows = []
    for row in run.scan_history(keys=list(METRICS), page_size=1000):
        if row.get('val/FID') is not None:
            rows.append({key: row.get(key) for key in METRICS})
    expected_steps = list(range(9999, 300000, 10000))
    actual_steps = [int(row['trainer/global_step']) for row in rows]
    if actual_steps != expected_steps:
        raise ValueError(
            f'historical validation steps changed: {actual_steps}'
        )
    remote_summary = dict(run.summary)
    summary_keys = (
        'trainer/global_step', 'Train/all', 'Train/every_token',
        'Train/masked', 'Train/no_masked', 'val/FID', 'val/Diversity',
        'val/TOP1', 'val/TOP2', 'val/TOP3', 'val/Matching',
    )
    summary_matches = {
        key: local_summary.get(key) == remote_summary.get(key)
        for key in summary_keys
    }
    if not all(summary_matches.values()):
        raise ValueError('remote W&B summary differs from local frozen summary')
    files = sorted(
        [
            {'name': item.name, 'size': item.size}
            for item in run.files()
        ],
        key=lambda item: item['name'],
    )
    source_names = {
        item['name'] for item in files
        if item['name'].startswith('code/')
        or item['name'].endswith('diff.patch')
    }
    payload = {
        'schema': 'hhc_historical_mmm_wandb_remote_history_v1',
        'status': 'pass',
        'run_path': RUN_PATH,
        'run_id': run.id,
        'run_name': run.name,
        'run_state': run.state,
        'history_artifact': {
            'qualified_name': HISTORY_ARTIFACT,
            'name': artifact.name,
            'type': artifact.type,
            'version': artifact.version,
            'state': artifact.state,
            'size': artifact.size,
            'digest': artifact.digest,
        },
        'local_summary': {
            'path': str(args.local_summary.resolve()),
            'sha256': sha256_file(args.local_summary),
        },
        'server_summary_exact_match': all(summary_matches.values()),
        'server_summary_field_matches': summary_matches,
        'validation_history': rows,
        'validation_milestones': len(rows),
        'run_files': files,
        'source_code_archive_available': bool(source_names),
        'source_code_archive_names': sorted(source_names),
        'interpretation': (
            'historical trajectory reference only; different VQ, architecture, '
            'seed and validation sampling protocol from the paper-config run'
        ),
        'implementation_sha256': sha256_file(__file__),
    }
    if run.state != 'finished' or payload['source_code_archive_available']:
        raise ValueError('historical run state/source-archive classification changed')
    atomic_json(args.output, payload)
    print(json.dumps({
        'output': str(args.output.resolve()),
        'milestones': len(rows),
        'server_summary_exact_match': True,
        'source_code_archive_available': False,
    }))


if __name__ == '__main__':
    main()
