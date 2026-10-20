#!/usr/bin/env python
"""Audit that retained MGT validation checkpoints equal the exact FID top-k."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


MODEL = re.compile(r'model-(\d+)-([0-9]+(?:\.[0-9]+)?)\.ckpt$')


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--validation-audit', type=Path, required=True)
    parser.add_argument('--target-step', type=int, required=True)
    parser.add_argument('--save-top-k', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    validation_path = args.validation_audit.resolve()
    validation = json.loads(validation_path.read_text(encoding='utf-8'))
    if validation.get('status') != 'pass' or validation.get(
            'maximum_absolute_difference') != 0.0:
        raise ValueError('validation comparison is not an exact replay')

    history = []
    for comparison in validation.get('comparisons', []):
        global_step = int(comparison['trainer/global_step'])
        checkpoint_step = global_step + 1
        if checkpoint_step > args.target_step:
            continue
        fid = float(comparison['candidate']['val/FID'])
        history.append({'step': checkpoint_step, 'fid': fid})
    target = [item for item in history if item['step'] == args.target_step]
    if len(target) != 1:
        raise ValueError(f'target step is absent or duplicated: {target}')

    ranking = sorted(history, key=lambda item: (item['fid'], item['step']))
    expected = ranking[:args.save_top_k]
    expected_steps = [item['step'] for item in expected]
    target_rank = next(
        index for index, item in enumerate(ranking, start=1)
        if item['step'] == args.target_step
    )

    retained = []
    for path in sorted(run_dir.glob('model-*.ckpt')):
        match = MODEL.fullmatch(path.name)
        if match is None:
            raise ValueError(f'invalid model checkpoint name: {path.name}')
        retained.append({
            'step': int(match.group(1)),
            'filename_fid': float(match.group(2)),
            'path': str(path),
            'bytes': path.stat().st_size,
        })
    actual_steps = [item['step'] for item in retained]
    if len(retained) != args.save_top_k:
        raise ValueError(
            f'expected {args.save_top_k} retained files, got {len(retained)}'
        )
    if sorted(actual_steps) != sorted(expected_steps):
        raise ValueError(
            f'top-k retention differs: expected={expected_steps}, '
            f'actual={actual_steps}'
        )
    exact_by_step = {item['step']: item['fid'] for item in history}
    for item in retained:
        if abs(item['filename_fid'] - exact_by_step[item['step']]) > 5.1e-5:
            raise ValueError(f'filename FID mismatch: {item}')

    target_retained = args.target_step in actual_steps
    expected_target_retained = target_rank <= args.save_top_k
    if target_retained != expected_target_retained:
        raise ValueError(
            'target retention differs from ranking: '
            f'rank={target_rank}, retained={target_retained}'
        )

    result = {
        'schema': 'hhc_mgt_topk_retention_audit_v1',
        'status': 'pass',
        'run_dir': str(run_dir),
        'save_top_k': args.save_top_k,
        'target': {
            'step': args.target_step,
            'fid': target[0]['fid'],
            'rank': target_rank,
            'retained': target_retained,
        },
        'expected_top_k': expected,
        'retained_checkpoints': retained,
        'validation_audit': {
            'path': str(validation_path),
            'bytes': validation_path.stat().st_size,
            'sha256': sha256_file(validation_path),
        },
        'implementation': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
