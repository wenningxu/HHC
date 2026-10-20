#!/usr/bin/env python3
"""Resolve exactly one new/resume/complete MGT diagnostic run safely."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LATEST = re.compile(r'latest-(\d+)\.ckpt$')
MODEL = re.compile(r'model-(\d+)-.+\.ckpt$')


def checkpoint_steps(directory: Path, pattern: re.Pattern[str]):
    rows = []
    for path in directory.glob('*.ckpt'):
        match = pattern.fullmatch(path.name)
        if match:
            rows.append((int(match.group(1)), path.resolve()))
    return sorted(rows)


def resolve(output_root: Path, experiment: str, stop_step: int) -> dict:
    output_root = output_root.resolve()
    candidates = sorted(
        path.resolve() for path in output_root.glob(f'*_{experiment}')
        if path.is_dir()
    )
    if not candidates:
        return {
            'action': 'new',
            'experiment': experiment,
            'output_root': str(output_root),
            'stop_step': stop_step,
        }
    if len(candidates) != 1:
        raise RuntimeError(
            f'{experiment}: expected at most one run directory, found '
            f'{len(candidates)}: {[str(path) for path in candidates]}'
        )
    run = candidates[0]
    manifest = run / 'training_manifest.json'
    if not manifest.is_file():
        raise RuntimeError(f'{run}: training manifest is missing')
    latest = checkpoint_steps(run, LATEST)
    models = checkpoint_steps(run, MODEL)
    exact_latest = [path for step, path in latest if step == stop_step]
    exact_models = [path for step, path in models if step == stop_step]
    if exact_latest and exact_models:
        return {
            'action': 'complete',
            'experiment': experiment,
            'run_directory': str(run),
            'resume_checkpoint': str(exact_latest[-1]),
            'validation_checkpoint': str(exact_models[-1]),
            'stop_step': stop_step,
        }
    eligible = [(step, path) for step, path in latest if 0 < step < stop_step]
    if not eligible:
        raise RuntimeError(
            f'{run}: no completed validation checkpoint at step {stop_step} '
            'and no resumable rolling checkpoint below it'
        )
    step, checkpoint = eligible[-1]
    return {
        'action': 'resume',
        'experiment': experiment,
        'run_directory': str(run),
        'resume_checkpoint': str(checkpoint),
        'resume_step': step,
        'stop_step': stop_step,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('output/t2i_trans'))
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--stop-step', type=int, required=True)
    args = parser.parse_args()
    if args.stop_step <= 0:
        raise ValueError('--stop-step must be positive')
    print(json.dumps(resolve(args.output_root, args.experiment, args.stop_step)))


if __name__ == '__main__':
    main()
