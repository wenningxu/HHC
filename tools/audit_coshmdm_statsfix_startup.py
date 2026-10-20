#!/usr/bin/env python
"""Audit the corrected CoShMDM stats contract and every completed epoch boundary."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


EXPECTED_RUNTIME = {
    'dataset': 'interx',
    'stats_prefix': 'interx_global',
    'normalization_mean_sha256': (
        'aa281b878b8a71c6d10c001059b29374921c18545c082102bf69a404982f568d'
    ),
    'normalization_std_sha256': (
        '30a83e92965a465301a3642eb55b2ba705d3dceaba05e45846aa433d3900b104'
    ),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {
        'path': str(path), 'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run-root', type=Path,
        default=Path('/mnt/8T-HDD/wenning/coshmdm-code/checkpoints/'
                     'CoShMDM-InterX-fair-statsfix-seed123-bs32x2'),
    )
    parser.add_argument(
        '--log', type=Path,
        default=Path('output/queue_logs/gpu0_cosh_statsfix_20260830.log'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/coshmdm_statsfix_startup_audit.json'),
    )
    args = parser.parse_args()
    runtime_path = args.run_root / 'meta/runtime_contract.json'
    event_paths = sorted((args.run_root / 'log').glob('events.out.tfevents.*'))
    require(runtime_path.is_file(), f'missing runtime manifest: {runtime_path}')
    require(event_paths, f'missing TensorBoard event files: {event_paths}')
    runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
    require(runtime.get('runtime_contract') == EXPECTED_RUNTIME,
            'runtime normalization contract changed')

    text = args.log.read_text(encoding='utf-8', errors='replace')
    rows = [
        (int(epoch), int(step))
        for epoch, step in re.findall(
            r'epoch:\s*(\d+)\s+niter:\s*(\d+)', text
        )
    ]
    require((0, 280) in rows and any(epoch == 1 and step >= 290
                                     for epoch, step in rows),
            'first epoch did not expose exactly 284 optimizer updates')
    dropped_by_epoch = {}
    for event_path in event_paths:
        events = EventAccumulator(str(event_path))
        events.Reload()
        for record in events.Scalars(
                'dropped_incomplete_accumulation_micro_batches'):
            dropped_by_epoch.setdefault(int(record.step), []).append(
                float(record.value)
            )
    require(dropped_by_epoch, 'no epoch-boundary discard scalars were recorded')
    completed_epochs = sorted(dropped_by_epoch)
    require(completed_epochs == list(range(completed_epochs[-1] + 1)),
            'epoch-boundary discard scalar steps are not contiguous from zero')
    require(all(value == 1.0 for values in dropped_by_epoch.values()
                for value in values),
            'a completed epoch did not discard exactly one half accumulation')

    payload = {
        'schema': 'hhc_coshmdm_statsfix_startup_audit_v2',
        'status': 'pass',
        'runtime_contract': EXPECTED_RUNTIME,
        'first_epoch': {
            'optimizer_updates': 284,
            'last_logged_decade_update': 280,
            'first_epoch_1_logged_update': min(
                step for epoch, step in rows if epoch == 1
            ),
            'dropped_incomplete_accumulation_micro_batches': 1,
        },
        'epoch_boundary_series': {
            'recorded_completed_epochs': len(completed_epochs),
            'first_epoch': completed_epochs[0],
            'last_epoch': completed_epochs[-1],
            'steps_contiguous_from_zero': True,
            'all_discard_values_equal_one': True,
            'duplicate_scalar_records': sum(
                len(values) - 1 for values in dropped_by_epoch.values()
            ),
        },
        'runtime_manifest': artifact(runtime_path),
        'event_files': [artifact(path) for path in event_paths],
        'training_log': artifact(args.log),
        'implementation': artifact(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
