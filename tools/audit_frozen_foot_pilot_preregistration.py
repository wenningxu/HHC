#!/usr/bin/env python3
"""Bind the frozen-foot single-factor plan before the released pilot ends."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--preregistration', type=Path,
        default=ROOT / 'configs/interx_shape_augmentation/'
        'frozen_initial_foot_contact_pilot_v1.json',
    )
    parser.add_argument(
        '--released-completion', type=Path,
        default=ROOT / 'output/'
        'interx_shape_aug_pilot_optimized_flathand_query10475_seed20260828/'
        '_completion_manifest.json',
    )
    parser.add_argument(
        '--gate', type=Path,
        default=ROOT / 'configs/interx_shape_augmentation/'
        'pilot_physical_gate_v1.json',
    )
    parser.add_argument(
        '--helper', type=Path,
        default=ROOT / 'optimization/foot_contact_constraints.py',
    )
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.preregistration.read_text(encoding='utf-8'))
    completion = json.loads(args.released_completion.read_text(encoding='utf-8'))
    gate = json.loads(args.gate.read_text(encoding='utf-8'))
    requested = int(completion['requested_samples'])
    completed = int(completion['completed_samples'])
    checks = {
        'schema_matches': plan.get('schema') == (
            'hhc_interx_shape_augmentation_frozen_foot_pilot_v1'
        ),
        'conditional_status': plan.get('status') == (
            'conditionally_preregistered'
        ),
        'released_pilot_was_incomplete': (
            completion.get('complete') is False and completed < requested
        ),
        'same_expected_sample_count': (
            plan['selection']['sample_count']
            == gate['expected_sample_count'] == requested
        ),
        'authorization_criteria_unchanged': (
            plan['authorization_gate']['criteria'] == gate['criteria']
        ),
        'single_factor_is_mask_policy': (
            plan['single_factor_change']['factor']
            == 'foot_contact_mask_policy'
        ),
        'three_axis_velocity_retained': (
            'x/y/z' in plan['frozen_unchanged_settings']['foot_velocity_loss']
        ),
        'released_skate_weight_retained': (
            plan['frozen_unchanged_settings']['foot_skate_weight'] == 100.0
        ),
    }
    passed = all(checks.values())
    payload = {
        'schema': 'hhc_frozen_foot_pilot_preregistration_audit_v1',
        'status': 'pass' if passed else 'fail',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'released_pilot_state_at_preregistration': {
            'completed_samples': completed,
            'requested_samples': requested,
            'complete': completion.get('complete'),
        },
        'checks': checks,
        'artifacts': {
            'preregistration': artifact(args.preregistration),
            'released_completion_snapshot': artifact(
                args.released_completion
            ),
            'authorization_gate': artifact(args.gate),
            'experimental_helper': artifact(args.helper),
            'implementation': artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
