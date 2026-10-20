#!/usr/bin/env python3
"""Verify that a frozen-foot pilot differs from the released pilot once."""

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


def without_policy(optimizer: dict) -> dict:
    result = dict(optimizer)
    result.pop('foot_contact_mask_policy', None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--released', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument(
        '--preregistration', type=Path,
        default=ROOT / 'configs/interx_shape_augmentation/'
        'frozen_initial_foot_contact_pilot_v1.json',
    )
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    released = json.loads(args.released.read_text(encoding='utf-8'))
    frozen = json.loads(args.frozen.read_text(encoding='utf-8'))
    prereg = json.loads(args.preregistration.read_text(encoding='utf-8'))
    immutable_top_level = (
        'schema', 'plan_sha256', 'plan_records_sha256',
        'contact_manifest_sha256', 'source_motion_sha256', 'selection',
        'requested_samples',
    )
    checks = {
        'preregistered_single_factor': (
            prereg['single_factor_change']['factor']
            == 'foot_contact_mask_policy'
        ),
        'released_is_complete': (
            released.get('complete') is True
            and released.get('completed_samples')
            == released.get('requested_samples') == 25
        ),
        'frozen_is_complete': (
            frozen.get('complete') is True
            and frozen.get('completed_samples')
            == frozen.get('requested_samples') == 25
        ),
        'all_inputs_and_selection_identical': all(
            released.get(key) == frozen.get(key)
            for key in immutable_top_level
        ),
        'all_optimizer_settings_except_policy_identical': (
            without_policy(released['optimizer'])
            == without_policy(frozen['optimizer'])
        ),
        'released_policy_is_original_dynamic': (
            released['optimizer'].get(
                'foot_contact_mask_policy', 'released_dynamic'
            ) == 'released_dynamic'
        ),
        'frozen_policy_is_iteration0_frozen': (
            frozen['optimizer'].get('foot_contact_mask_policy')
            == 'frozen_initial'
        ),
    }
    passed = all(checks.values())
    payload = {
        'schema': 'hhc_interx_frozen_foot_single_factor_audit_v1',
        'status': 'pass' if passed else 'fail',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'checks': checks,
        'declared_difference': {
            'field': 'optimizer.foot_contact_mask_policy',
            'released': 'released_dynamic',
            'frozen': 'frozen_initial',
        },
        'artifacts': {
            'released_completion': artifact(args.released),
            'frozen_completion': artifact(args.frozen),
            'preregistration': artifact(args.preregistration),
            'implementation': artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
