#!/usr/bin/env python3
"""Validate a random-shape-only augmentation arm independently of optimizer code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.interx_shape_augmentation_plan import (
    validate_interx_shape_augmentation_plan,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


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


def expected_person(raw: np.ndarray, person: int, donor: dict) -> dict:
    channel = slice(0, 3) if person == 0 else slice(3, 6)
    poses = np.zeros((raw.shape[0], 55, 3), dtype=np.float32)
    poses[:, 0] = raw[:, 0, channel]
    poses[:, 1:22] = raw[:, 1:22, channel]
    poses[:, 25:40] = raw[:, 25:40, channel]
    poses[:, 40:55] = raw[:, 40:55, channel]
    return {
        'poses': poses,
        'trans': raw[:, 55, channel].astype(np.float32),
        'betas': np.asarray(donor['betas'], dtype=np.float32).reshape(-1)[:10],
        'gender': str(donor['gender']),
        'mocap_framerate': 30,
    }


def validate_npz(path: Path, expected: dict) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError(f'{path}: NPZ keys differ from random-shape schema')
        for key in ('poses', 'trans', 'betas'):
            actual = np.asarray(archive[key])
            if (
                actual.shape != expected[key].shape
                or actual.dtype != expected[key].dtype
                or not np.array_equal(actual, expected[key])
            ):
                raise ValueError(f'{path}: {key} differs from deterministic input')
        if str(np.asarray(archive['gender']).item()) != expected['gender']:
            raise ValueError(f'{path}: gender differs from selected donor')
        if int(np.asarray(archive['mocap_framerate']).item()) != 30:
            raise ValueError(f'{path}: mocap framerate is not 30')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-samples', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    contract_path = run_dir / '_run_contract.json'
    completion_path = run_dir / '_completion_manifest.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    completion = json.loads(completion_path.read_text(encoding='utf-8'))
    plan_manifest, records_path, records, motion_path, _ = (
        validate_interx_shape_augmentation_plan(args.plan.resolve())
    )
    records_by_name = {row['augmented_name']: row for row in records}

    contract_identity = {
        key: value for key, value in contract.items()
        if key not in ('created_at_utc', 'contract_sha256')
    }
    selected = contract['selection']['selected_names']
    completed = completion.get('completed_names', [])
    checks = {
        'schema_matches': contract.get('schema')
        == 'hhc_interx_shape_augmentation_run_v1',
        'contract_hash_is_self_consistent': contract.get('contract_sha256')
        == canonical_sha256(contract_identity),
        'completion_is_complete': completion.get('complete') is True,
        'expected_count_matches': (
            contract.get('requested_samples') == args.expected_samples
            and completion.get('completed_samples') == args.expected_samples
            and len(selected) == args.expected_samples
            and len(completed) == args.expected_samples
            and len(set(completed)) == args.expected_samples
        ),
        'selected_and_completed_order_match': selected == completed,
        'completion_embeds_original_contract': all(
            completion.get(key) == value for key, value in contract.items()
        ),
        'mode_is_random_shape_only': contract.get('optimizer', {}).get('mode')
        == 'random_shape_only',
        'backend_was_not_instantiated': contract.get('optimizer', {}).get(
            'backend_lifetime'
        ) == 'not_instantiated_for_random_shape_only_control',
        'no_volumetric_checkpoints_loaded': contract.get('optimizer', {}).get(
            'volumetric_smpl_checkpoints'
        ) == {},
        'plan_hash_matches_current_frozen_plan': (
            contract.get('plan_sha256') == sha256_file(args.plan.resolve())
            and contract.get('plan_records_sha256') == sha256_file(records_path)
        ),
        'source_motion_hash_matches': contract.get('source_motion_sha256')
        == sha256_file(motion_path),
        'all_names_exist_in_plan': not (set(selected) - set(records_by_name)),
    }
    if not all(checks.values()):
        payload = {
            'schema': 'hhc_random_shape_augmentation_completion_audit_v1',
            'status': 'fail',
            'checks': checks,
        }
        atomic_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(2)

    sample_records = []
    with h5py.File(motion_path, 'r') as motions:
        for name in selected:
            record = records_by_name[name]
            raw = np.asarray(motions[record['source_name']][
                record['crop_start']:record['crop_stop']
            ], dtype=np.float32)
            if raw.shape != (record['crop_frames'], 56, 6):
                raise ValueError(f'{name}: source crop shape changed')
            donors = sorted(
                record['shape_donors'], key=lambda row: row['target_person']
            )
            p1 = run_dir / f'{name}_p1.npz'
            p2 = run_dir / f'{name}_p2.npz'
            sidecar_path = run_dir / f'{name}.json'
            sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
            if sidecar.get('run_contract_sha256') != contract['contract_sha256']:
                raise ValueError(f'{name}: run contract binding changed')
            if sidecar.get('plan_record_sha256') != canonical_sha256(record):
                raise ValueError(f'{name}: plan record binding changed')
            if sidecar.get('output_p1_sha256') != sha256_file(p1):
                raise ValueError(f'{name}: p1 content hash changed')
            if sidecar.get('output_p2_sha256') != sha256_file(p2):
                raise ValueError(f'{name}: p2 content hash changed')
            diagnostics = sidecar.get('diagnostics', {})
            if diagnostics != {
                'mode': 'random_shape_only_no_optimization',
                'iterations': 0,
                'temporal_mask_supplied_to_optimizer': False,
                'interaction_pairs_supplied_to_optimizer': False,
            }:
                raise ValueError(f'{name}: random-only diagnostics changed')
            validate_npz(p1, expected_person(raw, 0, donors[0]))
            validate_npz(p2, expected_person(raw, 1, donors[1]))
            sample_records.append({
                'name': name,
                'plan_record_sha256': canonical_sha256(record),
                'sidecar': artifact(sidecar_path),
                'p1': artifact(p1),
                'p2': artifact(p2),
            })

    payload = {
        'schema': 'hhc_random_shape_augmentation_completion_audit_v1',
        'status': 'pass',
        'interpretation': (
            'Random-shape outputs are bitwise deterministic copies of the '
            'frozen source crop with only the preregistered donor beta/gender; '
            'optimizer implementation hashes are therefore non-consumed '
            'historical provenance, not a reason to recompute this arm.'
        ),
        'checks': checks,
        'sample_count': len(sample_records),
        'artifacts': {
            'run_contract': artifact(contract_path),
            'completion': artifact(completion_path),
            'plan': artifact(args.plan),
            'plan_records': artifact(records_path),
            'source_motion': artifact(motion_path),
            'implementation': artifact(Path(__file__)),
        },
        'samples': sample_records,
    }
    atomic_json(args.output, payload)
    print(json.dumps({
        'schema': payload['schema'],
        'status': payload['status'],
        'sample_count': payload['sample_count'],
        'output': str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
