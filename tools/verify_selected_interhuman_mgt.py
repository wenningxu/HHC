#!/usr/bin/env python3
"""Verify that formal InterHuman evaluation uses the frozen val-FID selection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.interhuman_artifacts import sha256_file


def verify_selected_checkpoint(checkpoint_path, manifest_path, require_complete=True):
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'hhc_interhuman_mgt_training_manifest_v1':
        raise ValueError('unsupported InterHuman MGT training manifest')
    allowed = {'complete'} if require_complete else {'complete', 'pilot_decision_ready'}
    if manifest.get('status') not in allowed:
        raise ValueError(
            f"InterHuman MGT status {manifest.get('status')!r} is not evaluation-ready"
        )
    selected = Path(manifest.get('best_validation_checkpoint', '')).resolve()
    if checkpoint_path != selected:
        raise ValueError(f'formal checkpoint is not manifest-selected: {checkpoint_path} != {selected}')
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != manifest.get('best_validation_checkpoint_sha256'):
        raise ValueError('selected InterHuman MGT checkpoint hash changed')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint.get('schema') != 'hhc_interhuman_mgt_checkpoint_v1':
        raise ValueError('unsupported InterHuman MGT checkpoint')
    best_step = int(manifest['best_validation_step'])
    best_fid = float(manifest['best_validation_fid'])
    if int(checkpoint.get('optimizer_step', -1)) != best_step:
        raise ValueError('selected checkpoint optimizer step differs from manifest best step')
    if int(checkpoint.get('best_validation_step', -1)) != best_step:
        raise ValueError('selected checkpoint embeds another best-validation step')
    if float(checkpoint.get('best_validation_fid', float('nan'))) != best_fid:
        raise ValueError('selected checkpoint embeds another best-validation FID')
    if checkpoint.get('training_identity_sha256') != manifest.get('training_identity_sha256'):
        raise ValueError('selected checkpoint training identity differs from manifest')
    if checkpoint.get('vq_checkpoint_sha256') != manifest.get('vq', {}).get('checkpoint_sha256'):
        raise ValueError('selected checkpoint VQ differs from manifest')
    return {
        'schema': 'hhc_interhuman_mgt_selection_verification_v1',
        'status': 'pass',
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': actual_hash,
        'training_manifest': str(manifest_path),
        'training_manifest_sha256': sha256_file(manifest_path),
        'training_identity_sha256': manifest['training_identity_sha256'],
        'best_validation_step': best_step,
        'best_validation_fid': best_fid,
        'selection_split': 'official_interhuman_val_only',
        'selection_metric': 'FID',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--training-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-pilot', action='store_true')
    args = parser.parse_args()
    result = verify_selected_checkpoint(
        args.checkpoint, args.training_manifest,
        require_complete=not args.allow_pilot,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
