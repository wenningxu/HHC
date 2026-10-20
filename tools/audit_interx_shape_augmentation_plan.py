#!/usr/bin/env python
"""Emit a machine-readable audit of the frozen Inter-X shape plan."""

import argparse
import collections
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.interx_shape_augmentation_plan import (
    sha256_file,
    validate_interx_shape_augmentation_plan,
)


SCHEMA = 'hhc_interx_shape_augmentation_plan_audit_v1'


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'count': 0}
    return {
        'count': int(len(values)),
        'min': float(values.min()),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'p95': float(np.quantile(values, 0.95)),
        'max': float(values.max()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--plan', type=Path,
        default=Path(
            'configs/interx_shape_augmentation/timing_covered_seed123_v2/'
            'plan_manifest.json'
        ),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/interx_shape_augmentation_plan_v2_audit.json'),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plan_path = args.plan.resolve()
    manifest, records_path, records, motion_file, split = (
        validate_interx_shape_augmentation_plan(plan_path)
    )
    distances = [
        donor['beta_l2_from_source']
        for record in records for donor in record['shape_donors']
    ]
    donor_ids = [
        donor['shape_id'] for record in records for donor in record['shape_donors']
    ]
    donor_gender_pairs = collections.Counter(
        '-'.join(
            donor['gender']
            for donor in sorted(
                record['shape_donors'], key=lambda row: row['target_person']
            )
        )
        for record in records
    )
    source_action_counts = collections.Counter(
        record['action_id'] for record in records
    )
    crop_policy_counts = collections.Counter(
        record['crop_policy'] for record in records
    )
    report = {
        'schema': SCHEMA,
        'status': 'pass',
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'plan': str(plan_path),
        'plan_sha256': sha256_file(plan_path),
        'records': str(records_path),
        'records_sha256': sha256_file(records_path),
        'source_split': str(split),
        'source_split_sha256': sha256_file(split),
        'source_motion': str(motion_file),
        'source_motion_sha256': sha256_file(motion_file),
        'implementation': {
            'path': str((ROOT / 'utils/interx_shape_augmentation_plan.py').resolve()),
            'sha256': sha256_file(
                ROOT / 'utils/interx_shape_augmentation_plan.py'
            ),
        },
        'counts': {
            'official_train_samples': manifest['official_train_samples'],
            'eligible_timing_covered_samples': len(records),
            'positive_timing_samples': sum(
                record['timing_positive_frames_crop'] > 0 for record in records
            ),
            'explicit_zero_timing_samples': sum(
                record['timing_positive_frames_crop'] == 0 for record in records
            ),
            'excluded_missing_manual_timing': manifest[
                'excluded_missing_manual_timing'
            ],
            'excluded_outside_training_eligibility': manifest[
                'excluded_outside_training_eligibility'
            ],
            'unique_train_person_shapes': manifest[
                'unique_train_person_shapes'
            ],
            'unique_donor_shapes': manifest.get(
                'unique_donor_shapes', manifest['unique_train_person_shapes']
            ),
            'unique_donor_shapes_used': len(set(donor_ids)),
            'donor_assignments': len(donor_ids),
        },
        'donor_beta_l2_from_source': distribution(distances),
        'donor_pool_kind': manifest.get('donor_pool_kind', 'interx_train_native'),
        'external_donor_manifest': manifest.get('inputs', {}).get(
            'external_donor_manifest'
        ),
        'donor_gender_pair_counts': dict(sorted(donor_gender_pairs.items())),
        'crop_policy_counts': dict(sorted(crop_policy_counts.items())),
        'eligible_samples_by_action': dict(sorted(source_action_counts.items())),
        'invariants': {
            'official_split_unchanged': True,
            'validation_and_test_not_used': True,
            'every_eligible_sample_has_explicit_manual_binary_timing': True,
            'two_distinct_non_source_donor_shapes_per_pair': True,
            'caption_beta_timing_and_motion_sources_hash_verified': True,
            'missing_timing_is_excluded_not_imputed_zero': True,
            'donor_pool_artifact_hash_verified': True,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(json.dumps({
        'output': str(output),
        'status': report['status'],
        **report['counts'],
        'donor_beta_l2_mean': report['donor_beta_l2_from_source']['mean'],
        'donor_beta_l2_p95': report['donor_beta_l2_from_source']['p95'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
