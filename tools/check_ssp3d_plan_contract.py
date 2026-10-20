"""Validate an SSP plan and ensure cross-gender substitution is rejected."""
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_interx_shape_augmentation_plan import atomic_json, atomic_jsonl, sha256_file
from utils.interx_shape_augmentation_plan import validate_interx_shape_augmentation_plan


if __name__ == '__main__':
    manifest, _, rows, _, _ = validate_interx_shape_augmentation_plan(sys.argv[1])
    assert manifest['donor_pool_kind'] == 'ssp3d_transferred_smplx'
    for row in rows:
        for donor in row['shape_donors']:
            assert donor['gender'] == row['source_genders'][donor['target_person']]
    bad_rows = copy.deepcopy(rows)
    row = bad_rows[0]
    donor = row['shape_donors'][0]
    target = donor['target_person']
    used = {item['shape_id'] for item in row['shape_donors']}
    replacement = next(item for item in manifest['shape_pool']
                       if item['gender'] != row['source_genders'][target]
                       and item['shape_id'] not in used)
    row['shape_donors'][0] = {**replacement, 'target_person': target,
        'beta_l2_from_source': float(np.linalg.norm(
            np.asarray(replacement['betas']) - np.asarray(row['source_betas'][target])))}
    with tempfile.TemporaryDirectory(prefix='hhc_ssp_plan_check_') as directory:
        directory = Path(directory)
        records = directory / 'records.jsonl'
        atomic_jsonl(records, bad_rows)
        bad_manifest = {**manifest, 'records': str(records), 'records_sha256': sha256_file(records)}
        plan = directory / 'plan.json'
        atomic_json(plan, bad_manifest)
        try:
            validate_interx_shape_augmentation_plan(plan)
        except ValueError as error:
            assert 'must preserve source gender' in str(error), str(error)
        else:
            raise AssertionError('Cross-gender donor was accepted')
    print(f'{len(rows)} valid same-gender pairs; cross-gender tampering rejected')
