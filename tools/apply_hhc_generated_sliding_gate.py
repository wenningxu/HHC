#!/usr/bin/env python
"""Posthoc generated-motion sliding gate; not sampler/MM selection."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from tools.eval_vq_interx_reconstruction import sha256_file


def accept_candidate(control_speed, candidate_speed, control_transitions, candidate_transitions):
    return (np.asarray(candidate_speed) <= np.asarray(control_speed)
            ) & (np.asarray(candidate_transitions) >= np.asarray(control_transitions))


def pooled_ratio(speed, transitions):
    denominator = float(np.sum(transitions))
    return float(np.sum(speed) / denominator) if denominator else 0.0


def self_check():
    c_speed = np.array([2., 2., 2., 0.]); x_speed = np.array([1., 3., 2., 0.])
    c_trans = np.array([1, 2, 2, 0]); x_trans = np.array([1, 2, 1, 0])
    got = accept_candidate(c_speed, x_speed, c_trans, x_trans)
    assert got.tolist() == [True, False, False, True]
    assert pooled_ratio(np.zeros(2), np.zeros(2)) == 0
    assert pooled_ratio(np.array([1., 0.]), np.array([2, 0])) == .5
    selected_speed = np.where(got, x_speed, c_speed)
    selected_trans = np.where(got, x_trans, c_trans)
    assert pooled_ratio(selected_speed, selected_trans) <= pooled_ratio(c_speed, c_trans)
    assert np.array_equal(c_speed, [2., 2., 2., 0.])
    print('self-check: PASS')


def _physical(path, h5_path):
    data = json.load(open(path, encoding='utf-8'))
    if data.get('protocol') != 'interx_official_v1_physical_joint_v2':
        raise ValueError(f'physical protocol mismatch: {path}')
    if data.get('floor_height_policy') != 'valid_frames' or data.get('repeat') != 0:
        raise ValueError(f'physical floor/repeat mismatch: {path}')
    if data.get('generations_sha256') != sha256_file(h5_path):
        raise ValueError(f'physical/H5 SHA mismatch: {path}')
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control', required=False)
    p.add_argument('--candidate', required=False)
    p.add_argument('--control-physical', required=False)
    p.add_argument('--candidate-physical', required=False)
    p.add_argument('--output', required=False)
    p.add_argument('--self-check', action='store_true')
    a = p.parse_args()
    if a.self_check:
        self_check(); return
    for name in ('control', 'candidate', 'control_physical', 'candidate_physical', 'output'):
        if not getattr(a, name): p.error(f'--{name.replace("_", "-")} is required')
    paths = {k: Path(getattr(a, k)) for k in ('control', 'candidate', 'control_physical', 'candidate_physical', 'output')}
    if paths['output'].exists(): raise FileExistsError(paths['output'])
    cphys, xphys = _physical(paths['control_physical'], paths['control']), _physical(paths['candidate_physical'], paths['candidate'])
    if (cphys['requests_sha256'] != xphys['requests_sha256'] or cphys['samples'] != xphys['samples']
            or cphys.get('ground_height_threshold_m') != xphys.get('ground_height_threshold_m')
            or cphys.get('contact_threshold_m') != xphys.get('contact_threshold_m')):
        raise ValueError('physical request/sample mismatch')
    ca, xa = cphys['per_sample_counts'], xphys['per_sample_counts']
    if len(ca) != len(xa) or [x['name'] for x in ca] != [x['name'] for x in xa]:
        raise ValueError('physical sample order mismatch')
    fields = ('grounded_foot_speed_sum_mps', 'grounded_foot_transitions')
    arrays = {}
    for label, rows in (('control', ca), ('candidate', xa)):
        for field in fields:
            values = np.asarray([row[field] for row in rows])
            if not np.isfinite(values).all() or (values < 0).any(): raise ValueError(f'invalid {label} {field}')
            if field.endswith('transitions') and not np.equal(values, np.floor(values)).all(): raise ValueError(f'noninteger {label} transitions')
            arrays[label + '_' + field] = values
        if np.any((arrays[label + '_' + fields[1]] == 0) & (arrays[label + '_' + fields[0]] != 0)):
            raise ValueError('nonzero speed sum without grounded transitions')
    accepted = accept_candidate(arrays['control_grounded_foot_speed_sum_mps'], arrays['candidate_grounded_foot_speed_sum_mps'], arrays['control_grounded_foot_transitions'], arrays['candidate_grounded_foot_transitions'])
    with h5py.File(paths['control'], 'r') as c, h5py.File(paths['candidate'], 'r') as x:
        required = {'motions', 'names', 'lengths', 'completed_repeats', 'token_ids'}
        if not required.issubset(c) or not required.issubset(x): raise ValueError('missing required H5 field')
        for f in ('names', 'lengths'):
            if not np.array_equal(c[f][:], x[f][:]): raise ValueError(f'{f} mismatch')
        n = len(c['names'])
        if n == 0 or len(ca) != n or cphys['samples'] != n:
            raise ValueError('physical/H5 sample count mismatch')
        if c['names'][:].astype(str).tolist() != [row['name'] for row in ca]:
            raise ValueError('physical/H5 names/order mismatch')
        for source in (c, x):
            if source.attrs.get('requests_sha256') != cphys['requests_sha256']:
                raise ValueError('physical/H5 request SHA mismatch')
            if source['motions'].shape != (n, 1, 150, 672):
                raise ValueError('not single-repeat raw672 H5')
            if source['completed_repeats'].shape != (1,) or not source['completed_repeats'][:].all():
                raise ValueError('incomplete source')
            if source['lengths'].shape != (n,) or np.any((source['lengths'][:] < 1) | (source['lengths'][:] > 150)):
                raise ValueError('invalid lengths')
            token = source['token_ids']
            if token.ndim != 4 or token.shape[:2] != (n, 1) or token.shape[-1] != 3 or not np.issubdtype(token.dtype, np.integer):
                raise ValueError('invalid archived token shape/type')
            if np.any((source['lengths'][:] + 3) // 4 > token.shape[2]):
                raise ValueError('token archive shorter than requested lengths')
        if not np.isfinite(c['motions'][:]).all() or not np.isfinite(x['motions'][:]).all(): raise ValueError('nonfinite source motion')
        if c['token_ids'].shape != x['token_ids'].shape: raise ValueError('token_ids shape mismatch')
        motions = np.where(accepted[:, None, None, None], x['motions'][:], c['motions'][:])
        tokens = np.where(accepted[:, None, None, None], x['token_ids'][:], c['token_ids'][:]).astype(np.int32)
        lengths, names = c['lengths'][:], c['names'][:]
        if not np.isfinite(motions).all(): raise ValueError('nonfinite selected motion')
        paths['output'].parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(paths['output'], 'x') as out:
            out.create_dataset('motions', data=motions, compression='lzf')
            out.create_dataset('token_ids', data=tokens, compression='lzf')
            out.create_dataset('names', data=names)
            out.create_dataset('lengths', data=lengths)
            out.create_dataset('completed_repeats', data=np.array([False]))
            out.create_dataset('accepted_candidate', data=accepted.astype(np.bool_))
            out.attrs.update({'diagnostic_only': True, 'intervention': 'posthoc_generated_motion_sliding_gate',
                              'description': 'posthoc generated motion gate; not an unmodified sampler and not official MM',
                              'control_h5': str(paths['control'].resolve()), 'candidate_h5': str(paths['candidate'].resolve()),
                              'control_h5_sha256': sha256_file(paths['control']), 'candidate_h5_sha256': sha256_file(paths['candidate']),
                              'control_physical_json': str(paths['control_physical'].resolve()), 'candidate_physical_json': str(paths['candidate_physical'].resolve()),
                              'control_physical_sha256': sha256_file(paths['control_physical']), 'candidate_physical_sha256': sha256_file(paths['candidate_physical']),
                              'requests_sha256': cphys['requests_sha256'], 'selection_rule': 'candidate speed_sum <= control and candidate grounded_transitions >= control',
                              'selection_tool_sha256': sha256_file(__file__),
                              'candidate_sampling_metadata_json': json.dumps(dict(x.attrs), default=str),
                              'control_sampling_metadata_json': json.dumps({k: c.attrs[k] for k in c.attrs if k not in ('control_h5', 'candidate_h5')}, default=str)})
            out['completed_repeats'][:] = True
    print(json.dumps({'status': 'complete', 'samples': len(accepted), 'accepted_candidate': int(accepted.sum()), 'control_slide': pooled_ratio(arrays['control_grounded_foot_speed_sum_mps'], arrays['control_grounded_foot_transitions']), 'selected_slide': pooled_ratio(np.where(accepted, arrays['candidate_grounded_foot_speed_sum_mps'], arrays['control_grounded_foot_speed_sum_mps']), np.where(accepted, arrays['candidate_grounded_foot_transitions'], arrays['control_grounded_foot_transitions']))}, indent=2))


if __name__ == '__main__':
    main()
