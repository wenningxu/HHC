"""Describe VQ reconstruction errors by length/time; never generate or fit a model.

Root-centred and initial-offset-removed errors are oracle diagnostics only. They
are not output corrections, official metrics or a generation improvement.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from dataset.interhuman_generated_h5 import InterHumanGeneratedH5Dataset, sha256_file


def blank():
    return dict(samples=0, person_frames=0, full_distance_m=0., root_distance_m=0.,
                root_centered_distance_m=0., initial_offset_removed_root_distance_m=0.,
                initial_root_distance_m=0., final_root_distance_m=0.,
                quartile_root_sums_m=[0.] * 4, quartile_person_frames=[0] * 4)


def add(acc, delta):
    # [2, valid frames, 22, xyz], after shared normalization inversion.
    persons, length, joints, _ = delta.shape
    assert persons == 2 and joints == 22 and length > 0 and np.isfinite(delta).all()
    roots = delta[:, :, 0]
    root_distance = np.linalg.norm(roots, axis=-1)
    acc['samples'] += 1
    acc['person_frames'] += persons * length
    acc['full_distance_m'] += float(np.linalg.norm(delta, axis=-1).sum())
    acc['root_distance_m'] += float(root_distance.sum())
    acc['root_centered_distance_m'] += float(np.linalg.norm(delta - roots[:, :, None], axis=-1).sum())
    acc['initial_offset_removed_root_distance_m'] += float(np.linalg.norm(roots - roots[:, :1], axis=-1).sum())
    acc['initial_root_distance_m'] += float(root_distance[:, 0].sum())
    acc['final_root_distance_m'] += float(root_distance[:, -1].sum())
    quarter = np.minimum(np.arange(length) * 4 // length, 3)
    for q in range(4):
        acc['quartile_root_sums_m'][q] += float(root_distance[:, quarter == q].sum())
        acc['quartile_person_frames'][q] += int(persons * (quarter == q).sum())


def finish(acc):
    frames, samples = acc['person_frames'], acc['samples']
    if not frames:
        return {'samples': samples, 'person_frames': frames, 'metrics': None}
    return {'samples': samples, 'person_frames': frames, 'metrics': {
        'full_mpjpe_mm': 1000 * acc['full_distance_m'] / (22 * frames),
        'root_position_error_mm': 1000 * acc['root_distance_m'] / frames,
        'root_centered_22joint_mpjpe_mm_diagnostic': 1000 * acc['root_centered_distance_m'] / (22 * frames),
        'initial_offset_removed_root_error_mm_diagnostic': 1000 * acc['initial_offset_removed_root_distance_m'] / frames,
        'initial_frame_root_error_mm': 1000 * acc['initial_root_distance_m'] / (2 * samples),
        'final_frame_root_error_mm': 1000 * acc['final_root_distance_m'] / (2 * samples),
        'normalized_time_quartile_root_error_mm': [1000 * s / n if n else None for s, n in zip(acc['quartile_root_sums_m'], acc['quartile_person_frames'])],
    }}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--paired-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run CPU diagnostics with CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if os.name != 'nt':
        os.nice(10)
    reference = json.loads(args.paired_reference.read_text())
    assert reference['status'] == 'complete' and not reference['is_generation_evaluation']
    data = InterHumanValidationRequestDataset()
    assert len(data) == reference['samples'] == 480
    mean, std = np.asarray(data.adapter.mean, dtype=np.float64), np.asarray(data.adapter.std, dtype=np.float64)
    assert mean.shape == std.shape == (262,) and np.isfinite(std).all() and (std > 0).all()
    bins = [('14_63', 14, 64), ('64_127', 64, 128), ('128_191', 128, 192), ('192_255', 192, 256), ('256_299', 256, 300)]
    arms = {}
    datasets = []
    try:
        for arm, path, step in [('control', args.control, 5000), ('candidate', args.candidate, 35000)]:
            digest = sha256_file(path)
            assert digest == reference[arm + '_sha256']
            ds = InterHumanGeneratedH5Dataset(path, data.request_path, request_entry=data.manifest['request'], formal=False)
            datasets.append(ds)
            assert ds.metadata['vq_step'] == step and ds.metadata['method'] == 'hhc_native_smpl_vq_reconstruction'
            assert ds.metadata['vq_selection_scope'] == 'intermediate_val_selected_pilot'
            assert ds.metadata['split'] == 'val' and ds.metadata['status'] == 'complete'
            totals = blank()
            strata = {name: blank() for name, _, _ in bins}
            for index in range(len(data)):
                name, text, a, b, length = data[index]
                pn, pt, p, q, pl = ds[index]
                assert (name, text, length) == (pn, pt, pl)
                delta = np.stack([p[:length].double().numpy() - a[:length].double().numpy(),
                                  q[:length].double().numpy() - b[:length].double().numpy()])
                # Mean cancels. Positions are the first 66 normalized features.
                delta = (delta * std)[..., :66].reshape(2, length, 22, 3)
                bucket = [n for n, lo, hi in bins if lo <= length < hi]
                assert len(bucket) == 1
                add(totals, delta)
                add(strata[bucket[0]], delta)
            observed = finish(totals)
            for name in ('full_mpjpe_mm', 'root_position_error_mm'):
                assert abs(observed['metrics'][name] - reference['metrics'][name][arm]) < 1e-7
            assert sum(row['samples'] for row in strata.values()) == 480
            arms[arm] = {'vq_step': step, 'h5_sha256': digest, 'overall': observed,
                         'by_processed_length': {n: finish(v) for n, v in strata.items()}}
    finally:
        for ds in datasets:
            ds.close()
    result = {
        'schema': 'hhc_vq_length_time_geometry_diagnostic_v1', 'status': 'complete',
        'split': 'val', 'samples': 480, 'is_generation_evaluation': False,
        'scope': 'Same frozen480 requests, native VQ5000 versus35000; +30000 training updates, not matched-budget method gain.',
        'protocol': 'Official processed lengths T-1, valid-person-frame pooled. Root-centred uses all22 joints including zero root. Initial-offset-removed uses each persons own first-frame root error. Normalized time quarters use floor(4*t/L).',
        'limitations': ['Length strata contain different motions; length-error association is not causal.',
                       'Root-centred and initial-offset-removed measurements use GT and are oracle diagnostics, never output corrections or primary scores.',
                       'No model fitting, generation, test scoring, checkpoint load or GPU. No train window change authorized by this diagnostic alone.'],
        'arms': arms,
        'paired_reference_sha256': sha256_file(args.paired_reference),
        'request_sha256': data.manifest['request']['sha256'],
        'script_sha256': sha256_file(Path(__file__)),
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
