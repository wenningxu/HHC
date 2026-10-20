"""Bind per-sample original-GT proxy decisions to immutable scored inputs."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_interx_shape_augmentation import atomic_json, sha256_file

METRICS = ('foot_sliding_m_per_frame', 'nearest_vertex_squared_sum_m2_per_frame')


def checked_json(path, expected_hash):
    if sha256_file(path) != expected_hash:
        raise ValueError(f'Scored input changed: {path}')
    return json.loads(path.read_text())


def build_gate(report_path, condition):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    if report['status'] != 'diagnostic_only_nearest_vertex_proxy':
        raise ValueError('This gate requires the paired original-GT proxy report')
    folder = Path(report['condition_directories'][condition])
    completion_path = folder / '_completion_manifest.json'
    completion = checked_json(completion_path, report['condition_completion_sha256'][condition])
    names = completion['completed_names']
    if not completion['complete'] or len(names) != len(set(names)) or not names:
        raise ValueError('Incomplete or duplicate candidate set')
    if completion['completed_samples'] != len(names) or completion['requested_samples'] != len(names):
        raise ValueError('Requested samples must all be represented in the denominator')
    if report['source_motion_sha256'] != completion['source_motion_sha256']:
        raise ValueError('Reference source binding mismatch')
    if sha256_file(completion['source_motion']) != report['source_motion_sha256']:
        raise ValueError('Original-GT source changed')
    if [row['name'] for row in report['rows']] != names:
        raise ValueError('Score rows must exactly cover the complete candidate set')
    accepted, rejected = [], []
    for row in report['rows']:
        name = row['name']
        if Path(name).name != name:
            raise ValueError('Invalid sample name')
        binding = row['input_bindings'][condition]
        meta = checked_json(folder / f'{name}.json', binding['sidecar_sha256'])
        if row['frames'] != meta['crop_stop'] - meta['crop_start']:
            raise ValueError(f'Crop changed: {name}')
        for person in (1, 2):
            path = folder / f'{name}_p{person}.npz'
            expected = binding[f'p{person}_sha256']
            if expected != meta[f'output_p{person}_sha256'] or sha256_file(path) != expected:
                raise ValueError(f'Scored motion changed: {path}')
        gt, candidate = row['conditions']['original_gt'], row['conditions'][condition]
        if not all(math.isfinite(values[k]) and values[k] >= 0 for values in (gt, candidate) for k in METRICS):
            raise ValueError(f'Invalid physical metrics: {name}')
        passed = all(candidate[k] <= gt[k] for k in METRICS)
        if passed != row['proxy_pass_vs_original_gt'][condition]:
            raise ValueError(f'Inconsistent saved decision: {name}')
        (accepted if passed else rejected).append(name)
    return {
        'schema': 'hhc_original_gt_proxy_quality_gate_v1',
        'status': 'proxy_gate_complete',
        'scope': 'Nearest-vertex proxy only; not a volumetric or full augmentation acceptance claim',
        'condition': condition, 'report': str(report_path.resolve()),
        'report_sha256': sha256_file(report_path),
        'completion': str(completion_path.resolve()),
        'completion_sha256': report['condition_completion_sha256'][condition],
        'criteria': {k: 'candidate <= original_shape_gt; no tolerance' for k in METRICS},
        'denominator': len(names), 'accepted_count': len(accepted),
        'pass_fraction': len(accepted) / len(names),
        'accepted_names': accepted, 'rejected_names': rejected,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    gate = build_gate(args.report, args.condition)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, gate)
    print({key: gate[key] for key in ('denominator', 'accepted_count', 'pass_fraction')})
