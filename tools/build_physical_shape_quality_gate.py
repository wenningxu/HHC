"""Rebuild a strict production quality selection from bound physical scores."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diagnose_shape_support_transfer import production_ground_rows
from tools.run_interx_shape_augmentation import atomic_json, sha256_file

SCHEMA = 'hhc_physical_mesh_ground_quality_gate_v1'
METRICS = ('mean_foot_sliding_m_per_frame', 'paper_eq6_squared_sum_m2_per_frame')


def passes_quality(volumetric, sidecar):
    original, optimized = (volumetric[k] for k in ('original_shape', 'optimized'))
    for score in (original, optimized):
        if any(not math.isfinite(score[k]) or score[k] < 0 for k in METRICS):
            raise ValueError('nonfinite or negative quality metric')
        if score['frames'] != sidecar['crop_stop'] - sidecar['crop_start']:
            raise ValueError('score frame count differs from full crop')
    changed = all(not np.array_equal(np.asarray(sidecar['source_shapes']['betas'][i], dtype=np.float32),
                                    np.asarray(sidecar['augmented_shapes'][i]['betas'], dtype=np.float32))
                  for i in range(2))
    observable = all(sum(s[f'foot_sliding_p{i}']['contact_observations'] for i in (1, 2)) > 0
                     for s in (original, optimized))
    return changed and observable and all(optimized[k] <= original[k] for k in METRICS)


def build_physical_gate(report_path, completion_path, candidate_dir, floor_report_path):
    report_path, completion_path, candidate_dir, floor_report_path = map(
        lambda p: Path(p).resolve(), (report_path, completion_path, candidate_dir, floor_report_path))
    assert completion_path.name == '_completion_manifest.json'
    report = json.loads(report_path.read_text())
    if report['status'] != 'original_support_diagnostic_not_acceptance' or not report['volumetric_enabled']:
        raise ValueError('requires completed independent volumetric report')
    reference_path = candidate_dir / '_completion_manifest.json'
    reference = json.loads(reference_path.read_text()); floor = json.loads(floor_report_path.read_text())
    assert reference['complete'] and floor['completion_sha256'] == sha256_file(reference_path)
    assert report['floor_report_sha256'] == sha256_file(floor_report_path)
    assert sha256_file(reference['source_motion']) == reference['source_motion_sha256']
    assert sha256_file(reference['plan_records']) == reference['plan_records_sha256']
    floors = {(r['name'], r['condition']): r for r in floor['rows']}
    expected, bindings = production_ground_rows(completion_path.parent, reference, candidate_dir, floors)
    recorded = {str(Path(p).resolve()): h for p, h in report['ground_reports_sha256'].items()}
    assert all(recorded[str(Path(p).resolve())] == h for p, h in bindings.items())
    assert report['implementation_sha256'] == sha256_file(Path('tools/diagnose_shape_support_transfer.py'))
    assert report['penetration_implementation_sha256'] == sha256_file(Path('eval/calc_pene_from_npz.py'))
    metadata = report['volumetric_metadata']
    assert metadata['base_seed'] == 1234
    assert metadata['representation']['volumetric_body_encoding_samples'] == 1000
    assert metadata['representation']['smplx_flat_hand_mean'] is True
    for p, h in metadata['implementation_sha256'].items():
        assert sha256_file(p) == h
    for checkpoint in metadata['checkpoints'].values():
        assert sha256_file(checkpoint['path']) == checkpoint['sha256']
    names = [r['name'] for r in expected]
    assert names and len(names) == len(set(names))
    assert [r['name'] for r in report['rows']] == names, 'must score the whole completion in order'
    accepted = []
    for row, bound in zip(report['rows'], expected):
        assert row['input_npz_sha256'] == bound['output_npz_sha256']
        original_sidecar = candidate_dir / f'{row["name"]}.json'
        for condition in ('original_shape', 'new_shape'):
            assert floors[row['name'], condition]['sidecar_sha256'] == sha256_file(original_sidecar)
        side = json.loads((completion_path.parent / f'{row["name"]}.json').read_text())
        if passes_quality(row['volumetric'], side):
            accepted.append(row['name'])
    return {'schema': SCHEMA, 'status': 'physical_quality_gate_complete',
        'scope': 'Frozen neural SDF and standard foot sliding; not exact mesh-intersection ground truth.',
        'completion': str(completion_path), 'completion_sha256': sha256_file(completion_path),
        'report': str(report_path), 'report_sha256': sha256_file(report_path),
        'candidate_dir': str(candidate_dir), 'floor_report': str(floor_report_path),
        'floor_report_sha256': sha256_file(floor_report_path),
        'criteria': {**{k: 'optimized <= physical original-shape GT; no tolerance' for k in METRICS},
                     'shape': 'both people have changed float32 betas; fallback excluded',
                     'observability': 'positive foot contact observations in original and optimized pair'},
        'denominator': len(names), 'requested_names': names, 'accepted_names': accepted,
        'accepted_count': len(accepted), 'pass_fraction': len(accepted) / len(names),
        'rejected_names': [n for n in names if n not in set(accepted)]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('report', 'completion', 'candidate-dir', 'floor-report', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    gate = build_physical_gate(args.report, args.completion, args.candidate_dir, args.floor_report)
    atomic_json(args.output, gate)
    print({k: gate[k] for k in ('denominator', 'accepted_count', 'pass_fraction')})
