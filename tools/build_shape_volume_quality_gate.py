"""Cross-check original-GT gates with full-frame VolumetricSMPL scores."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_shape_gt_quality_gate import build_gate
from tools.run_interx_shape_augmentation import atomic_json, sha256_file
from tools.score_volumetric_smpl_npz import SCHEMA, canonical_sha256

METRICS = ('mean_foot_sliding_m_per_frame', 'paper_eq6_squared_sum_m2_per_frame')


def build_volume_gate(report_path, condition, references_path, gt_dir, candidate_dir):
    proxy = build_gate(report_path, condition)
    report = json.loads(Path(report_path).read_text())
    references_path = Path(references_path)
    references = json.loads(references_path.read_text())
    if (references['candidate_manifest_sha256'] != proxy['completion_sha256']
            or references['source_motion_sha256'] != report['source_motion_sha256']):
        raise ValueError('Reference export does not match the scored candidate/source')
    rows = report['rows']
    names = [row['name'] for row in rows]
    if [row['name'] for row in references['records']] != names:
        raise ValueError('Reference export must cover the full scored sample set')
    directories = [Path(gt_dir), Path(candidate_dir)]
    manifests = [json.loads((d / 'metrics.json').read_text()) for d in directories]
    for directory, manifest in zip(directories, manifests):
        expected_records = {(directory / 'samples' / f'{n}.json').resolve() for n in names}
        if (manifest['schema'] != SCHEMA or manifest['status'] != 'complete' or manifest['sample_count'] != len(names)
                or len(manifest['sample_records']) != len(names)
                or {Path(p).resolve() for p in manifest['sample_records']} != expected_records
                or canonical_sha256(manifest['scoring_contract']) != manifest['scoring_contract_sha256']):
            raise ValueError('Incomplete or inconsistent volume scoring manifest')
    if manifests[0]['scoring_contract_sha256'] != manifests[1]['scoring_contract_sha256']:
        raise ValueError('GT and candidate volume scoring protocols differ')
    accepted, volume_accepted, bindings = [], [], {}
    proxy_accepted = set(proxy['accepted_names'])
    for row, reference in zip(rows, references['records']):
        name = row['name']
        binding = row['input_bindings'][condition]
        if reference['sidecar_sha256'] != binding['sidecar_sha256'] or reference['frames'] != row['frames']:
            raise ValueError(f'Reference crop or metadata changed: {name}')
        paths = [d / 'samples' / f'{name}.json' for d in directories]
        scores = [json.loads(p.read_text()) for p in paths]
        if scores[0]['seed'] != scores[1]['seed']:
            raise ValueError(f'Volume seeds differ: {name}')
        for index, score in enumerate(scores):
            if (score['schema'] != SCHEMA or score['sample_key'] != name or score['metrics']['frames'] != row['frames']
                    or score['scoring_contract_sha256'] != manifests[index]['scoring_contract_sha256']):
                raise ValueError(f'Volume score identity/frame/protocol mismatch: {name}')
            for person in (1, 2):
                if index == 0:
                    expected = reference['outputs'][f'original_gt_p{person}']
                    source_path, source_hash = Path(expected['path']), expected['sha256']
                else:
                    source_path = Path(proxy['completion']).parent / f'{name}_p{person}.npz'
                    source_hash = binding[f'p{person}_sha256']
                if (Path(score[f'input_p{person}']).resolve() != source_path.resolve()
                        or score['input_sha256'][f'p{person}'] != source_hash
                        or sha256_file(source_path) != source_hash):
                    raise ValueError(f'Volume input changed: {source_path}')
            if not all(math.isfinite(score['metrics'][k]) and score['metrics'][k] >= 0 for k in METRICS):
                raise ValueError(f'Invalid volume metrics: {name}')
        passed = all(scores[1]['metrics'][k] <= scores[0]['metrics'][k] for k in METRICS)
        if passed:
            volume_accepted.append(name)
            if name in proxy_accepted:
                accepted.append(name)
        bindings[name] = {'gt_score_sha256': sha256_file(paths[0]), 'candidate_score_sha256': sha256_file(paths[1])}
    return {**proxy,
            'schema': 'hhc_original_gt_two_estimator_quality_gate_v1',
            'status': 'two_estimator_gate_complete',
            'scope': 'Both nearest-vertex and frozen neural SDF estimates; not exact mesh-intersection ground truth',
            'criteria': {**proxy['criteria'], **{k: 'candidate <= original_shape_gt; no tolerance' for k in METRICS}},
            'proxy_accepted_count': proxy['accepted_count'],
            'volume_accepted_count': len(volume_accepted), 'volume_accepted_names': volume_accepted,
            'accepted_names': accepted, 'accepted_count': len(accepted),
            'rejected_names': [name for name in names if name not in set(accepted)],
            'pass_fraction': len(accepted) / len(names),
            'references': str(references_path.resolve()), 'references_sha256': sha256_file(references_path),
            'gt_dir': str(directories[0].resolve()), 'candidate_dir': str(directories[1].resolve()),
            'volume_manifests': {str((d / 'metrics.json').resolve()): sha256_file(d / 'metrics.json') for d in directories},
            'volume_score_bindings': bindings}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('report', 'references', 'gt-dir', 'candidate-dir', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    parser.add_argument('--condition', required=True)
    args = parser.parse_args()
    gate = build_volume_gate(args.report, args.condition, args.references, args.gt_dir, args.candidate_dir)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, gate)
    print({k: gate[k] for k in ('denominator', 'proxy_accepted_count', 'volume_accepted_count', 'accepted_count', 'pass_fraction')})
