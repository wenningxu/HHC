"""Resume a production assessment without changing the frozen metric implementation."""
import argparse
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_shape_support_transfer as scorer
from tools.run_interx_shape_augmentation import sha256_file


def same_values(old, new):
    """Replay tolerance is diagnostic only; saved scores and quality gates stay exact."""
    assert type(old) is type(new), (type(old), type(new))
    if isinstance(old, dict):
        assert old.keys() == new.keys()
        for key in old:
            same_values(old[key], new[key])
    elif isinstance(old, list):
        assert len(old) == len(new)
        for x, y in zip(old, new):
            same_values(x, y)
    elif isinstance(old, float):
        assert math.isfinite(old) and math.isfinite(new)
        assert math.isclose(old, new, rel_tol=1e-6, abs_tol=1e-10), (old, new)
    else:
        assert old == new, (old, new)


def merged_rows(saved, fresh, names, complete=False):
    assert saved and fresh
    assert [r['name'] for r in saved] == names[:len(saved)]
    start = len(saved) - 1
    assert [r['name'] for r in fresh] == names[start:start + len(fresh)]
    same_values(saved[-1], fresh[0])
    rows = saved + fresh[1:]
    assert len(rows) <= len(names)
    if complete:
        assert [r['name'] for r in rows] == names
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-dir', 'floor-report', 'contact-dir', 'partial', 'reference-report', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    a = p.parse_args()
    assert a.device.startswith('cuda')
    assert not a.output.exists() and not a.output.with_suffix('.partial.json').exists()
    assert a.output.resolve() != a.partial.resolve()
    saved = json.loads(a.partial.read_text())
    reference = json.loads(a.reference_report.read_text())
    assert saved['status'] == 'running_not_acceptance'
    assert reference['status'] == 'original_support_diagnostic_not_acceptance'
    assert reference['volumetric_enabled'] and reference['penetration_enabled']
    assert reference['device'] == a.device
    # Legacy partials lack the scorer hash; bind to the frozen, completed sibling report.
    assert reference['implementation_sha256'] == sha256_file(scorer.__file__)
    assert reference['penetration_implementation_sha256'] == sha256_file('eval/calc_pene_from_npz.py')
    assert reference['standard_foot_sliding_implementation_sha256'] == sha256_file('tools/score_volumetric_smpl_npz.py')
    assert reference['contact_implementation_sha256']
    for path, digest in reference['contact_implementation_sha256'].items():
        assert sha256_file(path) == digest
    assert saved['volumetric_metadata'] == reference['volumetric_metadata']
    for path, digest in saved['volumetric_metadata']['implementation_sha256'].items():
        assert sha256_file(path) == digest
    for checkpoint in saved['volumetric_metadata']['checkpoints'].values():
        assert sha256_file(checkpoint['path']) == checkpoint['sha256']
    assert saved['floor_report_sha256'] == sha256_file(a.floor_report)
    for path, digest in saved['ground_reports_sha256'].items():
        assert sha256_file(path) == digest
    manifest = json.loads((a.candidate_dir / '_completion_manifest.json').read_text())
    names = manifest['completed_names']
    prefix = saved['rows']
    assert 0 < len(prefix) < len(names) and len(names) == len(set(names))
    assert [r['name'] for r in prefix] == names[:len(prefix)]
    assert sha256_file(manifest['plan_records']) == manifest['plan_records_sha256']
    records = {r['augmented_name']: r for r in map(json.loads, Path(manifest['plan_records']).read_text().splitlines())}
    for row in prefix:
        side = json.loads((a.candidate_dir / (row['name'] + '.json')).read_text())
        assert row['input_npz_sha256'] == [side[f'output_p{i}_sha256'] for i in (1, 2)]
        assert row['contacts']['contact_sha256'] == side['contact_pairs_sha256'] == sha256_file(a.contact_dir / (side['source_name'] + '.pkl'))
        record = records[row['name']]
        assert row['contacts']['timing_sha256'] == record['timing_sha256'] == sha256_file(record['timing_file'])
        assert set(row['volumetric']) == {'original_shape', 'optimized'}
        for score in row['volumetric'].values():
            assert score['frames'] == side['crop_stop'] - side['crop_start']
        same_values(row, row)  # Reject nonfinite saved numeric values.
    provenance = {'partial': str(a.partial), 'partial_sha256': sha256_file(a.partial),
                  'frozen_reference_report': str(a.reference_report),
                  'frozen_reference_report_sha256': sha256_file(a.reference_report),
                  'resume_implementation_sha256': sha256_file(__file__),
                  'saved_rows': len(prefix), 'replayed_name': prefix[-1]['name'],
                  'replay_rtol': 1e-6, 'replay_atol': 1e-10,
                  'legacy_partial_limitation': 'No scorer hash in old partial; frozen sibling report and replay provide corroboration, not a retroactive runtime attestation.'}
    original_rows, original_write = scorer.production_ground_rows, scorer.atomic_json

    def remaining(directory, reference_manifest, candidate_dir, floors):
        rows, bindings = original_rows(directory, reference_manifest, candidate_dir, floors)
        assert [r['name'] for r in rows] == names
        assert bindings == saved['ground_reports_sha256']
        return rows[len(prefix) - 1:], bindings

    def write(path, payload):
        for key in ('ground_reports_sha256', 'floor_report_sha256', 'volumetric_metadata'):
            assert payload[key] == saved[key]
        complete = payload['status'] == 'original_support_diagnostic_not_acceptance'
        payload = {**payload, 'rows': merged_rows(prefix, payload['rows'], names, complete),
                   'recovery': provenance}
        original_write(path, payload)
        print(f"Recovered total {len(payload['rows'])}/{len(names)}; complete={complete}", flush=True)

    argv = [scorer.__file__, '--candidate-dir', str(a.candidate_dir), '--floor-report', str(a.floor_report),
            '--ground-dir', str(a.candidate_dir), '--device', a.device, '--penetration', '--volumetric',
            '--contact-dir', str(a.contact_dir), '--output', str(a.output)]
    print(f'Validated {len(prefix)} saved rows; replaying last row then {len(names)-len(prefix)} missing rows.', flush=True)
    with patch.object(scorer, 'production_ground_rows', remaining), patch.object(scorer, 'atomic_json', write), patch.object(sys, 'argv', argv):
        scorer.main()


if __name__ == '__main__':
    main()
