"""Single-sample, original-geometry pseudo-times; never edits the source timing cache."""
import argparse
import json
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import check_shape_ground_after_optimization as checker
from tools.run_interx_shape_augmentation import sha256_file, atomic_json
from optimization.mesh_ground_support import MeshGroundSupportLoss


def geometric_times(distances):
    distances = np.asarray(distances, dtype=np.float64)
    assert distances.ndim == 2 and min(distances.shape) > 0
    assert np.isfinite(distances).all() and (distances >= 0).all()
    # Geometry-derived confidence, NOT historical annotation thresholds or human truth.
    weights = np.clip((.03 - distances) / .02, 0., 1.)
    assert (weights.sum(0) > 0).all(), 'Do not invent an event without geometric support'
    return weights


def install_times(loss, weights, expected_swapped):
    assert isinstance(loss, MeshGroundSupportLoss) and loss.relation_timing_scale == 0
    original = loss._determine_role_assignment
    def assign(*args, **kwargs):
        original(*args, **kwargs)
        assert loss.swapped == expected_swapped, 'Geometry timings are bound to a fixed global role assignment'
        assert len(args[0]) == len(weights) and len(args[4]) == weights.shape[1]
        loss._relation_timings = torch.as_tensor(weights, dtype=args[0].dtype, device=args[0].device)
    loss._determine_role_assignment = assign
    return loss


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--timing-evidence', type=Path, required=True)
    extra, remaining = parser.parse_known_args()
    inherited = argparse.ArgumentParser(add_help=False)
    for flag in ('candidate-dir', 'floor-report', 'contact-dir', 'contact-review', 'output-dir'):
        inherited.add_argument('--' + flag, type=Path, required=True)
    inherited.add_argument('--offset', type=int, default=0)
    inherited.add_argument('--limit', type=int, default=1)
    inherited.add_argument('--iterations', type=int, required=True)
    inherited.add_argument('--relation-timing-scale', type=float, default=0.)
    a, _ = inherited.parse_known_args(remaining)
    assert a.limit == 1 and 0 < a.iterations <= 100 and a.relation_timing_scale == 0
    assert not a.output_dir.exists()
    evidence = json.loads(extra.timing_evidence.read_text())
    assert evidence['status'] == 'paired_contact_diagnostic_not_acceptance'
    m = json.loads((a.candidate_dir / '_completion_manifest.json').read_text())
    name = m['completed_names'][a.offset]
    assert evidence['name'] == name and evidence['review_sha256'] == sha256_file(a.contact_review)
    sp = a.candidate_dir / f'{name}.json'; s = json.loads(sp.read_text())
    assert evidence['sidecar_sha256'] == sha256_file(sp)
    for path, digest in evidence['bindings'].items():
        assert sha256_file(Path(path)) == digest
        if path.endswith('/report.json'):
            assert json.loads(Path(path).read_text())['floor_report_sha256'] == sha256_file(a.floor_report)
    record = next(r for r in map(json.loads, Path(m['plan_records']).read_text().splitlines()) if r['augmented_name'] == name)
    assert evidence['timing_sha256'] == record['timing_sha256'] == sha256_file(Path(record['timing_file']))
    cp = a.contact_dir / f'{s["source_name"]}.pkl'
    entry = next(e for e in json.loads(a.contact_review.read_text())['entries'] if e['source_name'] == s['source_name'])
    contacts = checker.reviewed_contacts(entry, pickle.loads(cp.read_bytes()), sha256_file(cp), s['caption'])
    reviewed = evidence['results']['reviewed_annotation']
    assert reviewed['contacts'] == contacts
    distances = np.asarray([r['union_distance_m_by_frame'] for r in reviewed['conditions']['original_shape']['region_diagnostics']]).T
    assert distances.shape == (s['crop_stop'] - s['crop_start'], len(contacts))
    weights = geometric_times(distances)
    original_factory = checker.with_attraction_weight
    def factory(*args, **kwargs):
        return install_times(original_factory(*args, **kwargs), weights, reviewed['role_swapped_on_new_shape'])
    with patch.object(sys, 'argv', [sys.argv[0], *remaining]), patch.object(checker, 'with_attraction_weight', factory):
        checker.main()
    report_path = a.output_dir / 'report.json'; report = json.loads(report_path.read_text())
    report['relation_timing_override'] = {'evidence_sha256': sha256_file(extra.timing_evidence),
        'scope': 'Original-shape union distance: weight1 at <=1cm, linear to weight0 at >=3cm. May extend beyond old global timing. Auxiliary losses keep old timing. Not human truth.',
        'weights_by_frame': weights.tolist(), 'role_swapped': reviewed['role_swapped_on_new_shape']}
    report['implementation_sha256'][str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    atomic_json(report_path, report)


if __name__ == '__main__':
    main()
