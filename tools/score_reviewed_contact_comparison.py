"""Compare identical output motions under both old and reviewed contact labels."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.check_shape_ground_after_optimization import reviewed_contacts
from tools.run_interx_shape_augmentation import source_person_payload, sha256_file, atomic_json
from tools.diagnose_shape_contact_coordinates import fixed_region_contact
from eval.calc_pene_from_npz import PeneCalculator, to_torch_params, npz_to_amass_dict
from optimization.losses import DyadicInteractionLoss
from utils.interx_physical_ground import place_pair_on_mesh_ground, decode_physical_pair


def region_diagnostics(vertices_a, vertices_b, contacts, regions, winners):
    from pytorch3d.ops import knn_points
    def distances(ids_a, ids_b):
        x, y = vertices_a[:, ids_a].contiguous(), vertices_b[:, ids_b].contiguous()
        return knn_points(x, y, K=1).dists[..., 0].amin(1).clamp_min(0).sqrt().tolist()
    rows = []
    for i, contact in enumerate(contacts):
        union = [torch.unique(torch.cat([regions[r] for r in contact[key]])) for key in ('p1_ids', 'p2_ids')]
        row = {'union_distance_m_by_frame': distances(*union)}
        if i in winners:
            selected = winners[i]
            ids = [[int(r) for r, indices in regions.items() if torch.equal(indices, choice)] for choice in selected]
            assert all(len(group) == 1 for group in ids), 'winner must match one known region per person'
            row.update(winner_region_ids=[group[0] for group in ids], winner_distance_m_by_frame=distances(*selected))
        rows.append(row)
    return rows


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-dir', type=Path, required=True)
    p.add_argument('--floor-report', type=Path, required=True)
    p.add_argument('--contact-dir', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--ground-dir', action='append', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--region-diagnostics', action='store_true', help='Offline re-selection on each saved condition, not an optimization trajectory')
    a = p.parse_args(); assert not a.output.exists()
    assert len({p.resolve() for p in a.ground_dir}) == len(a.ground_dir)
    mp = a.candidate_dir / '_completion_manifest.json'; m = json.loads(mp.read_text())
    f = json.loads(a.floor_report.read_text()); assert m['complete'] and f['completion_sha256'] == sha256_file(mp)
    sp = a.candidate_dir / f'{a.name}.json'; s = json.loads(sp.read_text())
    assert a.name in m['completed_names'] and sha256_file(m['source_motion']) == m['source_motion_sha256']
    assert sha256_file(m['plan_records']) == m['plan_records_sha256']
    record = next(r for r in map(json.loads, Path(m['plan_records']).read_text().splitlines()) if r['augmented_name'] == a.name)
    cp = a.contact_dir / f'{s["source_name"]}.pkl'; assert sha256_file(cp) == s['contact_pairs_sha256']
    with cp.open('rb') as handle: old = pickle.load(handle)
    entry = next(e for e in json.loads(a.review.read_text())['entries'] if e['source_name'] == s['source_name'])
    corrected = reviewed_contacts(entry, old, sha256_file(cp), s['caption'])
    assert sha256_file(record['timing_file']) == record['timing_sha256']
    timing = torch.as_tensor(np.load(record['timing_file'])[s['crop_start']:s['crop_stop']], dtype=torch.float32)
    with h5py.File(m['source_motion'], 'r') as source: raw = source[s['source_name']][s['crop_start']:s['crop_stop']]
    calc = PeneCalculator('cpu'); loss = DyadicInteractionLoss(device='cpu'); pairs = {}; bindings = {}
    for label in ('original_shape', 'new_shape'):
        floor = next(r for r in f['rows'] if r['name'] == a.name and r['condition'] == label)
        assert floor['sidecar_sha256'] == sha256_file(sp)
        donors = ([{'betas': s['source_shapes']['betas'][i], 'gender': s['source_shapes']['genders'][i]} for i in (0, 1)]
                  if label == 'original_shape' else s['augmented_shapes'])
        pairs[label] = place_pair_on_mesh_ground([source_person_payload(raw, i, donors[i]) for i in (0, 1)],
                                               [r['mesh_floor_raw_m'] for r in floor['people']])
    for directory in a.ground_dir:
        rp = directory / 'report.json'; report = json.loads(rp.read_text())
        assert report['status'] == 'ground_diagnostic_complete_not_acceptance' and report['floor_report_sha256'] == sha256_file(a.floor_report)
        row = next(r for r in report['rows'] if r['name'] == a.name)
        assert row['input_sidecar_sha256'] == sha256_file(sp)
        if row.get('contact_review_sha256') is not None: assert row['contact_review_sha256'] == sha256_file(a.review)
        loaded = []; bindings[str(rp)] = sha256_file(rp)
        for i in (1, 2):
            path = directory / f'{a.name}_p{i}.npz'
            with np.load(path, allow_pickle=False) as z: loaded.append(npz_to_amass_dict(dict(z)))
            bindings[str(path)] = sha256_file(path)
        pairs[str(directory)] = decode_physical_pair(loaded, row['common_export_origin'])
    geometry = {}
    for label, people in pairs.items():
        assert all(len(person['trans']) == len(timing) for person in people)
        if label != 'original_shape':
            assert all(np.array_equal(person['betas'].reshape(-1), np.asarray(s['augmented_shapes'][i]['betas']).reshape(-1)) for i, person in enumerate(people))
        geometry[label] = [calc._compute_geometry(to_torch_params(person, 'cpu')) for person in people]
    results = {}
    for label, contacts in [('old_annotation', old), ('reviewed_annotation', corrected)]:
        x, y = geometry['new_shape']; loss._determine_role_assignment(x[0], y[0], x[2], y[2], contacts, timing)
        swapped = loss.swapped; scores = {}
        for condition, pair in geometry.items():
            x, y = pair[::-1] if swapped else pair
            loss.current_winner_regions = {}
            scores[condition] = {'semantic_attraction': float(loss._calc_semantic_contact_loss(x[0], y[0], x[2], y[2], contacts, timing)[0]),
                                 'fixed_regions': fixed_region_contact(x[0], y[0], contacts, loss.rid_to_vid_tensors, timing)}
            if a.region_diagnostics:
                scores[condition]['region_diagnostics'] = region_diagnostics(x[0], y[0], contacts, loss.rid_to_vid_tensors, loss.current_winner_regions)
                scores[condition]['attraction_by_relation'] = [float(loss._calc_semantic_contact_loss(x[0], y[0], x[2], y[2], [contact], timing)[0]) for contact in contacts]
                assert np.isclose(np.mean(scores[condition]['attraction_by_relation']) if contacts else 0., scores[condition]['semantic_attraction'], rtol=1e-5, atol=1e-7)
        results[label] = {'role_swapped_on_new_shape': swapped, 'contacts': contacts, 'conditions': scores}
    atomic_json(a.output, {'status': 'paired_contact_diagnostic_not_acceptance', 'name': a.name, 'results': results,
                'scope': 'Each label set selects roles once on identical new-shape initialization; same historical timing for every condition. Not true contact labels.',
                'review_sha256': sha256_file(a.review), 'sidecar_sha256': sha256_file(sp), 'timing_sha256': record['timing_sha256'],
                'timing_by_frame': timing.tolist() if a.region_diagnostics else None,
                'bindings': bindings, 'implementation_sha256': {str(path): sha256_file(path) for path in
                    (Path(__file__), Path('tools/check_shape_ground_after_optimization.py'), Path('tools/diagnose_shape_contact_coordinates.py'), Path('optimization/losses.py'))}})


if __name__ == '__main__':
    main()
