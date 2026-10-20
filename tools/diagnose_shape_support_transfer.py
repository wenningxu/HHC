"""Assess shape transfer against frozen original-shape near-ground foot times."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.calc_pene_from_npz import PeneCalculator, to_torch_params, npz_to_amass_dict
from tools.run_interx_shape_augmentation import atomic_json, sha256_file, source_person_payload, canonical_sha256
from utils.interx_physical_ground import place_pair_on_mesh_ground, decode_physical_pair
from optimization.metric_aligned_foot_sliding import FOOT_JOINT_INDICES
from tools.score_volumetric_smpl_npz import foot_sliding


def paired_volumetric_metrics(backend, original, optimized, name):
    from tools.score_volumetric_smpl_npz import score_pair, stable_sample_seed
    assert len(original) == len(optimized) == 2
    lengths = [len(p['trans']) for p in original + optimized]
    assert lengths[0] > 0 and len(set(lengths)) == 1, lengths
    seed = stable_sample_seed(1234, name)
    return {label: score_pair(backend, *[to_torch_params(p, backend.device) for p in pair], seed)
            for label, pair in [('original_shape', original), ('optimized', optimized)]}


def production_ground_rows(directory, reference_manifest, candidate_dir, floors):
    """Validate production provenance before adapting it to the existing scorer."""
    mp = directory / '_completion_manifest.json'
    cp = directory / '_run_contract.json'
    manifest = json.loads(mp.read_text()); contract = json.loads(cp.read_text())
    assert manifest['complete'] and manifest['completed_samples'] == manifest['requested_samples'] == len(manifest['completed_names'])
    assert all(manifest[k] == v for k, v in contract.items())
    assert contract['contract_sha256'] == canonical_sha256({k: v for k, v in contract.items() if k not in ('contract_sha256', 'created_at_utc')})
    assert manifest['optimizer']['ground_alignment_output_coordinates'] == 'physical_pair_common_p1_origin'
    for key in ('source_motion_sha256', 'plan_records_sha256'):
        assert manifest[key] == reference_manifest[key]
    rows = []; bindings = {str(p): sha256_file(p) for p in (mp, cp)}
    for name in manifest['completed_names']:
        sp = directory / f'{name}.json'; side = json.loads(sp.read_text())
        original_sp = candidate_dir / f'{name}.json'; original = json.loads(original_sp.read_text())
        assert side['run_contract_sha256'] == manifest['contract_sha256']
        assert side['augmented_name'] == name
        for key in ('plan_record_sha256', 'source_name', 'crop_start', 'crop_stop', 'source_shapes', 'augmented_shapes'):
            assert side[key] == original[key], key
        ground = side['diagnostics']['ground_alignment']
        assert ground['output_coordinates'] == 'physical_pair_common_p1_origin'
        for label, field in [('original_shape', 'original_mesh_floors_m'), ('new_shape', 'new_mesh_floors_m')]:
            assert ground[field] == [p['mesh_floor_raw_m'] for p in floors[name, label]['people']]
        hashes = [side[f'output_p{i}_sha256'] for i in (1, 2)]
        assert hashes == [sha256_file(directory / f'{name}_p{i}.npz') for i in (1, 2)]
        bindings[str(sp)] = sha256_file(sp)
        rows.append({'name': name, 'common_export_origin': ground['common_export_origin'],
                     'input_sidecar_sha256': sha256_file(original_sp), 'output_npz_sha256': hashes})
    return rows, bindings


def frozen_support_horizontal_motion(reference_heights, joints):
    """Ankle/toe displacement at original mesh support times; no speed filtering."""
    assert reference_heights.shape == (len(joints), 2)
    assert joints.ndim == 3 and joints.shape[1] > max(FOOT_JOINT_INDICES) and joints.shape[2] == 3
    assert torch.isfinite(reference_heights).all() and torch.isfinite(joints).all()
    mask = (reference_heights[:-1] <= .01).repeat_interleave(2, dim=1)
    feet = joints[:, FOOT_JOINT_INDICES]
    speed = torch.linalg.vector_norm(torch.diff(feet[..., [0, 2]], dim=0), dim=-1)[mask]
    return {'observations': int(speed.numel()),
            'sum_m': float(speed.sum()),
            'mean_m_per_frame': float(speed.mean()) if speed.numel() else None,
            'max_m_per_frame': float(speed.max()) if speed.numel() else None}


def support_comparison(reference, candidate):
    assert reference.shape == candidate.shape and reference.ndim == 2 and reference.shape[1] == 2
    assert torch.isfinite(reference).all() and torch.isfinite(candidate).all()
    mask = reference <= .01
    delta = (candidate - reference)[mask]
    result = {'original_support_observations': int(mask.sum()),
            'lost_near_ground_observations': int((mask & (candidate > .01)).sum()),
            'max_height_on_original_support_m': float(candidate[mask].max()) if delta.numel() else None,
            'max_abs_delta_from_original_m': float(delta.abs().max()) if delta.numel() else None,
            'max_up_delta_from_original_m': float(delta.max()) if delta.numel() else None}
    for order in (1, 2):
        for label, values in (('original', reference), ('candidate', candidate), ('error', candidate - reference)):
            result[f'{label}_max_abs_d{order}_y_m_per_frame{order}'] = (
                float(torch.diff(values, n=order, dim=0).abs().max()) if len(values) > order else None)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-dir', type=Path, required=True)
    p.add_argument('--floor-report', type=Path, required=True)
    p.add_argument('--ground-dir', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--penetration', action='store_true', help='Also measure nearest-vertex proxy on CUDA; not independent volumetric acceptance')
    p.add_argument('--volumetric', action='store_true', help='Independent full-crop original/optimized VolumetricSMPL scoring')
    p.add_argument('--contact-dir', type=Path, help='Optional fixed-region contact diagnostic using original plan timings')
    a = p.parse_args()
    assert not a.output.exists()
    assert not a.penetration or a.device.startswith('cuda')
    assert not a.volumetric or a.device.startswith('cuda')
    partial_output = a.output.with_suffix('.partial.json')
    assert not a.volumetric or not partial_output.exists()
    mp = a.candidate_dir / '_completion_manifest.json'
    m = json.loads(mp.read_text()); f = json.loads(a.floor_report.read_text())
    assert m['complete'] and f['completion_sha256'] == sha256_file(mp)
    assert sha256_file(m['source_motion']) == m['source_motion_sha256']
    floors = {(r['name'], r['condition']): r for r in f['rows']}
    selected = []; seen = set(); reports = {}
    for directory in a.ground_dir:
        rp = directory / 'report.json'
        if rp.exists():
            report = json.loads(rp.read_text())
            assert report['status'] == 'ground_diagnostic_complete_not_acceptance'
            assert report['floor_report_sha256'] == sha256_file(a.floor_report)
            reports[str(rp)] = sha256_file(rp)
            ground_rows = report['rows']
        else:
            ground_rows, bindings = production_ground_rows(directory, m, a.candidate_dir, floors)
            reports.update(bindings)
        for row in ground_rows:
            assert row['name'] not in seen
            seen.add(row['name']); selected.append((directory, row))
    assert selected
    calc = PeneCalculator(a.device)
    volume = None
    volume_metadata = {}
    if a.volumetric:
        from optimization.volumetric_smpl_backend import VolumetricSMPLBackend
        volume = VolumetricSMPLBackend(device=a.device, max_query_points_per_call=512)
        volume_metadata = {'checkpoints': volume.checkpoint_records,
            'representation': volume.representation_contract, 'base_seed': 1234,
            'scope': 'Full exact crop; physical original-shape GT versus decoded optimized pair, no re-grounding; shared per-name seed.',
            'implementation_sha256': {str(p): sha256_file(p) for p in
                [Path('tools/score_volumetric_smpl_npz.py'), Path('optimization/volumetric_smpl_backend.py')]}}
    contact_loss = None
    if a.contact_dir:
        from optimization.losses import DyadicInteractionLoss
        from tools.diagnose_shape_contact_coordinates import fixed_region_contact
        contact_loss = DyadicInteractionLoss(device=a.device)
        assert sha256_file(m['plan_records']) == m['plan_records_sha256']
        records = {r['augmented_name']: r for r in map(json.loads, Path(m['plan_records']).read_text().splitlines())}
    feet = [torch.unique(torch.cat([calc.rid_to_vid_tensors[r] for r in group]))
            for group in ((1, 67), (42, 65))]
    rows = []
    with h5py.File(m['source_motion'], 'r') as source, torch.no_grad():
        for directory, row in selected:
            name = row['name']; sp = a.candidate_dir / f'{name}.json'; s = json.loads(sp.read_text())
            assert sha256_file(sp) == row['input_sidecar_sha256']
            raw = source[s['source_name']][s['crop_start']:s['crop_stop']]
            heights = {}; joint_sequences = {}; penetration = {}; contact_geometry = {}
            for condition in ('original_shape', 'new_shape'):
                floor = floors[(name, condition)]
                assert floor['sidecar_sha256'] == sha256_file(sp)
                donors = ([{'betas': s['source_shapes']['betas'][i], 'gender': s['source_shapes']['genders'][i]}
                           for i in range(2)] if condition == 'original_shape' else s['augmented_shapes'])
                people = place_pair_on_mesh_ground([source_person_payload(raw, i, donors[i]) for i in range(2)],
                                                   [v['mesh_floor_raw_m'] for v in floor['people']])
                if condition == 'original_shape':
                    original_people = people
                heights[condition] = []
                joint_sequences[condition] = []
                geometries = []
                for person in people:
                    geometry = calc._compute_geometry(to_torch_params(person, calc.device))
                    verts, joints = geometry[:2]
                    if a.penetration or contact_loss is not None:
                        geometries.append(geometry)
                    joint_sequences[condition].append(joints)
                    heights[condition].append(torch.stack([verts[:, ids, 1].min(1).values for ids in feet], 1))
                if a.penetration:
                    (va, _, na), (vb, _, nb) = geometries
                    penetration[condition] = float(calc._calc_penetration_metric_per_frame(va, vb, na, nb).mean())
                if contact_loss is not None:
                    contact_geometry[condition] = geometries
            loaded = []
            npz_hashes = []
            for i in range(2):
                npz_path = directory / f'{name}_p{i+1}.npz'
                npz_hashes.append(sha256_file(npz_path))
                with np.load(npz_path, allow_pickle=False) as z:
                    loaded.append(npz_to_amass_dict(dict(z)))
            if 'output_npz_sha256' in row:
                assert npz_hashes == row['output_npz_sha256']
            people = decode_physical_pair(loaded, row['common_export_origin'])
            stats = []; geometries = []
            for i, person in enumerate(people):
                assert np.array_equal(person['betas'].reshape(-1), np.asarray(s['augmented_shapes'][i]['betas']).reshape(-1))
                geometry = calc._compute_geometry(to_torch_params(person, calc.device))
                verts, joints = geometry[:2]
                if a.penetration or contact_loss is not None:
                    geometries.append(geometry)
                final = torch.stack([verts[:, ids, 1].min(1).values for ids in feet], 1)
                stats.append({'initial_new_shape': support_comparison(heights['original_shape'][i], heights['new_shape'][i]),
                              'optimized': support_comparison(heights['original_shape'][i], final)})
                stats[-1]['frozen_support_horizontal_motion'] = {
                    label: frozen_support_horizontal_motion(heights['original_shape'][i], values)
                    for label, values in [('original', joint_sequences['original_shape'][i]),
                                          ('initial', joint_sequences['new_shape'][i]), ('optimized', joints)]}
                stats[-1]['standard_foot_sliding'] = {
                    label: foot_sliding(values)
                    for label, values in [('original', joint_sequences['original_shape'][i]),
                                          ('initial', joint_sequences['new_shape'][i]), ('optimized', joints)]}
            if a.penetration:
                (va, _, na), (vb, _, nb) = geometries
                penetration['optimized'] = float(calc._calc_penetration_metric_per_frame(va, vb, na, nb).mean())
                assert all(np.isfinite(x) and x >= 0 for x in penetration.values())
            rows.append({'name': name, 'people': stats, 'input_npz_sha256': npz_hashes,
                         'nearest_vertex_squared_sum_m2_per_frame': penetration})
            if contact_loss is not None:
                contact_geometry['optimized'] = geometries
                cp = a.contact_dir / f'{s["source_name"]}.pkl'
                assert sha256_file(cp) == s['contact_pairs_sha256']
                record = records[name]
                assert sha256_file(record['timing_file']) == record['timing_sha256']
                with cp.open('rb') as handle:
                    contacts = pickle.load(handle)
                timing = torch.as_tensor(np.load(record['timing_file'])[s['crop_start']:s['crop_stop']], device=a.device, dtype=torch.float32)
                assert len(timing) == len(raw)
                (va, _, na), (vb, _, nb) = contact_geometry['new_shape']
                contact_loss._determine_role_assignment(va, vb, na, nb, contacts, timing)
                contact_stats = {}
                for label, pair in contact_geometry.items():
                    x, y = pair[::-1] if contact_loss.swapped else pair
                    attraction = float(contact_loss._calc_semantic_contact_loss(x[0], y[0], x[2], y[2], contacts, timing)[0])
                    assert np.isfinite(attraction)
                    contact_stats[label] = {'semantic_attraction': attraction,
                        'fixed_regions': fixed_region_contact(x[0], y[0], contacts, contact_loss.rid_to_vid_tensors, timing)}
                rows[-1]['contacts'] = {'role_swapped_initial_physical': contact_loss.swapped,
                    'contact_sha256': sha256_file(cp), 'timing_sha256': record['timing_sha256'], 'conditions': contact_stats}
            if volume is not None:
                rows[-1]['volumetric'] = paired_volumetric_metrics(volume, original_people, people, name)
                atomic_json(partial_output, {'status': 'running_not_acceptance',
                    'ground_reports_sha256': reports, 'floor_report_sha256': sha256_file(a.floor_report),
                    'volumetric_metadata': volume_metadata, 'rows': rows})
                print(f'Volumetric completed {len(rows)}/{len(selected)}: {name}', flush=True)
    atomic_json(a.output, {'status': 'original_support_diagnostic_not_acceptance',
        'scope': 'Original-shape mesh foot minimum <=1cm freezes times, not true contact labels. Each shape uses its own full-sequence constant floor. No projection or selection by outcome.',
        'ground_reports_sha256': reports, 'floor_report_sha256': sha256_file(a.floor_report),
        'horizontal_motion_scope': 'Original mesh foot height <=1cm at interval start, ankle/toe XZ displacement, m/frame, no speed filter. Diagnostic, not official footskating metric.',
        'standard_foot_sliding_scope': 'Existing score_volumetric_smpl_npz.foot_sliding formula on HBP joints; per-candidate skeleton floor/mask. Not VolumetricSMPL backend verification.',
        'standard_foot_sliding_implementation_sha256': sha256_file(Path(__file__).with_name('score_volumetric_smpl_npz.py')),
        'device': a.device, 'penetration_enabled': a.penetration,
        'volumetric_enabled': a.volumetric, 'volumetric_metadata': volume_metadata,
        'penetration_scope': 'Original/new shape use own full-sequence physical mesh floor; optimized common origin decoded. Exact crop, both directions, no timing filtering; nearest-vertex proxy, not volumetric acceptance.',
        'penetration_implementation_sha256': sha256_file(Path('eval/calc_pene_from_npz.py')),
        'contact_scope': 'Optional: fixed annotated unions, timing>0 with timing weights; roles selected once on physical new-shape initialization. Unsigned proximity not true contact or annotation correctness.',
        'contact_implementation_sha256': ({str(p): sha256_file(p) for p in [Path('optimization/losses.py'), Path('tools/diagnose_shape_contact_coordinates.py')]} if a.contact_dir else {}),
        'implementation_sha256': sha256_file(Path(__file__)), 'rows': rows})
    print(f'Completed {len(rows)} paired samples: {a.output}', flush=True)


if __name__ == '__main__':
    test_joints = torch.zeros(3, 12, 3)
    test_joints[1:, :, 0] = torch.tensor([.1, .2])[:, None]
    test_heights = torch.tensor([[0., .02], [0., .02], [0., .02]])
    horizontal = frozen_support_horizontal_motion(test_heights, test_joints)
    assert horizontal['observations'] == 4 and abs(horizontal['mean_m_per_frame'] - .1) < 1e-7
    assert frozen_support_horizontal_motion(test_heights + 1, test_joints)['mean_m_per_frame'] is None
    assert frozen_support_horizontal_motion(test_heights[:1], test_joints[:1])['observations'] == 0
    reference = torch.tensor([[0., .02], [.005, .03]])
    candidate = torch.tensor([[.02, .02], [.006, .03]])
    check = support_comparison(reference, candidate)
    assert check['original_support_observations'] == 2 and check['lost_near_ground_observations'] == 1
    assert abs(check['max_abs_delta_from_original_m'] - .02) < 1e-7
    assert support_comparison(reference + 1, candidate)['max_abs_delta_from_original_m'] is None
    assert support_comparison(reference, reference + .1)['error_max_abs_d1_y_m_per_frame1'] < 1e-7
    assert check['candidate_max_abs_d2_y_m_per_frame2'] is None
    main()
