"""Ground-only assessment of the unchanged optimizer with mesh-aligned inputs."""
import argparse
import json
import pickle
import sys
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.calc_pene_from_npz import PeneCalculator, to_torch_params
from optimize.optimize_full_body import optimize_full_body
from optimization.losses import DyadicInteractionLoss
from optimization.mesh_ground_support import MeshGroundSupportLoss
from tools.run_interx_shape_augmentation import atomic_json, atomic_npz, canonical_sha256, sha256_file, source_person_payload
from utils.interx_physical_ground import place_pair_on_mesh_ground, encode_physical_pair, decode_physical_pair


def ground_state(vertices, foot_regions, initial_support=None):
    heights = torch.stack([vertices[:, idx, 1].min(1).values for idx in foot_regions], dim=1)
    support = heights <= .01 if initial_support is None else initial_support
    assert torch.isfinite(vertices).all() and support.shape == heights.shape
    return {'min_vertex_y_m': float(vertices[..., 1].min()),
            'ground_loss': float(torch.relu(-vertices[..., 1]).mean()),
            'below_ground_vertex_fraction_1mm': float((vertices[..., 1] < -.001).float().mean()),
            'initial_support_observations': int(support.sum()),
            'initial_support_max_height_m': float(heights[support].max()) if bool(support.any()) else None,
            'initial_support_mean_height_m': float(heights[support].mean()) if bool(support.any()) else None}, support


def reviewed_contacts(entry, original, digest, caption):
    from eval.extract_interx_contacts_codex import map_contact_rows
    assert entry.get('review_status') == 'approved_for_diagnostic_only' and not entry.get('do_not_apply', False)
    assert entry['pickle_sha256'] == digest and entry['old_contacts'] == original and entry['caption'] == caption
    assert entry['new_contacts'], 'Do not silently drop all semantic relations'
    for row in entry['new_contacts']:
        assert all(np.isfinite(row[k]) and row[k] >= 0 for k in ('w_p1', 'w_p2'))
        ratio = '5:5' if abs(row['w_p1'] - .5) < .01 else ('9:1' if row['w_p1'] > .5 else '1:9')
        assert map_contact_rows([{'p1_part': row['p1_part'], 'p2_part': row['p2_part'],
                                 'contact_type': row['contact_type'], 'role_ratio': ratio}])[0] == row
    return entry['new_contacts']


def gradient_snapshot(terms, variables):
    gradients = {}
    for term, value in terms.items():
        values = torch.autograd.grad(value, list(variables.values()), retain_graph=True, allow_unused=True) if value.requires_grad else [None] * len(variables)
        gradients[term] = {key: torch.zeros_like(var) if grad is None else grad.detach()
                           for (key, var), grad in zip(variables.items(), values)}
    rows = {}
    for key in variables:
        reference = gradients['attract'][key].flatten()
        rows[key] = {}
        for term in terms:
            grad = gradients[term][key].flatten()
            assert torch.isfinite(grad).all()
            dot = float(grad @ reference); denominator = float(grad.norm() * reference.norm())
            rows[key][term] = {'norm': float(grad.norm()), 'dot_attract': dot,
                              'cosine_attract': dot / denominator if denominator > 0 else None}
    return {'weighted_values': {key: float(value.detach()) for key, value in terms.items()}, 'parameters': rows}


def with_attraction_weight(factory, weight, *args, gradient_records=None, gradient_epochs=(), **kwargs):
    assert np.isfinite(weight) and weight > 0
    loss = factory(*args, **kwargs)
    loss.weights['attract'] = float(weight)
    if gradient_records is not None:
        def audit(module, inputs, output):
            epoch = int(inputs[5])
            if epoch not in gradient_epochs:
                return
            total, parts = output
            variables = {f'P{i+1}.{key}': value for i, person in enumerate(inputs[:2])
                         for key, value in person.items() if torch.is_tensor(value) and value.requires_grad}
            terms = {key: parts[key] * module.weights['smooth' if key == 'acc' else key]
                     for key in ('attract', 'pene', 'ground', 'skate', 'reg', 'smooth', 'acc')}
            terms['total'] = total
            assert torch.allclose(sum(value for key, value in terms.items() if key != 'total'), total)
            gradient_records.append({'epoch': epoch, **gradient_snapshot(terms, variables)})
        loss.register_forward_hook(audit)
    return loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-dir', type=Path, required=True)
    p.add_argument('--floor-report', type=Path, required=True)
    p.add_argument('--contact-dir', type=Path, required=True)
    p.add_argument('--contact-review', type=Path, help='Explicitly reviewed, hash-bound diagnostic overrides; never changes the source cache')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--limit', type=int, default=4)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--iterations', type=int)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--reuse-body-models', action='store_true')
    p.add_argument('--ground-policy', choices=('released', 'mesh_support', 'mesh_support_max', 'mesh_support_original', 'mesh_support_original_soft'), default='released')
    p.add_argument('--ground-weight', type=float, default=20.)
    p.add_argument('--ground-temporal-weight', type=float, default=0.)
    p.add_argument('--ground-horizontal-weight', type=float, default=0.)
    p.add_argument('--relation-timing-scale', type=float, default=0.)
    p.add_argument('--attraction-weight', type=float, default=1., help='Isolated diagnostic only; other loss weights unchanged')
    p.add_argument('--gradient-audit', action='store_true', help='Extra backwards at first/middle/last iteration; timing includes diagnostic overhead')
    a = p.parse_args()
    reviews = [] if a.contact_review is None else json.loads(a.contact_review.read_text())['entries']
    assert len({r['source_name'] for r in reviews}) == len(reviews)
    assert a.limit > 0 and a.offset >= 0 and not a.output_dir.exists()
    assert a.iterations is None or a.iterations > 0
    assert a.learning_rate is None or (np.isfinite(a.learning_rate) and a.learning_rate > 0)
    assert np.isfinite(a.ground_weight) and a.ground_weight > 0
    assert np.isfinite(a.attraction_weight) and a.attraction_weight > 0
    assert np.isfinite(a.ground_temporal_weight) and a.ground_temporal_weight >= 0
    assert np.isfinite(a.ground_horizontal_weight) and a.ground_horizontal_weight >= 0
    assert not a.ground_horizontal_weight or a.ground_policy in ('mesh_support_original', 'mesh_support_original_soft')
    assert np.isfinite(a.relation_timing_scale) and a.relation_timing_scale >= 0
    assert not a.relation_timing_scale or a.ground_policy in ('mesh_support_original', 'mesh_support_original_soft')
    assert a.ground_policy != 'released' or a.ground_temporal_weight == 0
    assert a.ground_policy != 'released' or a.ground_weight == 20., 'released retains its original weight'
    mp = a.candidate_dir / '_completion_manifest.json'; m = json.loads(mp.read_text())
    config = dict(m['optimizer'])
    if a.iterations is not None:
        config['iterations'] = a.iterations
    if a.learning_rate is not None:
        config['learning_rate'] = a.learning_rate
    assert not a.relation_timing_scale or config['iterations'] <= 100, 'pilot excludes anchored phase'
    f = json.loads(a.floor_report.read_text())
    assert f['completion_sha256'] == sha256_file(mp) and m['complete']
    assert sha256_file(m['source_motion']) == m['source_motion_sha256']
    assert sha256_file(m['plan_records']) == m['plan_records_sha256']
    records = {r['augmented_name']: r for r in map(json.loads, Path(m['plan_records']).read_text().splitlines())}
    floor_rows = {r['name']: r for r in f['rows'] if r['condition'] == 'new_shape'}
    selected = m['completed_names'][a.offset:a.offset + a.limit]
    assert len(selected) == a.limit and len(set(selected)) == a.limit
    assert all(name in floor_rows for name in selected)
    calc = PeneCalculator('cpu')
    foot_regions = [torch.unique(torch.cat([calc.rid_to_vid_tensors[k] for k in group])) for group in ((1, 67), (42, 65))]
    synthetic = torch.zeros(2, 2, 3); synthetic[1, 0, 1] = -.002
    check, _ = ground_state(synthetic, [torch.tensor([0]), torch.tensor([1])])
    assert abs(check['min_vertex_y_m'] + .002) < 1e-8 and check['below_ground_vertex_fraction_1mm'] == .25
    a.output_dir.mkdir(parents=True)
    rows = []
    static_body_models = {} if a.reuse_body_models else None
    with h5py.File(m['source_motion'], 'r') as source:
        for name in selected:
            sp = a.candidate_dir / f'{name}.json'; s = json.loads(sp.read_text()); r = records[name]
            assert canonical_sha256(r) == s['plan_record_sha256']
            assert floor_rows[name]['sidecar_sha256'] == sha256_file(sp)
            floors = [x['mesh_floor_raw_m'] for x in floor_rows[name]['people']]
            raw = source[s['source_name']][s['crop_start']:s['crop_stop']]
            people = [source_person_payload(raw, i, s['augmented_shapes'][i]) for i in range(2)]
            physical = place_pair_on_mesh_ground(people, floors)
            reference_params = None
            if a.ground_policy in ('mesh_support_original', 'mesh_support_original_soft'):
                original_floor = next(x for x in f['rows'] if x['name'] == name and x['condition'] == 'original_shape')
                assert original_floor['sidecar_sha256'] == sha256_file(sp)
                original_people = [source_person_payload(raw, i, {'betas': s['source_shapes']['betas'][i],
                                   'gender': s['source_shapes']['genders'][i]}) for i in range(2)]
                original_people = place_pair_on_mesh_ground(original_people, [x['mesh_floor_raw_m'] for x in original_floor['people']])
                reference_params = [to_torch_params(person, a.device) for person in original_people]
            cp = a.contact_dir / f'{s["source_name"]}.pkl'
            assert sha256_file(cp) == s['contact_pairs_sha256'] and sha256_file(r['timing_file']) == r['timing_sha256']
            with cp.open('rb') as handle:
                contacts = pickle.load(handle)
            if a.contact_review is not None:
                entry = next(e for e in reviews if e['source_name'] == s['source_name'])
                contacts = reviewed_contacts(entry, contacts, sha256_file(cp), s['caption'])
            timing = torch.tensor(np.load(r['timing_file'])[s['crop_start']:s['crop_stop']], dtype=torch.float32, device=a.device)
            torch.manual_seed(1234)
            # Scoped substitution only in this diagnostic process; production stays unchanged.
            loss_type = DyadicInteractionLoss if a.ground_policy == 'released' else partial(
                MeshGroundSupportLoss, ground_weight=a.ground_weight,
                temporal_weight=a.ground_temporal_weight,
                horizontal_weight=a.ground_horizontal_weight,
                relation_timing_scale=a.relation_timing_scale,
                reference_params=reference_params,
                support_reduction=('height_decay' if a.ground_policy == 'mesh_support_original_soft' else
                                   'mean' if a.ground_policy == 'mesh_support' else 'frame_max'))
            gradient_records = [] if a.gradient_audit else None
            with patch('optimize.optimize_full_body.DyadicInteractionLoss', partial(with_attraction_weight, loss_type, a.attraction_weight,
                       gradient_records=gradient_records, gradient_epochs=(0, (config['iterations']-1)//2, config['iterations']-1))):
                if str(a.device).startswith('cuda'):
                    torch.cuda.synchronize(a.device)
                started = time.perf_counter()
                refined = optimize_full_body(*physical, temporal_mask=timing, interaction_pairs=contacts,
                    num_epochs=config['iterations'], lr=config['learning_rate'], device=a.device,
                    static_body_models=static_body_models,
                    signed_distance_backend=config['signed_distance_backend'],
                    foot_contact_mask_policy=config['foot_contact_mask_policy'],
                    penetration_loss_reduction=config['penetration_loss_reduction'],
                    aux_temporal_weighting=config['aux_temporal_weighting'])
                if str(a.device).startswith('cuda'):
                    torch.cuda.synchronize(a.device)
                optimizer_call_seconds = time.perf_counter() - started
            for i in range(2):
                refined[i]['gender'] = physical[i]['gender']
                assert np.array_equal(refined[i]['betas'], physical[i]['betas'])
            encoded, origin = encode_physical_pair(refined)
            loaded = []
            for i, payload in enumerate(encoded):
                path = a.output_dir / f'{name}_p{i+1}.npz'; atomic_npz(path, payload)
                with np.load(path, allow_pickle=False) as z:
                    loaded.append(dict(z))
            decoded = decode_physical_pair(loaded, origin)
            stats = []
            with torch.no_grad():
                for i in range(2):
                    vi, _, _ = calc._compute_geometry(to_torch_params(physical[i], calc.device))
                    vf, _, _ = calc._compute_geometry(to_torch_params(refined[i], calc.device))
                    vr, _, _ = calc._compute_geometry(to_torch_params(decoded[i], calc.device))
                    before, support = ground_state(vi, foot_regions)
                    after, _ = ground_state(vf, foot_regions, support)
                    initial_heights = torch.stack([vi[:, idx, 1].min(1).values for idx in foot_regions], dim=1)
                    final_heights = torch.stack([vf[:, idx, 1].min(1).values for idx in foot_regions], dim=1)
                    support_delta = (final_heights - initial_heights)[support]
                    after['support_height_delta_max_abs_m'] = float(support_delta.abs().max()) if support_delta.numel() else None
                    after['support_height_delta_max_up_m'] = float(support_delta.max()) if support_delta.numel() else None
                    after['support_height_delta_min_m'] = float(support_delta.min()) if support_delta.numel() else None
                    after['root_y_delta_max_abs_m'] = float(np.abs(refined[i]['trans'][:, 1] - physical[i]['trans'][:, 1]).max())
                    body_delta = (refined[i]['pose_body'] - physical[i]['pose_body']).reshape(-1, 3)
                    after['body_axis_angle_delta_max_norm_rad'] = float(np.linalg.norm(body_delta, axis=1).max())
                    assert float((vf-vr).abs().max()) < 2e-6
                    stats.append({'before': before, 'after': after, 'export_roundtrip_max_m': float((vf-vr).abs().max())})
            row = {'name': name, 'common_export_origin': origin.tolist(), 'people': stats,
                   'gradient_audit': gradient_records,
                   'contacts_used': contacts,
                   'contact_review_sha256': sha256_file(a.contact_review) if a.contact_review else None,
                   'optimizer_call_seconds': optimizer_call_seconds,
                   'input_sidecar_sha256': sha256_file(sp)}
            rows.append(row); atomic_json(a.output_dir / f'{name}.json', row)
            print(json.dumps(row), flush=True)
    atomic_json(a.output_dir / 'report.json', {'status': 'ground_diagnostic_complete_not_acceptance',
        'scope': 'Mesh-ground inputs, explicitly selected ground-only loss policy; other optimization settings unchanged. No posthoc floor projection. Assessment fields here retain new-shape initial support. mesh_support_original instead uses original-shape near-plane vertex heights/mask in the loss; use diagnose_shape_support_transfer.py for original-support assessment. Export origin must be applied when decoding saved NPZ.',
        'ground_policy': a.ground_policy,
        'contact_review_sha256': sha256_file(a.contact_review) if a.contact_review else None,
        'reuse_body_models': a.reuse_body_models,
        'ground_weight': a.ground_weight,
        'attraction_weight': a.attraction_weight,
        'ground_temporal_weight': a.ground_temporal_weight,
        'ground_horizontal_weight': a.ground_horizontal_weight,
        'relation_timing_scale_m': a.relation_timing_scale,
        'relation_timing_policy': 'Optional frozen original-geometry relative distance weights inside manual timing; baseline role selection and per-relation normalized loss; not human per-relation annotations.',
        'relation_timing_helper_sha256': sha256_file(Path('tools/diagnose_shape_contact_coordinates.py')),
        'horizontal_support_policy': 'mean relu(current XZ ankle/toe speed - original speed), frozen original mesh foot<=1cm at interval start; multiplied by ground_weight*ground_horizontal_weight',
        'support_weighting': ('exp(-relu(original_vertex_y-0.01)/0.02), per-frame weighted max then all-frame mean'
                              if a.ground_policy == 'mesh_support_original_soft' else 'hard original or initial mask as selected'),
        'selection_offset': a.offset, 'selection_limit': a.limit,
        'optimizer': config, 'timing_scope': 'Optimizer call including model/loss initialization; excludes source/floor preparation and CPU/export assessment.',
        'floor_report_sha256': sha256_file(a.floor_report),
        'implementation_sha256': {str(x): sha256_file(x) for x in (Path(__file__), Path('utils/interx_physical_ground.py'), Path('optimize/optimize_full_body.py'), Path('optimization/losses.py'), Path('optimization/mesh_ground_support.py'))}, 'rows': rows})


if __name__ == '__main__':
    main()
