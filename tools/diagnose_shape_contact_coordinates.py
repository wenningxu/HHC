"""Small paired semantic-loss diagnostic; not a physical acceptance gate."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from pytorch3d.ops import knn_points

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.calc_pene_from_npz import npz_to_amass_dict, to_torch_params
from optimization.losses import DyadicInteractionLoss
from tools.run_interx_shape_augmentation import (
    atomic_json, canonical_sha256, sha256_file, source_person_payload,
)


def shifted(geometry, floor):
    v, j, n = geometry
    delta = v.new_tensor([0, float(floor), 0])
    return v - delta, j - delta, n


@torch.no_grad()
def relative_relation_timings(distances, manual_timing, scale_m):
    """Frozen geometric pseudo-times; equal/sole relations retain manual timing.

    Relative distance cannot reject a frame where all annotations are wrong.
    These weights are not new human contact labels or acceptance evidence.
    """
    assert distances.ndim == 2 and manual_timing.shape == distances.shape[:1]
    assert np.isfinite(scale_m) and scale_m > 0
    assert torch.isfinite(distances).all() and bool((distances >= 0).all())
    assert torch.isfinite(manual_timing).all() and bool(((manual_timing >= 0) & (manual_timing <= 1)).all())
    if distances.shape[1] == 0:
        return distances.clone()
    excess = distances - distances.amin(dim=1, keepdim=True)
    return manual_timing[:, None] * torch.exp(-excess / scale_m)


def fixed_region_contact(vertices_a, vertices_b, contacts, regions, timing):
    """Unsigned vertex distance on fixed annotated region unions, active frames only.

    This measures proximity, not mesh nonpenetration or correctness of annotation.
    """
    assert torch.isfinite(timing).all() and bool((timing >= 0).all())
    active = timing > 0
    if not contacts or not bool(active.any()):
        return []
    weights = timing[active] / timing[active].sum()
    rows = []
    for contact in contacts:
        ids = [contact[key] for key in ('p1_ids', 'p2_ids')]
        assert all(ids) and all(r in regions for group in ids for r in group)
        indices = [torch.unique(torch.cat([regions[r] for r in group])) for group in ids]
        a, b = [v[active][:, idx].contiguous() for v, idx in zip((vertices_a, vertices_b), indices)]
        assert a.shape[1] and b.shape[1] and torch.isfinite(a).all() and torch.isfinite(b).all()
        da = knn_points(a, b, K=1).dists[..., 0].clamp_min(0).sqrt()
        db = knn_points(b, a, K=1).dists[..., 0].clamp_min(0).sqrt()
        minimum = da.min(dim=1).values
        # Fixed 5% support diagnostic, not the optimizer's contact-type top-k.
        support = sum(d.topk(max(1, int(d.shape[1] * .05)), largest=False).values.mean(1)
                      for d in (da, db)) / 2
        rows.append({'p1_region_ids': ids[0], 'p2_region_ids': ids[1],
                     'active_frames': int(active.sum()),
                     'weighted_min_vertex_distance_m': float((minimum * weights).sum()),
                     'weighted_symmetric_nearest_5pct_m': float((support * weights).sum()),
                     'weighted_contact_frame_fraction': {
                         str(t): float(((minimum <= t) * weights).sum()) for t in (.01, .02, .05)}})
    return rows


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-dir', type=Path, required=True)
    p.add_argument('--contact-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=4)
    a = p.parse_args()
    assert a.limit > 0 and not a.output.exists()
    manifest_path = a.candidate_dir / '_completion_manifest.json'
    m = json.loads(manifest_path.read_text())
    assert m['complete'] and sha256_file(m['source_motion']) == m['source_motion_sha256']
    assert sha256_file(m['plan_records']) == m['plan_records_sha256']
    records = {r['augmented_name']: r for r in map(json.loads, Path(m['plan_records']).read_text().splitlines())}
    loss = DyadicInteractionLoss(device='cpu')
    # Runnable geometry check: only translation changes, normals and pair deltas do not.
    g = (torch.randn(2, 3, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3))
    assert torch.allclose(shifted(shifted(g, 1.5), -1.5)[0], g[0], atol=1e-6)
    assert shifted(g, 1.5)[2] is g[2]
    test_a = torch.zeros(2, 1, 3)
    test_b = test_a.clone(); test_b[:, :, 0] = torch.tensor([.015, .9])[:, None]
    test_contacts = [{'p1_ids': [0], 'p2_ids': [0]}]
    test_regions = {0: torch.tensor([0])}
    result = fixed_region_contact(test_a, test_b, test_contacts, test_regions, torch.tensor([1., 0.]))[0]
    assert abs(result['weighted_min_vertex_distance_m'] - .015) < 1e-6
    assert result['weighted_contact_frame_fraction'] == {'0.01': 0., '0.02': 1., '0.05': 1.}
    assert not fixed_region_contact(test_a, test_b, test_contacts, test_regions, torch.zeros(2))
    rows = []
    with h5py.File(m['source_motion'], 'r') as source:
        for name in m['completed_names'][:a.limit]:
            sidecar = a.candidate_dir / f'{name}.json'
            s = json.loads(sidecar.read_text()); record = records[name]
            assert canonical_sha256(record) == s['plan_record_sha256']
            contact_path = a.contact_dir / f'{s["source_name"]}.pkl'
            assert sha256_file(contact_path) == s['contact_pairs_sha256']
            assert sha256_file(record['timing_file']) == record['timing_sha256']
            with contact_path.open('rb') as handle:
                contacts = pickle.load(handle)
            timing = torch.tensor(np.load(record['timing_file'])[s['crop_start']:s['crop_stop']], dtype=torch.float32)
            raw = source[s['source_name']][:]
            geometries, floors = {}, {}
            for label in ('original_gt', 'random_shape', 'optimized'):
                geometries[label], floors[label] = [], []
                for i in range(2):
                    donor = ({'betas': s['source_shapes']['betas'][i], 'gender': s['source_shapes']['genders'][i]}
                             if label == 'original_gt' else s['augmented_shapes'][i])
                    if label == 'optimized':
                        path = a.candidate_dir / f'{name}_p{i+1}.npz'
                        assert sha256_file(path) == s[f'output_p{i+1}_sha256']
                        with np.load(path, allow_pickle=False) as z:
                            payload = npz_to_amass_dict(z)
                        floor = floors['random_shape'][i]
                    else:
                        floor = float('inf')
                        for start in range(0, len(raw), 32):
                            payload = source_person_payload(raw[start:start+32], i, donor)
                            _, joints, _ = loss._compute_geometry(to_torch_params(payload, 'cpu'))
                            floor = min(floor, float(joints[:, :55, 1].min()))
                        payload = source_person_payload(raw[s['crop_start']:s['crop_stop']], i, donor)
                    assert len(payload['trans']) == len(timing)
                    geometries[label].append(loss._compute_geometry(to_torch_params(payload, 'cpu')))
                    floors[label].append(floor)
            def attraction(pair):
                x, y = pair
                return float(loss._calc_semantic_contact_loss(x[0], y[0], x[2], y[2], contacts, timing)[0])
            initial = geometries['random_shape']
            swapped = attraction(initial[::-1]) < attraction(initial)
            values, fixed_contacts = {}, {}
            for coordinate in ('raw', 'per_person_full_sequence_joint_floor_fixed'):
                values[coordinate] = {}
                fixed_contacts[coordinate] = {}
                for label, pair in geometries.items():
                    if coordinate != 'raw':
                        pair = [shifted(g, f) for g, f in zip(pair, floors[label])]
                    ordered = pair[::-1] if swapped else pair
                    values[coordinate][label] = attraction(ordered)
                    fixed_contacts[coordinate][label] = fixed_region_contact(
                        ordered[0][0], ordered[1][0], contacts, loss.rid_to_vid_tensors, timing)
                assert np.isfinite(list(values[coordinate].values())).all()
            row = {'name': name, 'sidecar_sha256': sha256_file(sidecar),
                   'active_contact': bool(contacts) and bool(timing.sum() > 0),
                   'role_swapped_initial_raw': swapped, 'full_sequence_floors_m': floors,
                   'semantic_attraction_dynamic_regions': values,
                   'fixed_annotated_region_unsigned_proximity': fixed_contacts}
            rows.append(row); print(json.dumps(row), flush=True)
    atomic_json(a.output, {'status': 'diagnostic_only_not_acceptance',
        'scope': 'Released dynamic-region ReLU attraction plus unsigned nearest-vertex proximity on fixed annotated region unions. Timing-weighted fractions at diagnostic thresholds only, not annotation correctness or nonpenetration. Fixed raw-initial roles. Full-sequence joint-min alignment is a diagnostic, not proven sole-ground recovery; optimized uses initial new-shape offsets without posthoc regrounding.',
        'completion_sha256': sha256_file(manifest_path),
        'implementation_sha256': {str(x): sha256_file(x) for x in (Path(__file__), Path('optimization/losses.py'))},
        'rows': rows})


if __name__ == '__main__':
    main()
