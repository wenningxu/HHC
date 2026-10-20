"""Verify initial mesh-ground placement and serialized common-origin roundtrip.

No optimization, contact scoring, or production asset modification.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.calc_pene_from_npz import PeneCalculator, to_torch_params
from tools.run_interx_shape_augmentation import source_person_payload, sha256_file, atomic_json
from utils.interx_physical_ground import place_pair_on_mesh_ground, encode_physical_pair, decode_physical_pair


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=4)
    a = parser.parse_args()
    assert a.limit > 0 and not a.output.exists()
    mp = a.candidate_dir / '_completion_manifest.json'
    m = json.loads(mp.read_text())
    assert m['complete'] and sha256_file(m['source_motion']) == m['source_motion_sha256']
    calc = PeneCalculator('cpu')
    foot_ids = torch.unique(torch.cat([calc.rid_to_vid_tensors[i] for i in (1, 42, 65, 67)]))
    rows = []
    with h5py.File(m['source_motion'], 'r') as source:
        for name in m['completed_names'][:a.limit]:
            sp = a.candidate_dir / f'{name}.json'
            s = json.loads(sp.read_text()); raw = source[s['source_name']][:]
            for condition in ('original_shape', 'new_shape'):
                donors = ([{'betas': s['source_shapes']['betas'][i], 'gender': s['source_shapes']['genders'][i]}
                           for i in range(2)] if condition == 'original_shape' else s['augmented_shapes'])
                people = [source_person_payload(raw, i, donors[i]) for i in range(2)]
                geometries = [calc._compute_geometry(to_torch_params(p, calc.device), compute_normals=False) for p in people]
                floors = [float(g[0][..., 1].min()) for g in geometries]
                physical = place_pair_on_mesh_ground(people, floors)
                encoded, origin = encode_physical_pair(physical)
                reloaded = []
                for person in encoded:
                    buffer = io.BytesIO(); np.savez_compressed(buffer, **person); buffer.seek(0)
                    with np.load(buffer, allow_pickle=False) as payload:
                        reloaded.append(dict(payload))
                decoded = decode_physical_pair(reloaded, origin)
                stats = []
                for i, (raw_person, placed, restored, geometry) in enumerate(zip(people, physical, decoded, geometries)):
                    for key in raw_person:
                        if key != 'trans':
                            assert np.array_equal(np.asarray(raw_person[key]), np.asarray(restored[key])), key
                    np.testing.assert_allclose(np.diff(placed['trans'], axis=0), np.diff(raw_person['trans'], axis=0), atol=5e-7)
                    np.testing.assert_allclose(restored['trans'], placed['trans'], atol=5e-7)
                    vertices, joints, _ = geometry
                    restored_vertices, _, _ = calc._compute_geometry(to_torch_params(restored, calc.device), compute_normals=False)
                    expected = vertices - vertices.new_tensor([0, floors[i], 0])
                    error = float((expected - restored_vertices).abs().max())
                    minimum = float(restored_vertices[..., 1].min())
                    assert error < 2e-6 and abs(minimum) < 2e-6
                    assert not bool((restored_vertices[..., 1] < -2e-6).any())
                    joint_floor = float(joints[:, :55, 1].min())
                    low_vertex = int(vertices[..., 1].argmin()) % vertices.shape[1]
                    stats.append({'person': i, 'mesh_floor_raw_m': floors[i],
                        'lowest_vertex_in_foot_regions': bool((foot_ids == low_vertex).any()),
                        'joint_floor_ground_loss': float(torch.relu(-(vertices[..., 1] - joint_floor)).mean()),
                        'decoded_mesh_min_y_m': minimum,
                        'decoded_ground_loss': float(torch.relu(-restored_vertices[..., 1]).mean()),
                        'roundtrip_mesh_max_error_m': error})
                np.testing.assert_allclose(encoded[1]['trans'] - encoded[0]['trans'],
                                           physical[1]['trans'] - physical[0]['trans'], atol=5e-7)
                row = {'name': name, 'sidecar_sha256': sha256_file(sp), 'condition': condition,
                       'full_sequence_frames': len(raw), 'common_export_origin': origin.tolist(), 'people': stats}
                rows.append(row); print(json.dumps(row), flush=True)
    atomic_json(a.output, {'status': 'initial_ground_roundtrip_verified_only',
        'scope': 'Full-sequence mesh-min placement, actual gender/beta, constant per-person vertical offsets, serialized common-origin roundtrip. Not optimization, stance timing, object-support recovery, or whole-dataset acceptance.',
        'completion_sha256': sha256_file(mp),
        'implementation_sha256': {str(p): sha256_file(p) for p in (Path(__file__), Path('utils/interx_physical_ground.py'))},
        'rows': rows})


if __name__ == '__main__':
    main()
