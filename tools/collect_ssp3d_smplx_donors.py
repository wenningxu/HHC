"""Audit completed SSP-3D transfers and retain only finite 10D fits below 1 cm."""
import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.collect_humanml3d_amass_smplx_donors import (
    artifact, atomic_json, atomic_npz, canonical_shape_id, numpy_value,
    obj_vertices, shape_statistics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--fit-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    diagnostic_path = args.input_dir / 'diagnostic.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    assets = diagnostic['assets']
    for key in ('labels', 'mapping'):
        assert artifact(Path(assets[key]['path'])) == assets[key], key
    for gender in ('male', 'female'):
        for record in assets[gender].values():
            assert artifact(Path(record['path'])) == record
    with open(assets['mapping']['path'], 'rb') as handle:
        matrix = csr_matrix(pickle.load(handle, encoding='latin1')['matrix'])
    source_records = {row['id']: row for row in diagnostic['rows']}
    fit_dir = args.fit_dir or args.input_dir / 'fitted'
    accepted, rejected, betas, seen = [], [], [], set()
    meshes = sorted((args.input_dir / 'meshes').glob('*/*.obj'))
    if not meshes:
        raise ValueError('No prepared source meshes')
    for mesh in meshes:
        gender, identity = mesh.parent.name, mesh.stem
        source = source_records[identity]
        assert source['gender'] == gender
        assert canonical_shape_id(gender, source['source_betas']) == identity
        fit_path = fit_dir / gender / f'{identity}.pkl'
        if not fit_path.is_file():
            raise FileNotFoundError(f'Fit incomplete: {fit_path}')
        with fit_path.open('rb') as handle:
            fit = pickle.load(handle)
        beta = numpy_value(fit['betas'])
        vertices = numpy_value(fit['vertices'])
        assert beta.shape == (1, 10) and vertices.shape == (1, 10475, 3)
        distances = np.linalg.norm(vertices[0] - matrix @ obj_vertices(mesh), axis=1)
        rmse = float(np.sqrt(np.mean(distances ** 2)))
        row = {'source_identity_ids': [identity], 'gender': gender,
               'source_filenames': source['source_filenames'],
               'transfer_vertex_rmse_m': rmse,
               'transfer_vertex_p95_m': float(np.quantile(distances, .95)),
               'fit': artifact(fit_path), 'source_mesh': artifact(mesh)}
        if not np.isfinite(beta).all() or not np.isfinite(distances).all() or rmse > .01:
            rejected.append(row)
            continue
        shape_id = canonical_shape_id(gender, beta[0])
        if shape_id in seen:
            raise ValueError('Duplicate fitted shapes: inspect before pooling')
        seen.add(shape_id)
        accepted.append({'shape_id': shape_id, **row})
        betas.append(beta[0].astype(np.float32))
    counts = Counter(row['gender'] for row in accepted)
    if min(counts.get('female', 0), counts.get('male', 0)) < 2:
        raise ValueError(f'Need >=2 accepted donors per gender: {counts}; rejected={len(rejected)}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / 'ssp3d_smplx_donors.npz'
    atomic_npz(npz_path, shape_id=np.asarray([r['shape_id'] for r in accepted]),
               gender=np.asarray([r['gender'] for r in accepted]), betas=np.stack(betas),
               transfer_vertex_rmse_m=np.asarray([r['transfer_vertex_rmse_m'] for r in accepted]))
    payload = {
        'schema': 'hhc_ssp3d_smplx_shape_donor_pool_v1', 'status': 'complete',
        'source_dataset': 'SSP-3D', 'source_url': diagnostic['source_url'],
        'source_annotations_license': 'MIT; SMPL/SMPL-X models retain their own licenses',
        'diagnostic_input': artifact(diagnostic_path), 'assets': assets,
        'donor_pool': artifact(npz_path), 'donor_count': len(accepted),
        'prepared_source_shapes': len(meshes), 'available_source_shapes': len(source_records),
        'maximum_allowed_transfer_vertex_rmse_m': .01,
        'gender_counts': dict(counts), 'shape_statistics': shape_statistics(np.stack(betas)),
        'records': accepted, 'rejected': rejected,
        'scope': 'Shape transfer only; augmentation foot/penetration gate is separate',
    }
    atomic_json(args.output_dir / 'ssp3d_smplx_donor_manifest.json', payload)
    print({'accepted': len(accepted), 'rejected': len(rejected), 'gender_counts': dict(counts)}, flush=True)


if __name__ == '__main__':
    main()
