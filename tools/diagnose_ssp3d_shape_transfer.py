"""Measure SSP-3D SMPL->SMPL-X linear transfer; diagnostic, not donor approval."""
import argparse
import builtins
import inspect
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.collect_humanml3d_amass_smplx_donors import (
    artifact, atomic_json, atomic_npz, canonical_shape_id, shape_statistics,
)
from tools.prepare_humanml3d_amass_shape_transfer import transfer_config, write_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path,
                        default=ROOT / 'dataset/ssp3d/ssp_3d/labels.npz')
    parser.add_argument('--mapping', type=Path, default=Path(
        '/home/wenning/projects/prosepose/essentials/body_model_utils/smpl_to_smplx.pkl'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-transfer-probes', action='store_true')
    parser.add_argument('--prepare-all-transfers', action='store_true')
    args = parser.parse_args()
    # Compatibility only for loading licensed, trusted local legacy SMPL pickles.
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    for name in ('bool', 'int', 'float', 'complex', 'object', 'str'):
        if name not in np.__dict__:
            setattr(np, name, getattr(builtins, name))
    if 'unicode' not in np.__dict__:
        np.unicode = str
    with args.mapping.open('rb') as handle:
        mapping = pickle.load(handle, encoding='latin1')
    matrix = csr_matrix(mapping['matrix'])
    assert matrix.shape == (10475, 6890)
    assert np.isfinite(matrix.data).all()
    assert np.allclose(np.asarray(matrix.sum(axis=1)), 1, atol=1e-6)
    del mapping
    with np.load(args.labels, allow_pickle=False) as labels:
        source_betas = labels['shapes'].astype(np.float64)
        genders = labels['genders'].astype(str)
        filenames = labels['fnames'].astype(str)
    assert source_betas.shape == (len(genders), 10)
    assert np.isfinite(source_betas).all() and set(genders) <= {'m', 'f'}
    rows, targets, sources, ids, target_genders = [], [], [], [], []
    assets = {'labels': artifact(args.labels), 'mapping': artifact(args.mapping)}
    for gender, code in (('female', 'f'), ('male', 'm')):
        smpl_path = ROOT / f'body_models/smpl/SMPL_{gender.upper()}.pkl'
        smplx_path = ROOT / f'body_models/smplx/SMPLX_{gender.upper()}.npz'
        assets[gender] = {'smpl': artifact(smpl_path), 'smplx': artifact(smplx_path)}
        with smpl_path.open('rb') as handle:
            smpl = pickle.load(handle, encoding='latin1')
        source_dirs = np.asarray(smpl['shapedirs'])[:, :, :10]
        source_template = np.asarray(smpl['v_template'])
        with np.load(smplx_path, allow_pickle=False) as smplx:
            target_dirs = smplx['shapedirs'][:, :, :10].astype(np.float64)
            target_template = smplx['v_template'].astype(np.float64)
        mapped_dirs = (matrix @ source_dirs.reshape(6890, 30)).reshape(10475, 3, 10)
        template_delta = matrix @ source_template - target_template
        design = target_dirs.reshape(-1, 10)
        transfer = np.linalg.lstsq(design, mapped_dirs.reshape(-1, 10), rcond=None)[0]
        # Translation is a nuisance coordinate, not a shape parameter.
        centered_design = (target_dirs - target_dirs.mean(axis=0)).reshape(-1, 10)
        unique = np.unique(source_betas[genders == code], axis=0)
        for beta in unique:
            displacement = np.einsum('vci,i->vc', mapped_dirs, beta)
            target_beta = transfer @ beta
            predicted = np.einsum('vci,i->vc', target_dirs, target_beta)
            absolute_delta = template_delta + displacement
            centered_delta = absolute_delta - absolute_delta.mean(axis=0)
            absolute_beta = np.linalg.lstsq(centered_design, centered_delta.ravel(), rcond=None)[0]
            absolute_residual = centered_delta - (
                centered_design @ absolute_beta).reshape(-1, 3)
            distances = np.linalg.norm(predicted - displacement, axis=1)
            selected = (genders == code) & np.all(source_betas == beta, axis=1)
            identity = canonical_shape_id(gender, beta)
            if args.prepare_all_transfers or (args.prepare_transfer_probes and (
                    np.array_equal(beta, unique[0]) or np.array_equal(beta, unique[-1]))):
                mesh_dir = args.output_dir / 'meshes' / gender
                write_obj(mesh_dir / f'{identity}.obj', source_template +
                          np.einsum('vci,i->vc', source_dirs, beta), np.asarray(smpl['f']))
            rows.append({
                'id': identity, 'gender': gender,
                'source_filenames': filenames[selected].tolist(),
                'source_betas': beta.tolist(), 'target_betas': target_beta.tolist(),
                'displacement_vertex_rmse_m': float(np.sqrt(np.mean(distances ** 2))),
                'displacement_vertex_p95_m': float(np.quantile(distances, .95)),
                'displacement_relative_l2_error': float(np.linalg.norm(predicted - displacement)
                    / max(np.linalg.norm(displacement), 1e-12)),
                'absolute_fit_vertex_rmse_m': float(np.sqrt(np.mean(np.sum(absolute_residual ** 2, axis=1)))),
                'absolute_fit_betas': absolute_beta.tolist(),
            })
            targets.append(target_beta); sources.append(beta)
            ids.append(identity); target_genders.append(gender)
        if args.prepare_transfer_probes or args.prepare_all_transfers:
            config = transfer_config(args.output_dir / 'meshes' / gender,
                                     args.output_dir / 'fitted' / gender,
                                     args.mapping, ROOT / 'body_models', gender, 1)
            config['logger_level'] = 'WARNING'
            atomic_json(args.output_dir / 'configs' / f'{gender}.yaml', config)
    assert len(ids) == len(set(ids)) and np.isfinite(targets).all()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / 'candidate_shapes.npz'
    atomic_npz(npz_path, identity_id=np.asarray(ids), gender=np.asarray(target_genders),
               source_betas=np.asarray(sources), betas=np.asarray(targets, dtype=np.float32))
    summary = {}
    for metric in ('displacement_vertex_rmse_m', 'absolute_fit_vertex_rmse_m'):
        values = np.asarray([row[metric] for row in rows])
        summary[metric] = {'mean': float(values.mean()), 'max': float(values.max()),
                           'below_1cm': int((values <= .01).sum())}
    atomic_json(args.output_dir / 'diagnostic.json', {
        'status': 'diagnostic_only_not_approved_donors', 'source': 'SSP-3D',
        'source_url': 'https://github.com/akashsengupta1997/SSP-3D',
        'method': 'ProsePose full-mesh shape-displacement least squares, gender matched',
        'assets': assets, 'candidates': artifact(npz_path), 'summary': summary,
        'shape_statistics': shape_statistics(np.asarray(targets)), 'rows': rows,
        'acceptance': 'Not approved: inspect absolute and displacement errors before augmentation.',
    })
    print(summary, flush=True)


if __name__ == '__main__':
    main()
