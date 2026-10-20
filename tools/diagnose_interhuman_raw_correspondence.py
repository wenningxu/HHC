"""Compare official raw parameters against released 492D arrays; never modify data."""
import argparse
import builtins
import hashlib
import inspect
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation
import smplx
import torch


def frame_indices(length, rate):
    return np.floor(np.arange(length) * rate / 30.0).astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('dataset/interhuman_official'))
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--geometry-frames', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert frame_indices(5, 59.94).tolist() == [0, 1, 3, 5, 7]
    # Same trusted legacy model compatibility used by diagnose_ssp3d_shape_transfer.
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    for name in ('bool', 'int', 'float', 'complex', 'object', 'str'):
        if name not in np.__dict__:
            setattr(np, name, getattr(builtins, name))
    if 'unicode' not in np.__dict__:
        np.unicode = str
    models = {gender: smplx.create('body_models', model_type='smpl', gender=gender).eval()
              for gender in ('female', 'male', 'neutral')}
    candidates = sorted((args.root / 'motions').glob('*.pkl'), key=lambda p: int(p.stem))
    eligible = [p for p in candidates if all(
        (args.root / 'motions_processed' / who / (p.stem + '.npy')).is_file()
        for who in ('person1', 'person2'))]
    selected = [eligible[i] for i in np.unique(np.linspace(0, len(eligible)-1, min(args.limit, len(eligible))).astype(int))]
    results = []
    for path in selected:
        raw = pickle.loads(path.read_bytes())
        for who in ('person1', 'person2'):
            p = raw[who]
            target_path = args.root / 'motions_processed' / who / (path.stem + '.npy')
            y = np.load(target_path)
            n = len(y)
            idx = frame_indices(n, raw['mocap_framerate'])
            row = {'id': path.stem, 'person': who, 'raw_gender': str(p['gender']),
                   'raw_frames': len(p['trans']), 'processed_shape': list(y.shape),
                   'fps': float(raw['mocap_framerate']),
                   'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'processed_sha256': hashlib.sha256(target_path.read_bytes()).hexdigest()}
            if idx[-1] >= len(p['trans']):
                row['error'] = 'floor_time_mapping_out_of_bounds'
                results.append(row)
                continue
            rot = Rotation.from_rotvec(p['pose_body'][idx].reshape(-1, 3)).as_matrix()
            cols = rot[:, :, :2].reshape(n, -1)
            rows = rot[:, :2, :].reshape(n, -1)
            row.update(rotation_columns_mae=float(np.abs(cols-y[:, 186:312]).mean()),
                       rotation_columns_max=float(np.abs(cols-y[:, 186:312]).max()),
                       rotation_rows_mae=float(np.abs(rows-y[:, 186:312]).mean()))
            frames = np.unique(np.linspace(0, n-1, min(n, args.geometry_frames)).astype(int))
            source = idx[frames]
            b = y[frames, :66].reshape(-1, 22, 3)
            pose = np.pad(p['pose_body'][source], ((0, 0), (0, 6)))
            metrics = {}
            for gender, model in models.items():
                for shape in ('original', 'zero'):
                    beta = p['betas'][None] if shape == 'original' else np.zeros((1, 10), np.float32)
                    with torch.no_grad():
                        out = model(global_orient=torch.from_numpy(p['root_orient'][source]),
                                    body_pose=torch.from_numpy(pose), transl=torch.from_numpy(p['trans'][source]),
                                    betas=torch.from_numpy(beta).expand(len(source), -1))
                    a = out.joints[:, :22].numpy()
                    metrics[f'{gender}_{shape}'] = {
                        'mpjpe_mm': float(np.linalg.norm(a-b, axis=-1).mean()*1000),
                        'max_joint_error_mm': float(np.linalg.norm(a-b, axis=-1).max()*1000)}
            row['geometry'] = metrics
            row['best_geometry'] = min(metrics, key=lambda k: metrics[k]['mpjpe_mm'])
            results.append(row)
        if len(results) % 32 == 0:
            print(f'checked {len(results)//2}/{len(selected)} pairs', flush=True)
    valid = [r for r in results if 'geometry' in r]
    summary = {'sample_pairs': len(selected), 'available_raw_files': len(candidates),
               'eligible_available_pairs': len(eligible), 'person_records': len(results),
               'errors': [r for r in results if 'error' in r],
               'rotation_columns_mae_mean': float(np.mean([r['rotation_columns_mae'] for r in valid])),
               'rotation_columns_max': max(r['rotation_columns_max'] for r in valid),
               'best_geometry_counts': {k: sum(r['best_geometry'] == k for r in valid) for k in valid[0]['geometry']},
               'geometry_mean_mpjpe_mm': {k: float(np.mean([r['geometry'][k]['mpjpe_mm'] for r in valid]))
                                         for k in valid[0]['geometry']}}
    payload = {'protocol': 'same ID/person; floor(k*fps/30); no alignment; SMPL first22; deterministic geometry frames',
               'summary': summary, 'records': results,
               'model_sha256': {g: hashlib.sha256(Path(f'body_models/smpl/SMPL_{g.upper()}.pkl').read_bytes()).hexdigest()
                                for g in models}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
