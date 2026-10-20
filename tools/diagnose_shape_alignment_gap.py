#!/usr/bin/env python
"""Compare penetration before/after restoring each person's floor offset."""

import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval.calc_pene_from_npz import PeneCalculator, npz_to_amass_dict, to_torch_params


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def score(calculator, paths, floors):
    params = []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            params.append(to_torch_params(npz_to_amass_dict(saved), calculator.device))
    frames = params[0]['trans'].shape[0]
    assert all(p[k].shape[0] == frames for p in params for k in
               ('trans', 'root_orient', 'pose_body', 'pose_lhand', 'pose_rhand'))
    raw = [calculator._compute_geometry(p) for p in params]
    aligned = [(v.clone(), j, n) for (v, j, n) in raw]
    for (verts, _, _), floor in zip(aligned, floors):
        verts[:, :, 1] -= float(floor)
    def metric(g):
        return float(calculator._calc_penetration_metric_per_frame(g[0][0], g[1][0], g[0][2], g[1][2]).mean())
    return metric(raw), metric(aligned), frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--optimized-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    a = p.parse_args()
    calc = PeneCalculator(a.device)
    rows = []
    for sidecar in sorted(x for x in a.optimized_dir.glob('*.json') if not x.name.startswith('_')):
        meta = json.loads(sidecar.read_text())
        name = meta['augmented_name']
        paths = [a.optimized_dir / f'{name}_p{i}.npz' for i in (1, 2)]
        hashes = [sha256(x) for x in paths]
        expected = [meta[f'output_p{i}_sha256'] for i in (1, 2)]
        if hashes != expected:
            raise RuntimeError(f'hash mismatch: {name}: {hashes} != {expected}')
        floors = [meta['diagnostics']['ground_alignment'][f'p{i}_floor_y_m'] for i in (1, 2)]
        raw, aligned, frames = score(calc, paths, floors)
        assert frames == meta['crop_stop'] - meta['crop_start']
        assert np.isfinite([raw, aligned, *floors]).all()
        rows.append({'name': name, 'raw': raw, 'aligned': aligned, 'offsetgap': aligned - raw,
                     'floors_y_m': floors, 'frames': frames, 'output_hashes': hashes})
    result = {'schema': 'hhc_shape_alignment_gap_v1', 'samples': len(rows), 'rows': rows,
              'aggregate': {'raw_mean': float(np.mean([r['raw'] for r in rows])),
                            'aligned_mean': float(np.mean([r['aligned'] for r in rows])),
                            'offsetgap_mean': float(np.mean([r['offsetgap'] for r in rows])),
                            'offsetgap_max_abs': float(np.max(np.abs([r['offsetgap'] for r in rows])))}}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['aggregate'], indent=2))


if __name__ == '__main__':
    main()
