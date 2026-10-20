"""CPU paired-reconstruction interventions; hybrid-feature FID is diagnostic only."""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from dataset.interhuman_generated_h5 import InterHumanGeneratedH5Dataset
from models.interhuman_evaluator_wrapper import (
    CHECKPOINT_PATH, CHECKPOINT_BYTES, CONFIG_PATH, CONFIG_SHA256,
    EVALUATOR_SOURCE, EVALUATOR_SOURCE_SHA256, _require_hash,
    load_official_metrics_module,
)

SOURCE = '/home/wenning/HHC_interhuman_runs/interhuman_vq_validation/official_intermask_reconstruction.h5'
GROUPS = {'P': (0, 66), 'V': (66, 132), 'R': (132, 258)}


def rotation(value):
    r = value[..., 132:258].reshape(*value.shape[:-1], 21, 6)
    # Stored InterHuman uses flattened first two columns (interleaved), not 3+3.
    first, second = r[..., ::2], r[..., 1::2]
    n1 = first.norm(dim=-1, keepdim=True)
    x = first / n1.clamp_min(1e-12)
    cross = torch.linalg.cross(x, second)
    nc = cross.norm(dim=-1, keepdim=True)
    if (n1 < 1e-6).any() or (nc < 1e-6).any():
        raise ValueError('Degenerate 6D rotation: cannot define projection/angle unambiguously')
    z = cross / nc
    y = torch.linalg.cross(z, x)
    return r, torch.stack((x, y, z), dim=-1)


def project(value):
    result = value.clone()
    _, matrix = rotation(value)
    result[..., 132:258] = matrix[..., :2].flatten(-2).flatten(-2)
    return result


def write_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=16)
    parser.add_argument('--coalitions', action='store_true')
    parser.add_argument('--stats-only', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261006)
    start = time.monotonic()
    print(json.dumps({'pid': os.getpid(), 'device': 'cpu', 'threads': 2, 'limit': args.limit}), flush=True)
    dataset = InterHumanValidationRequestDataset()
    entry = dataset.manifest['request']
    generated = InterHumanGeneratedH5Dataset(SOURCE, dataset.request_path, entry, formal=False)
    assert generated.metadata['method'] == 'intermask_official_vq_reconstruction'
    assert generated.metadata['status'] == 'complete' and generated.metadata['split'] == 'val'
    assert 2 <= args.limit <= len(dataset)
    mean, std = map(torch.from_numpy, (dataset.adapter.mean, dataset.adapter.std))
    gt, pred = [], []
    sums, counts, maxima = {}, {}, {}

    def record(name, value):
        value = value.double()
        sums[name] = sums.get(name, 0.) + float(value.sum())
        counts[name] = counts.get(name, 0) + value.numel()
        maxima[name] = max(maxima.get(name, 0.), float(value.max()))

    for index in range(len(dataset)):
        a, b = dataset[index], generated[index]
        assert a[0:2] == b[0:2] and a[4] == b[4]
        length = a[4]
        g = torch.stack(a[2:4], dim=1)[:length] * std + mean
        p = torch.stack(b[2:4], dim=1)[:length] * std + mean
        record('mpjpe_m', (p[..., :66] - g[..., :66]).reshape(length, 2, 22, 3).norm(dim=-1))
        record('velocity_l2_m_per_frame', (p[..., 66:132] - g[..., 66:132]).reshape(length, 2, 22, 3).norm(dim=-1))
        for key, (lo, hi) in GROUPS.items():
            record(key + '_normalized_squared_error', ((p[..., lo:hi] - g[..., lo:hi]) / std[lo:hi]).square())
        rg, mg = rotation(g)
        rp, mp = rotation(p)
        cos = ((mg * mp).sum(dim=(-1, -2)) - 1) / 2
        record('rotation_angle_deg', torch.rad2deg(torch.acos(cos.clamp(-1, 1))))
        for name, value, r in [('gt', g, rg), ('prediction', p, rp)]:
            record(name + '_rotation_abs_norm_error', (r.reshape(*r.shape[:-1], 3, 2).norm(dim=-2) - 1).abs())
            record(name + '_rotation_abs_dot', (r[..., ::2] * r[..., 1::2]).sum(-1).abs())
            record(name + '_rotation_projection_squared_error', (project(value)[..., 132:258] - value[..., 132:258]).square())
            record(name + '_velocity_position_residual_m_per_frame',
                   (value[:-1, :, 66:132] - value[1:, :, :66] + value[:-1, :, :66]).reshape(-1, 22, 3).norm(dim=-1))
        if index < args.limit:
            gt.append(g)
            pred.append(p)
    generated.close()
    result = {'status': 'stats_complete', 'source': SOURCE, 'generation_metadata': generated.metadata,
              'stats_samples': len(dataset), 'embedding_samples': args.limit,
              'stats': {k: {'mean': sums[k] / counts[k], 'max': maxima[k], 'count': counts[k]} for k in sums},
              'protocol': 'frozen 480 validation requests, raw 262D, CPU2, seed20261006',
              'request_sha256': entry['request_sha256']}
    write_json(args.output, result)
    print(json.dumps({'stage': 'stats_complete', 'seconds': time.monotonic()-start,
                      'mpjpe_m': result['stats']['mpjpe_m']['mean'],
                      'angle_deg': result['stats']['rotation_angle_deg']['mean']}), flush=True)
    if args.stats_only:
        return
    _require_hash(CONFIG_PATH, CONFIG_SHA256)
    _require_hash(EVALUATOR_SOURCE, EVALUATOR_SOURCE_SHA256)
    assert CHECKPOINT_PATH.stat().st_size == CHECKPOINT_BYTES
    spec = importlib.util.spec_from_file_location('official_ablation', EVALUATOR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MotionEncoder(SimpleNamespace(**yaml.safe_load(CONFIG_PATH.read_text()))).eval()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('model.'): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict({k.removeprefix('motion_encoder.'): v for k, v in state.items()
                          if k.startswith('motion_encoder.')}, strict=True)
    scale = float(state['latent_scale'].item())
    projected = project(gt[0])
    assert torch.allclose(project(projected), projected, atol=2e-6)
    assert torch.allclose(rotation(projected)[1], rotation(gt[0])[1], atol=2e-6)
    assert torch.equal(projected[..., :132], gt[0][..., :132])
    labels = ['GT', 'PVR', 'project_GT', 'project_PVR']
    if args.coalitions:
        labels = ['GT', 'P', 'V', 'R', 'PV', 'PR', 'VR', 'PVR', 'project_GT', 'project_PVR']
    embeddings = {}
    with torch.inference_mode():
        for label in labels:
            values = []
            for index, (g, p) in enumerate(zip(gt, pred)):
                x = g.clone()
                for key, (lo, hi) in GROUPS.items():
                    if key in label.removeprefix('project_') and label != 'GT':
                        x[..., lo:hi] = p[..., lo:hi]
                if label.startswith('project_'):
                    x = project(x)
                embedding = model({'motions': x.flatten(1)[None],
                                   'mask': torch.ones(1, len(x))})['motion_emb']
                values.append((torch.nn.functional.normalize(embedding, dim=-1) * scale).numpy()[0])
            embeddings[label] = np.stack(values)
            np.savez(Path(args.output).with_suffix('.embeddings.npz'), **embeddings)
            print(json.dumps({'stage': label, 'samples': len(values), 'seconds': time.monotonic()-start}), flush=True)
    metrics = load_official_metrics_module()
    stats = {k: metrics.calculate_activation_statistics(v, 6) for k, v in embeddings.items()}
    fids = {k: float(metrics.calculate_frechet_distance(*stats['GT'], *v)) for k, v in stats.items()}
    projection_fid = float(metrics.calculate_frechet_distance(*stats['project_GT'], *stats['project_PVR']))
    result.update(status='complete', fid_vs_raw_gt=fids, fid_both_projected=projection_fid,
                  latent_scale=scale, metric_activation_scale=6,
                  seconds=time.monotonic()-start,
                  embedding_l2_vs_gt={k: float(np.linalg.norm(v-embeddings['GT'], axis=-1).mean()) for k,v in embeddings.items()},
                  limitations='Paired VQ reconstruction, not text generation. Feature hybrids may be inconsistent; FID effects are nonadditive. Projection is a representation diagnostic, not noise removal.')
    if args.coalitions:
        shapley = {}
        for key in GROUPS:
            other = [k for k in GROUPS if k != key]
            value = 0.
            for bits in range(4):
                subset = {other[j] for j in range(2) if bits & (1 << j)}
                a = ''.join(k for k in GROUPS if k in subset) or 'GT'
                b = ''.join(k for k in GROUPS if k in subset | {key})
                weight = math.factorial(len(subset))*math.factorial(2-len(subset))/6
                value += weight * (fids[b]-fids[a])
            shapley[key] = value
        assert abs(sum(shapley.values())-(fids['PVR']-fids['GT'])) < 1e-8
        result['diagnostic_shapley_fid'] = shapley
    write_json(args.output, result)
    print(json.dumps({'status': 'complete', 'fids': fids, 'fid_both_projected': projection_fid,
                      'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
