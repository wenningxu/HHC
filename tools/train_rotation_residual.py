"""Train-split-only temporal rotation residuals; validation FID is never a training loss."""
import argparse
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
import sys

import h5py
import numpy as np
import torch
from torch import nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ablate_intermask_feature_fid import rotation
from tools.fit_interhuman_position_ik import fit, load_model, unit
from models.interhuman_evaluator_wrapper import (
    CHECKPOINT_PATH, CONFIG_PATH, CONFIG_SHA256, EVALUATOR_SOURCE,
    EVALUATOR_SOURCE_SHA256, _require_hash, load_official_metrics_module,
)


def features(x, positions):
    # x = [person_batch,time,262]. Remove each frame's global translation/yaw.
    p = x[..., :66].reshape(*x.shape[:-1], 22, 3)
    right = unit(p[..., 2, :]-p[..., 1, :])
    up = unit(p[..., 3, :]-p[..., 0, :])
    forward = unit(torch.linalg.cross(right, up))
    up = unit(torch.linalg.cross(forward, right))
    basis = torch.stack((right, up, forward), -1)
    canonical = (p-p[..., :1, :])@basis
    canonical = canonical.flatten(-2) if positions else torch.zeros_like(x[..., :66])
    return torch.cat((canonical, x[..., 132:258]), -1)


class Corrector(nn.Module):
    def __init__(self, positions):
        super().__init__()
        self.positions = positions
        self.net = nn.Sequential(nn.Conv1d(192, 192, 3, padding=1, padding_mode='replicate'), nn.GELU(),
                                 nn.Conv1d(192, 192, 3, padding=1, padding_mode='replicate'), nn.GELU(),
                                 nn.Conv1d(192, 126, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        delta = .25*torch.tanh(self.net(features(x, self.positions).transpose(1, 2)).transpose(1, 2))
        return x[..., 132:258]+delta


def angles(a, b):
    # Reuse the tested column-interleaved 6D decoder.
    def matrix(r):
        x = torch.zeros(*r.shape[:-1], 262, device=r.device)
        x[..., 132:258] = r
        return rotation(x)[1]
    aa, bb = matrix(a), matrix(b)
    c = ((aa*bb).sum((-1, -2))-1)/2
    return torch.acos(c.clamp(-1, 1))*180/torch.pi


def matrices(r):
    r = r.reshape(*r.shape[:-1], 21, 3, 2)
    x = torch.nn.functional.normalize(r[..., 0], dim=-1)
    z = torch.nn.functional.normalize(torch.linalg.cross(x, r[..., 1]), dim=-1)
    return torch.stack((x, torch.linalg.cross(z, x), z), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['position_rotation', 'rotation_only'], required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--encoder-device', default='cpu')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--geometric', action='store_true')
    parser.add_argument('--selection', choices=['holdout_angle', 'final'], default='holdout_angle')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261007)
    arrays = np.load(args.pairs)
    gt_np, pred_np = arrays['gt'], arrays['recon']
    ids = arrays['ids'].astype(str)
    assert gt_np.shape == pred_np.shape == (1024, 64, 2, 262)
    assert len(set(ids)) == 1024
    splits = Path('dataset/interhuman_official/split')
    assert set(ids) <= set((splits/'train.txt').read_text().splitlines())
    assert not set(ids) & set((splits/'val.txt').read_text().splitlines())
    assert not set(ids) & set((splits/'test.txt').read_text().splitlines())
    split = 896
    assert int(arrays['split_index']) == split
    def tensor(a):
        return torch.from_numpy(a.copy()).permute(0, 2, 1, 3).reshape(-1, 64, 262).to(args.device)
    gt, pred = tensor(gt_np), tensor(pred_np)
    boundary = split*2
    model = Corrector(args.mode == 'position_rotation').to(args.device)
    assert torch.equal(model(pred[:2]), pred[:2, :, 132:258]), 'Start exactly at input rotation'
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scale = gt[:boundary, :, 132:258].flatten(0, 1).std(0).clamp_min(.05)
    baseline_angle = float(angles(pred[boundary:, :, 132:258], gt[boundary:, :, 132:258]).mean())
    best, best_step, best_state = baseline_angle, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        assert saved['mode'] == args.mode and saved['train_ids'] == ids[:split].tolist()
        best_state, best_step = saved['model'], saved['step']
    started = time.monotonic()
    for step in range(1, (0 if args.checkpoint else args.steps)+1):
        model.train()
        index = torch.randint(boundary, (64,), device=args.device)
        x, target = pred[index].clone(), gt[index, :, 132:258]
        identity = torch.rand(64, device=args.device) < .25
        x[identity] = gt[index[identity]]
        if args.geometric:
            noisy = (~identity) & (torch.rand(64, device=args.device) < 1/3)
            n = int(noisy.sum())
            if n:
                x[noisy] = gt[index[noisy]]
                axis = torch.randn(n, 64, 21, 3, device=args.device)*(.12/3**.5)
                a, b, c = axis.unbind(-1)
                zero = torch.zeros_like(a)
                skew = torch.stack((zero, -c, b, c, zero, -a, -b, a, zero), -1).reshape(n, 64, 21, 3, 3)
                changed = torch.matrix_exp(skew)@matrices(target[noisy])
                x[noisy, :, 132:258] = changed[..., :2].flatten(-2).flatten(-2)
                x[noisy, :, :66] += torch.randn_like(x[noisy, :, :66])*.01
        out = model(x)
        mse = ((out-target)/scale).square().mean()
        velocity = (((out[:, 1:]-out[:, :-1])-(target[:, 1:]-target[:, :-1]))/scale).square().mean()
        loss = mse+.1*velocity
        if args.geometric:
            loss = (matrices(out)-matrices(target)).square().mean()+.02*mse+.01*velocity
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                score = float(angles(model(pred[boundary:]), gt[boundary:, :, 132:258]).mean())
            if score < best:
                best, best_step = score, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = {'step': step, 'loss': float(loss.detach()), 'holdout_angle_deg': score, 'best_angle_deg': best, 'best_step': best_step}
            print(json.dumps(row), flush=True)
            with (args.output/'training.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
    if not args.checkpoint and args.selection == 'final':
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_step = args.steps
        model.eval()
        with torch.no_grad():
            best = float(angles(model(pred[boundary:]), gt[boundary:, :, 132:258]).mean())
    model.load_state_dict(best_state)
    model.eval()
    torch.save({'model': best_state, 'mode': args.mode, 'step': best_step, 'train_ids': ids[:split].tolist(),
                'holdout_ids': ids[split:].tolist(), 'seed': 20261007}, args.output/'best.pth')
    training = {'train_ids': ids[:split].tolist(), 'holdout_ids': ids[split:].tolist(), 'steps': args.steps,
                'selected_step': best_step, 'baseline_holdout_angle_deg': baseline_angle, 'best_holdout_angle_deg': best,
                'seconds': time.monotonic()-started, 'selection': 'minimum mean projected geodesic angle on 128 held-out TRAIN IDs; never validation FID',
                'objective': 'train normalized raw6D MSE + .1 temporal difference MSE; 25% clean identity input, 75% real frozen-VQ reconstruction; head zero init and .25 tanh bound'}
    if args.geometric:
        training['objective'] = 'SO3 matrix chordal MSE + .02 normalized6D MSE + .01 temporal difference MSE; approx50% realVQ/25%clean/25%synthetic legal rotation noise RMS .12rad and position noise .01m'
    if args.selection == 'final':
        training['selection'] = 'Fixed final-step candidate for downstream validation metric comparison; holdout angle is diagnostic, not an acceptance gate'
    if args.checkpoint:
        training = json.loads((args.checkpoint.parent/'summary.json').read_text())['training']
        training['reloaded_checkpoint'] = str(args.checkpoint)
    _require_hash(CONFIG_PATH, CONFIG_SHA256)
    _require_hash(EVALUATOR_SOURCE, EVALUATOR_SOURCE_SHA256)
    spec = importlib.util.spec_from_file_location('official_residual_eval', EVALUATOR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = module.MotionEncoder(SimpleNamespace(**yaml.safe_load(CONFIG_PATH.read_text()))).to(args.encoder_device).eval()
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items()}
    encoder.load_state_dict({k.removeprefix('motion_encoder.'): v for k, v in state.items()
                            if k.startswith('motion_encoder.')}, strict=True)
    latent = float(state['latent_scale'])
    skeleton = load_model()
    rest, parents = skeleton.J_regressor@skeleton.v_template, skeleton.parents[:22].tolist()
    offsets = rest[:22].clone()
    for j in range(1, 22):
        offsets[j] -= rest[parents[j]]
    metrics = load_official_metrics_module()
    caches = {'intermask': Path('output/intermask_generation_rotation_20261006/generated480.h5'),
              'intergen': Path('/dev/shm/hhc_intergen_20261006.CtYnWA/full480.h5')}
    results = {}
    with torch.no_grad():
        for name, path in caches.items():
            embeddings = {k: [] for k in ('gt', 'gt_corrected', 'raw', 'half', 'full')}
            diagnostics = {k: [] for k in ('raw', 'half', 'full')}
            with h5py.File(path) as h:
                assert h.attrs['status'] == 'complete'
                metadata = json.loads(h.attrs['metadata'])
                assert metadata['limit'] == 480
                for i in range(480):
                    g, p = [torch.from_numpy(h[str(i)][s][:]).to(args.device) for s in ('gt', 'gen')]
                    assert json.loads(h[str(i)].attrs['request'])['identifier'] not in set(ids), 'train/val leakage'
                    values = {'gt': g, 'raw': p}
                    for tag, x, alpha in [('gt_corrected', g, 1.), ('half', p, .5), ('full', p, 1.)]:
                        y = x.clone()
                        r = model(x.transpose(0, 1)).transpose(0, 1)
                        y[..., 132:258] += alpha*(r-x[..., 132:258])
                        assert torch.equal(y[..., :132], x[..., :132]) and torch.equal(y[..., 258:], x[..., 258:])
                        assert torch.isfinite(y).all()
                        values[tag] = y
                    for tag, x in values.items():
                        emb = encoder({'motions': x.flatten(1)[None].to(args.encoder_device), 'mask': torch.ones(1, len(x), device=args.encoder_device)})['motion_emb']
                        embeddings[tag].append((torch.nn.functional.normalize(emb, dim=-1)*latent).cpu().numpy()[0])
                        if tag in diagnostics:
                            _, d = fit(x.cpu(), offsets, parents)
                            diagnostics[tag].append({'fk_mm': d['fk_before_mm'], 'step_deg': d['step_before_deg']})
            embeddings = {k: np.stack(v) for k, v in embeddings.items()}
            np.savez(args.output/f'{name}_embeddings.npz', **embeddings)
            stats = {k: metrics.calculate_activation_statistics(v, 6) for k, v in embeddings.items()}
            results[name] = {'fid': {k: float(metrics.calculate_frechet_distance(*stats['gt'], *v)) for k, v in stats.items()},
                             'diagnostics': {k: {m: float(np.mean([r[m] for r in rows])) for m in rows[0]} for k, rows in diagnostics.items()},
                             'source_metadata': metadata}
            print(json.dumps({'model': name, **results[name]}), flush=True)
    (args.output/'summary.json').write_text(json.dumps({'status': 'complete', 'mode': args.mode, 'training': training,
             'results': results, 'limitations': 'Train-only supervised postprocessor; validation480 single-seed exploratory, not test. Position/velocity/contact unchanged. FK uses approximate template directions and own estimated bone lengths.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
