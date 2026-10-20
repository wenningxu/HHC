"""Position-only swing IK: preserve original twist, no paired-GT fitting."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.reconstruct_interhuman_raw_body import load_model
from tools.ablate_intermask_feature_fid import rotation
from models.interhuman_evaluator_wrapper import (
    CHECKPOINT_PATH, CONFIG_PATH, CONFIG_SHA256, EVALUATOR_SOURCE,
    EVALUATOR_SOURCE_SHA256, _require_hash, load_official_metrics_module,
)


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def proper(matrix):
    u, _, vh = torch.linalg.svd(matrix)
    sign = torch.ones_like(u[..., 0, :])
    sign[..., -1] = torch.linalg.det(u @ vh)
    return (u * sign[..., None, :]) @ vh


def swing(a, b):
    a, b = unit(a), unit(b)
    v = torch.linalg.cross(a, b)
    c = (a*b).sum(-1)
    k = torch.zeros(*v.shape[:-1], 3, 3)
    k[..., 0, 1], k[..., 0, 2] = -v[..., 2], v[..., 1]
    k[..., 1, 0], k[..., 1, 2] = v[..., 2], -v[..., 0]
    k[..., 2, 0], k[..., 2, 1] = -v[..., 1], v[..., 0]
    result = torch.eye(3) + k + (k@k)/(1+c).clamp_min(1e-7)[..., None, None]
    # Antiparallel bones need an explicit, deterministic perpendicular axis.
    axis = torch.nn.functional.one_hot(a.abs().argmin(-1), 3).float()
    axis = unit(torch.linalg.cross(a, axis))
    flip = 2*axis[..., :, None]*axis[..., None, :]-torch.eye(3)
    return torch.where((c < -0.99999)[..., None, None], flip, result)


@torch.enable_grad()
def refine_from_prediction(p, original, root, offsets, parents, steps):
    """Zero residual starts exactly at the generated (valid SO3) rotation."""
    original, root, p = original.detach(), root.detach(), p.detach()
    lengths = torch.stack([(p[..., j, :]-p[..., parents[j], :]).norm(dim=-1).mean(0)
                           for j in range(1, 22)], dim=-1)
    bones = unit(offsets[1:])[None]*lengths[..., None]
    delta = torch.zeros(*original.shape[:-2], 3, requires_grad=True)

    def pose():
        x, y, z = delta.unbind(-1)
        zero = torch.zeros_like(x)
        k = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*delta.shape[:-1], 3, 3)
        theta = delta.norm(dim=-1)
        a = torch.sinc(theta/torch.pi)[..., None, None]
        b = (.5*torch.sinc(theta/(2*torch.pi)).square())[..., None, None]
        return (torch.eye(3)+a*k+b*(k@k))@original

    assert torch.equal(pose().detach(), original), 'Initialization must equal prediction'

    def loss(local):
        gs, xyz = [root], [p[..., 0, :]]
        for j in range(1, 22):
            par = parents[j]
            xyz.append(xyz[par]+(gs[par]@bones[:, j-1, :, None]).squeeze(-1))
            gs.append(gs[par]@local[..., j-1, :, :])
        position = (torch.stack(xyz, -2)-p).square().sum(-1).mean()
        prior = delta.square().sum(-1).mean()
        temporal = (delta[1:]-delta[:-1]).square().sum(-1).mean()
        return position + .02*prior + .01*temporal

    optimizer = torch.optim.Adam([delta], lr=.01)
    best_pose, best_loss = original.clone(), float(loss(pose()).detach())
    initial_loss = best_loss
    for _ in range(steps):
        optimizer.zero_grad()
        objective = loss(pose())
        assert torch.isfinite(objective)
        objective.backward()
        assert torch.isfinite(delta.grad).all()
        optimizer.step()
        with torch.no_grad():
            local = pose()
            score = float(loss(local))
            if score < best_loss:
                best_pose, best_loss = local.clone(), score
    return best_pose, {'objective_before': initial_loss, 'objective_after': best_loss,
                       'optimizer_steps': steps}


def fit(value, offsets, parents, strength=1., temporal=0., iterations=0):
    p = value[..., :66].reshape(len(value), 2, 22, 3)
    original = rotation(value)[1]
    children = [[k for k in range(1, 22) if parents[k] == j] for j in range(22)]
    fitted, globals_ = [], []
    for j in range(22):
        prior = torch.eye(3).expand(len(value), 2, 3, 3) if j == 0 else globals_[parents[j]] @ original[..., j-1, :, :]
        kids = children[j]
        if len(kids) == 1:
            k = kids[0]
            before = (prior @ offsets[k, :, None]).squeeze(-1)
            g = swing(before, p[..., k, :]-p[..., j, :]) @ prior
        elif kids:
            a = unit(offsets[kids])
            b = unit(p[..., kids, :]-p[..., j:j+1, :])
            g = proper(torch.einsum('...ki,kj->...ij', b, a)+1e-4*prior)
        else:
            g = prior
        globals_.append(g)
        if j:
            fitted.append(globals_[parents[j]].transpose(-1, -2) @ g)
    fitted = torch.stack(fitted, dim=-3)
    optimization = {}
    if iterations:
        fitted, optimization = refine_from_prediction(p, original, globals_[0], offsets, parents, iterations)
    fitted = proper((1-strength)*original+strength*fitted)
    if temporal:
        neighbors = .5*(torch.cat((fitted[:1], fitted[:-1]))+torch.cat((fitted[1:], fitted[-1:])))
        fitted = proper((1-temporal)*fitted+temporal*neighbors)
    out = value.clone()
    out[..., 132:258] = fitted[..., :2].flatten(-2).flatten(-2)
    # Leaves are unconstrained by positions: preserve even their original raw 6D.
    for j in range(1, 22):
        if not children[j]:
            lo = 132+(j-1)*6
            out[..., lo:lo+6] = value[..., lo:lo+6]
    assert torch.equal(out[..., :132], value[..., :132])
    assert torch.equal(out[..., 258:], value[..., 258:])
    assert torch.isfinite(out).all()
    root = globals_[0]
    # Shape-free diagnostic: per-sequence mean bone lengths, template directions.
    lengths = torch.stack([(p[..., j, :]-p[..., parents[j], :]).norm(dim=-1).mean(0)
                           for j in range(1, 22)], dim=-1)
    fixed = unit(offsets[1:])[None]*lengths[..., None]

    def fk_error(local):
        gs, xyz = [root], [p[..., 0, :]]
        for j in range(1, 22):
            par = parents[j]
            xyz.append(xyz[par]+(gs[par]@fixed[:, j-1, :, None]).squeeze(-1))
            gs.append(gs[par]@local[..., j-1, :, :])
        return float((torch.stack(xyz, dim=-2)-p).norm(dim=-1).mean()*1000)

    def step_angle(local):
        rel = local[1:].transpose(-1, -2)@local[:-1]
        cos = (rel.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2
        return float(torch.acos(cos.clamp(-1, 1)).mean()*180/torch.pi)

    solved = rotation(out)[1]
    cos = ((original*solved).sum((-1, -2))-1)/2
    return out, {'fk_before_mm': fk_error(original), 'fk_after_mm': fk_error(solved),
                 'rotation_change_deg': float(torch.acos(cos.clamp(-1, 1)).mean()*180/torch.pi),
                 'step_before_deg': step_angle(original), 'step_after_deg': step_angle(solved), **optimization}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--strength', type=float, default=1.)
    parser.add_argument('--temporal', type=float, default=0.)
    parser.add_argument('--iterations', type=int, default=0)
    args = parser.parse_args()
    assert 0 < args.strength <= 1 and 0 <= args.temporal <= 1
    torch.set_num_threads(2)
    torch.manual_seed(20261007)
    model = load_model()
    rest = model.J_regressor@model.v_template
    parents = model.parents[:22].tolist()
    offsets = rest[:22].clone()
    for j in range(1, 22):
        offsets[j] -= rest[parents[j]]
    # Minimal exact-skeleton roundtrip and antiparallel checks.
    a = torch.tensor([[1., 0, 0]])
    assert torch.allclose((swing(a, -a)@a[..., None]).squeeze(-1), -a, atol=1e-6)
    xyz = rest[:22]-rest[0]
    test = torch.zeros(2, 2, 262)
    test[..., :66] = xyz.flatten()
    test[..., 132:258] = torch.eye(3)[:, :2].flatten().repeat(21)
    fitted, check = fit(test, offsets, parents)
    assert check['fk_after_mm'] < .1 and check['rotation_change_deg'] < .1, check
    _require_hash(CONFIG_PATH, CONFIG_SHA256)
    _require_hash(EVALUATOR_SOURCE, EVALUATOR_SOURCE_SHA256)
    spec = importlib.util.spec_from_file_location('official_ik', EVALUATOR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = module.MotionEncoder(SimpleNamespace(**yaml.safe_load(CONFIG_PATH.read_text()))).eval()
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items()}
    encoder.load_state_dict({k.removeprefix('motion_encoder.'): v for k, v in state.items()
                            if k.startswith('motion_encoder.')}, strict=True)
    scale = float(state['latent_scale'])
    embeddings = {k: [] for k in ('gt', 'gt_ik', 'gen', 'gen_ik')}
    diagnostics = {'gt': [], 'gen': []}
    started = time.monotonic()
    with h5py.File(args.cache) as handle, torch.no_grad():
        assert handle.attrs['status'] == 'complete'
        metadata = json.loads(handle.attrs['metadata'])
        assert 2 <= args.limit <= metadata['limit']
        for i in range(args.limit):
            for side in ('gt', 'gen'):
                value = torch.from_numpy(handle[str(i)][side][:])
                solved, d = fit(value, offsets, parents, args.strength, args.temporal, args.iterations)
                diagnostics[side].append(d)
                for key, x in ((side, value), (side+'_ik', solved)):
                    emb = encoder({'motions': x.flatten(1)[None], 'mask': torch.ones(1, len(x))})['motion_emb']
                    embeddings[key].append((torch.nn.functional.normalize(emb, dim=-1)*scale).numpy()[0])
            if (i+1) % 16 == 0:
                print(json.dumps({'samples': i+1, 'seconds': time.monotonic()-started}), flush=True)
    embeddings = {k: np.stack(v) for k, v in embeddings.items()}
    metrics = load_official_metrics_module()
    stats = {k: metrics.calculate_activation_statistics(v, 6) for k, v in embeddings.items()}
    fid = {k: float(metrics.calculate_frechet_distance(*stats['gt'], *v)) for k, v in stats.items()}
    result = {'status': 'complete', 'samples': args.limit, 'source': str(args.cache), 'metadata': metadata,
              'fid_vs_original_gt': fid, 'diagnostics': diagnostics,
              'mean_diagnostics': {s: {k: float(np.mean([d[k] for d in rows])) for k in rows[0]}
                                   for s, rows in diagnostics.items()},
              'seconds': time.monotonic()-started,
              'strength': args.strength, 'temporal': args.temporal, 'iterations': args.iterations,
              'optimizer': 'When iterations>0: SO3 residual initialized to zero about predicted rotations; Adam .01; position MSE + .02 residual-angle-square + .01 residual temporal-difference-square; retain best objective. Root inferred from pelvis directions and fixed.',
              'method': ('Iterative SO3 residual optimization initialized exactly at predicted rotations.' if args.iterations else 'Closed-form swing IK with predicted twist prior.')+' Zero-shape female rest directions; positions/velocities/contact unchanged; leaf rotations preserved. Optional polar blend and temporal neighbors.',
              'limitations': 'Bone lengths fit from each own position sequence, not known SMPL shape. No GT supplied to generation IK. Temporal smoothing is a soft prior, not a physical constraint. Small-sample FID is diagnostic.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix('.embeddings.npz'), **embeddings)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('diagnostics', 'metadata')}), flush=True)


if __name__ == '__main__':
    main()
