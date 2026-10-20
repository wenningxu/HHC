"""Validation-only P/V/SO(3) sensitivity with equal standardized energy.

No FID, no training, and no test-set reads. Legal rotation perturbations still
need not agree with positions or mesh contacts; this is a mechanism probe.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from dataset.interhuman_fair_requests import canonical_sha256
from models.interhuman_evaluator_wrapper import CONFIG_SHA256, EVALUATOR_SOURCE_SHA256, _require_hash


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def rotation_matrix(six):
    columns = six.reshape(*six.shape[:-1], 21, 3, 2)
    a = torch.nn.functional.normalize(columns[..., 0], dim=-1)
    c = torch.nn.functional.normalize(torch.linalg.cross(a, columns[..., 1]), dim=-1)
    return torch.stack((a, torch.linalg.cross(c, a), c), dim=-1)


def perturb(x, std, group, epsilon, seed):
    generator = torch.Generator().manual_seed(seed)
    length = len(x)
    budget = epsilon * (length * 2 * 66)**.5
    changed = x.clone()
    angle = 0.
    if group == 'rotation_so3':
        base = rotation_matrix(x[..., 132:258])
        axis = torch.nn.functional.normalize(torch.randn(length, 2, 21, 3, generator=generator), dim=-1)
        zeros = torch.zeros_like(axis[..., 0])
        a, b, c = axis.unbind(-1)
        skew = torch.stack((zeros, -c, b, c, zeros, -a, -b, a, zeros), -1).reshape(*axis.shape[:-1], 3, 3)
        identity = torch.eye(3)
        def candidate(theta):
            delta = identity + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)
            return ((delta @ base)[..., :2]).flatten(-3)
        low, high = 0., .5
        def energy(theta):
            return ((candidate(theta) - x[..., 132:258]) / std[132:258]).norm()
        assert float(energy(high)) > budget
        for _ in range(24):
            mid = (low + high) / 2
            if float(energy(mid)) < budget: low = mid
            else: high = mid
        angle = (low + high) / 2
        changed[..., 132:258] = candidate(angle)
        actual = rotation_matrix(changed[..., 132:258])
        assert torch.max(torch.abs(actual.transpose(-1, -2) @ actual - identity)) < 2e-5
        assert torch.min(torch.linalg.det(actual)) > .9999
    elif group == 'rotation_6d_gaussian':
        direction = torch.randn(length, 2, 126, generator=generator) * std[132:258]
        direction *= budget / (direction / std[132:258]).norm()
        changed[..., 132:258] += direction
    else:
        # A three-frequency smooth direction; matched for P-only and P+deltaP.
        t = torch.linspace(0., 1., length + 1)
        coefficients = torch.randn(3, 2, 66, generator=generator)
        phase = torch.rand(3, 2, 66, generator=generator) * (2 * torch.pi)
        trajectory = sum(torch.sin((k + 1) * 2 * torch.pi * t[:, None, None] + phase[k]) * coefficients[k]
                         for k in range(3))
        direction = torch.zeros_like(x)
        if group == 'velocity':
            direction[..., 66:132] = trajectory[:-1] * std[66:132]
        else:
            direction[..., :66] = trajectory[:-1] * std[:66]
            if group == 'position_consistent_velocity':
                direction[..., 66:132] = trajectory.diff(dim=0) * std[:66]
        direction *= budget / (direction[..., :258] / std[:258]).norm()
        changed += direction
        if group == 'position_consistent_velocity':
            assert (changed[1:, :, :66] - changed[:-1, :, :66] - changed[:-1, :, 66:132]).abs().max() < 1e-5
    achieved = float(((changed[..., :258] - x[..., :258]) / std[:258]).norm())
    assert abs(achieved / budget - 1.) < 2e-4, (group, achieved, budget)
    return changed, {'energy_target': budget, 'energy_achieved': achieved,
                     'rotation_angle_deg': angle * 180 / np.pi,
                     'position_rms_m': float((changed[..., :66] - x[..., :66]).square().mean().sqrt()),
                     'velocity_rms_m_per_frame': float((changed[..., 66:132] - x[..., 66:132]).square().mean().sqrt())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--directions', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    assert 1 <= args.samples <= 480 and args.directions >= 1
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261016)
    external = ROOT / 'third_party/InterMask'
    source, config, checkpoint = external / 'models/evaluator/evaluator_models.py', external / 'checkpoints/eval_model/eval_model.yaml', external / 'checkpoints/eval_model/interclip.ckpt'
    _require_hash(source, EVALUATOR_SOURCE_SHA256)
    _require_hash(config, CONFIG_SHA256)
    _require_hash(checkpoint, '4e14b8ee5c2d965778fe003fb37d98ed8e5c7e0940668515fc7dcfa2831c17f8')
    spec = importlib.util.spec_from_file_location('official_legal_probe', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.MotionEncoder(SimpleNamespace(**yaml.safe_load(config.read_text()))).eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)['state_dict']
    state = {k.removeprefix('model.'): v for k, v in state.items()}
    model.load_state_dict({k.removeprefix('motion_encoder.'): v for k, v in state.items() if k.startswith('motion_encoder.')}, strict=True)
    adapter = OfficialInterHumanAdapter(data_root=ROOT/'dataset/interhuman_official', intermask_root=external,
            mean_path=external/'data/stats/global_mean.npy', std_path=external/'data/stats/global_std.npy')
    request_path = ROOT/'output/interhuman_validation_requests/validation.json'
    _require_hash(request_path, '2053a04a5f6169ef9ae52ecb286c1d4dcc0df1de3a21aaa01024a801c0c2308c')
    payload = json.loads(request_path.read_text())
    assert canonical_sha256(payload['requests']) == payload['request_sha256']
    # Evenly spaced positions chosen before scoring, not sorted by sequence or result.
    indices = np.linspace(0, 479, args.samples, dtype=int).tolist()
    requests = [payload['requests'][i] for i in indices]
    pairs = []
    for request in requests:
        replay = adapter.frozen_text_pair_request(request['identifier'], caption_index=request['caption_index'],
                crop_start=request['crop_start'], max_raw_frames=300, left_right_swap=False,
                anchor_person=request['anchor_person'], output_person_swap=request['output_person_swap'],
                normalize=False, pad_to=300)
        assert replay['processed_length'] == request['processed_length'] and replay['caption'] == request['caption']
        n = replay['processed_length']
        pairs.append(torch.from_numpy(np.stack((replay['motion1'][:n], replay['motion2'][:n]), axis=1)).float())

    @torch.inference_mode()
    def encode(values):
        output = []
        for begin in range(0, len(values), 4):
            chunk = values[begin:begin+4]
            maximum = max(len(x) for x in chunk)
            x = torch.zeros(len(chunk), maximum, 2, 262)
            mask = torch.zeros(len(chunk), maximum)
            for i, value in enumerate(chunk):
                x[i, :len(value)] = value; mask[i, :len(value)] = 1
            z = model({'motions': x.flatten(2), 'mask': mask})['motion_emb']
            output.append(torch.nn.functional.normalize(z, dim=-1))
        return torch.cat(output)

    baseline = encode(pairs)
    contact = [x.clone() for x in pairs]
    for x in contact: x[..., 258:] += 100
    assert torch.equal(encode(contact), baseline)
    rows = []; std = torch.from_numpy(adapter.std); started = time.monotonic()
    for direction in range(args.directions):
        for epsilon in (.02, .1):
            for group in ('position', 'velocity', 'position_consistent_velocity', 'rotation_so3', 'rotation_6d_gaussian'):
                changes = [perturb(x, std, group, epsilon, 20261016 + direction * 1000 + i) for i, x in enumerate(pairs)]
                encoded = encode([x for x, _ in changes])
                rows.append({'group': group, 'direction': direction, 'epsilon': epsilon,
                    'unit_embedding_l2': (encoded - baseline).norm(dim=-1).tolist(),
                    'cosine_distance': (1 - (encoded * baseline).sum(-1)).tolist(),
                    'perturbation': [info for _, info in changes]})
                (args.output/'progress.json').write_text(json.dumps({'finished_arms':len(rows), 'total_arms':args.directions*10, 'elapsed_s':time.monotonic()-started}), encoding='utf-8')
                print(json.dumps({'group':group,'epsilon':epsilon,'direction':direction,'mean':float((encoded-baseline).norm(dim=-1).mean())}), flush=True)
    report = {'status':'complete','scope':'validation_mechanism_only','device':'cpu','threads':args.threads,
              'request_sha256':payload['request_sha256'],'selected_positions':indices,'requests':requests,
              'lengths':[len(x) for x in pairs],'directions':args.directions,'contact_max_abs':0.,'rows':rows,
              'sources':{'script':sha(__file__),'evaluator':sha(source),'checkpoint':sha(checkpoint)},
              'limitations':'No FID or text distances. Unit-normalized motion embeddings only. Smooth P perturbations change bone lengths; legal local rotations need not match P. This does not estimate model error contributions or establish a method ranking.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('complete', flush=True)


if __name__ == '__main__':
    main()
