"""Reconstruct InterHuman's model-consumed 192D body subset from official raw pkl.

This does NOT reconstruct the full 492D SMPL-H file or modify released data.
Female template and zero serialized wrist rotations are empirically identified
release conventions, not assertions that the raw gender labels are correct.
"""
import argparse
import builtins
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import smplx
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_model():
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    for name in ('bool', 'int', 'float', 'complex', 'object', 'str'):
        if name not in np.__dict__:
            setattr(np, name, getattr(builtins, name))
    if 'unicode' not in np.__dict__:
        np.unicode = str
    return smplx.create(str(ROOT / 'body_models'), model_type='smpl', gender='female').eval()


@torch.no_grad()
def reconstruct_person(person, frame_rate, model):
    assert float(frame_rate) > 0
    total = len(person['trans'])
    if total == 0:
        raise ValueError('Empty raw motion has no processed counterpart (released IDs 3945 and 4106).')
    idx = np.arange(0, total, float(frame_rate)/30).astype(np.int64)
    body = torch.from_numpy(person['pose_body'][idx]).float()
    orient = torch.from_numpy(person['root_orient'][idx]).float()
    beta = torch.from_numpy(person['betas']).float().reshape(1, 10)
    shaped = model.v_template.unsqueeze(0) + blend_shapes(beta, model.shapedirs)
    rest = vertices2joints(model.J_regressor, shaped).expand(len(idx), -1, -1)
    pose = torch.cat([orient, body, torch.zeros((len(idx), 6))], dim=1)
    matrices = batch_rodrigues(pose.reshape(-1, 3)).reshape(len(idx), 24, 3, 3)
    joints, _ = batch_rigid_transform(matrices, rest, model.parents)
    positions = joints[:, :22] + torch.from_numpy(person['trans'][idx]).float()[:, None]
    rotations = matrices[:, 1:22].clone()
    rotations[:, -2:] = torch.eye(3)  # Published 6D wrist fields are identity.
    features = torch.cat([positions.reshape(len(idx), 66),
                          rotations[:, :, :, :2].reshape(len(idx), 126)], dim=1).numpy()
    assert features.shape == (len(idx), 192) and np.isfinite(features).all()
    return features, idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT / 'dataset/interhuman_official')
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--online', action='store_true', help='Also compare shared official online 262D processing')
    parser.add_argument('--require-complete-download', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.require_complete_download:
        manifest = json.loads((ROOT / 'output/interhuman_raw_download_20261003.json').read_text())
        assert manifest['status'] == 'complete'
        assert manifest['verified_count'] == manifest['count']
    model = load_model()
    paths = sorted((args.root / 'motions').glob('*.pkl'), key=lambda p: int(p.stem))
    eligible = [p for p in paths if all((args.root / 'motions_processed' / w / (p.stem+'.npy')).exists()
                                      for w in ('person1', 'person2'))]
    selected = [eligible[i] for i in np.unique(np.linspace(0, len(eligible)-1, min(args.limit, len(eligible))).astype(int))]
    adapter = None
    if args.online:
        from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
        adapter = OfficialInterHumanAdapter(data_root=args.root)
    records = []
    schemas, rates, genders, shape_counts = Counter(), Counter(), Counter(), Counter()
    hand_reference = None
    hand_rotation_max_deviation = 0.0
    for path in selected:
        raw = pickle.loads(path.read_bytes())
        rates[str(raw['mocap_framerate'])] += 1
        generated, released = [], []
        row = {'id': path.stem, 'people': []}
        for who in ('person1', 'person2'):
            schemas[','.join(sorted(raw[who]))] += 1
            genders[str(raw[who]['gender'])] += 1
            shape_counts[hashlib.sha256(np.asarray(raw[who]['betas']).tobytes()).hexdigest()] += 1
            a, idx = reconstruct_person(raw[who], raw['mocap_framerate'], model)
            y = np.load(args.root / 'motions_processed' / who / (path.stem+'.npy'))
            if hand_reference is None:
                hand_reference = y[0, 312:].copy()
            hand_rotation_max_deviation = max(hand_rotation_max_deviation,
                                               float(abs(y[:, 312:]-hand_reference).max()))
            b = np.concatenate([y[:, :66], y[:, 186:312]], axis=1)
            if a.shape != b.shape:
                row['people'].append({'person': who, 'error': 'length_mismatch',
                                      'reconstructed': list(a.shape), 'released': list(b.shape)})
                continue
            error = np.linalg.norm((a[:, :66]-b[:, :66]).reshape(-1, 22, 3), axis=-1)
            row['people'].append({'person': who, 'frames': len(a), 'mpjpe_mm': float(error.mean()*1000),
                                  'max_joint_error_mm': float(error.max()*1000),
                                  'rotation_max': float(abs(a[:, 66:]-b[:, 66:]).max())})
            generated.append(a)
            released.append(b)
        if adapter and len(generated) == 2:
            a = np.stack(adapter.process_loaded_pair(*generated, normalize=False))
            b = np.stack(adapter.process_loaded_pair(*released, normalize=False))
            row['online262'] = {
                'positions_max_m': float(abs(a[:, :, :66]-b[:, :, :66]).max()),
                'velocities_max_m_per_frame': float(abs(a[:, :, 66:132]-b[:, :, 66:132]).max()),
                'rotations_max': float(abs(a[:, :, 132:258]-b[:, :, 132:258]).max()),
                'contact_disagreements': int(np.count_nonzero(a[:, :, 258:] != b[:, :, 258:])),
                'contact_values': int(a[:, :, 258:].size),
                'normalized_max': float(abs((a-b)/adapter.std).max())}
        records.append(row)
        if len(records) % 128 == 0:
            print(f'checked {len(records)}/{len(selected)} pairs', flush=True)
    valid = [p for r in records for p in r['people'] if 'error' not in p]
    online = [r['online262'] for r in records if 'online262' in r]
    summary = {'pairs': len(records), 'person_records': len(valid),
               'raw_person_schemas': dict(schemas), 'frame_rates': dict(rates),
               'raw_genders': dict(genders), 'unique_beta_vectors': len(shape_counts),
               'hand_rotation_max_deviation_from_first': hand_rotation_max_deviation,
               'raw_without_processed': [p.stem for p in paths if p not in eligible],
               'processed_without_raw': sorted({p.stem for p in (args.root/'motions_processed/person1').glob('*.npy')}
                                               - {p.stem for p in paths}),
               'errors': [r for r in records if any('error' in p for p in r['people'])],
               'mpjpe_mm_mean': float(np.mean([p['mpjpe_mm'] for p in valid])),
               'max_joint_error_mm': max(p['max_joint_error_mm'] for p in valid),
               'rotation_max': max(p['rotation_max'] for p in valid)}
    if online:
        summary['online262_max'] = {k: max(r[k] for r in online) for k in online[0] if 'contact' not in k}
        summary['contact_disagreements'] = sum(r['contact_disagreements'] for r in online)
        summary['contact_values'] = sum(r['contact_values'] for r in online)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {'model': str((ROOT/'body_models/smpl/SMPL_FEMALE.pkl').resolve()),
                  'model_sha256': hashlib.sha256((ROOT/'body_models/smpl/SMPL_FEMALE.pkl').read_bytes()).hexdigest(),
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'protocol': 'same ID/person, floor(arange time*fps/30), original betas, female SMPL FK, wrist 6D identity, no fitted alignment'}
    args.output.write_text(json.dumps({'summary': summary, 'records': records, 'provenance': provenance}, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
