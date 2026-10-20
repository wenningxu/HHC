"""Small CPU-only local sensitivity probe; not a benchmark or causal bias test."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from models.interhuman_evaluator_wrapper import (
    CHECKPOINT_PATH, CONFIG_PATH, CONFIG_SHA256, EVALUATOR_SOURCE,
    EVALUATOR_SOURCE_SHA256, _require_hash,
)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261006)
    _require_hash(CONFIG_PATH, CONFIG_SHA256)
    _require_hash(EVALUATOR_SOURCE, EVALUATOR_SOURCE_SHA256)
    spec = importlib.util.spec_from_file_location('official_probe', EVALUATOR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MotionEncoder(SimpleNamespace(**yaml.safe_load(CONFIG_PATH.read_text()))).eval()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('model.'): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict({k.removeprefix('motion_encoder.'): v for k, v in state.items()
                           if k.startswith('motion_encoder.')}, strict=True)
    adapter = OfficialInterHumanAdapter()
    ids, pairs = [], []
    for identifier in (adapter.data_root / 'split/val.txt').read_text().splitlines():
        if not identifier or not adapter._has_pair(identifier):
            continue
        pair = adapter.process_pair(identifier, normalize=False)
        if len(pair[0]) >= 64:
            ids.append(identifier)
            pairs.append(np.stack([m[:64] for m in pair], axis=1))
        if len(pairs) == 4:
            break
    x = torch.from_numpy(np.stack(pairs))
    std = torch.from_numpy(adapter.std)
    groups = {'position': (0, 66), 'velocity': (66, 132), 'rotation6d': (132, 258)}

    @torch.inference_mode()
    def encode(value):
        result = model({'motions': value.flatten(2), 'mask': torch.ones(value.shape[:2])})['motion_emb']
        return torch.nn.functional.normalize(result, dim=-1)

    base = encode(x)
    contacts = x.clone()
    contacts[..., 258:] += 100
    contact_max = float((encode(contacts) - base).abs().max())
    assert contact_max == 0, contact_max
    consistency = float((x[:, 1:, :, :66] - x[:, :-1, :, :66] - x[:, :-1, :, 66:132]).abs().max())
    assert consistency < 1e-5, consistency
    rows = []
    # Equal standardized total energy: ||delta/std|| = epsilon * sqrt(T * 2 * 66).
    # Each group gets its own isotropic direction; repeat seeds expose direction noise.
    for seed in (20261006, 20261007):
        generator = torch.Generator().manual_seed(seed)
        for name, (start, end) in groups.items():
            noise = torch.randn(x[..., start:end].shape, generator=generator)
            noise /= noise.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            noise *= (64 * 2 * 66) ** 0.5
            for epsilon in (0.02, 0.1):
                changed = x.clone()
                changed[..., start:end] += epsilon * noise * std[start:end]
                distance = (encode(changed) - base).norm(dim=-1)
                rows.append({'group': name, 'seed': seed, 'epsilon': epsilon,
                             'unit_embedding_l2': distance.tolist(), 'mean': float(distance.mean())})
    # A temporally smooth, position/velocity-consistent translation of actor 2.
    for amplitude in (0.01, 0.05):
        changed = x.clone()
        trajectory = amplitude * torch.sin(torch.arange(65) * (2 * torch.pi / 64))
        changed[:, :, 1, 0:66:3] += trajectory[:64, None]
        changed[:, :, 1, 66:132:3] += trajectory.diff()[:, None]
        assert (changed[:, 1:, :, :66] - changed[:, :-1, :, :66]
                - changed[:, :-1, :, 66:132]).abs().max() < 1e-5
        rows.append({'group': 'actor2_smooth_x_translation_consistent_velocity',
                     'amplitude_m': amplitude,
                     'unit_embedding_l2': (encode(changed) - base).norm(dim=-1).tolist()})
    result = {'ids': ids, 'frames': 64, 'device': 'cpu', 'threads': 2,
              'seed': 20261006, 'contact_embedding_max_abs': contact_max,
              'position_velocity_consistency_max_abs': consistency,
              'latent_scale': float(state['latent_scale'].item()),
              'source_sha256': EVALUATOR_SOURCE_SHA256,
              'limitations': 'Four short validation prefixes; raw feature noise breaks manifold consistency; no FID, text, or baseline generation comparison.',
              'rows': rows}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
