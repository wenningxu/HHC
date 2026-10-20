"""Evaluate a frozen rotation corrector against its untouched generation cache."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.train_rotation_residual import Corrector
from dataset.interhuman_fair_requests import InterHumanFrozenRequestDataset, sha256_file
from models.interhuman_evaluator_wrapper import (
    CHECKPOINT_PATH, CONFIG_PATH, CONFIG_SHA256, EVALUATOR_SOURCE,
    EVALUATOR_SOURCE_SHA256, _require_hash,
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replication', type=int, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    assert not args.output.exists(), 'Do not overwrite completed embeddings'
    torch.set_num_threads(2)
    dataset = InterHumanFrozenRequestDataset(args.replication)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    training_ids = set(saved['train_ids']) | set(saved['holdout_ids'])
    assert not training_ids & {r['identifier'] for r in dataset.requests}
    corrector = Corrector(saved['mode'] == 'position_rotation').to(args.device).eval()
    corrector.load_state_dict(saved['model'], strict=True)
    _require_hash(CONFIG_PATH, CONFIG_SHA256)
    _require_hash(EVALUATOR_SOURCE, EVALUATOR_SOURCE_SHA256)
    spec = importlib.util.spec_from_file_location('frozen_rotation_encoder', EVALUATOR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = module.MotionEncoder(SimpleNamespace(**yaml.safe_load(CONFIG_PATH.read_text()))).eval()
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items()}
    encoder.load_state_dict({k.removeprefix('motion_encoder.'): v for k, v in state.items()
                            if k.startswith('motion_encoder.')}, strict=True)
    latent = float(state['latent_scale'])
    embeddings = {k: [] for k in ('gt', 'raw', 'corrected')}
    with h5py.File(args.cache) as cache, torch.inference_mode():
        assert cache.attrs['status'] == 'complete'
        metadata = json.loads(cache.attrs['metadata'])
        assert metadata['request_sha256'] == dataset.payload['request_sha256']
        assert metadata['limit'] == len(dataset) == len(cache)
        for i, request in enumerate(dataset.requests):
            group = cache[str(i)]
            assert json.loads(group.attrs['request']) == request
            gt, raw = [torch.from_numpy(group[key][:]) for key in ('gt', 'gen')]
            corrected = raw.clone()
            corrected[..., 132:258] = corrector(raw.transpose(0, 1).to(args.device)).transpose(0, 1).cpu()
            assert torch.equal(corrected[..., :132], raw[..., :132])
            assert torch.equal(corrected[..., 258:], raw[..., 258:])
            assert torch.isfinite(corrected).all()
            for key, x in [('gt', gt), ('raw', raw), ('corrected', corrected)]:
                emb = encoder({'motions': x.flatten(1)[None], 'mask': torch.ones(1, len(x))})['motion_emb']
                embeddings[key].append((torch.nn.functional.normalize(emb, dim=-1)*latent).numpy()[0])
            if (i+1) % 96 == 0:
                print(json.dumps({'encoded': i+1, 'total': len(dataset)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez(handle, **{k: np.stack(v) for k, v in embeddings.items()})
    args.output.with_suffix('.provenance.json').write_text(json.dumps({
        'status': 'complete', 'split': 'test', 'replication': args.replication,
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha256_file(args.checkpoint),
        'selected_step': saved['step'], 'source_metadata': metadata,
        'request_sha256': dataset.payload['request_sha256'],
        'protocol': 'Frozen full-strength correction, CPU official encoder; unchanged P/V/contact. Single replication, not 20-rep result.',
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
