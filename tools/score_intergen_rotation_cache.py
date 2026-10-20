#!/usr/bin/env python3
"""Encode a complete frozen InterGen validation cache and score its raw output."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from models.interhuman_evaluator_wrapper import (
    CHECKPOINT_PATH, CONFIG_PATH, CONFIG_SHA256, EVALUATOR_SOURCE,
    EVALUATOR_SOURCE_SHA256, _require_hash,
)


def metrics_path(output):
    name = output.name
    if name.endswith('.embeddings.npz'):
        name = name[:-len('.embeddings.npz')]
    elif name.endswith('_embeddings.npz'):
        name = name[:-len('_embeddings.npz')]
    else:
        name = output.stem
    return output.with_name(f'{name}_metrics.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.cache = args.cache.resolve(strict=True)
    args.output = args.output.resolve()
    score_output = metrics_path(args.output)
    if args.output.exists() or score_output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output} or {score_output}')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA device requested but unavailable: {device}')
    dataset = InterHumanValidationRequestDataset()
    assert len(dataset) == 480
    _require_hash(CONFIG_PATH, CONFIG_SHA256)
    _require_hash(EVALUATOR_SOURCE, EVALUATOR_SOURCE_SHA256)
    spec = importlib.util.spec_from_file_location('frozen_intergen_rotation_encoder', EVALUATOR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config = SimpleNamespace(**yaml.safe_load(CONFIG_PATH.read_text()))
    assert config.INPUT_DIM == 258
    encoder = module.MotionEncoder(config).to(device).eval()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False, mmap=True)
    state = {key.removeprefix('model.'): value for key, value in checkpoint['state_dict'].items()}
    encoder.load_state_dict({key.removeprefix('motion_encoder.'): value for key, value in state.items()
                             if key.startswith('motion_encoder.')}, strict=True)
    latent_scale = float(state['latent_scale'])
    embeddings = {'GT': [], 'raw': []}

    with h5py.File(args.cache, 'r') as cache, torch.inference_mode():
        assert cache.attrs['status'] == 'complete'
        metadata = json.loads(cache.attrs['metadata'])
        assert metadata['request_sha256'] == dataset.payload['request_sha256']
        assert metadata['split'] == 'validation'
        assert metadata['limit'] == len(dataset) == len(cache)
        assert set(cache.keys()) == {str(i) for i in range(len(dataset))}
        for index, request in enumerate(dataset.requests):
            group = cache[str(index)]
            assert json.loads(group.attrs['request']) == request
            gt, raw = (group[key][:] for key in ('gt', 'gen'))
            assert gt.ndim == raw.ndim == 3 and gt.shape == raw.shape
            assert gt.shape[1:] == (2, 262) and 0 < gt.shape[0] <= 300
            assert np.isfinite(gt).all() and np.isfinite(raw).all()
            for name, value in (('GT', gt), ('raw', raw)):
                x = torch.from_numpy(value.astype(np.float32, copy=False))
                result = encoder({
                    'motions': x.flatten(1)[None].to(device),
                    'mask': torch.ones(1, len(x), device=device),
                })['motion_emb']
                emb = torch.nn.functional.normalize(result, dim=-1) * latent_scale
                assert torch.isfinite(emb).all()
                embeddings[name].append(emb.cpu().numpy()[0])
            if (index + 1) % 96 == 0:
                print(json.dumps({'encoded': index + 1, 'total': len(dataset)}), flush=True)

    arrays = {key: np.stack(values) for key, values in embeddings.items()}
    assert all(value.shape == (480, 512) and np.isfinite(value).all()
               for value in arrays.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez(handle, **arrays)

    text_cache = ROOT / 'output/rotation_methods_20261007/text_embeddings.npz'
    subprocess.run([
        sys.executable, str(ROOT / 'tools/score_rotation_method_metrics.py'),
        '--embeddings', str(args.output), '--text-cache', str(text_cache), '--device', 'cpu',
    ], check=True)


if __name__ == '__main__':
    main()
