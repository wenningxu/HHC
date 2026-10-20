"""Direct paired-request GT position/rotation swaps, not aligned-motion error attribution."""
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
from models.interhuman_evaluator_wrapper import (
    CHECKPOINT_PATH, CONFIG_PATH, CONFIG_SHA256, EVALUATOR_SOURCE,
    EVALUATOR_SOURCE_SHA256, _require_hash, load_official_metrics_module,
)


def replace_block(pred, gt, start, end):
    assert 0 < len(pred) <= len(gt)
    result = pred.clone()
    result[..., start:end] = gt[:len(pred), ..., start:end]
    assert torch.equal(result[..., :start], pred[..., :start])
    assert torch.equal(result[..., end:], pred[..., end:])
    assert torch.equal(result[..., start:end], gt[:len(pred), ..., start:end])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--original-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261006)
    _require_hash(CONFIG_PATH, CONFIG_SHA256)
    _require_hash(EVALUATOR_SOURCE, EVALUATOR_SOURCE_SHA256)
    spec = importlib.util.spec_from_file_location('official_gt_swap', EVALUATOR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MotionEncoder(SimpleNamespace(**yaml.safe_load(CONFIG_PATH.read_text()))).eval()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('model.'): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict({k.removeprefix('motion_encoder.'): v for k, v in state.items()
                          if k.startswith('motion_encoder.')}, strict=True)
    scale = float(state['latent_scale'])
    embeddings = {k: [] for k in ('GT', 'raw', 'GT_position', 'GT_rotation')}
    started = time.monotonic()
    with h5py.File(args.cache) as handle, torch.inference_mode():
        assert handle.attrs['status'] == 'complete'
        metadata = json.loads(handle.attrs['metadata'])
        assert metadata['limit'] == 480 and metadata['frame_cap'] == 0
        for i in range(480):
            group = handle[str(i)]
            gt, pred = [torch.from_numpy(group[k][:]) for k in ('gt', 'gen')]
            assert gt.shape[1:] == pred.shape[1:] == (2, 262)
            assert len(gt) == group.attrs['gt_length'] and len(pred) == group.attrs['generated_length']
            assert 0 <= len(gt) - len(pred) <= 3
            assert torch.isfinite(gt).all() and torch.isfinite(pred).all()
            values = {'GT': gt, 'raw': pred,
                      'GT_position': replace_block(pred, gt, 0, 66),
                      'GT_rotation': replace_block(pred, gt, 132, 258)}
            for key, value in values.items():
                out = model({'motions': value.flatten(1)[None],
                             'mask': torch.ones(1, len(value))})['motion_emb']
                embeddings[key].append((torch.nn.functional.normalize(out, dim=-1)*scale).numpy()[0])
            if (i+1) % 96 == 0:
                print(json.dumps({'samples': i+1, 'seconds': time.monotonic()-started}), flush=True)
    embeddings = {k: np.stack(v) for k, v in embeddings.items()}
    metrics = load_official_metrics_module()
    stats = {k: metrics.calculate_activation_statistics(v, 6) for k, v in embeddings.items()}
    fids = {k: float(metrics.calculate_frechet_distance(*stats['GT'], *v)) for k, v in stats.items()}
    original = json.loads(args.original_result.read_text())
    assert original['metadata']['request_sha256'] == metadata['request_sha256']
    assert abs(fids['raw']-original['fid_symmetric']['raw']) < 1e-4, (fids, original['fid_symmetric'])
    result = {'status': 'complete', 'samples': 480, 'source_cache': str(args.cache),
              'metadata': metadata, 'fid': fids,
              'fid_drop': {k: fids['raw']-fids[k] for k in ('GT_position', 'GT_rotation')},
              'seconds': time.monotonic()-started,
              'protocol': 'Same-request GT; replace only P[0:66] or R[132:258], both persons; V/contact unchanged. GT prefix to generated length, original full-length GT reference unchanged.',
              'limitations': 'Generated motions and paired-caption GT are not frame-aligned. Hybrid inputs may be inconsistent; FID drops are diagnostic intervention effects, not additive causal contributions.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix('.embeddings.npz'), **embeddings)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
