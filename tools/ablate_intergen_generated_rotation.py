"""Released InterGen sampling into the shared symmetric-ablation cache format."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.interhuman_fair_requests import (
    InterHumanFrozenRequestDataset, InterHumanValidationRequestDataset, sha256_file,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--frames', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--replication', type=int)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    assert args.limit >= 1 and (args.frames == 0 or args.frames >= 4)
    args.source, args.checkpoint, args.cache = (x.resolve() for x in (args.source, args.checkpoint, args.cache))
    assert args.checkpoint.stat().st_size == 2701617525
    device = torch.device(args.device)
    if device.type == 'cuda':
        assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    dataset = (InterHumanValidationRequestDataset() if args.replication is None
               else InterHumanFrozenRequestDataset(args.replication))
    assert args.limit <= len(dataset)
    # Dataset resolves paths before the official sampler's relative data paths.
    os.chdir(args.source)
    sys.path.insert(0, str(args.source))
    from models.intergen import InterGen
    cfg = SimpleNamespace(**yaml.safe_load((args.source / 'configs/model.yaml').read_text()))
    assert cfg.STRATEGY == 'ddim50' and cfg.CFG_WEIGHT == 3.5
    for name, value in [('global_mean.npy', dataset.adapter.mean), ('global_std.npy', dataset.adapter.std)]:
        assert np.array_equal(np.load(args.source / 'data' / name), value)
    model = InterGen(cfg).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    state = {key.removeprefix('model.'): value for key, value in checkpoint['state_dict'].items()}
    loaded = model.load_state_dict(state, strict=True)
    metadata = dict(model='InterGen', source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source, text=True).strip(),
                    checkpoint_sha256=sha256_file(args.checkpoint), checkpoint_bytes=args.checkpoint.stat().st_size,
                    missing_keys=loaded.missing_keys, unexpected_keys=loaded.unexpected_keys,
                    request_sha256=dataset.payload['request_sha256'],
                    split='validation' if args.replication is None else 'frozen_replication',
                    replication=args.replication,
                    seed=args.seed, limit=args.limit, frame_cap=args.frames,
                    sampler={'strategy': cfg.STRATEGY, 'cfg_weight': cfg.CFG_WEIGHT, 'batch_size': 1},
                    protocol='frozen requests; official unrounded requested length; raw denormalized 262D',
                    device=str(device), threads=2, pid=os.getpid())
    del checkpoint, state
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(metadata), flush=True)
    started = time.monotonic()
    with h5py.File(args.cache, 'x') as handle, torch.inference_mode():
        handle.attrs['metadata'] = json.dumps(metadata)
        handle.attrs['status'] = 'running'
        for index in range(args.limit):
            name, caption, first, second, length = dataset[index]
            length = min(length, args.frames) if args.frames else length
            generated = model.forward_test({'text': [caption], 'motion_lens': torch.tensor([length], device=device)})['output'][0].reshape(length, 2, 262)
            assert torch.isfinite(generated).all()
            gt = torch.stack((first, second), dim=1)[:length]
            group = handle.create_group(str(index))
            group.attrs['request'] = json.dumps(dataset.requests[index])
            group.attrs['caption'] = caption
            group.attrs['gt_length'] = length
            group.attrs['generated_length'] = length
            for side, value in [('gt', gt), ('gen', generated)]:
                raw = value.cpu().numpy() * dataset.adapter.std + dataset.adapter.mean
                group.create_dataset(side, data=raw.astype(np.float32), compression='lzf')
            handle.flush()
            assert args.cache.stat().st_size < 1_000_000_000
            print(json.dumps({'generated': index+1, 'length': length, 'seconds': time.monotonic()-started, 'pid': os.getpid()}), flush=True)
        handle.attrs['generation_seconds'] = time.monotonic()-started
        handle.attrs['status'] = 'complete'


if __name__ == '__main__':
    main()
