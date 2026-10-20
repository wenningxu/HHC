#!/usr/bin/env python3
"""Build deterministic official-train pairs for the rotation residual model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path('/mnt/8T-HDD/wenning/InterMask')
DEFAULT_OUTPUT = Path('/dev/shm/hhc_rotation_train_20261007.npz')
MANIFEST = ROOT / 'output/rotation_methods_20261007/training_pairs_manifest.json'
SEED = 20261007
FRAMES = 64
RAW_WINDOW = FRAMES + 1  # released preprocessing drops its initial frame
VQ_SHA256 = 'af57bbdde94ed4d4a01f3ba5f9fe24f0012c2a37b3e38c2fa57a2a91ebcc1272'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=1024, help='use 4 for smoke')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 1024:
        raise ValueError('--limit must be in [1,1024]')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for official VQ reconstruction')
    torch.cuda.set_device(0)

    sys.path.insert(0, str(ROOT))
    from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
    bridge_path = ROOT / 'tools/export_official_intermask_interhuman_vq_reconstruction.py'
    spec = importlib.util.spec_from_file_location('official_vq_bridge', bridge_path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    bridge.require_official_sources()
    adapter = OfficialInterHumanAdapter()

    train_ids = adapter.split_ids('train', require_motion=True)
    eligible = [name for name in train_ids if adapter._pair_raw_length(name) >= RAW_WINDOW]
    random.Random(SEED).shuffle(eligible)
    ids = eligible[:args.limit]
    if len(ids) != args.limit:
        raise RuntimeError(f'only {len(ids)} legal official train IDs for requested {args.limit}')

    # Full run uses motion IDs for the held-out split; smoke remains train-only.
    split_index = min(896, len(ids))
    split = np.asarray(['train'] * split_index + ['internal_holdout'] * (len(ids) - split_index), dtype='U16')
    vq, checkpoint_meta = bridge.build_official_vq('cuda:0')
    raw_gt, raw_recon = [], []
    for index, identifier in enumerate(ids):
        p1, p2 = adapter._load_raw_pair(identifier, False, FRAMES)
        start_max = min(p1.shape[0], p2.shape[0]) - RAW_WINDOW
        # Deterministic per-ID crop start, independent of iteration order.
        start = int.from_bytes(hashlib.sha256(f'{SEED}:{identifier}'.encode()).digest()[:8], 'little') % (start_max + 1)
        pair = adapter.process_loaded_pair(p1[start:start + RAW_WINDOW], p2[start:start + RAW_WINDOW], normalize=True)
        normalized = np.stack(pair)
        if normalized.shape != (2, FRAMES, 262) or not np.isfinite(normalized).all():
            raise ValueError(f'invalid normalized pair {identifier}: {normalized.shape}')
        x = torch.from_numpy(normalized.transpose(1, 0, 2).copy()).to('cuda:0')
        pred = bridge.reconstruct_person(vq, x.transpose(0, 1).reshape(2, FRAMES, 262), torch.full((2,), FRAMES, device='cuda:0', dtype=torch.long))
        # VQ sees each person independently; restore [T,2,262] layout.
        recon_norm = pred.transpose(0, 1).cpu().numpy()
        gt_raw = normalized.transpose(1, 0, 2) * adapter.std + adapter.mean
        recon_raw = recon_norm * adapter.std + adapter.mean
        if not np.isfinite(gt_raw).all() or not np.isfinite(recon_raw).all():
            raise ValueError(f'non-finite raw positions/velocities for {identifier}')
        raw_gt.append(gt_raw.astype(np.float32))
        raw_recon.append(recon_raw.astype(np.float32))
        if (index + 1) % 64 == 0 or index + 1 == len(ids):
            print(json.dumps({'processed': index + 1, 'total': len(ids), 'id': identifier}), flush=True)

    gt, recon = np.stack(raw_gt), np.stack(raw_recon)
    if gt.shape != (len(ids), FRAMES, 2, 262) or recon.shape != gt.shape:
        raise ValueError(f'bad final shapes GT={gt.shape}, recon={recon.shape}')
    if not np.isfinite(gt[..., :132]).all() or not np.isfinite(recon[..., :132]).all():
        raise ValueError('raw position/velocity channels contain non-finite values')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    np.savez_compressed(args.output, ids=np.asarray(ids, dtype='U64'), split=split,
                        split_index=np.asarray(split_index, dtype=np.int64), gt=gt, recon=recon,
                        crop_frames=np.full(len(ids), RAW_WINDOW, np.int16))
    size = args.output.stat().st_size
    if size >= 400_000_000:
        raise RuntimeError(f'npz exceeds 400 MB ceiling: {size}')
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema': 'hhc_rotation_training_pairs_v1', 'status': 'complete',
        'seed': SEED, 'official_split': 'train', 'requested_ids': 1024,
        'selected_ids': len(ids), 'eligible_official_train_ids': len(eligible),
        'selection': f'seeded shuffle of official train IDs with both raw people at least {RAW_WINDOW} frames',
        'crop': '65 raw contiguous frames; released preprocessing drops first frame, yielding 64 processed frames; per-ID deterministic SHA256(seed,id) start',
        'split_index': split_index,
        'motion_id_split': {'train': int(np.count_nonzero(split == 'train')),
                            'internal_holdout': int(np.count_nonzero(split == 'internal_holdout'))},
        'shapes': {'ids': list(np.asarray(ids).shape), 'gt': list(gt.shape),
                   'recon': list(recon.shape), 'split': list(split.shape), 'split_index': []},
        'npz': str(args.output), 'npz_bytes': size, 'npz_sha256': digest(args.output),
        'vq_checkpoint_sha256': VQ_SHA256, 'vq_checkpoint_metadata': checkpoint_meta,
        'normalization_mean_sha256': digest(adapter.mean_path),
        'normalization_std_sha256': digest(adapter.std_path),
        'source_script_sha256': digest(Path(__file__)),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    with np.load(args.output, allow_pickle=False) as saved:
        assert saved['gt'].shape == saved['recon'].shape == (len(ids), FRAMES, 2, 262)
        assert len(set(saved['ids'].tolist())) == len(ids)
        assert int(saved['split_index']) == split_index
    print(json.dumps({'status': 'complete', 'pid': __import__('os').getpid(),
                      'output': str(args.output), 'bytes': size,
                      'keys': ['ids', 'split', 'split_index', 'gt', 'recon', 'crop_frames'],
                      'shapes': manifest['shapes'], 'manifest': str(args.manifest)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
