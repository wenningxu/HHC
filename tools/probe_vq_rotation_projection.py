#!/usr/bin/env python3
"""InterHuman VQ rotation-only projection diagnostic on frozen 480 caches."""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path('/mnt/8T-HDD/wenning/InterMask')
SHM = Path('/dev/shm/hhc_vq_projection_20261007')
OUT = ROOT / 'output/rotation_methods_20261007'
SOURCES = {
    'intermask': ROOT / 'output/intermask_generation_rotation_20261006/generated480.h5',
    'intergen': Path('/dev/shm/hhc_intergen_20261006.CtYnWA/full480.h5'),
}
BASELINES = {
    'intermask': ROOT / 'output/intermask_generation_rotation_20261006/generated480.json',
    'intergen': ROOT / 'output/intergen_generation_rotation_20261006/full480.json',
}
ALPHAS = (0.25, 0.5, 1.0)


def reconstruct(args):
    # Load the released model through the existing namespace bridge before HHC imports.
    sys.path.insert(0, str(OFFICIAL))
    sys.path.insert(1, str(ROOT / 'tools'))
    from export_official_intermask_interhuman_vq_reconstruction import (
        build_official_vq, require_official_sources,
    )
    import h5py
    import numpy as np
    import torch
    from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter

    require_official_sources()
    adapter = OfficialInterHumanAdapter()
    device = torch.device('cuda:1')
    assert torch.cuda.is_available() and torch.cuda.device_count() > 1
    torch.cuda.set_device(device)
    model, checkpoint = build_official_vq(device)
    SHM.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for tag, source in SOURCES.items():
        destination = SHM / f'{tag}_rotations.h5'
        if destination.exists():
            raise FileExistsError(destination)
        with h5py.File(source, 'r') as src, h5py.File(destination, 'x') as dst, torch.inference_mode():
            metadata = json.loads(src.attrs['metadata'])
            assert src.attrs['status'] == 'complete' and metadata['limit'] == 480
            assert metadata['frame_cap'] == 0 and metadata['request_sha256'] == '751efc9ced91e2a6ad4eec4a7b9316946761fe13f86d09af7d5c04a9226b7f93'
            dst.attrs['status'] = 'running'
            dst.attrs['source'] = str(source)
            dst.attrs['request_sha256'] = metadata['request_sha256']
            dst.attrs['vq_checkpoint_sha256'] = 'af57bbdde94ed4d4a01f3ba5f9fe24f0012c2a37b3e38c2fa57a2a91ebcc1272'
            for i in range(480):
                group = src[str(i)]
                raw = torch.from_numpy(group['gen'][:]).to(device=device, dtype=torch.float32)
                length = int(group.attrs['generated_length'])
                assert raw.shape == (length, 2, 262) and torch.isfinite(raw).all()
                normalized = (raw - torch.from_numpy(adapter.mean).to(device)) / torch.from_numpy(adapter.std).to(device)
                padded = ((length + 3) // 4) * 4
                if padded > length:
                    normalized = torch.cat((normalized, normalized[-1:].expand(padded - length, 2, 262)), dim=0)
                vq_input = normalized.permute(1, 0, 2).contiguous()
                pred, _, _ = model(vq_input)
                assert pred.shape == vq_input.shape, (tag, i, length, pred.shape, vq_input.shape)
                decoded_length = pred.shape[1]
                pred = pred[:, :length].permute(1, 0, 2).contiguous()
                assert pred.shape == (length, 2, 262) and torch.isfinite(pred).all()
                recon_raw = pred * torch.from_numpy(adapter.std).to(device) + torch.from_numpy(adapter.mean).to(device)
                rotation = recon_raw[..., 132:258].cpu().numpy().astype(np.float32)
                assert rotation.shape == (length, 2, 126)
                dst.create_dataset(str(i), data=rotation, compression='lzf')
                if i == 3:
                    print(json.dumps({'event': 'smoke_pass', 'model': tag, 'samples': 4,
                                      'device': str(device), 'last_length': length,
                                      'vq_output_length': decoded_length}), flush=True)
                if (i + 1) % 96 == 0:
                    print(json.dumps({'event': 'reconstructed', 'model': tag,
                                      'samples': i + 1, 'seconds': time.monotonic() - started,
                                      'pid': os.getpid()}), flush=True)
            dst.attrs['status'] = 'complete'
    print(json.dumps({'event': 'reconstruction_complete', 'seconds': time.monotonic() - started,
                      'pid': os.getpid(), 'checkpoint': checkpoint}), flush=True)


def evaluate(args):
    sys.path.insert(0, str(ROOT))
    import h5py
    import numpy as np
    import torch
    import yaml
    from models.interhuman_evaluator_wrapper import (
        CHECKPOINT_PATH, CONFIG_PATH, CONFIG_SHA256, EVALUATOR_SOURCE,
        EVALUATOR_SOURCE_SHA256, _require_hash, load_official_metrics_module,
    )

    torch.set_num_threads(2)
    _require_hash(CONFIG_PATH, CONFIG_SHA256)
    _require_hash(EVALUATOR_SOURCE, EVALUATOR_SOURCE_SHA256)
    spec = importlib.util.spec_from_file_location('official_vq_rotation_projection', EVALUATOR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MotionEncoder(SimpleNamespace(**yaml.safe_load(CONFIG_PATH.read_text()))).eval()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('model.'): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict({k.removeprefix('motion_encoder.'): v for k, v in state.items()
                           if k.startswith('motion_encoder.')}, strict=True)
    latent_scale = float(state['latent_scale'])
    del checkpoint, state
    metrics = load_official_metrics_module()
    start = time.monotonic()
    with torch.inference_mode():
        for tag, source in SOURCES.items():
            baseline = json.loads(BASELINES[tag].read_text())
            variants = {name: [] for name in ('GT', 'raw', *(f'alpha_{a:.2f}' for a in ALPHAS))}
            with h5py.File(source) as src, h5py.File(SHM / f'{tag}_rotations.h5') as recon:
                metadata = json.loads(src.attrs['metadata'])
                assert src.attrs['status'] == recon.attrs['status'] == 'complete'
                assert metadata['request_sha256'] == baseline['metadata']['request_sha256'] == recon.attrs['request_sha256']
                wrist_identity = torch.tensor([1, 0, 0, 1, 0, 0] * 2, dtype=torch.float32)
                wrist_error = max(float((torch.from_numpy(src[str(i)]['gt'][:, :, 246:258]) - wrist_identity).abs().max())
                                  for i in range(480))
                wrist_ok = wrist_error < 1e-5
                if wrist_ok:
                    variants['release_wrist_identity'] = []
                else:
                    print(json.dumps({'event': 'wrist_identity_skipped', 'model': tag,
                                      'max_abs_error': wrist_error}), flush=True)
                for i in range(480):
                    group = src[str(i)]
                    gt = torch.from_numpy(group['gt'][:])
                    raw = torch.from_numpy(group['gen'][:])
                    rotation = torch.from_numpy(recon[str(i)][:])
                    assert gt.shape[1:] == raw.shape[1:] == (2, 262)
                    assert len(gt) == int(group.attrs['gt_length'])
                    assert len(raw) == int(group.attrs['generated_length'])
                    assert 0 <= len(gt) - len(raw) <= 3
                    assert rotation.shape == (len(raw), 2, 126)
                    gt_embedding = model({'motions': gt.flatten(1)[None], 'mask': torch.ones(1, len(gt))})['motion_emb']
                    variants['GT'].append((torch.nn.functional.normalize(gt_embedding, dim=-1) * latent_scale).numpy()[0])
                    raw_embedding = model({'motions': raw.flatten(1)[None], 'mask': torch.ones(1, len(raw))})['motion_emb']
                    variants['raw'].append((torch.nn.functional.normalize(raw_embedding, dim=-1) * latent_scale).numpy()[0])
                    for alpha in ALPHAS:
                        candidate = raw.clone()
                        candidate[..., 132:258] = raw[..., 132:258] * (1.0 - alpha) + rotation * alpha
                        assert torch.equal(candidate[..., :132], raw[..., :132])
                        assert torch.equal(candidate[..., 258:], raw[..., 258:])
                        assert torch.isfinite(candidate).all()
                        if alpha == 1.0:
                            assert torch.equal(candidate[..., 132:258], rotation)
                        key = f'alpha_{alpha:.2f}'
                        embedding = model({'motions': candidate.flatten(1)[None], 'mask': torch.ones(1, len(candidate))})['motion_emb']
                        variants[key].append((torch.nn.functional.normalize(embedding, dim=-1) * latent_scale).numpy()[0])
                    if wrist_ok:
                        wrist = raw.clone()
                        wrist[..., 246:258] = wrist_identity
                        assert torch.equal(wrist[..., :246], raw[..., :246])
                        assert torch.equal(wrist[..., 258:], raw[..., 258:])
                        embedding = model({'motions': wrist.flatten(1)[None], 'mask': torch.ones(1, len(wrist))})['motion_emb']
                        variants['release_wrist_identity'].append((torch.nn.functional.normalize(embedding, dim=-1) * latent_scale).numpy()[0])
                    if (i + 1) % 96 == 0:
                        print(json.dumps({'event': 'scored', 'model': tag, 'samples': i + 1,
                                          'seconds': time.monotonic() - start, 'pid': os.getpid()}), flush=True)
            embeddings = {k: np.stack(v) for k, v in variants.items()}
            stats = {k: metrics.calculate_activation_statistics(v, 6) for k, v in embeddings.items()}
            fids = {k: float(metrics.calculate_frechet_distance(*stats['GT'], *v)) for k, v in stats.items()}
            assert abs(fids['raw'] - baseline['fid_symmetric']['raw']) < 1e-4, (tag, fids['raw'], baseline['fid_symmetric']['raw'])
            result = {
                'status': 'complete', 'model': tag, 'samples': 480,
                'source_cache': str(source), 'vq_reconstruction': str(SHM / f'{tag}_rotations.h5'),
                'request_sha256': metadata['request_sha256'], 'seed': metadata.get('seed'),
                'protocol': '480 frozen validation requests; VQ trained checkpoint frozen; normalize raw 262D with official mean/std, edge-pad to multiple of four, decode and crop; blend only raw R[132:258] for both persons; P/V/contact remain bitwise identical; 6D linear blend without re-orthogonalization; official MotionEncoder, latent_scale and FID activation scale 6.',
                'alpha': {k: fids[f'alpha_{k:.2f}'] for k in ALPHAS},
                'alpha_fid_delta_vs_raw': {k: fids['raw'] - fids[f'alpha_{k:.2f}'] for k in ALPHAS},
                'release_wrist_identity': ({'gt_identity_max_abs_error': wrist_error,
                                            'gen_fid': fids['release_wrist_identity'],
                                            'fid_delta_vs_raw': fids['raw'] - fids['release_wrist_identity'],
                                            'protocol': 'gen only; set R[246:258] to [1,0,0,1,0,0] x 2; no GT action copied'}
                                           if wrist_ok else {'skipped': True,
                                                             'gt_identity_max_abs_error': wrist_error}),
                'raw_fid_recomputed': fids['raw'], 'raw_fid_reference': baseline['fid_symmetric']['raw'],
                'latent_scale': latent_scale, 'metric_activation_scale': 6,
                'limitations': 'Validation diagnostic only, not formal test. Single frozen generation per model; 6D linear blending is not projected to SO(3), and hybrid motions can be internally inconsistent.',
            }
            stem = OUT / f'vq_projection_{tag}'
            stem.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
            np.savez_compressed(stem.with_suffix('.embeddings.npz'), **embeddings)
            print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('reconstruct', 'evaluate'), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == 'reconstruct':
        reconstruct(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
