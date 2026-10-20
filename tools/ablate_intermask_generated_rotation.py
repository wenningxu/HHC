"""Released text-to-motion sampling and symmetric feature interventions (diagnostic)."""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path('/mnt/8T-HDD/wenning/InterMask')
sys.path.insert(0, str(ROOT))
from dataset.interhuman_fair_requests import (
    InterHumanFrozenRequestDataset, InterHumanValidationRequestDataset, sha256_file,
)


def generate(args):
    # Import the established official namespace bridge before any HHC models.
    from export_official_intermask_interhuman_vq_reconstruction import build_official_vq, require_official_sources
    from models.mask_transformer.transformer import MaskTransformer
    require_official_sources()
    dataset = (InterHumanValidationRequestDataset() if args.replication is None
               else InterHumanFrozenRequestDataset(args.replication))
    assert args.limit <= len(dataset)
    device = torch.device(args.device)
    if device.type == 'cuda':
        assert torch.cuda.is_available()
    vq, metadata = build_official_vq(device)
    opt = SimpleNamespace(num_tokens=1024, dataset_name='interhuman')
    model = MaskTransformer(code_dim=512, cond_mode='text', latent_dim=384,
                            ff_size=1024, num_layers=6, num_heads=6, dropout=.2,
                            clip_dim=768, cond_drop_prob=.1,
                            clip_version='ViT-L/14@336px', opt=opt).to(device).eval()
    checkpoint_path = OFFICIAL / 'checkpoints/interhuman/trans_default/model/best_fid.tar'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint.get('t2m_transformer', checkpoint.get('trans')), strict=False)
    assert not unexpected and all(k.startswith('clip_') for k in missing)
    metadata.update(transformer_sha256=sha256_file(checkpoint_path),
                    vq_sha256=sha256_file(OFFICIAL / 'checkpoints/interhuman/vq_default/model/best_fid.tar'),
                    transformer_epoch=int(checkpoint['ep']), seed=args.seed,
                    sampler={'timesteps':20, 'cond_scale':2, 'topk_filter_thres':.9,
                             'temperature':1, 'gsample':False, 'batch_size':1},
                    source_sha256={str(p.relative_to(OFFICIAL)):sha256_file(p) for p in
                                   [OFFICIAL/'models/mask_transformer/transformer.py',
                                    OFFICIAL/'checkpoints/interhuman/trans_default/opt.txt']},
                    request_sha256=dataset.payload['request_sha256'],
                    split='validation' if args.replication is None else 'frozen_replication',
                    replication=args.replication, device=str(device),
                    limit=args.limit, frame_cap=args.frames,
                    protocol='frozen requests; official per-request floor(length/4)*4 generation')
    del checkpoint
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    with h5py.File(args.cache, 'x') as handle, torch.inference_mode():
        handle.attrs['metadata'] = json.dumps(metadata)
        handle.attrs['status'] = 'running'
        for index in range(args.limit):
            name, caption, first, second, length = dataset[index]
            length = min(length, args.frames) if args.frames else length
            ids = model.generate([caption], torch.tensor([length // 4], device=device), 20, 2,
                                 topk_filter_thres=.9, temperature=1)
            ids1, ids2 = ids.chunk(2, dim=1)
            pred = torch.stack([vq.forward_decoder(x.unsqueeze(-1))[0] for x in (ids1, ids2)], dim=1)
            gt = torch.stack((first, second), dim=1)[:length]
            group = handle.create_group(str(index))
            group.attrs['request'] = json.dumps(dataset.requests[index])
            group.attrs['caption'] = caption
            group.attrs['gt_length'] = length
            group.attrs['generated_length'] = len(pred)
            for key, value in [('gt',gt), ('gen',pred)]:
                raw = value.cpu().numpy() * dataset.adapter.std + dataset.adapter.mean
                group.create_dataset(key, data=raw.astype(np.float32), compression='lzf')
            handle.flush()
            print(json.dumps({'generated':index+1, 'seconds':time.monotonic()-started,
                              'length':len(pred), 'pid':os.getpid()}), flush=True)
        handle.attrs['generation_seconds'] = time.monotonic()-started
        handle.attrs['status'] = 'complete'


def smooth(value):
    return .25 * torch.cat((value[:1], value[:-1])) + .5 * value + .25 * torch.cat((value[1:], value[-1:]))


def intervention(value, label):
    from ablate_intermask_feature_fid import project
    x = value.clone()
    if label in ('R_project', 'R_smooth'):
        x = project(x)
        if label == 'R_smooth':
            x[..., 132:258] = smooth(x[..., 132:258])
            x = project(x)
    if label in ('P_smooth', 'P_smooth_V_consistent'):
        x[..., :66] = smooth(x[..., :66])
    if label in ('V_consistent', 'P_smooth_V_consistent'):
        # Last velocity needs an unobserved next frame: preserve it identically on both sides.
        x[:-1, :, 66:132] = x[1:, :, :66] - x[:-1, :, :66]
    return x


def evaluate(args):
    import yaml
    from ablate_intermask_feature_fid import rotation, project
    from models.interhuman_evaluator_wrapper import (
        CHECKPOINT_PATH, CONFIG_PATH, CONFIG_SHA256, EVALUATOR_SOURCE,
        EVALUATOR_SOURCE_SHA256, _require_hash, load_official_metrics_module,
    )
    _require_hash(CONFIG_PATH, CONFIG_SHA256)
    _require_hash(EVALUATOR_SOURCE, EVALUATOR_SOURCE_SHA256)
    spec = importlib.util.spec_from_file_location('official_generated_ablation', EVALUATOR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MotionEncoder(SimpleNamespace(**yaml.safe_load(CONFIG_PATH.read_text()))).eval()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('model.'):v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict({k.removeprefix('motion_encoder.'):v for k,v in state.items()
                           if k.startswith('motion_encoder.')}, strict=True)
    scale = float(state['latent_scale'])
    labels = ['raw', 'R_project', 'R_smooth', 'P_smooth', 'V_consistent', 'P_smooth_V_consistent']
    values = {}
    with h5py.File(args.cache) as handle:
        assert handle.attrs['status'] == 'complete'
        metadata = json.loads(handle.attrs['metadata'])
        for side in ('gt', 'gen'):
            values[side] = [torch.from_numpy(handle[str(i)][side][:]) for i in range(metadata['limit'])]
    # Runnable contract check: column-interleaved identity and channel isolation.
    test = values['gt'][0]
    projected = project(test)
    assert torch.allclose(project(projected), projected, atol=2e-6)
    assert torch.equal(projected[..., :132], test[..., :132])
    assert torch.equal(intervention(test, 'P_smooth')[..., 66:], test[..., 66:])
    consistent = intervention(test, 'V_consistent')
    assert torch.allclose(consistent[:-1, :, 66:132], consistent[1:, :, :66]-consistent[:-1, :, :66])
    embeddings, diagnostics = {}, {}
    start = time.monotonic()
    with torch.inference_mode():
        for label in labels:
            for side in ('gt', 'gen'):
                embedded, norm_error, dot_error, jitter, residual = [], [], [], [], []
                for value in values[side]:
                    x = intervention(value, label)
                    emb = model({'motions':x.flatten(1)[None], 'mask':torch.ones(1,len(x))})['motion_emb']
                    embedded.append((torch.nn.functional.normalize(emb, dim=-1)*scale).numpy()[0])
                    r, matrix = rotation(x)
                    cols = r.reshape(*r.shape[:-1],3,2)
                    norm_error.append((cols.norm(dim=-2)-1).abs().flatten())
                    dot_error.append((cols[...,0]*cols[...,1]).sum(-1).abs().flatten())
                    relative = matrix[:-1].transpose(-1,-2) @ matrix[1:]
                    angles = torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))
                    jitter.append(torch.rad2deg(angles).flatten())
                    residual.append((x[:-1,:,66:132]-x[1:,:,:66]+x[:-1,:,:66]).reshape(-1,22,3).norm(dim=-1).flatten())
                key = side+'_'+label
                embeddings[key] = np.stack(embedded)
                diagnostics[key] = {'rotation_norm_abs_error':float(torch.cat(norm_error).mean()),
                                    'rotation_dot_abs':float(torch.cat(dot_error).mean()),
                                    'rotation_step_deg':float(torch.cat(jitter).mean()),
                                    'velocity_position_residual_m_per_frame':float(torch.cat(residual).mean())}
                np.savez(Path(args.cache).with_suffix('.embeddings.npz'), **embeddings)
                print(json.dumps({'encoded':key, 'samples':len(embedded), 'seconds':time.monotonic()-start}), flush=True)
    metrics = load_official_metrics_module()
    stats = {k:metrics.calculate_activation_statistics(v,6) for k,v in embeddings.items()}
    fid = {label:float(metrics.calculate_frechet_distance(*stats['gt_'+label],*stats['gen_'+label])) for label in labels}
    result = {'status':'complete', 'metadata':metadata, 'fid_symmetric':fid,
              'diagnostics':diagnostics, 'latent_scale':scale, 'metric_activation_scale':6,
              'evaluation_seconds':time.monotonic()-start,
              'limitations':'Validation diagnostic, single generation seed; small-sample FID is not a benchmark. Both GT and generation receive the same intervention. FID changes are nonadditive, not percentage contributions. Smoothing can remove real motion; no frame-aligned GT replacement.'}
    Path(args.cache).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--replication', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    assert args.limit >= 1 and (args.frames == 0 or args.frames >= 4)
    torch.set_num_threads(2)
    print(json.dumps({'pid':os.getpid(), 'device':args.device, 'threads':2}), flush=True)
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    if args.evaluate:
        evaluate(args)
    else:
        generate(args)


if __name__ == '__main__':
    main()
