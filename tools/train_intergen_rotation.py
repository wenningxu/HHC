"""Train InterGen with root rotation and a rotation-primary FK loss."""
import argparse
import fcntl
import json
import math
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
from dataset.interhuman_root_rotation import RootRotationData
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset, sha256_file
from tools.intergen_rotation_objective import rotation_objective


def initialize(model, checkpoint, mean, std):
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    old = {k.removeprefix('model.'): v for k, v in saved['state_dict'].items()}
    state = model.state_dict()
    changed = []
    for key, value in state.items():
        source = old[key]
        if source.shape == value.shape:
            state[key] = source
        elif key == 'decoder.net.motion_embed.weight':
            value.zero_()
            value[:, :262] = source
            changed.append(key)
        elif key == 'decoder.net.out.linear.weight':
            value.zero_()
            value[:262] = source
            changed.append(key)
        elif key == 'decoder.net.out.linear.bias':
            value[:262] = source
            value[262:] = (torch.tensor([1., 0, 0, 1, 0, 0])-mean[262:])/std[262:]
            changed.append(key)
        else:
            raise ValueError(f'Unexpected checkpoint size mismatch: {key}')
    assert len(changed) == 3 and set(state) == set(old)
    model.load_state_dict(state, strict=True)
    return changed


def statistics(data, output, limit=0):
    rng = random.Random(20261008)
    sums, squares, count = np.zeros(6), np.zeros(6), 0
    ids = data.ids[:limit] if limit else data.ids
    for i, identifier in enumerate(ids):
        length = len(data.source(identifier)[0][0])
        start = rng.randrange(length-300) if length > 300 else 0
        x, _ = data.pair(identifier, start, bool(i % 2), 1+i % 2, False)
        root = x[..., 262:].reshape(-1, 6).astype(np.float64)
        sums += root.sum(0)
        squares += (root*root).sum(0)
        count += len(root)
        if (i+1) % 500 == 0:
            print(json.dumps({'stats_train_ids': i+1, 'total': len(ids)}), flush=True)
    root_mean = sums/count
    root_std = np.sqrt(np.maximum(squares/count-root_mean**2, 0)).clip(.05)
    mean = np.concatenate((data.adapter.mean, root_mean)).astype(np.float32)
    std = np.concatenate((data.adapter.std, root_std)).astype(np.float32)
    np.savez(output/'normalization.npz', mean=mean, std=std)
    (output/'normalization.json').write_text(json.dumps({
        'train_ids': ids, 'root_frames_people': count,
        'protocol': 'Preserve official first262 stats for pretrained model; appended root stats: one deterministic <=300-frame training crop per ID, alternating mirror/anchor, seed20261008; std floor .05. No val/test.',
    }, indent=2)+'\n')
    return torch.from_numpy(mean), torch.from_numpy(std)


def validate(model, mean, std, dataset, output, step, seed, device):
    destination = output/f'validation_{step:06d}'
    complete = destination/'validation_metrics.json'
    if complete.exists():
        return json.loads(complete.read_text())['arms']['raw']
    cache = Path('/dev/shm')/f'{output.name}_val_{step:06d}.h5'
    # Preserve interrupted validation artifacts rather than overwriting them.
    for path in (destination, cache):
        if path.exists():
            path.rename(path.with_name(path.name+f'.interrupted_{time.time_ns()}'))
    destination.mkdir(exist_ok=False)
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model.eval()
    with h5py.File(cache, 'x') as h, torch.no_grad():
        h.attrs['status'] = 'running'
        h.attrs['metadata'] = json.dumps({'split': 'validation', 'limit': len(dataset),
            'request_sha256': dataset.payload['request_sha256'], 'step': step, 'seed': seed,
            'protocol': 'InterGen268 DDIM50 CFG3.5; denormalize then drop only appended root6; no GT shape/root supplied to generation'})
        for i in range(len(dataset)):
            name, caption, a, b, length = dataset[i]
            pred = model.forward_test({'text': [caption], 'motion_lens': torch.tensor([length], device=device)})['output']
            pred = pred[0].reshape(length, 2, 268).float()*std+mean
            pred = pred[..., :262].cpu().numpy()
            gt = torch.stack((a, b), 1)[:length].numpy()*dataset.adapter.std+dataset.adapter.mean
            assert pred.shape == gt.shape and np.isfinite(pred).all()
            group = h.create_group(str(i))
            group.attrs['request'] = json.dumps(dataset.requests[i])
            group.create_dataset('gt', data=gt, compression='lzf')
            group.create_dataset('gen', data=pred, compression='lzf')
        h.attrs['status'] = 'complete'
    subprocess.run([sys.executable, str(ROOT/'tools/score_intergen_rotation_cache.py'), '--cache', str(cache),
                    '--output', str(destination/'validation_embeddings.npz')], cwd=ROOT, check=True)
    result = json.loads((destination/'validation_metrics.json').read_text())['arms']['raw']
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'third_party/intergen_base_20261008')
    parser.add_argument('--checkpoint', type=Path, default=Path('/dev/shm/hhc_intergen_20261006.CtYnWA/intergen.ckpt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--micro-batch', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--scratch', action='store_true', help='No pretrained motion-generator weights; keep frozen CLIP')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-weights-only', type=Path, help='Explicit scratch weight continuation with fresh optimizer/RNG; never full-state resume')
    parser.add_argument('--check-resume-only', action='store_true')
    parser.add_argument('--write-recovery-receipt', action='store_true', help='Publish hash/step after atomic recovery saves for the local backup controller')
    args = parser.parse_args()
    args.steps = args.steps if args.steps is not None else (704000 if args.scratch else 10000)
    args.lr = args.lr if args.lr is not None else (1e-4 if args.scratch else 1e-5)
    args.eval_every = args.eval_every if args.eval_every is not None else (10000 if args.scratch else 2000)
    assert args.steps > 0 and args.micro_batch > 0 and args.accumulation > 0 and args.eval_every > 0
    args.output = args.output.resolve()
    args.source = args.source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    assert not args.check_resume_only or args.resume
    assert not (args.resume and args.smoke)
    assert not args.resume_weights_only or (args.scratch and not args.resume and not args.smoke)
    args.output.mkdir(parents=True, exist_ok=args.resume or args.scratch)
    if args.scratch and not args.resume and (args.output/'latest.pth').exists():
        assert torch.load(args.output/'latest.pth', map_location='cpu', mmap=True, weights_only=False)['step'] == 0, 'Existing training requires --resume'
    run_lock = (args.output/'.training.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.output/'STATUS.json').write_text(json.dumps({'status': 'preparing_train_root_statistics', 'step': 0})+'\n')
    torch.set_num_threads(2)
    random.seed(20261008)
    np.random.seed(20261008)
    torch.manual_seed(20261008)
    data = RootRotationData()
    if args.resume or (args.scratch and (args.output/'normalization.npz').exists()):
        stats = np.load(args.output/'normalization.npz')
        mean, std = torch.from_numpy(stats['mean']), torch.from_numpy(stats['std'])
    else:
        mean, std = statistics(data, args.output, limit=8 if args.smoke else 0)
    validation = InterHumanValidationRequestDataset(adapter=data.adapter)
    os.chdir(args.source)
    sys.path.insert(0, str(args.source))
    from models.intergen import InterGen
    cfg = SimpleNamespace(**yaml.safe_load((args.source/'configs/model.yaml').read_text()))
    cfg.INPUT_DIM = 268
    model = InterGen(cfg)
    if args.scratch:
        # Identity rotations avoid singular FK at the official zero output head.
        with torch.no_grad():
            assert torch.count_nonzero(model.decoder.net.out.linear.weight) == 0
            identity = torch.tensor([1., 0, 0, 1, 0, 0])
            for start in list(range(132, 258, 6))+[262]:
                model.decoder.net.out.linear.bias[start:start+6] = (identity-mean[start:start+6])/std[start:start+6]
        changed = []
    else:
        changed = initialize(model, args.checkpoint, mean, std)
    model.positional_embedding.requires_grad_(False)
    model = model.to(args.device)
    mean, std = mean.to(args.device), std.to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=2e-5)
    loader = torch.utils.data.DataLoader(data, batch_size=args.micro_batch, shuffle=True, drop_last=True, num_workers=0)
    iterator = iter(loader)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    configuration = {**vars(args), 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source, text=True).strip(),
        'base_checkpoint_sha256': None if args.scratch else sha256_file(args.checkpoint), 'extended_keys': changed,
        'effective_batch': args.micro_batch*args.accumulation, 'seed': 20261008,
        'objective_source_sha256': sha256_file(ROOT/'tools/intergen_rotation_objective.py'),
        'data_source_sha256': sha256_file(ROOT/'dataset/interhuman_root_rotation.py'),
        'trainer_source_sha256': sha256_file(Path(__file__)),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'protocol': 'Official architecture/diffusion/CFG, pretrained fine-tuning (NOT from-scratch reproduction); bf16 autocast, fp32 FK/objective. Additional root6 predicted, stripped at eval. Original 262 position/velocity heads retained as weak auxiliaries.'}
    if args.scratch:
        configuration['checkpoint'] = None
        configuration['protocol'] = 'Motion generator and trainable text-adaptation layers initialized from scratch; frozen pretrained CLIP only. Same rotation/FK loss and 268D representation. Identity output-rotation bias, no motion checkpoint read.'
    if not args.resume:
        (args.output/'config.json').write_text(json.dumps(configuration, default=str, indent=2)+'\n')
    print(json.dumps(configuration, default=str), flush=True)
    best, best_step = float('inf'), None
    start_step = 0
    recovery = args.output/'recovery.pt' if args.scratch else Path('/dev/shm')/(args.output.name+'_optimizer.pt')
    if args.resume:
        recover = torch.load(recovery, map_location='cpu', mmap=True, weights_only=False)
        saved = recover.get('checkpoint')
        if saved is None:
            saved = torch.load(args.output/'latest.pth', map_location='cpu', mmap=True, weights_only=False)
        assert saved['step'] == recover['step'], 'Weights/optimizer steps must match'
        assert saved['config'].get('scratch', False) == args.scratch, 'Cannot mix scratch and finetune checkpoints'
        for key in ('steps', 'micro_batch', 'accumulation', 'lr', 'eval_every', 'base_checkpoint_sha256', 'objective_source_sha256', 'data_source_sha256'):
            assert saved['config'][key] == configuration[key], f'Resume configuration mismatch: {key}'
        assert set(saved['trainable_state']) == names
        assert torch.equal(saved['mean'], mean.cpu()) and torch.equal(saved['std'], std.cpu())
        assert all(torch.isfinite(x).all() for x in saved['trainable_state'].values())
        assert all(torch.isfinite(x).all() for v in recover['optimizer']['state'].values() for x in v.values() if isinstance(x, torch.Tensor))
        model.load_state_dict(saved['trainable_state'], strict=False)
        optimizer.load_state_dict(recover['optimizer'])
        start_step = saved['step']
        best, best_step = saved['best_fid'], saved['best_step']
        if 'continuation' in saved['config']:
            configuration['continuation'] = saved['config']['continuation']
        random.setstate(recover['python_rng'])
        np.random.set_state(recover['numpy_rng'])
        torch.set_rng_state(recover['torch_rng'])
        torch.cuda.set_rng_state_all(recover['cuda_rng'])
        record = {'status': 'resume_checked' if args.check_resume_only else 'resumed', 'step': start_step,
                  'best_fid': best, 'best_step': best_step, 'pid': os.getpid(),
                  'note': 'Optimizer/RNG restored; dataloader cursor not saved, not bitwise resume'}
        print(json.dumps(record), flush=True)
        (args.output/'STATUS.json').write_text(json.dumps(record, indent=2)+'\n')
        del saved, recover
        if args.check_resume_only:
            return
    elif args.resume_weights_only:
        saved = torch.load(args.resume_weights_only, map_location='cpu', mmap=True, weights_only=False)
        assert saved['config'].get('scratch') is True
        for key in ('steps', 'micro_batch', 'accumulation', 'lr', 'eval_every', 'base_checkpoint_sha256', 'objective_source_sha256', 'data_source_sha256'):
            assert saved['config'][key] == configuration[key], f'Weight continuation mismatch: {key}'
        assert set(saved['trainable_state']) == names
        assert torch.equal(saved['mean'], mean.cpu()) and torch.equal(saved['std'], std.cpu())
        assert all(torch.isfinite(x).all() for x in saved['trainable_state'].values())
        model.load_state_dict(saved['trainable_state'], strict=False)
        start_step = saved['step']
        best, best_step = saved['best_fid'], saved['best_step']
        configuration['continuation'] = {'mode': 'scratch_weights_only', 'source_step': start_step,
            'source_sha256': sha256_file(args.resume_weights_only), 'optimizer_restored': False,
            'rng_restored': False, 'reason': 'Original compute unavailable; latest complete optimizer/RNG backup not available'}
        (args.output/'config.json').write_text(json.dumps(configuration, default=str, indent=2)+'\n')
        print(json.dumps(configuration['continuation']), flush=True)
        del saved

    def save(step, name):
        # Store only changed/trainable weights; frozen CLIP comes from pinned base.
        payload = {'trainable_state': {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names},
                   'step': step, 'mean': mean.cpu(), 'std': std.cpu(), 'config': configuration,
                   'best_fid': best, 'best_step': best_step}
        temporary = args.output/(name+'.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, args.output/name)
        return payload

    def save_recovery(step, payload):
        temporary = recovery.with_suffix('.tmp')
        torch.save({'step': step, 'checkpoint': payload, 'optimizer': optimizer.state_dict(),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'note': 'Dataloader cursor not saved; not bitwise resume'}, temporary)
        os.replace(temporary, recovery)
        if args.write_recovery_receipt:
            receipt = {'step': step, 'file': str(recovery), 'bytes': recovery.stat().st_size,
                       'sha256': sha256_file(recovery), 'time_unix': time.time()}
            receipt_tmp = args.output/'RECOVERY.json.tmp'
            receipt_tmp.write_text(json.dumps(receipt, indent=2)+'\n')
            os.replace(receipt_tmp, args.output/'RECOVERY.json')

    if args.scratch and not args.resume and not args.smoke:
        save_recovery(start_step, save(start_step, 'latest.pth'))

    if not args.smoke and not args.resume and not args.scratch:
        (args.output/'STATUS.json').write_text(json.dumps({'status': 'initial_validation', 'step': 0})+'\n')
        baseline = validate(model, mean, std, validation, args.output, 0, 20261006, args.device)
        best, best_step = baseline['FID'], 0
        save(0, 'best.pth')
        (args.output/'STATUS.json').write_text(json.dumps({'status': 'running', 'step': 0,
            'metrics': baseline, 'best_fid': best, 'best_step': 0}, indent=2)+'\n')

    started = time.monotonic()
    for step in range(start_step+1, args.steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        warmup = 10*(len(data)//(args.micro_batch*args.accumulation)) if args.scratch else 100
        lr = args.lr*min(step/max(warmup, 1), 1)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        for _ in range(args.accumulation):
            try:
                texts, raw, lengths, offsets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                texts, raw, lengths, offsets = next(iterator)
            raw, lengths, offsets = raw.to(args.device), lengths.to(args.device), offsets.to(args.device)
            target = ((raw-mean)/std).flatten(2)
            t, _ = model.decoder.sampler.sample(len(raw), args.device)
            mask = (torch.arange(raw.shape[1], device=args.device)[None] < lengths[:, None]).float()[..., None].expand(-1, -1, 2)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                cond = model.text_process({'text': list(texts)})['cond']
                cond, _ = model.decoder.mask_cond(cond, .1)
                noisy = model.decoder.diffusion.q_sample(target, t)
                pred = model.decoder.net(noisy, t, mask=mask, cond=cond)
            losses = rotation_objective(pred, target, offsets, lengths, t, mean, std)
            assert torch.isfinite(losses['total']), 'Nonfinite loss'
            (losses['total']/args.accumulation).backward()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.)+float(value.detach())/args.accumulation
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
        if step % 50 == 0 or step == 1 or args.smoke:
            row = {'step': step, 'lr': lr, 'seconds': time.monotonic()-started, 'grad_norm': float(grad), **totals}
            print(json.dumps(row), flush=True)
            with (args.output/'training.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            if not args.smoke:
                (args.output/'STATUS.json').write_text(json.dumps({'status': 'training', 'step': step,
                    'best_fid': best, 'best_step': best_step, 'last_loss': totals['total']}, indent=2)+'\n')
        if not args.smoke and (step % args.eval_every == 0 or step == args.steps):
            metric = validate(model, mean, std, validation, args.output, step, 20261006, args.device)
            if metric['FID'] < best:
                best, best_step = metric['FID'], step
                save(step, 'best.pth')
            checkpoint_payload = save(step, 'latest.pth')
            (args.output/'STATUS.json').write_text(json.dumps({'status': 'running', 'step': step, 'metrics': metric,
                'best_fid': best, 'best_step': best_step}, indent=2)+'\n')
        elif step % 500 == 0 or args.smoke:
            checkpoint_payload = save(step, 'latest.pth')
        if step % 500 == 0 and not args.smoke:
            save_recovery(step, checkpoint_payload)
    if args.smoke:
        model.eval()
        with torch.no_grad():
            generated = model.forward_test({'text': ['two people shake hands'], 'motion_lens': torch.tensor([32], device=args.device)})['output']
            generated = generated.reshape(1, 32, 2, 268).float()*std+mean
            exported = generated[..., :262]
            assert exported.shape == (1, 32, 2, 262) and torch.isfinite(exported).all()
        print(json.dumps({'sampling_smoke': 'passed', 'root_removed_shape': list(exported.shape)}), flush=True)
    (args.output/'STATUS.json').write_text(json.dumps({'status': 'smoke_complete' if args.smoke else 'complete',
        'step': args.steps, 'best_fid': best if best_step is not None else None, 'best_step': best_step}, indent=2)+'\n')


if __name__ == '__main__':
    main()
