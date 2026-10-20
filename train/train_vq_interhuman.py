#!/usr/bin/env python3
"""Train HHC's paired body/root two-channel VQ on official InterHuman.

The formal entry point is intentionally gated behind an explicit
``--confirm-after-interx`` flag.  Preparing or importing this module cannot
silently start the InterHuman stage before the active Inter-X experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Sampler

from dataset.interhuman_fair_requests import validation_request_loader
from dataset.interhuman_hhc_adapter import InterHumanVQWindowDataset
from models.interhuman_evaluator_wrapper import (
    InterHumanEvaluatorWrapper,
    load_official_metrics_module,
)
from models.model_tools import build_model
from utils.interhuman_vq_evaluation import evaluate_interhuman_vq
from utils.interhuman_vq_loss import InterHumanPairVQLoss
from utils.quantizer_state import quantizer_training_state, restore_quantizer_training_state
from utils.runtime import evaluating, preserving_rng_state
from utils.training_telemetry import (
    GradientClipTelemetry,
    ThroughputMeter,
    gpu_peak_memory_gib,
    parameter_and_gradient_telemetry,
)


PILOT_CONTRACT = ROOT / 'configs/interhuman_vq_pilot_v1.json'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    torch.save(value, temporary)
    os.replace(temporary, path)


class DeterministicEpochBatchSampler(Sampler):
    """Epoch permutations keyed by seed, resumable at an exact optimizer step."""

    def __init__(self, dataset_size, batch_size, total_steps, seed, start_step=0):
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.total_steps = int(total_steps)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.steps_per_epoch = self.dataset_size // self.batch_size
        if self.steps_per_epoch <= 0:
            raise ValueError('InterHuman VQ dataset is smaller than one batch')
        if not 0 <= self.start_step <= self.total_steps:
            raise ValueError('start_step must lie within total_steps')

    def __len__(self):
        return self.total_steps - self.start_step

    def __iter__(self):
        step = self.start_step
        while step < self.total_steps:
            epoch = step // self.steps_per_epoch
            batch_in_epoch = step % self.steps_per_epoch
            permutation = np.random.default_rng(self.seed + epoch).permutation(
                self.dataset_size
            )
            while batch_in_epoch < self.steps_per_epoch and step < self.total_steps:
                start = batch_in_epoch * self.batch_size
                yield permutation[start:start + self.batch_size].tolist()
                batch_in_epoch += 1
                step += 1


def learning_rate_multiplier(step, total_steps, warmup_steps):
    if step <= warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def codebook_metrics(model):
    result = {}
    for name, quantizer in zip(model.channel_names, model.quantizers):
        count = quantizer.code_count
        if count is None:
            used = 0
            total_mass = 0.0
        else:
            count = count.detach().float()
            used = int((count >= 1.0).sum().item())
            total_mass = float(count.sum().item())
        result[f'train/codebook_{name}_used'] = used
        result[f'train/codebook_{name}_utilization'] = used / quantizer.nb_code
        result[f'train/codebook_{name}_ema_mass'] = total_mass
    return result


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm-after-interx', action='store_true')
    parser.add_argument('--name', default='hhc_interhuman_vq_bodyroot_evalfix_v2')
    parser.add_argument('--output-root', default='output/vq_interhuman')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--total-steps', type=int, default=30000)
    parser.add_argument('--run-until-step', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--window-size', type=int, default=64)
    parser.add_argument('--window-stride', type=int, default=10)
    parser.add_argument('--nb-code', type=int, default=1024)
    parser.add_argument('--code-dim', type=int, default=512)
    parser.add_argument('--interhuman-latent-dims', default='384,128')
    parser.add_argument('--interhuman-quantizer', choices=('legacy', 'fp32_window_reset_v1'), default='legacy')
    parser.add_argument('--down-t', type=int, default=2)
    parser.add_argument('--stride-t', type=int, default=2)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--dilation-growth-rate', type=int, default=3)
    parser.add_argument('--output-emb-width', type=int, default=512)
    parser.add_argument('--vq-act', default='relu')
    parser.add_argument('--vq-norm', default=None)
    parser.add_argument('--mu', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--telemetry-interval', type=int, default=50)
    parser.add_argument('--validation-interval', type=int, default=1000)
    parser.add_argument('--checkpoint-interval', type=int, default=1000)
    parser.add_argument('--resume')
    parser.add_argument('--wandb-project', default='NewInter')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    parser.add_argument('--commit', type=float, default=0.02)
    parser.add_argument('--loss-explicit', type=float, default=1.0)
    parser.add_argument('--loss-vel', type=float, default=100.0)
    parser.add_argument('--loss-bn', type=float, default=5.0)
    parser.add_argument('--loss-geo', type=float, default=0.01)
    parser.add_argument('--loss-fc', type=float, default=500.0)
    return parser.parse_args()


def validate_args(args, pilot):
    if getattr(args, 'interhuman_quantizer', 'legacy') != pilot['model'].get('quantizer', 'legacy'):
        raise ValueError('InterHuman quantizer differs from registered pilot')
    if pilot['model']['feature_partition'] != [512, 12] or pilot['model']['codebooks'] != 2:
        raise ValueError('InterHuman VQ requires the two-part body/root pilot contract')
    if not args.confirm_after_interx:
        raise ValueError('InterHuman training is gated; pass --confirm-after-interx only after Inter-X finishes')
    if args.wandb_mode != 'online':
        raise ValueError('all new InterHuman training must use online W&B')
    frozen = {
        'total_steps': pilot['optimization']['full_optimizer_steps'],
        'batch_size': pilot['optimization']['micro_batch_pairs'],
        'window_size': pilot['data']['window_size'],
        'window_stride': pilot['data']['window_stride'],
        'nb_code': pilot['model']['codes_per_codebook'],
        'code_dim': pilot['model']['code_dimension_total'],
        'interhuman_latent_dims': ','.join(map(str, pilot['model']['latent_partition'])),
        'down_t': 2,
        'stride_t': 2,
        'width': pilot['model']['width'],
        'depth': pilot['model']['depth'],
        'lr': pilot['optimization']['learning_rate'],
        'weight_decay': pilot['optimization']['weight_decay'],
        'warmup_steps': pilot['optimization']['warmup_steps'],
        'validation_interval': pilot['automatic_validation']['interval_optimizer_steps'],
    }
    drift = {
        key: {'actual': getattr(args, key), 'expected': value}
        for key, value in frozen.items() if getattr(args, key) != value
    }
    if drift:
        raise ValueError(f'InterHuman VQ pilot arguments drift from preregistration: {drift}')
    if args.output_emb_width != args.code_dim:
        raise ValueError('InterHuman output_emb_width must equal code_dim')
    if min(args.log_interval, args.telemetry_interval, args.checkpoint_interval) <= 0:
        raise ValueError('logging/checkpoint intervals must be positive')
    if args.run_until_step not in {
        int(pilot['optimization']['pilot_optimizer_steps']),
        int(pilot['optimization']['full_optimizer_steps']),
    }:
        raise ValueError('run_until_step must be the VQ pilot decision or full endpoint')


@torch.no_grad()
def validation_losses(model, loader, loss_module, device, precision):
    sums = {}
    samples = 0
    with preserving_rng_state(), evaluating(model):
        for motion in loader:
            motion = motion.to(device).float()
            with torch.autocast(
                device_type='cuda', dtype=torch.bfloat16,
                enabled=precision == 'bf16' and torch.device(device).type == 'cuda',
            ):
                prediction, commitment, perplexity = model(motion, type='full')
            values = loss_module(motion, prediction.float(), commitment.float())
            batch_size = motion.shape[0]
            samples += batch_size
            for name, value in values.items():
                sums[name] = sums.get(name, 0.0) + float(value) * batch_size
            for index, name in enumerate(model.channel_names):
                key = f'perplexity_{name}'
                sums[key] = sums.get(key, 0.0) + float(perplexity[index]) * batch_size
    return {name: value / samples for name, value in sums.items()}


def main():
    args = parse_args()
    pilot_path = (ROOT / 'configs/interhuman_vq_pilot_ema_v3.json'
                  if args.interhuman_quantizer == 'fp32_window_reset_v1' else PILOT_CONTRACT)
    pilot = json.loads(pilot_path.read_text())
    validate_args(args, pilot)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA requested but unavailable: {device}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = (ROOT / args.output_root / args.name).resolve()
    checkpoint_dir = run_dir / 'checkpoints'
    manifest_path = run_dir / 'training_manifest.json'
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError('resume requires the original training manifest')
        existing_manifest = json.loads(manifest_path.read_text())
        wandb_run_id = existing_manifest['observability']['wandb_run_id']
    else:
        if run_dir.exists():
            raise FileExistsError(f'new InterHuman VQ run directory already exists: {run_dir}')
        wandb_run_id = wandb.util.generate_id()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = InterHumanVQWindowDataset(
        split='train', window_size=args.window_size,
        window_stride=args.window_stride, cache_pairs=16,
    )
    val_loss_dataset = InterHumanVQWindowDataset(
        split='val', window_size=args.window_size,
        window_stride=args.window_stride, cache_pairs=8,
    )
    start_step = 0
    best_fid = float('inf')
    best_step = -1
    resume_rng_state = None
    net = build_model(args, 'interhuman_vq').to(device)
    loss_module = InterHumanPairVQLoss(
        commit=args.commit,
        loss_explicit=args.loss_explicit,
        loss_vel=args.loss_vel,
        loss_bn=args.loss_bn,
        loss_geo=args.loss_geo,
        loss_fc=args.loss_fc,
    ).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    if args.resume:
        checkpoint = torch.load(Path(args.resume).resolve(), map_location=device)
        net.load_state_dict(checkpoint['net'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        restore_quantizer_training_state(net, checkpoint['quantizer_training_state'])
        resume_rng_state = checkpoint['rng_state']
        start_step = int(checkpoint['optimizer_step'])
        best_fid = float(checkpoint['best_validation_fid'])
        best_step = int(checkpoint['best_validation_step'])

    source_paths = (
        ROOT / 'train/train_vq_interhuman.py',
        ROOT / 'configs/online_wandb_training_policy.json',
        ROOT / 'dataset/interhuman_hhc_adapter.py',
        ROOT / 'dataset/interhuman_fair_requests.py',
        ROOT / 'models/model_tools.py',
        ROOT / 'models/vqvae_sep3_interhuman.py',
        ROOT / 'models/encdec.py',
        ROOT / 'models/quantize_cnn.py',
        ROOT / 'models/interhuman_quantizer.py',
        ROOT / 'models/interhuman_evaluator_wrapper.py',
        ROOT / 'utils/interhuman_representation.py',
        ROOT / 'utils/interhuman_vq_loss.py',
        ROOT / 'utils/interhuman_vq_evaluation.py',
        ROOT / 'utils/quantizer_state.py',
        ROOT / 'utils/training_telemetry.py',
        ROOT / 'utils/runtime.py',
        pilot_path,
    )
    manifest_contract = {
        'schema': 'hhc_interhuman_vq_training_manifest_v1',
        'resolved_arguments': {
            key: value for key, value in vars(args).items()
            if key not in {'resume', 'run_until_step'}
        },
        'dataset': {
            'train_windows': len(train_dataset),
            'val_loss_windows': len(val_loss_dataset),
            'validation_requests': str(ROOT / 'output/interhuman_validation_requests/manifest.json'),
        },
        'sources': {str(path): sha256_file(path) for path in source_paths},
        'observability': {
            'wandb_mode': 'online',
            'wandb_project': args.wandb_project,
            'wandb_run_id': wandb_run_id,
            'telemetry_interval_optimizer_steps': args.telemetry_interval,
        },
    }
    manifest_contract['contract_sha256'] = canonical_sha256({
        key: value for key, value in manifest_contract.items()
        if key != 'observability'
    } | {
        'observability': {
            key: value for key, value in manifest_contract['observability'].items()
            if key != 'wandb_run_id'
        }
    })
    manifest_contract['initial_run_until_step'] = (
        existing_manifest['initial_run_until_step']
        if args.resume else int(args.run_until_step)
    )
    manifest_contract['current_execution_run_until_step'] = int(args.run_until_step)
    if args.resume:
        if existing_manifest['contract_sha256'] != manifest_contract['contract_sha256']:
            raise ValueError('resume arguments/sources differ from original InterHuman VQ run')
    else:
        atomic_json(manifest_path, {**manifest_contract, 'status': 'running'})

    batch_sampler = DeterministicEpochBatchSampler(
        len(train_dataset), args.batch_size, args.total_steps,
        args.seed, start_step,
    )
    train_loader_generator = torch.Generator().manual_seed(args.seed + 700001)
    val_loader_generator = torch.Generator().manual_seed(args.seed + 700002)
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
        generator=train_loader_generator,
    )
    val_loss_loader = DataLoader(
        val_loss_dataset, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
        generator=val_loader_generator,
    )
    val_metric_loader = validation_request_loader(num_workers=0)
    evaluator = InterHumanEvaluatorWrapper.from_official_checkpoint(device)
    metric_module = load_official_metrics_module()

    wandb.init(
        project=args.wandb_project,
        group='vqvae_interhuman_hhc',
        name=args.name,
        id=wandb_run_id,
        resume='allow',
        mode='online',
        config=vars(args),
    )
    # The official evaluator is instantiated after checkpoint loading and its
    # module construction consumes RNG. Restore at the final boundary before
    # the first resumed batch, not earlier, or continuation masks drift.
    if resume_rng_state is not None:
        restore_rng_state(resume_rng_state)
    clipper = GradientClipTelemetry(args.gradient_clip)
    throughput = ThroughputMeter()
    log_sums = {}
    log_count = 0

    if not start_step < args.run_until_step <= args.total_steps:
        raise ValueError(
            f'run_until_step={args.run_until_step} must exceed resume step '
            f'{start_step} and not exceed total_steps={args.total_steps}'
        )

    def save_checkpoint(path, step):
        atomic_torch_save(path, {
            'schema': 'hhc_interhuman_vq_checkpoint_v1',
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'quantizer_training_state': quantizer_training_state(net),
            'rng_state': capture_rng_state(),
            'optimizer_step': step,
            'best_validation_fid': best_fid,
            'best_validation_step': best_step,
            'training_contract_sha256': manifest_contract['contract_sha256'],
        })

    for step, motion in enumerate(train_loader, start=start_step + 1):
        multiplier = learning_rate_multiplier(step, args.total_steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group['lr'] = args.lr * multiplier
        motion = motion.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type='cuda', dtype=torch.bfloat16,
            enabled=args.precision == 'bf16' and device.type == 'cuda',
        ):
            prediction, commitment, perplexity = net(motion, type='full')
        losses = loss_module(motion, prediction.float(), commitment.float())
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError(
                f'non-finite InterHuman VQ loss at optimizer step {step}: '
                f'{ {name: float(value.detach()) for name, value in losses.items()} }'
            )
        losses['total'].backward()
        throughput.update(motion.shape[0])
        emit_telemetry = step % args.telemetry_interval == 0
        telemetry = {}
        if emit_telemetry:
            telemetry.update(parameter_and_gradient_telemetry(net))
        telemetry.update(clipper.clip(net, emit_metrics=emit_telemetry))
        optimizer.step()
        log_count += 1
        for name, value in losses.items():
            log_sums[name] = log_sums.get(name, 0.0) + float(value.detach())
        for index, name in enumerate(net.channel_names):
            key = f'perplexity_{name}'
            log_sums[key] = log_sums.get(key, 0.0) + float(perplexity[index].detach())

        if emit_telemetry:
            telemetry.update(codebook_metrics(net))
            pairs_per_second = throughput.measure_and_reset()
            telemetry.update({
                'train/optimizer_step': step,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'train/pairs_per_second': pairs_per_second,
                'train/optimizer_steps_per_second': pairs_per_second / args.batch_size,
                'train/gpu_peak_memory_gib': gpu_peak_memory_gib(device),
            })
            wandb.log(telemetry, step=step)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
        if step % args.log_interval == 0:
            record = {
                f'train/{name}': value / log_count for name, value in log_sums.items()
            }
            record['train/optimizer_step'] = step
            record['train/learning_rate'] = optimizer.param_groups[0]['lr']
            wandb.log(record, step=step)
            log_sums.clear()
            log_count = 0

        if step % args.validation_interval == 0:
            val_losses = validation_losses(
                net, val_loss_loader, loss_module, device, args.precision
            )
            val_metrics = evaluate_interhuman_vq(
                net, val_metric_loader, evaluator, metric_module,
                device=device, mean=train_dataset.adapter.mean,
                std=train_dataset.adapter.std, random_seed=20260130,
            )
            wandb.log({
                **{f'val/{name}': value for name, value in val_losses.items()},
                **{f'val/{name}': value for name, value in val_metrics.items()},
                'val/best_fid': min(best_fid, val_metrics['fid']),
            }, step=step)
            if val_metrics['fid'] < best_fid:
                best_fid = float(val_metrics['fid'])
                best_step = step
                save_checkpoint(checkpoint_dir / 'best_validation_fid.pth', step)

        # Save after same-step validation so resume state includes the newest
        # best-FID bookkeeping rather than the previous validation result.
        if step % args.checkpoint_interval == 0:
            save_checkpoint(checkpoint_dir / f'step_{step:09d}.pth', step)
            save_checkpoint(checkpoint_dir / 'latest.pth', step)

        if step >= args.run_until_step:
            break

    if step != args.run_until_step:
        raise RuntimeError(
            f'InterHuman VQ stopped at {step}, expected {args.run_until_step}'
        )
    best_path = checkpoint_dir / 'best_validation_fid.pth'
    if not best_path.is_file() or best_step < 0:
        raise RuntimeError('InterHuman VQ ended without a validation-selected checkpoint')
    if step < args.total_steps:
        save_checkpoint(checkpoint_dir / 'latest.pth', step)
        atomic_json(manifest_path, {
            **manifest_contract,
            'status': 'pilot_decision_ready',
            'current_optimizer_step': step,
            'best_validation_fid': best_fid,
            'best_validation_step': best_step,
            'best_validation_checkpoint': str(best_path),
            'best_validation_checkpoint_sha256': sha256_file(best_path),
            'resume_checkpoint': str(checkpoint_dir / 'latest.pth'),
        })
        wandb.finish()
        return
    save_checkpoint(checkpoint_dir / 'last.pth', step)
    atomic_json(manifest_path, {
        **manifest_contract,
        'status': 'complete',
        'final_optimizer_step': step,
        'best_validation_fid': best_fid,
        'best_validation_step': best_step,
        'best_validation_checkpoint': str(best_path),
        'best_validation_checkpoint_sha256': sha256_file(best_path),
    })
    wandb.finish()


if __name__ == '__main__':
    main()
