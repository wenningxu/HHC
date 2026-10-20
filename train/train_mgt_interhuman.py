#!/usr/bin/env python3
"""Gated, reproducible HHC MGT training on official InterHuman."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import clip
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from dataset.interhuman_fair_requests import validation_request_loader
from dataset.interhuman_generated_h5 import mm_repeat_seed
from dataset.interhuman_mgt import (
    InterHumanMGTDrawBatchSampler,
    InterHumanMGTTrainDataset,
)
from models.interhuman_evaluator_wrapper import (
    InterHumanEvaluatorWrapper,
    load_official_metrics_module,
)
from models.model_tools import build_model
from utils.interhuman_artifacts import load_hhc_interhuman_vq, sha256_file
from utils.interhuman_mgt_evaluation import (
    evaluate_interhuman_generator,
    evaluate_interhuman_validation_multimodality,
    generate_interhuman_pair,
)
from utils.interhuman_mgt_training import (
    freeze_interhuman_vq_for_mgt,
    interhuman_mgt_objective,
    prepare_interhuman_mgt_batch,
)
from utils.runtime import evaluating, preserving_rng_state
from utils.training_telemetry import (
    GradientClipTelemetry,
    ThroughputMeter,
    gpu_peak_memory_gib,
    parameter_and_gradient_telemetry,
)


CONTRACT_PATH = ROOT / 'configs/interhuman_mgt_pilot_v1.json'
VALIDATION_MM_PATH = (
    ROOT / 'output/interhuman_validation_mm_requests/validation_mm.json'
)


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    torch.save(value, temporary)
    os.replace(temporary, path)


def verify_vq_wandb_audit(audit_path, vq_manifest_path):
    audit_path = Path(audit_path).resolve()
    vq_manifest_path = Path(vq_manifest_path).resolve()
    audit = json.loads(audit_path.read_text())
    checks = {
        'schema': audit.get('schema') == 'hhc_wandb_online_metric_history_audit_v1',
        'status': audit.get('status') == 'pass',
        'profile': audit.get('metric_profile') == 'interhuman_vq',
        'endpoint': int(audit.get('expected_final_step', -1)) == 30000,
        'remote_finished': audit.get('remote_state') == 'finished',
        'manifest_path': Path(audit.get('manifest', {}).get('path', '')).resolve()
        == vq_manifest_path,
        'manifest_hash': audit.get('manifest', {}).get('sha256')
        == sha256_file(vq_manifest_path),
    }
    if not all(checks.values()):
        raise ValueError(f'InterHuman VQ W&B audit is not valid for MGT: {checks}')
    return {
        'path': str(audit_path),
        'sha256': sha256_file(audit_path),
        'run_path': audit['run_path'],
        'metric_profile': audit['metric_profile'],
        'expected_final_step': audit['expected_final_step'],
    }


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


class TextCLIP(torch.nn.Module):
    """Expose the same global and word CLIP features used by HHC Inter-X."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, text):
        word = self.model.token_embedding(text).type(self.model.dtype)
        word = word + self.model.positional_embedding.type(self.model.dtype)
        word = self.model.transformer(word.permute(1, 0, 2))
        word = self.model.ln_final(word).permute(1, 0, 2).float()
        return self.model.encode_text(text).float(), word


@contextmanager
def temporary_conditioning_contract(transformer, contract):
    targets = [transformer, transformer.trans_base]
    previous = [(target, target.conditioning_contract) for target in targets]
    for target in targets:
        target.conditioning_contract = contract
    try:
        yield
    finally:
        for target, value in previous:
            target.conditioning_contract = value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm-after-interx', action='store_true')
    parser.add_argument('--name', default='hhc_interhuman_mgt_pilot_v1')
    parser.add_argument('--contract', default=str(CONTRACT_PATH))
    parser.add_argument('--output-root', default='output/mgt_interhuman')
    parser.add_argument('--vq-checkpoint', required=True)
    parser.add_argument('--vq-training-manifest', required=True)
    parser.add_argument('--vq-wandb-audit', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--evaluator-device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=114514)
    parser.add_argument('--total-steps', type=int, default=300000)
    parser.add_argument('--run-until-step', type=int, default=50000)
    parser.add_argument('--micro-batch-size', type=int, default=64)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--validation-interval', type=int, default=10000)
    parser.add_argument('--checkpoint-interval', type=int, default=1000)
    parser.add_argument('--telemetry-interval', type=int, default=50)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--validation-generation-batch-size', type=int, default=8)
    parser.add_argument('--resume')
    parser.add_argument('--wandb-project', default='NewInter')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    return parser.parse_args()


def validate_args(args, contract):
    if not args.confirm_after_interx:
        raise ValueError(
            'InterHuman MGT training is gated; pass --confirm-after-interx only '
            'after the Inter-X stages and InterHuman VQ selection finish'
        )
    if args.wandb_mode != 'online':
        raise ValueError('all new InterHuman training must use online W&B')
    expected = {
        'seed': contract['training']['seed'],
        'total_steps': contract['training']['full_optimizer_steps'],
        'micro_batch_size': contract['training']['micro_batch_pairs'],
        'gradient_accumulation': contract['training']['gradient_accumulation'],
        'lr': contract['training']['learning_rate'],
        'weight_decay': contract['training']['weight_decay'],
        'gradient_clip': contract['training']['gradient_clip_norm'],
        'precision': contract['training']['precision'],
        'validation_interval': contract['automatic_validation']['interval_optimizer_steps'],
        'checkpoint_interval': contract['training']['checkpoint_interval_optimizer_steps'],
    }
    drift = {
        key: {'actual': getattr(args, key), 'expected': value}
        for key, value in expected.items() if getattr(args, key) != value
    }
    if drift:
        raise ValueError(f'InterHuman MGT arguments drift from preregistration: {drift}')
    if min(
        args.micro_batch_size, args.gradient_accumulation, args.num_workers + 1,
        args.validation_generation_batch_size, args.validation_interval,
        args.checkpoint_interval, args.telemetry_interval, args.log_interval,
    ) <= 0:
        raise ValueError('InterHuman MGT batch/interval arguments must be positive')
    if args.run_until_step not in {
        int(contract['training']['pilot_decision_step']),
        int(contract['training']['full_optimizer_steps']),
    }:
        raise ValueError(
            'run_until_step must be the preregistered pilot decision or full endpoint'
        )


def transformer_arguments(contract, vq):
    model = contract['model']
    if int(vq.num_code) != int(model['codes_per_channel']):
        raise ValueError(
            f'selected VQ code count {vq.num_code} != MGT {model["codes_per_channel"]}'
        )
    return SimpleNamespace(
        nb_code=int(model['codes_per_channel']),
        embed_dim_gpt=int(model['embed_dim']),
        clip_dim=512,
        block_size=int(model['block_size']),
        num_layers=int(model['base_layers_total']),
        num_local_layer=int(model['local_cross_attention_layers']),
        n_head_gpt=int(model['attention_heads']),
        drop_out_rate=float(model['dropout']),
        ff_rate=int(model['feedforward_rate']),
        contact_conditioning=False,
        shape=False,
        paper_architecture=bool(model['paper_architecture']),
        conditioning_contract=model['conditioning_contract_train'],
    )


def trainable_transformer_state(transformer):
    return {
        key: value for key, value in transformer.state_dict().items()
        if not key.startswith('trans_base.vqvae.')
    }


def load_trainable_transformer_state(transformer, state):
    result = transformer.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    invalid_missing = [
        key for key in result.missing_keys
        if not key.startswith('trans_base.vqvae.')
    ]
    if unexpected or invalid_missing:
        raise ValueError(
            f'InterHuman MGT state mismatch: missing={invalid_missing}, '
            f'unexpected={unexpected}'
        )


def main():
    args = parse_args()
    contract_path = Path(args.contract).resolve()
    contract = json.loads(contract_path.read_text())
    validate_args(args, contract)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA requested but unavailable: {device}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    vq_wandb_audit = verify_vq_wandb_audit(
        args.vq_wandb_audit, args.vq_training_manifest
    )

    run_dir = (ROOT / args.output_root / args.name).resolve()
    checkpoint_dir = run_dir / 'checkpoints'
    manifest_path = run_dir / 'training_manifest.json'
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError('resume requires the original MGT training manifest')
        existing_manifest = json.loads(manifest_path.read_text())
        wandb_run_id = existing_manifest['observability']['wandb_run_id']
    else:
        if run_dir.exists():
            raise FileExistsError(f'new InterHuman MGT run directory exists: {run_dir}')
        wandb_run_id = wandb.util.generate_id()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    vq, vq_provenance = load_hhc_interhuman_vq(
        args.vq_checkpoint, args.vq_training_manifest, device=device,
    )
    freeze_interhuman_vq_for_mgt(vq)
    model_args = transformer_arguments(contract, vq)
    transformer = build_model(
        model_args, 'interhuman_transformer', vqvae=vq
    ).to(device)
    freeze_interhuman_vq_for_mgt(vq)
    trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=tuple(contract['training']['betas']),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=contract['training']['scheduler_milestones'],
        gamma=contract['training']['scheduler_gamma'],
    )
    start_step = 0
    best_fid = float('inf')
    best_step = -1
    resume_rng_state = None
    if args.resume:
        checkpoint = torch.load(Path(args.resume).resolve(), map_location=device)
        if checkpoint.get('schema') != 'hhc_interhuman_mgt_checkpoint_v1':
            raise ValueError('unsupported InterHuman MGT resume checkpoint')
        load_trainable_transformer_state(transformer, checkpoint['transformer'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        resume_rng_state = checkpoint['rng_state']
        start_step = int(checkpoint['optimizer_step'])
        best_fid = float(checkpoint['best_validation_fid'])
        best_step = int(checkpoint['best_validation_step'])

    train_dataset = InterHumanMGTTrainDataset(base_seed=args.seed)
    sources = (
        Path(__file__).resolve(), contract_path,
        ROOT / 'configs/online_wandb_training_policy.json',
        ROOT / 'dataset/interhuman_mgt.py',
        ROOT / 'dataset/interhuman_hhc_adapter.py',
        ROOT / 'models/two_person_transformer.py',
        ROOT / 'models/interhuman_transformer.py',
        ROOT / 'models/model_tools.py',
        ROOT / 'models/motion_transformer.py',
        ROOT / 'utils/interhuman_mgt_batch.py',
        ROOT / 'utils/interhuman_mgt_training.py',
        ROOT / 'utils/interhuman_mgt_evaluation.py',
        ROOT / 'utils/interhuman_artifacts.py',
        ROOT / 'utils/motion_masks.py',
    )
    manifest = {
        'schema': 'hhc_interhuman_mgt_training_manifest_v1',
        'resolved_arguments': {
            key: value for key, value in vars(args).items()
            if key not in {'resume', 'run_until_step'}
        },
        'initial_run_until_step': int(args.run_until_step),
        'contract_path': str(contract_path),
        'contract_sha256': sha256_file(contract_path),
        'vq': vq_provenance,
        'vq_wandb_audit': vq_wandb_audit,
        'dataset': {
            'train_records': len(train_dataset),
            'validation_requests': str(
                ROOT / 'output/interhuman_validation_requests/manifest.json'
            ),
            'validation_mm_requests': str(VALIDATION_MM_PATH),
            'validation_mm_requests_sha256': sha256_file(VALIDATION_MM_PATH),
        },
        'sources': {str(path): sha256_file(path) for path in sources},
        'observability': {
            'wandb_mode': 'online', 'wandb_project': args.wandb_project,
            'wandb_run_id': wandb_run_id,
        },
    }
    if args.resume:
        manifest['initial_run_until_step'] = int(
            existing_manifest['initial_run_until_step']
        )
    manifest['current_execution_run_until_step'] = int(args.run_until_step)
    identity_payload = dict(manifest)
    identity_payload.pop('initial_run_until_step')
    identity_payload.pop('current_execution_run_until_step')
    identity_payload['observability'] = {
        key: value for key, value in manifest['observability'].items()
        if key != 'wandb_run_id'
    }
    manifest['training_identity_sha256'] = canonical_sha256(identity_payload)
    if args.resume:
        if existing_manifest['training_identity_sha256'] != manifest['training_identity_sha256']:
            raise ValueError('resume arguments/sources/VQ differ from original MGT run')
        if checkpoint['training_identity_sha256'] != manifest['training_identity_sha256']:
            raise ValueError('resume checkpoint differs from MGT training manifest')
    else:
        atomic_json(manifest_path, {**manifest, 'status': 'running'})

    total_micro_steps = args.total_steps * args.gradient_accumulation
    start_micro_step = start_step * args.gradient_accumulation
    train_sampler = InterHumanMGTDrawBatchSampler(
        train_dataset, args.micro_batch_size, total_micro_steps,
        args.seed, start_micro_step,
    )
    train_loader_generator = torch.Generator().manual_seed(args.seed + 700001)
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0,
        generator=train_loader_generator,
    )
    validation_loader = validation_request_loader(num_workers=0)
    validation_seeds_by_name = {
        row['identifier']: mm_repeat_seed(row['sample_seed'], 0)
        for row in validation_loader.dataset.requests
    }
    if len(validation_seeds_by_name) != len(validation_loader.dataset.requests):
        raise ValueError('official InterHuman validation identifiers must be unique')
    validation_mm_payload = json.loads(VALIDATION_MM_PATH.read_text())

    clip_base, _ = clip.load('ViT-B/32', device=device, jit=False)
    clip.model.convert_weights(clip_base)
    clip_encoder = TextCLIP(clip_base.eval())
    for parameter in clip_encoder.parameters():
        parameter.requires_grad_(False)
    evaluator = InterHumanEvaluatorWrapper.from_official_checkpoint(
        args.evaluator_device
    )
    metrics_module = load_official_metrics_module()
    wandb.init(
        project=args.wandb_project, group='mgt_interhuman_hhc', name=args.name,
        id=wandb_run_id, resume='allow', mode='online', config=vars(args),
    )
    # Runtime construction (especially CLIP/InterCLIP module initialization)
    # consumes RNG. Restore only after every such object exists so the first
    # resumed corruption mask is exactly the saved next draw.
    if resume_rng_state is not None:
        restore_rng_state(resume_rng_state)
    clipper = GradientClipTelemetry(args.gradient_clip)
    throughput = ThroughputMeter()

    def save_checkpoint(path, step):
        atomic_torch_save(path, {
            'schema': 'hhc_interhuman_mgt_checkpoint_v1',
            'transformer': trainable_transformer_state(transformer),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'rng_state': capture_rng_state(),
            'optimizer_step': int(step),
            'best_validation_fid': float(best_fid),
            'best_validation_step': int(best_step),
            'training_identity_sha256': manifest['training_identity_sha256'],
            'vq_checkpoint_sha256': vq_provenance['checkpoint_sha256'],
        })

    def validate(step):
        @torch.no_grad()
        def validation_objective():
            totals = {}
            samples = 0
            mask_generator = torch.Generator(device=device).manual_seed(20260131)
            batch_size = args.validation_generation_batch_size
            for names, texts, motion1, motion2, lengths in validation_loader:
                for start in range(0, len(texts), batch_size):
                    end = min(start + batch_size, len(texts))
                    mini_batch = (
                        list(names[start:end]), list(texts[start:end]),
                        motion1[start:end], motion2[start:end], lengths[start:end],
                    )
                    prepared = prepare_interhuman_mgt_batch(
                        vq, mini_batch, device, num_codes=model_args.nb_code,
                        pkeep=0.5,
                        paper_architecture=model_args.paper_architecture,
                        generator=mask_generator,
                    )
                    text_tokens = clip.tokenize(
                        prepared['captions'], truncate=True
                    ).to(device)
                    clip_feature, words = clip_encoder(text_tokens)
                    with torch.autocast(
                        device_type='cuda', dtype=torch.bfloat16,
                        enabled=args.precision == 'bf16' and device.type == 'cuda',
                    ):
                        logits = transformer(
                            prepared['masked_input'], clip_feature,
                            src_mask=prepared['attention_mask'], word_emb=words,
                        )
                        values = interhuman_mgt_objective(logits.float(), prepared)
                    count = end - start
                    for key, value in values.items():
                        totals[key] = totals.get(key, 0.0) + float(value) * count
                    samples += count
            if samples != 480:
                raise ValueError(
                    f'InterHuman MGT val loss requires 480 samples, got {samples}'
                )
            return {f'objective_{key}': value / samples for key, value in totals.items()}

        def generate_batch(texts, lengths, sample_seeds=None):
            first, second = [], []
            batch_size = args.validation_generation_batch_size
            for start in range(0, len(texts), batch_size):
                end = min(start + batch_size, len(texts))
                text_tokens = clip.tokenize(texts[start:end], truncate=True).to(device)
                clip_feature, words = clip_encoder(text_tokens)
                generated1, generated2 = generate_interhuman_pair(
                    transformer, vq, clip_feature, words, lengths[start:end].to(device),
                    sampling_steps=contract['automatic_validation']['outer_maskgit_steps'],
                    temperature=contract['automatic_validation']['temperature'],
                    cfg_scale=contract['automatic_validation']['cfg_scale'],
                    sample_seeds=(
                        None if sample_seeds is None else sample_seeds[start:end]
                    ),
                )
                first.append(generated1.cpu())
                second.append(generated2.cpu())
            return torch.cat(first), torch.cat(second)

        with preserving_rng_state(), evaluating(transformer), evaluating(vq):
            result = validation_objective()
            with temporary_conditioning_contract(
                transformer, contract['model']['conditioning_contract_inference']
            ):
                result.update(evaluate_interhuman_generator(
                    generate_batch, validation_loader, evaluator, metrics_module,
                    random_seed=20260130,
                    sample_seeds_by_name=validation_seeds_by_name,
                ))
                result.update(evaluate_interhuman_validation_multimodality(
                    lambda texts, lengths, repeat: generate_batch(
                        texts, lengths,
                        [
                            mm_repeat_seed(row['sample_seed'], repeat)
                            for row in validation_mm_payload['requests']
                        ],
                    ),
                    validation_mm_payload, evaluator, metrics_module,
                    random_seed=20260132,
                ))
        wandb.log({f'val/{key}': value for key, value in result.items()}, step=step)
        return result

    optimizer.zero_grad(set_to_none=True)
    metric_sums = {}
    micro_in_update = 0
    optimizer_step = start_step
    if not start_step < args.run_until_step <= args.total_steps:
        raise ValueError(
            f'run_until_step={args.run_until_step} must exceed resume step '
            f'{start_step} and not exceed total_steps={args.total_steps}'
        )
    for batch in train_loader:
        transformer.train()
        vq.eval()
        prepared = prepare_interhuman_mgt_batch(
            vq, batch, device, num_codes=model_args.nb_code, pkeep=0.5,
            paper_architecture=model_args.paper_architecture,
        )
        text_tokens = clip.tokenize(prepared['captions'], truncate=True).to(device)
        clip_feature, words = clip_encoder(text_tokens)
        with torch.autocast(
            device_type='cuda', dtype=torch.bfloat16,
            enabled=args.precision == 'bf16' and device.type == 'cuda',
        ):
            logits = transformer(
                prepared['masked_input'], clip_feature,
                src_mask=prepared['attention_mask'], word_emb=words,
            )
            objective = interhuman_mgt_objective(logits.float(), prepared)
            scaled_loss = objective['loss'] / args.gradient_accumulation
        if not all(torch.isfinite(value).all() for value in objective.values()):
            raise FloatingPointError(
                f'non-finite InterHuman MGT objective at step {optimizer_step + 1}'
            )
        scaled_loss.backward()
        micro_in_update += 1
        throughput.update(args.micro_batch_size)
        for key, value in objective.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach())
        if micro_in_update < args.gradient_accumulation:
            continue

        optimizer_step += 1
        emit_telemetry = optimizer_step % args.telemetry_interval == 0
        telemetry = (
            parameter_and_gradient_telemetry(transformer)
            if emit_telemetry else {}
        )
        telemetry.update(clipper.clip(transformer, emit_metrics=emit_telemetry))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if emit_telemetry:
            telemetry.update({
                'train/samples_per_second': throughput.measure_and_reset(),
                'train/GPU_memory_GiB': gpu_peak_memory_gib(device),
            })
        if optimizer_step % args.log_interval == 0 or emit_telemetry:
            logged = {
                f'train/{key}': value / args.gradient_accumulation
                for key, value in metric_sums.items()
            }
            logged['train/optimizer_step'] = optimizer_step
            logged['train/learning_rate'] = optimizer.param_groups[0]['lr']
            logged.update(telemetry)
            wandb.log(logged, step=optimizer_step)
        metric_sums.clear()
        micro_in_update = 0

        if optimizer_step % args.validation_interval == 0:
            result = validate(optimizer_step)
            if result['fid'] < best_fid:
                best_fid = float(result['fid'])
                best_step = optimizer_step
                save_checkpoint(checkpoint_dir / 'best_validation_fid.pth', optimizer_step)
        if optimizer_step % args.checkpoint_interval == 0:
            save_checkpoint(checkpoint_dir / 'latest.pth', optimizer_step)
        if optimizer_step >= args.run_until_step:
            break

    if optimizer_step != args.run_until_step or micro_in_update:
        raise RuntimeError(
            f'InterHuman MGT stopped at incomplete step {optimizer_step}, '
            f'micro_in_update={micro_in_update}'
        )
    if optimizer_step < args.total_steps:
        save_checkpoint(checkpoint_dir / 'latest.pth', optimizer_step)
        selected_checkpoint = checkpoint_dir / 'best_validation_fid.pth'
        if not selected_checkpoint.is_file() or best_step < 0:
            raise RuntimeError('pilot ended without a validation-selected MGT checkpoint')
        paused = {
            **manifest,
            'status': 'pilot_decision_ready',
            'current_optimizer_step': optimizer_step,
            'best_validation_fid': best_fid,
            'best_validation_step': best_step,
            'best_validation_checkpoint': str(selected_checkpoint),
            'best_validation_checkpoint_sha256': sha256_file(selected_checkpoint),
            'resume_checkpoint': str(checkpoint_dir / 'latest.pth'),
        }
        atomic_json(manifest_path, paused)
        wandb.finish()
        return
    save_checkpoint(checkpoint_dir / 'last.pth', optimizer_step)
    selected_checkpoint = checkpoint_dir / 'best_validation_fid.pth'
    if not selected_checkpoint.is_file() or best_step < 0:
        raise RuntimeError('training ended without a validation-selected MGT checkpoint')
    completed = {
        **manifest,
        'status': 'complete',
        'completed_at_unix': time.time(),
        'best_validation_fid': best_fid,
        'best_validation_step': best_step,
        'best_validation_checkpoint': str(selected_checkpoint),
        'best_validation_checkpoint_sha256': sha256_file(selected_checkpoint),
    }
    atomic_json(manifest_path, completed)
    wandb.finish()


if __name__ == '__main__':
    main()
