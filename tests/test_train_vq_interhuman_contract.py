import json
from pathlib import Path
from types import SimpleNamespace

from train.train_vq_interhuman import (
    DeterministicEpochBatchSampler,
    learning_rate_multiplier,
    validate_args,
)


def test_deterministic_batch_sampler_resumes_exact_suffix():
    full = list(DeterministicEpochBatchSampler(23, 5, 11, seed=7, start_step=0))
    resumed = list(DeterministicEpochBatchSampler(23, 5, 11, seed=7, start_step=6))
    assert resumed == full[6:]
    assert all(len(set(batch)) == 5 for batch in full)
    for epoch_start in (0, 4, 8):
        epoch = full[epoch_start:epoch_start + 4]
        flattened = [item for batch in epoch for item in batch]
        assert len(flattened) == len(set(flattened))


def test_lr_schedule_has_warmup_and_reaches_zero():
    assert learning_rate_multiplier(1, 30000, 1000) == 0.001
    assert learning_rate_multiplier(1000, 30000, 1000) == 1.0
    assert 0 < learning_rate_multiplier(10000, 30000, 1000) < 1
    assert learning_rate_multiplier(30000, 30000, 1000) == 0.0


def test_launch_requires_gate_online_wandb_and_frozen_pilot_args():
    pilot = json.loads(open('configs/interhuman_vq_pilot_v1.json').read())
    values = {
        'confirm_after_interx': True,
        'wandb_mode': 'online',
        'total_steps': 30000,
        'run_until_step': 10000,
        'batch_size': 64,
        'window_size': 64,
        'window_stride': 10,
        'nb_code': 1024,
        'code_dim': 512,
        'interhuman_latent_dims': '384,128',
        'down_t': 2,
        'stride_t': 2,
        'width': 512,
        'depth': 2,
        'lr': 0.0002,
        'weight_decay': 0.0,
        'warmup_steps': 1000,
        'validation_interval': 1000,
        'output_emb_width': 512,
        'log_interval': 20,
        'telemetry_interval': 50,
        'checkpoint_interval': 1000,
    }
    validate_args(SimpleNamespace(**values), pilot)
    validate_args(SimpleNamespace(**dict(values, run_until_step=30000)), pilot)
    for key, bad in (
        ('confirm_after_interx', False),
        ('wandb_mode', 'offline'),
        ('nb_code', 4096),
        ('run_until_step', 20000),
    ):
        changed = dict(values, **{key: bad})
        try:
            validate_args(SimpleNamespace(**changed), pilot)
        except ValueError:
            pass
        else:
            raise AssertionError(f'launch drift {key}={bad!r} was accepted')


def test_vq_pilot_is_a_pause_inside_one_fixed_full_schedule():
    pilot = json.loads(open('configs/interhuman_vq_pilot_v1.json').read())
    assert pilot['optimization']['pilot_optimizer_steps'] == 10000
    assert pilot['optimization']['full_optimizer_steps'] == 30000
    assert pilot['optimization']['full_optimizer_steps'] > 20608
    source = open('train/train_vq_interhuman.py').read()
    assert "default=30000" in source
    assert "default=10000" in source
    assert "if key not in {'resume', 'run_until_step'}" in source
    assert "if step >= args.run_until_step" in source
    assert "'status': 'pilot_decision_ready'" in source
    assert "'resume_checkpoint': str(checkpoint_dir / 'latest.pth')" in source
    assert source.count("record['train/optimizer_step'] = step") == 1


def test_interhuman_vq_wandb_profile_matches_emitted_metric_names():
    policy = json.loads(Path('configs/online_wandb_training_policy.json').read_text())
    profile = policy['metric_profiles']['interhuman_vq']
    expected_train = {
        *(f'train/{name}' for name in (
            'total', 'reconstruction', 'explicit_position', 'position_velocity',
            'bone_length', 'rotation_geodesic', 'foot_sliding', 'commitment',
        )),
        *(f'train/perplexity_{name}' for name in ('body', 'root')),
        *(f'train/codebook_{name}_utilization' for name in ('body', 'root')),
        'train/learning_rate', 'train/gradient_l2_norm', 'train/gradient_max_abs',
        'train/weight_l2_norm', 'train/weight_max_abs',
        'train/gradient_to_weight_l2_ratio', 'train/gradient_value_count',
        'train/weight_value_count', 'train/nonfinite_gradients',
        'train/nonfinite_weights', 'train/gradient_l2_norm_post_clip',
        'train/gradient_clip_hit_rate', 'train/pairs_per_second',
        'train/optimizer_steps_per_second', 'train/gpu_peak_memory_gib',
    }
    expected_val = {
        *(f'val/{name}' for name in (
            'total', 'reconstruction', 'explicit_position', 'position_velocity',
            'bone_length', 'rotation_geodesic', 'foot_sliding', 'commitment',
            'fid', 'matching', 'top1', 'top2', 'top3', 'diversity', 'mpjpe_m',
            'normalized_feature_mae',
        )),
    }
    assert set(profile['required_training_metrics']) == expected_train
    assert set(profile['required_validation_metrics']) == expected_val


def test_archived_dense_pilot_keeps_training_protocol_and_model_shape():
    import sys
    from unittest.mock import patch
    from train.train_vq_interhuman import parse_args
    from models.model_tools import build_model

    dense = json.loads(Path('configs/interhuman_vq_pilot_dense_v4.json').read_text())
    previous = json.loads(Path('configs/interhuman_vq_pilot_ema_v3.json').read_text())
    for key in ('data', 'optimization', 'loss', 'automatic_validation', 'wandb'):
        assert dense[key] == previous[key]
    with patch.object(sys, 'argv', ['train', '--confirm-after-interx', '--down-t', '1',
                                    '--interhuman-quantizer', 'fp32_window_reset_v1']):
        args = parse_args()
    # v4 was rejected at 10k; its launch source is preserved in its run archive.
    # The active trainer is restored byte-for-byte to the selected v3 source.
    model = build_model(args, 'interhuman_vq')
    assert model.temporal_downsample == 2 and len(model.quantizers) == 2
    assert sum(p.numel() for p in model.parameters()) == dense['model']['parameters'] == 13154828
