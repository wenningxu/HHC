import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.interhuman_transformer import InterHumanText2MotionTransformer
from models.vqvae_sep3_interhuman import VQVAE_SEP3_InterHuman
from train.train_mgt_interhuman import (
    load_trainable_transformer_state,
    trainable_transformer_state,
    validate_args,
    verify_vq_wandb_audit,
)
from tools.verify_selected_interhuman_mgt import verify_selected_checkpoint
from utils.interhuman_artifacts import sha256_file


CONTRACT = json.loads(Path('configs/interhuman_mgt_pilot_v1.json').read_text())


def _valid_args():
    return SimpleNamespace(
        confirm_after_interx=True,
        wandb_mode='online',
        seed=CONTRACT['training']['seed'],
        total_steps=CONTRACT['training']['full_optimizer_steps'],
        run_until_step=CONTRACT['training']['pilot_decision_step'],
        micro_batch_size=CONTRACT['training']['micro_batch_pairs'],
        gradient_accumulation=CONTRACT['training']['gradient_accumulation'],
        lr=CONTRACT['training']['learning_rate'],
        weight_decay=CONTRACT['training']['weight_decay'],
        gradient_clip=CONTRACT['training']['gradient_clip_norm'],
        precision=CONTRACT['training']['precision'],
        validation_interval=CONTRACT['automatic_validation']['interval_optimizer_steps'],
        checkpoint_interval=CONTRACT['training']['checkpoint_interval_optimizer_steps'],
        num_workers=0, validation_generation_batch_size=2,
        telemetry_interval=50, log_interval=20,
    )


def _small_transformer():
    vq_args = SimpleNamespace(mu=0.99, interhuman_latent_dims='24,8')
    vq = VQVAE_SEP3_InterHuman(
        vq_args, nb_code=16, code_dim=32, output_emb_width=32,
        down_t=2, stride_t=2, width=32, depth=1,
        dilation_growth_rate=2,
    )
    for parameter in vq.parameters():
        parameter.requires_grad_(False)
    return InterHumanText2MotionTransformer(
        vqvae=vq, num_vq=16, embed_dim=32, clip_dim=32,
        block_size=75, num_layers=1, num_local_layer=1, n_head=4,
        drop_out_rate=0.0, fc_rate=2, contact=False, shape=False,
        paper_architecture=False,
        conditioning_contract='historical_mmm_shape_dropout',
    )


def test_interhuman_mgt_trainer_requires_gate_online_and_frozen_args():
    args = _valid_args()
    validate_args(args, CONTRACT)
    args.confirm_after_interx = False
    try:
        validate_args(args, CONTRACT)
    except ValueError as exc:
        assert 'gated' in str(exc)
    else:
        raise AssertionError('InterHuman MGT launch gate was bypassed')
    args.confirm_after_interx = True
    args.wandb_mode = 'offline'
    try:
        validate_args(args, CONTRACT)
    except ValueError as exc:
        assert 'online W&B' in str(exc)
    else:
        raise AssertionError('offline InterHuman MGT logging was accepted')


def test_interhuman_mgt_has_only_preregistered_pilot_and_full_stop_points():
    args = _valid_args()
    validate_args(args, CONTRACT)
    args.run_until_step = CONTRACT['training']['full_optimizer_steps']
    validate_args(args, CONTRACT)
    args.run_until_step = 60000
    try:
        validate_args(args, CONTRACT)
    except ValueError as exc:
        assert 'pilot decision or full endpoint' in str(exc)
    else:
        raise AssertionError('unregistered InterHuman MGT stop point was accepted')


def test_mgt_checkpoint_state_excludes_external_frozen_vq_and_loads_strictly():
    source = _small_transformer()
    state = trainable_transformer_state(source)
    assert state and not any(key.startswith('trans_base.vqvae.') for key in state)
    destination = _small_transformer()
    load_trainable_transformer_state(destination, state)
    destination_state = destination.state_dict()
    assert all(torch.equal(value, destination_state[key]) for key, value in state.items())


def test_mgt_trainer_contains_required_online_validation_and_telemetry_paths():
    source = Path('train/train_mgt_interhuman.py').read_text()
    for required in (
        "mode='online'", 'parameter_and_gradient_telemetry',
        'GradientClipTelemetry', 'evaluate_interhuman_generator',
        'evaluate_interhuman_validation_multimodality',
        'validation_objective', "f'objective_{key}'",
        "'best_validation_fid.pth'", "'latest.pth'",
        'training_identity_sha256', 'preserving_rng_state',
        "logged['train/optimizer_step'] = optimizer_step",
        "ROOT / 'models/interhuman_transformer.py'",
        "ROOT / 'models/model_tools.py'",
        "ROOT / 'configs/online_wandb_training_policy.json'",
    ):
        assert required in source


def test_resume_rng_boundary_follows_runtime_init_and_loader_rng_is_isolated():
    for path in ('train/train_mgt_interhuman.py', 'train/train_vq_interhuman.py'):
        source = Path(path).read_text()
        restore = source.rindex('restore_rng_state(resume_rng_state)')
        assert restore > source.rindex('wandb.init(')
        assert restore < source.rindex('GradientClipTelemetry(')
        assert 'generator=train_loader_generator' in source

    torch.manual_seed(12345)
    before = torch.random.get_rng_state().clone()
    loader = DataLoader(
        TensorDataset(torch.arange(4)), batch_size=2,
        generator=torch.Generator().manual_seed(99),
    )
    next(iter(loader))
    assert torch.equal(torch.random.get_rng_state(), before)


def test_formal_selection_is_manifest_and_val_fid_bound():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint_path = root / 'best_validation_fid.pth'
        manifest_path = root / 'training_manifest.json'
        torch.save({
            'schema': 'hhc_interhuman_mgt_checkpoint_v1',
            'optimizer_step': 30000,
            'best_validation_step': 30000,
            'best_validation_fid': 0.25,
            'training_identity_sha256': 'training-id',
            'vq_checkpoint_sha256': 'vq-id',
        }, checkpoint_path)
        manifest = {
            'schema': 'hhc_interhuman_mgt_training_manifest_v1',
            'status': 'complete',
            'best_validation_checkpoint': str(checkpoint_path),
            'best_validation_checkpoint_sha256': sha256_file(checkpoint_path),
            'best_validation_step': 30000,
            'best_validation_fid': 0.25,
            'training_identity_sha256': 'training-id',
            'vq': {'checkpoint_sha256': 'vq-id'},
        }
        manifest_path.write_text(json.dumps(manifest))
        result = verify_selected_checkpoint(checkpoint_path, manifest_path)
        assert result['status'] == 'pass'
        assert result['selection_split'] == 'official_interhuman_val_only'

        manifest['status'] = 'pilot_decision_ready'
        manifest_path.write_text(json.dumps(manifest))
        try:
            verify_selected_checkpoint(checkpoint_path, manifest_path)
        except ValueError as exc:
            assert 'not evaluation-ready' in str(exc)
        else:
            raise AssertionError('formal test accepted a pilot-only MGT manifest')


def test_interhuman_finalizer_verifies_selected_checkpoint_first():
    source = Path('tools/finalize_hhc_interhuman_common50.sh').read_text()
    verifier = source.index('tools/verify_selected_interhuman_mgt.py')
    exporter = source.index('tools/export_hhc_interhuman.py')
    assert verifier < exporter
    audit = source.index('tools/audit_wandb_online_metric_history.py')
    assert audit < verifier
    assert '--metric-profile interhuman_mgt' in source
    assert '--expected-final-step 300000' in source
    trainer = Path('train/train_mgt_interhuman.py').read_text()
    assert "'best_validation_checkpoint_sha256': sha256_file(selected_checkpoint)" in trainer


def test_mgt_requires_completed_vq_online_wandb_evidence():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / 'vq_manifest.json'
        manifest.write_text('{}')
        audit = root / 'vq_wandb_audit.json'
        audit.write_text(json.dumps({
            'schema': 'hhc_wandb_online_metric_history_audit_v1',
            'status': 'pass',
            'metric_profile': 'interhuman_vq',
            'expected_final_step': 30000,
            'remote_state': 'finished',
            'run_path': 'entity/NewInter/run',
            'manifest': {
                'path': str(manifest.resolve()),
                'sha256': sha256_file(manifest),
            },
        }))
        result = verify_vq_wandb_audit(audit, manifest)
        assert result['expected_final_step'] == 30000
        payload = json.loads(audit.read_text())
        payload['remote_state'] = 'running'
        audit.write_text(json.dumps(payload))
        try:
            verify_vq_wandb_audit(audit, manifest)
        except ValueError as exc:
            assert 'not valid for MGT' in str(exc)
        else:
            raise AssertionError('MGT accepted unfinished VQ W&B history')


def test_interhuman_mgt_wandb_profile_matches_emitted_metric_names():
    policy = json.loads(Path('configs/online_wandb_training_policy.json').read_text())
    profile = policy['metric_profiles']['interhuman_mgt']
    expected_train = {
        *(f'train/{name}' for name in (
            'loss', 'accuracy_valid', 'accuracy_masked', 'accuracy_body',
            'accuracy_root', 'masked_fraction',
        )),
        'train/learning_rate', 'train/gradient_l2_norm', 'train/gradient_max_abs',
        'train/weight_l2_norm', 'train/weight_max_abs',
        'train/gradient_to_weight_l2_ratio', 'train/gradient_value_count',
        'train/weight_value_count', 'train/nonfinite_gradients',
        'train/nonfinite_weights', 'train/gradient_l2_norm_post_clip',
        'train/gradient_clip_hit_rate', 'train/samples_per_second',
        'train/GPU_memory_GiB',
    }
    expected_val = {
        *(f'val/objective_{name}' for name in (
            'loss', 'accuracy_valid', 'accuracy_masked', 'accuracy_body',
            'accuracy_root',
        )),
        *(f'val/{name}' for name in (
            'fid', 'matching', 'top1', 'top2', 'top3', 'diversity',
            'multimodality',
        )),
    }
    assert set(profile['required_training_metrics']) == expected_train
    assert set(profile['required_validation_metrics']) == expected_val
