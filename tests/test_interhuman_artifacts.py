import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from models.vqvae_sep3_interhuman import VQVAE_SEP3_InterHuman
from models.interhuman_transformer import InterHumanText2MotionTransformer
from utils.interhuman_artifacts import (
    load_hhc_interhuman_mgt,
    load_hhc_interhuman_vq,
    sha256_file,
)


def _fixture(root):
    arguments = {
        'nb_code': 16, 'code_dim': 32, 'output_emb_width': 32,
        'down_t': 2, 'stride_t': 2, 'width': 32, 'depth': 1,
        'dilation_growth_rate': 2, 'vq_act': 'relu', 'vq_norm': None,
        'mu': 0.99, 'interhuman_latent_dims': '24,8',
    }
    model = VQVAE_SEP3_InterHuman(
        SimpleNamespace(**arguments), arguments['nb_code'], arguments['code_dim'],
        arguments['output_emb_width'], arguments['down_t'], arguments['stride_t'],
        arguments['width'], arguments['depth'], arguments['dilation_growth_rate'],
        arguments['vq_act'], arguments['vq_norm'],
    )
    manifest = {
        'schema': 'hhc_interhuman_vq_training_manifest_v1',
        'contract_sha256': 'contract',
        'resolved_arguments': arguments,
        'sources': {},
    }
    checkpoint_path = root / 'best.pth'
    torch.save({
        'schema': 'hhc_interhuman_vq_checkpoint_v1',
        'net': model.state_dict(),
        'training_contract_sha256': 'contract',
        'optimizer_step': 10,
        'best_validation_fid': 0.25,
        'best_validation_step': 10,
    }, checkpoint_path)
    manifest.update({
        'status': 'complete',
        'best_validation_checkpoint': str(checkpoint_path),
        'best_validation_checkpoint_sha256': sha256_file(checkpoint_path),
        'best_validation_fid': 0.25,
        'best_validation_step': 10,
    })
    manifest_path = root / 'training_manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    return checkpoint_path, manifest_path


def test_hhc_interhuman_vq_loader_reconstructs_exact_frozen_model():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint, manifest = _fixture(Path(directory))
        model, provenance = load_hhc_interhuman_vq(checkpoint, manifest)
        assert model.num_code == 16 and model.latent_dims == (24, 8)
        assert all(not parameter.requires_grad for parameter in model.parameters())
        assert provenance['optimizer_step'] == 10
        assert provenance['best_validation_fid'] == 0.25


def test_hhc_interhuman_vq_loader_rejects_manifest_drift():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint, manifest = _fixture(Path(directory))
        payload = json.loads(manifest.read_text())
        payload['contract_sha256'] = 'different'
        manifest.write_text(json.dumps(payload))
        try:
            load_hhc_interhuman_vq(checkpoint, manifest)
        except ValueError as exc:
            assert 'checkpoint/manifest contract mismatch' in str(exc)
        else:
            raise AssertionError('mismatched InterHuman VQ artifact was accepted')


def test_hhc_interhuman_vq_loader_rejects_pilot_or_nonselected_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint, manifest = _fixture(Path(directory))
        payload = json.loads(manifest.read_text())
        payload['status'] = 'pilot_decision_ready'
        manifest.write_text(json.dumps(payload))
        try:
            load_hhc_interhuman_vq(checkpoint, manifest)
        except ValueError as exc:
            assert 'completed full endpoint' in str(exc)
        else:
            raise AssertionError('pilot-only InterHuman VQ was accepted downstream')


def test_hhc_interhuman_mgt_loader_binds_checkpoint_manifest_and_vq():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        vq_checkpoint, vq_manifest = _fixture(root)
        vq, vq_info = load_hhc_interhuman_vq(vq_checkpoint, vq_manifest)
        model_values = {
            'codes_per_channel': 16, 'embed_dim': 32, 'block_size': 75,
            'base_layers_total': 1, 'local_cross_attention_layers': 1,
            'attention_heads': 4, 'dropout': 0.0, 'feedforward_rate': 2,
            'paper_architecture': False,
            'conditioning_contract_train': 'historical_mmm_shape_dropout',
            'conditioning_contract_inference': 'paper_text_cfg',
        }
        contract_path = root / 'mgt_contract.json'
        contract_path.write_text(json.dumps({'model': model_values}))
        transformer = InterHumanText2MotionTransformer(
            vqvae=vq, num_vq=16, embed_dim=32, clip_dim=512,
            block_size=75, num_layers=1, num_local_layer=1, n_head=4,
            drop_out_rate=0.0, fc_rate=2, contact=False, shape=False,
            paper_architecture=False,
            conditioning_contract='historical_mmm_shape_dropout',
        )
        transformer_state = {
            key: value for key, value in transformer.state_dict().items()
            if not key.startswith('trans_base.vqvae.')
        }
        mgt_manifest = root / 'mgt_manifest.json'
        vq_wandb_audit = root / 'vq_wandb_audit.json'
        vq_wandb_audit.write_text('{}')
        mgt_manifest.write_text(json.dumps({
            'schema': 'hhc_interhuman_mgt_training_manifest_v1',
            'contract_path': str(contract_path),
            'contract_sha256': sha256_file(contract_path),
            'training_identity_sha256': 'mgt-identity',
            'sources': {}, 'vq': vq_info,
            'vq_wandb_audit': {
                'path': str(vq_wandb_audit),
                'sha256': sha256_file(vq_wandb_audit),
            },
        }))
        mgt_checkpoint = root / 'mgt.pth'
        torch.save({
            'schema': 'hhc_interhuman_mgt_checkpoint_v1',
            'transformer': transformer_state,
            'training_identity_sha256': 'mgt-identity',
            'vq_checkpoint_sha256': vq_info['checkpoint_sha256'],
            'optimizer_step': 20, 'best_validation_fid': 0.4,
            'best_validation_step': 20,
        }, mgt_checkpoint)
        loaded, loaded_vq, provenance = load_hhc_interhuman_mgt(
            mgt_checkpoint, mgt_manifest
        )
        assert loaded.trans_base.vqvae is loaded_vq
        assert loaded.conditioning_contract == 'paper_text_cfg'
        assert loaded.trans_base.conditioning_contract == 'paper_text_cfg'
        assert all(not parameter.requires_grad for parameter in loaded.parameters())
        assert provenance['optimizer_step'] == 20
