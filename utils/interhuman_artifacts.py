"""Fail-closed loaders for HHC InterHuman training artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models.model_tools import build_model


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest_sources(manifest):
    changed = {}
    for path_string, expected in manifest.get('sources', {}).items():
        path = Path(path_string)
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            changed[path_string] = {'expected': expected, 'actual': actual}
    if changed:
        raise ValueError(f'InterHuman training sources changed: {changed}')


def load_hhc_interhuman_vq(
    checkpoint_path,
    training_manifest_path,
    *,
    device='cpu',
    verify_sources=True,
):
    """Strictly reconstruct a VQ from its immutable training manifest."""
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest_path = Path(training_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'hhc_interhuman_vq_training_manifest_v1':
        raise ValueError('unsupported HHC InterHuman VQ training manifest')
    if manifest.get('status') != 'complete':
        raise ValueError('HHC InterHuman VQ is not at its completed full endpoint')
    selected_path = Path(manifest.get('best_validation_checkpoint', '')).resolve()
    if checkpoint_path != selected_path:
        raise ValueError('requested InterHuman VQ is not the manifest-selected checkpoint')
    if sha256_file(checkpoint_path) != manifest.get('best_validation_checkpoint_sha256'):
        raise ValueError('manifest-selected InterHuman VQ checkpoint hash changed')
    if verify_sources:
        _verify_manifest_sources(manifest)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint.get('schema') != 'hhc_interhuman_vq_checkpoint_v1':
        raise ValueError('unsupported HHC InterHuman VQ checkpoint')
    if checkpoint.get('training_contract_sha256') != manifest.get('contract_sha256'):
        raise ValueError('InterHuman VQ checkpoint/manifest contract mismatch')
    if int(checkpoint.get('optimizer_step', -1)) != int(manifest['best_validation_step']):
        raise ValueError('InterHuman VQ checkpoint is not from the manifest best step')
    if float(checkpoint.get('best_validation_fid', float('nan'))) != float(
        manifest['best_validation_fid']
    ):
        raise ValueError('InterHuman VQ checkpoint embeds another best FID')
    arguments = dict(manifest.get('resolved_arguments', {}))
    required = {
        'nb_code', 'code_dim', 'output_emb_width', 'down_t', 'stride_t',
        'width', 'depth', 'dilation_growth_rate', 'vq_act', 'vq_norm',
        'mu', 'interhuman_latent_dims',
    }
    missing = sorted(required - set(arguments))
    if missing:
        raise ValueError(f'InterHuman VQ manifest lacks model arguments: {missing}')
    model = build_model(SimpleNamespace(**arguments), 'interhuman_vq')
    model.load_state_dict(checkpoint['net'], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    provenance = {
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'training_manifest': str(manifest_path),
        'training_manifest_sha256': sha256_file(manifest_path),
        'training_contract_sha256': manifest['contract_sha256'],
        'optimizer_step': int(checkpoint['optimizer_step']),
        'best_validation_fid': float(checkpoint['best_validation_fid']),
        'best_validation_step': int(checkpoint['best_validation_step']),
    }
    return model, provenance


def load_hhc_interhuman_mgt(
    checkpoint_path,
    training_manifest_path,
    *,
    device='cpu',
    verify_sources=True,
    inference_conditioning=True,
):
    """Strictly reconstruct the MGT and its separately bound frozen VQ."""
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest_path = Path(training_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'hhc_interhuman_mgt_training_manifest_v1':
        raise ValueError('unsupported HHC InterHuman MGT training manifest')
    if verify_sources:
        _verify_manifest_sources(manifest)
    contract_path = Path(manifest['contract_path']).resolve()
    if sha256_file(contract_path) != manifest['contract_sha256']:
        raise ValueError('InterHuman MGT preregistration contract changed')
    contract = json.loads(contract_path.read_text())
    vq_wandb_audit = manifest.get('vq_wandb_audit', {})
    vq_wandb_audit_path = Path(vq_wandb_audit.get('path', '')).resolve()
    if (
        not vq_wandb_audit_path.is_file()
        or sha256_file(vq_wandb_audit_path) != vq_wandb_audit.get('sha256')
    ):
        raise ValueError('InterHuman MGT-bound VQ W&B audit changed')
    vq_info = manifest['vq']
    for key, path_key in (
        ('checkpoint_sha256', 'checkpoint'),
        ('training_manifest_sha256', 'training_manifest'),
    ):
        if sha256_file(vq_info[path_key]) != vq_info[key]:
            raise ValueError(f'InterHuman MGT-bound VQ {path_key} changed')
    vq, loaded_vq_info = load_hhc_interhuman_vq(
        vq_info['checkpoint'], vq_info['training_manifest'], device=device,
        verify_sources=verify_sources,
    )
    if loaded_vq_info['checkpoint_sha256'] != vq_info['checkpoint_sha256']:
        raise ValueError('InterHuman MGT resolved a different VQ checkpoint')
    model_values = contract['model']
    if int(vq.num_code) != int(model_values['codes_per_channel']):
        raise ValueError('InterHuman MGT/VQ codebook sizes differ')
    arguments = SimpleNamespace(
        nb_code=int(model_values['codes_per_channel']),
        embed_dim_gpt=int(model_values['embed_dim']), clip_dim=512,
        block_size=int(model_values['block_size']),
        num_layers=int(model_values['base_layers_total']),
        num_local_layer=int(model_values['local_cross_attention_layers']),
        n_head_gpt=int(model_values['attention_heads']),
        drop_out_rate=float(model_values['dropout']),
        ff_rate=int(model_values['feedforward_rate']),
        contact_conditioning=False, shape=False,
        paper_architecture=bool(model_values['paper_architecture']),
        conditioning_contract=model_values['conditioning_contract_train'],
    )
    transformer = build_model(
        arguments, 'interhuman_transformer', vqvae=vq
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint.get('schema') != 'hhc_interhuman_mgt_checkpoint_v1':
        raise ValueError('unsupported HHC InterHuman MGT checkpoint')
    if checkpoint.get('training_identity_sha256') != manifest.get('training_identity_sha256'):
        raise ValueError('InterHuman MGT checkpoint/manifest identity mismatch')
    if checkpoint.get('vq_checkpoint_sha256') != vq_info['checkpoint_sha256']:
        raise ValueError('InterHuman MGT checkpoint was trained with another VQ')
    result = transformer.load_state_dict(checkpoint['transformer'], strict=False)
    invalid_missing = [
        key for key in result.missing_keys
        if not key.startswith('trans_base.vqvae.')
    ]
    if invalid_missing or result.unexpected_keys:
        raise ValueError(
            f'InterHuman MGT state mismatch: missing={invalid_missing}, '
            f'unexpected={list(result.unexpected_keys)}'
        )
    if inference_conditioning:
        inference_contract = model_values['conditioning_contract_inference']
        transformer.conditioning_contract = inference_contract
        transformer.trans_base.conditioning_contract = inference_contract
    transformer.eval()
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    provenance = {
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'training_manifest': str(manifest_path),
        'training_manifest_sha256': sha256_file(manifest_path),
        'training_identity_sha256': manifest['training_identity_sha256'],
        'optimizer_step': int(checkpoint['optimizer_step']),
        'best_validation_fid': float(checkpoint['best_validation_fid']),
        'best_validation_step': int(checkpoint['best_validation_step']),
        'vq': loaded_vq_info,
        'vq_wandb_audit': vq_wandb_audit,
        'model_contract': model_values,
    }
    return transformer, vq, provenance
