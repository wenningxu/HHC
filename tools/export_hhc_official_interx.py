#!/usr/bin/env python
"""Export paper HHC generations for a frozen official Inter-X request file."""

import argparse
import hashlib
import json
import os
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clip
import h5py
import numpy as np
import torch

from models.model_tools import build_model
from train.train_t2i_trans import (
    TextCLIP,
    load_transformer_checkpoint,
    merge_interx_lookup_weights,
)
from utils.eval_trans import downsample_contact_mask
from utils.motion_process import MotionNormalizerTorchInterX


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_conditioning_contracts(args):
    """Keep training dropout identity separate from inference-time CFG."""
    training_contract = (
        'historical_mmm_shape_dropout'
        if args.mgt_contract == 'historical_mmm'
        else 'paper_text_cfg'
    )
    inference_contract = (
        training_contract
        if args.inference_conditioning_contract == 'training_contract'
        else args.inference_conditioning_contract
    )
    return training_contract, inference_contract


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resume_attributes(source, expected):
    for key, default in (('confidence_noise_scale', 0.0), ('confidence_noise_scope', 'all'),
                         ('confidence_noise_until_step', 0),
                         ('confidence_ranking', 'probability'), ('channel_schedule', 'interleaved'),
                         ('save_token_ids', False), ('root_replay_control', False)):
        if source.attrs.get(key, default) != expected.get(key, default):
            raise ValueError(f'resume {key} mismatch')
    legacy = source.attrs.get('beta_policy', 'requested')
    for key in ('mgt_beta_policy', 'vq_beta_policy'):
        if source.attrs.get(key, legacy) != expected.get(key, expected.get('beta_policy', 'requested')):
            raise ValueError(f'resume beta policy mismatch ({key})')
    if source.attrs.get('beta_policy', 'requested') != expected.get('beta_policy', 'requested'):
        raise ValueError('resume beta policy mismatch')
    mismatches = {
        key: (source.attrs.get(key), value)
        for key, value in expected.items()
        if key not in source.attrs or source.attrs.get(key) != value
    }
    if mismatches:
        raise ValueError(f'resume provenance mismatch: {mismatches}')


def generation_betas(betas, policy):
    if policy == 'requested':
        return betas
    if policy == 'zero':
        return np.zeros_like(betas)
    raise ValueError(f'unknown beta policy: {policy}')


def resolve_beta_policies(beta_policy, mgt_beta_policy=None, vq_beta_policy=None):
    return (mgt_beta_policy or beta_policy, vq_beta_policy or beta_policy)


def token_archive(target, shape, resume, name='token_ids'):
    if resume:
        stored = target.get(name)
        if stored is None or stored.shape != shape or stored.dtype != np.dtype('int32'):
            raise ValueError('resume token archive is missing or has incompatible shape/dtype')
        return stored
    return target.create_dataset(name, shape=shape, dtype=np.int32,
                                 chunks=(1, 1, shape[2], shape[3]), compression='lzf')


def validate_contact_map_provenance(
    vq_checkpoint, contact_map, stats_dir='dataset/interx',
    stats_prefix='Interx', contact_map_contract='paper_formula_train_only',
):
    if contact_map_contract == 'mmm_legacy_full_axis_bug':
        audit_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'output', 'hhc_contact_guidance_map_audit.json',
        )
        if not os.path.isfile(audit_path):
            raise FileNotFoundError(
                f'legacy contact-map audit not found: {audit_path}'
            )
        with open(audit_path, 'r', encoding='utf-8') as handle:
            audit = json.load(handle)
        legacy = audit.get('mmm_legacy_full_axis_bug_map', {})
        policy = audit.get('policy', {})
        vq = audit.get('vq_checkpoint', {})
        audit_script = audit.get('audit_script', {})
        legacy_source = legacy.get('source_implementation', {})
        legacy_source_path = str(legacy_source.get('path') or '')
        expected_audit_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'audit_hhc_contact_guidance_maps.py',
        )
        expected = {
            'schema': (audit.get('schema'), 'hhc_contact_guidance_map_audit_v1'),
            'status': (audit.get('status'), 'pass'),
            'audit_script_path': (
                os.path.abspath(audit_script.get('path', '')),
                os.path.abspath(expected_audit_script),
            ),
            'audit_script_sha256': (
                audit_script.get('sha256'), sha256_file(expected_audit_script)
            ),
            'legacy_source_implementation_sha256': (
                legacy_source.get('sha256'),
                sha256_file(legacy_source_path)
                if os.path.isfile(legacy_source_path) else None,
            ),
            'copied_path': (
                os.path.abspath(legacy.get('copied_path', '')),
                os.path.abspath(contact_map),
            ),
            'contact_map_sha256': (
                legacy.get('sha256'), sha256_file(contact_map)
            ),
            'copy_byte_identical': (
                legacy.get('copy_byte_identical'), True
            ),
            'exact_bug_replay_match': (
                legacy.get('exact_bug_replay_match'), True
            ),
            'uses_test_split': (legacy.get('uses_test_split'), True),
            'test_samples_used': (legacy.get('test_samples_used'), 1706),
            'vq_checkpoint_path': (
                os.path.abspath(vq.get('path', '')),
                os.path.abspath(vq_checkpoint),
            ),
            'vq_checkpoint_sha256': (
                vq.get('sha256'), sha256_file(vq_checkpoint)
            ),
            'fair_main_table_policy': (
                policy.get('fair_main_table'), 'paper_formula_fair_map'
            ),
        }
        mismatches = {
            key: values for key, values in expected.items()
            if values[0] != values[1]
        }
        if 'author-implementation reproduction sensitivity only' not in str(
                policy.get('legacy_map_role', '')):
            mismatches['legacy_map_role'] = (
                policy.get('legacy_map_role'),
                'must be explicitly excluded from the fair main table',
            )
        if mismatches:
            raise RuntimeError(
                f'legacy contact-guidance provenance mismatch: {mismatches}'
            )
        return audit_path
    if contact_map_contract != 'paper_formula_train_only':
        raise ValueError(
            f'unknown contact-map contract: {contact_map_contract}'
        )
    # A fixed VQ can back multiple immutable training overlays, each with its
    # own train-only contact map.  Bind provenance to the selected map's cache
    # root, then verify that cache still points to the requested VQ hash.
    contact_map_directory = os.path.dirname(os.path.abspath(contact_map))
    manifest_path = os.path.join(
        contact_map_directory, 'codebook', '_cache_manifest.json'
    )
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f'MGT cache provenance manifest not found: {manifest_path}')
    with open(manifest_path, 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    # Historical MMM token caches were produced through
    # utils.motion_process.process_motion, whose normalization contract is
    # ``<prefix>_{Mean,Std}_root.npy``.  Generation/decoding deliberately uses
    # the raw-motion ``<prefix>_{Mean,Std}.npy`` pair instead.  These are two
    # distinct contracts: bind the contact map to the statistics that created
    # its cached token IDs, while the output HDF5 metadata below independently
    # binds the generated motions to the decoder statistics.
    root_normalized_cache_contracts = {
        'historical_mmm_exact',
        'deterministic_historical_root_v1',
    }
    cache_suffix = (
        '_root'
        if manifest.get('token_cache_contract') in root_normalized_cache_contracts
        else ''
    )
    expected = {
        'status': 'complete',
        'contact_map_scope': 'train_only',
        'contact_map_estimator': 'empirical_frequency_unseen_neutral_0.5',
        'vq_checkpoint_sha256': sha256_file(vq_checkpoint),
        'contact_map_sha256': sha256_file(contact_map),
        'normalization_mean_sha256': sha256_file(os.path.join(
            stats_dir, f'{stats_prefix}_Mean{cache_suffix}.npy'
        )),
        'normalization_std_sha256': sha256_file(os.path.join(
            stats_dir, f'{stats_prefix}_Std{cache_suffix}.npy'
        )),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items() if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f'contact-guidance provenance mismatch: {mismatches}')
    return manifest_path


def resolve_contact_timing(args, request, indices, names, lengths):
    """Resolve an explicitly-labelled inference-time timing source."""
    if not args.contact_guidance:
        return (
            np.zeros((len(indices), 150), dtype=np.int64),
            'none', '', '',
        )
    source = args.contact_timing_source
    if source == 'request_oracle':
        contact = request['contact'][indices].astype(np.int64)
        source_path = os.path.abspath(args.requests)
        source_hash = sha256_file(args.requests)
    elif source == 'full_valid_weak':
        contact = np.zeros((len(indices), 150), dtype=np.int64)
        for index, valid_length in enumerate(lengths):
            contact[index, :int(valid_length)] = 1
        source_path = ''
        source_hash = ''
    elif source == 'external_text_predictor':
        if not args.contact_timing_file:
            raise ValueError(
                '--contact-timing-file is required for external_text_predictor'
            )
        timing_path = os.path.abspath(args.contact_timing_file)
        with np.load(timing_path, allow_pickle=False) as timing:
            required = {'names', 'contact', 'requests_sha256'}
            missing = sorted(required - set(timing.files))
            if missing:
                raise ValueError(
                    f'external contact timing lacks arrays: {missing}'
                )
            timing_names = timing['names'].astype(str)
            contact = timing['contact'].astype(np.int64)
            bound_request_hash = str(timing['requests_sha256'].item())
        full_request_names = request['names'].astype(str)
        if np.array_equal(timing_names, full_request_names):
            contact = contact[indices]
        elif not np.array_equal(timing_names, names):
            raise ValueError(
                'external contact timing names/order differ from either the '
                'full or selected frozen requests'
            )
        if bound_request_hash != sha256_file(args.requests):
            raise ValueError(
                'external contact timing was predicted for another request file'
            )
        source_path = timing_path
        source_hash = sha256_file(timing_path)
    else:
        raise ValueError(f'unsupported contact timing source: {source}')
    if contact.shape != (len(indices), 150):
        raise ValueError(
            f'contact timing must have shape {(len(indices), 150)}, got '
            f'{contact.shape}'
        )
    if not np.isin(contact, (0, 1)).all():
        raise ValueError('contact timing must be binary')
    for index, valid_length in enumerate(lengths):
        if np.any(contact[index, int(valid_length):]):
            raise ValueError(
                f'contact timing is nonzero after valid length for {names[index]}'
            )
    return contact, source, source_path, source_hash


PAPER_VQ_CONTRACT = {
    'dataname': 'hhi',
    'motion_rep': 'raw',
    'nb_code': 8096,
    'code_dim': 256,
    'output_emb_width': 512,
    'down_t': 2,
    'stride_t': 2,
    'width': 512,
    'depth': 3,
    'dilation_growth_rate': 3,
    'vq_act': 'relu',
    'vq_norm': None,
    'mu': 0.99,
    'quantizer': 'ema_reset',
}

# The original MMM run called this the 8192-code setting, while its serialized
# tensors contain 8096 learnable entries per body part.  Keep that historical
# name at the CLI boundary and bind inference to the exact checkpoint schema.
ORIGINAL_8192_VQ_CONTRACT = {
    **PAPER_VQ_CONTRACT,
    'code_dim': 64,
    'shape_condition_dim': 8,
}

PRECLEANUP_RAW_12288_VQ_CONTRACT = {
    **PAPER_VQ_CONTRACT,
    'nb_code': 12288,
    'shape_condition_dim': 8,
}


def paper_model_args(
    device, vq_checkpoint, vq_payload=None, contract='current',
    mgt_contract='current', mgt_shape=True,
):
    """Build inference arguments from, and bind them to, the VQ checkpoint."""
    if vq_payload is None:
        vq_payload = torch.load(
            vq_checkpoint, map_location='cpu', weights_only=False
        )
    if contract == 'original_8192':
        expected_contract = ORIGINAL_8192_VQ_CONTRACT
    elif contract == 'precleanup_raw12288':
        expected_contract = PRECLEANUP_RAW_12288_VQ_CONTRACT
    else:
        expected_contract = PAPER_VQ_CONTRACT
    saved_args = vq_payload.get('args') if isinstance(vq_payload, dict) else None
    if saved_args is None:
        if contract not in ('original_8192', 'precleanup_raw12288'):
            raise ValueError(
                'VQ checkpoint does not contain saved architecture arguments'
            )
        # The 2025-11 original checkpoint predates argument snapshots.  Its
        # codebook and conditioning tensor dimensions are verified below.
        saved_args = dict(expected_contract)
    elif not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    mismatches = {
        key: (saved_args.get(key), expected)
        for key, expected in expected_contract.items()
        if saved_args.get(key) != expected
    }
    if mismatches:
        raise ValueError(f'VQ checkpoint violates the paper contract: {mismatches}')

    state_dict = vq_payload.get('net', {})
    codebook = state_dict.get('quantizer_upper.codebook')
    condition = state_dict.get('condition_mlp.2.weight')
    if codebook is None or tuple(codebook.shape) != (
            expected_contract['nb_code'], expected_contract['code_dim'] * 3 // 8):
        raise ValueError(
            'VQ codebook tensor violates the selected contract: '
            f'{None if codebook is None else tuple(codebook.shape)}'
        )
    expected_condition_dim = expected_contract.get(
        'shape_condition_dim', expected_contract['code_dim'] // 32
    )
    if condition is None or int(condition.shape[0]) != expected_condition_dim:
        raise ValueError(
            'VQ shape-condition tensor violates the selected contract: '
            f'{None if condition is None else tuple(condition.shape)}'
        )

    if mgt_contract == 'historical_mmm':
        mgt_args = dict(
            embed_dim_gpt=1024, clip_dim=512, block_size=51,
            num_layers=9, num_local_layer=2, n_head_gpt=16,
            drop_out_rate=0.1, ff_rate=4, contact_conditioning=False,
            shape=bool(mgt_shape), paper_architecture=False,
            conditioning_contract='historical_mmm_shape_dropout',
        )
    else:
        mgt_args = dict(
            embed_dim_gpt=512, clip_dim=512, block_size=51,
            num_layers=9, num_local_layer=9, n_head_gpt=8,
            drop_out_rate=0.1, ff_rate=4, contact_conditioning=False,
            shape=bool(mgt_shape), paper_architecture=True,
            conditioning_contract='paper_text_cfg',
        )

    return SimpleNamespace(
        device=str(device),
        **expected_contract,
        **mgt_args, vq_checkpoint=vq_checkpoint,
    )


def load_models(args, device):
    vq_payload = torch.load(
        args.vq_checkpoint, map_location='cpu', weights_only=False
    )
    model_args = paper_model_args(
        device, args.vq_checkpoint, vq_payload=vq_payload,
        contract=args.vq_contract, mgt_contract=args.mgt_contract,
        mgt_shape=args.mgt_shape,
    )
    tokenizer = build_model(model_args, 'interx_vq')
    transformer = build_model(model_args, 'interx_transformer', vqvae=tokenizer)

    # Build the parent transformer before restoring the tokenizer because the
    # released constructor recursively initializes its child modules.
    vq_state = dict(vq_payload['net'])
    tokenizer.load_state_dict(merge_interx_lookup_weights(vq_state), strict=True)
    tokenizer.to(device).eval()

    load_transformer_checkpoint(
        transformer, args.checkpoint, preserve_vqvae=True
    )
    training_contract, inference_contract = resolve_conditioning_contracts(args)
    if transformer.trans_base.conditioning_contract != training_contract:
        raise RuntimeError(
            'constructed MGT conditioning contract differs from its training '
            f'identity: {transformer.trans_base.conditioning_contract} != '
            f'{training_contract}'
        )
    # The historical MMM dropout contract is needed to reconstruct training,
    # but the pre-cleanup common-50 exporter used text CFG while retaining
    # requested body shape.  Switching only after an exact checkpoint load
    # preserves topology/weights and makes that inference choice explicit.
    transformer.trans_base.conditioning_contract = inference_contract
    transformer.conditioning_contract = inference_contract
    transformer.training_conditioning_contract = training_contract
    transformer.inference_conditioning_contract = inference_contract
    transformer.to(device).eval()

    clip_model, _ = clip.load('ViT-B/32', device=device, jit=False)
    clip.model.convert_weights(clip_model)
    if torch.device(device).type == 'cpu':
        clip_model.float()
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    return tokenizer, transformer, TextCLIP(clip_model).to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', required=True, help='paper MGT Lightning checkpoint')
    parser.add_argument(
        '--vq-checkpoint',
        default='checkpoints/vq_precleanup_raw12288/net_175000.pth',
    )
    parser.add_argument(
        '--vq-contract',
        choices=('current', 'original_8192', 'precleanup_raw12288'),
        default='precleanup_raw12288',
        help='Exact tokenizer schema; original_8192 is the 8096-entry/64-dim MMM checkpoint.',
    )
    parser.add_argument(
        '--mgt-contract', choices=('current', 'historical_mmm'), default='current',
        help='Transformer architecture; historical_mmm is the 1024-dim/16-head/2-local-layer line.',
    )
    parser.add_argument(
        '--mgt-shape', action=argparse.BooleanOptionalAction, default=True,
        help=(
            'Whether the MGT checkpoint contains beta/gender conditioning. '
            'This does not disable shape conditioning in the VQ decoder.'
        ),
    )
    parser.add_argument(
        '--inference-conditioning-contract',
        choices=(
            'training_contract', 'paper_text_cfg',
            'historical_mmm_shape_dropout',
        ),
        default='training_contract',
        help=(
            'CFG conditioning semantics used only after exact checkpoint load. '
            'Historical common-50 replay uses paper_text_cfg; training_contract '
            'is retained for historical validation diagnostics.'
        ),
    )
    parser.add_argument('--stats-dir', default='dataset/interx')
    parser.add_argument(
        '--beta-policy', choices=('requested', 'zero'), default='requested',
        help='Diagnostic zero betas for both MGT and VQ; requested genders remain fixed.',
    )
    parser.add_argument('--mgt-beta-policy', choices=('requested', 'zero'), default=None)
    parser.add_argument('--vq-beta-policy', choices=('requested', 'zero'), default=None)
    parser.add_argument('--stats-prefix', default='Interx')
    parser.add_argument('--contact-map', default=None,
                        help='Defaults to code_contact_probs.npy beside --vq-checkpoint.')
    parser.add_argument(
        '--contact-map-contract',
        choices=('paper_formula_train_only', 'mmm_legacy_full_axis_bug'),
        default='paper_formula_train_only',
        help=(
            'Strict provenance contract. The legacy contract is a disclosed '
            'author-implementation sensitivity only and is excluded from the '
            'fair main table.'
        ),
    )
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--resume', action='store_true', help='Resume completed repeat blocks')
    parser.add_argument('--indices-key', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--contact-guidance', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--contact-timing-source',
        choices=('none', 'request_oracle', 'full_valid_weak', 'external_text_predictor'),
        default='request_oracle',
        help=(
            'Explicit timing assumption. none is required when contact guidance '
            'is disabled; request_oracle uses frozen GT labels; '
            'full_valid_weak marks the entire valid clip; external_text_predictor '
            'loads a request-bound binary mask.'
        ),
    )
    parser.add_argument('--contact-timing-file')
    parser.add_argument('--sampling-steps', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument(
        '--temperature-end', type=float, default=None,
        help=(
            'Optional linear final temperature for a disclosed sampling '
            'diagnostic. Omit it to retain constant-temperature paper sampling.'
        ),
    )
    parser.add_argument('--cfg-scale', type=float, default=3.5)
    parser.add_argument('--confidence-noise-scale', type=float, default=0.0)
    parser.add_argument('--confidence-noise-until-step', type=int, default=0,
                        help='Stop ranking noise after this per-channel step; 0 retains the full schedule.')
    parser.add_argument('--confidence-noise-scope', choices=('all', 'root'), default='all')
    parser.add_argument('--confidence-ranking', choices=('probability', 'log_odds', 'max_probability'), default='probability')
    parser.add_argument('--channel-schedule', choices=('interleaved', 'partwise'), default='interleaved')
    parser.add_argument('--save-token-ids', action='store_true',
                        help='Archive exact sampled tokens for decoder-only diagnostics; no sampler change.')
    parser.add_argument('--root-replay-control', action='store_true',
                        help='Diagnostic two-pass sampling with per-stage generated-control root replay.')
    parser.add_argument('--contact-guidance-scale', type=float, default=1.0)
    args = parser.parse_args()
    if not 0 <= args.confidence_noise_until_step <= args.sampling_steps:
        parser.error('--confidence-noise-until-step must be in [0, sampling-steps]')
    if args.root_replay_control and (args.channel_schedule != 'interleaved' or not args.save_token_ids):
        parser.error('root replay requires interleaved scheduling and --save-token-ids')
    if not np.isfinite(args.confidence_noise_scale) or args.confidence_noise_scale < 0:
        parser.error('--confidence-noise-scale must be finite and nonnegative')
    if args.contact_map is None:
        args.contact_map = os.path.join(
            os.path.dirname(os.path.abspath(args.vq_checkpoint)),
            'code_contact_probs.npy',
        )

    seed_everything(args.seed)
    device = torch.device(args.device)
    normalizer = MotionNormalizerTorchInterX(
        'raw', stats_dir=args.stats_dir, stats_prefix=args.stats_prefix
    )
    request = np.load(args.requests, allow_pickle=False)
    indices = np.arange(len(request['lengths']))
    if args.indices_key is not None:
        indices = request[args.indices_key].astype(np.int64)
    if args.limit is not None:
        indices = indices[:args.limit]
    names = request['names'][indices].astype(str)
    captions = request['captions'][indices].astype(str)
    lengths = request['lengths'][indices].astype(np.int64)
    mgt_beta_policy, vq_beta_policy = resolve_beta_policies(
        args.beta_policy, args.mgt_beta_policy, args.vq_beta_policy)
    requested_betas = request['betas'][indices].astype(np.float32)
    mgt_betas = generation_betas(requested_betas, mgt_beta_policy)
    vq_betas = generation_betas(requested_betas, vq_beta_policy)
    gender = request['gender'][indices].astype(np.int64)
    contact, timing_source, timing_path, timing_hash = resolve_contact_timing(
        args, request, indices, names, lengths
    )
    training_conditioning, inference_conditioning = (
        resolve_conditioning_contracts(args)
    )

    expected_attributes = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'vq_checkpoint': os.path.abspath(args.vq_checkpoint),
        'vq_checkpoint_sha256': sha256_file(args.vq_checkpoint),
        'vq_contract': args.vq_contract,
        'mgt_contract': args.mgt_contract,
        'mgt_shape_conditioning': bool(args.mgt_shape),
        'training_conditioning_contract': training_conditioning,
        'inference_conditioning_contract': inference_conditioning,
        'normalization_mean': normalizer.mean_path,
        'normalization_mean_sha256': sha256_file(normalizer.mean_path),
        'normalization_std': normalizer.std_path,
        'normalization_std_sha256': sha256_file(normalizer.std_path),
        'requests': os.path.abspath(args.requests),
        'requests_sha256': sha256_file(args.requests),
        'seed': args.seed,
        'seed_strategy': 'base_seed + repeat_index',
        'sampler': 'sample_intra_cas',
        'sampler_semantics': 'paper_high_confidence_tokens_fixed',
        'cascade_order': 'root,body,hands',
        'sampling_steps': args.sampling_steps,
        'sampling_steps_unit': 'outer_maskgit_iterations',
        'maximum_active_channel_stages_per_outer_iteration': 3,
        'maximum_model_forward_calls_per_outer_iteration_with_cfg': 6,
        'resolved_channels_skip_forward_but_not_outer_iterations': True,
        'temperature': args.temperature,
        'temperature_end': (
            'constant' if args.temperature_end is None else args.temperature_end
        ),
        'temperature_schedule': (
            'constant' if args.temperature_end is None else 'linear_by_outer_step'
        ),
        'cfg_scale': args.cfg_scale,
        'contact_guidance': args.contact_guidance,
        'contact_guidance_scale': args.contact_guidance_scale,
        'contact_timing_source': timing_source,
        'contact_timing_file': timing_path,
        'contact_timing_file_sha256': timing_hash,
        'contact_map': (
            os.path.abspath(args.contact_map) if args.contact_guidance else ''
        ),
        'contact_map_sha256': (
            sha256_file(args.contact_map) if args.contact_guidance else ''
        ),
        'contact_map_contract': (
            args.contact_map_contract if args.contact_guidance else 'unused'
        ),
        'batch_size': args.batch_size,
        'generation_length_policy': 'exact_request_length',
    }
    if args.mgt_beta_policy is not None or args.vq_beta_policy is not None:
        expected_attributes.update({'mgt_beta_policy': mgt_beta_policy,
                                    'vq_beta_policy': vq_beta_policy,
                                    'diagnostic_split_beta_policies': True})
    if (args.confidence_noise_until_step != 0 or args.confidence_noise_scale != 0 or args.confidence_ranking != 'probability'
            or args.channel_schedule != 'interleaved'):
        expected_attributes.update({
            'confidence_noise_scale': args.confidence_noise_scale,
            'confidence_noise_scope': args.confidence_noise_scope,
            'confidence_noise_schedule': 'linear_to_zero',
            'sampler_semantics': 'diagnostic_annealed_noisy_confidence_tokens_fixed',
            'sampling_implementation_sha256': sha256_file(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'models', 'two_person_transformer.py',
            )),
            'diagnostic_not_formal_result': True,
        })
        if args.confidence_noise_until_step:
            expected_attributes.update(confidence_noise_until_step=args.confidence_noise_until_step,
                                       confidence_noise_schedule='linear_to_zero_with_hard_cutoff')
        if args.confidence_ranking != 'probability' or args.channel_schedule != 'interleaved':
            expected_attributes.update({
                'confidence_ranking': args.confidence_ranking,
                'channel_schedule': args.channel_schedule,
                'sampler_semantics': 'diagnostic_configurable_confidence_tokens_fixed',
            })
        if args.channel_schedule == 'partwise':
            expected_attributes.update({
                'sampling_steps_unit': 'maskgit_iterations_per_channel',
                'maximum_active_channel_stages_per_outer_iteration': 1,
                'maximum_model_forward_calls_per_outer_iteration_with_cfg': 2,
                'temperature_schedule': ('constant' if args.temperature_end is None
                                         else 'linear_by_channel_step'),
            })

    if args.beta_policy != 'requested':
        expected_attributes.update({
            'beta_policy': args.beta_policy,
            'diagnostic_changed_factor': 'zero_beta_requested_gender',
            'diagnostic_not_formal_result': True,
        })

    if args.save_token_ids:
        expected_attributes.update({'save_token_ids': True,
                                    'token_ids_layout': 'sample,repeat,time,body_hand_root'})
    if args.root_replay_control:
        expected_attributes.update({
            'root_replay_control': True,
            'root_replay_control_ranking': 'probability',
            'root_replay_control_noise_scale': 0.0,
            'root_replay_rng_policy': 'preserve_control_batch_chain',
            'root_replay_passes_per_batch': 2,
            'sampler_semantics': 'diagnostic_generated_control_root_trajectory_replay',
            'diagnostic_not_formal_result': True,
            'root_replay_implementation_sha256': sha256_file(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'probe_hhc_root_replay.py')),
            'root_replay_trace_sha256': sha256_file(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'trace_hhc_sampling.py')),
        })

    if args.resume and os.path.isfile(args.output):
        with h5py.File(args.output, 'r') as existing:
            validate_resume_attributes(existing, expected_attributes)
            expected_shape = (len(indices), args.repeats, 150, 56 * 12)
            complete = (
                existing['motions'].shape == expected_shape
                and np.array_equal(existing['lengths'][:], lengths)
                and np.array_equal(existing['names'][:].astype(str), names)
                and 'completed_repeats' in existing
                and existing['completed_repeats'][:].astype(bool).all()
            )
        if complete and not args.save_token_ids:
            print({'output': args.output, 'status': 'all repeats already complete'}, flush=True)
            return

    tokenizer, transformer, text_encoder = load_models(args, device)
    sample = transformer
    if args.root_replay_control:
        from functools import partial
        from tools.probe_hhc_root_replay import sample_with_control_root_replay
        sample = partial(sample_with_control_root_replay, transformer)
    contact_probabilities = None
    if args.contact_guidance:
        validate_contact_map_provenance(
            args.vq_checkpoint, args.contact_map,
            stats_dir=args.stats_dir, stats_prefix=args.stats_prefix,
            contact_map_contract=args.contact_map_contract,
        )
        contact_probabilities = torch.from_numpy(np.load(args.contact_map)).float().to(device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    resume_existing = args.resume and os.path.isfile(args.output)
    target = h5py.File(args.output, 'r+' if resume_existing else 'w')
    try:
        if resume_existing:
            validate_resume_attributes(target, expected_attributes)
            expected_shape = (len(indices), args.repeats, 150, 56 * 12)
            if target['motions'].shape != expected_shape:
                raise ValueError(f'resume shape mismatch: {target["motions"].shape} != {expected_shape}')
            if not np.array_equal(target['lengths'][:], lengths):
                raise ValueError('resume lengths mismatch')
            if not np.array_equal(target['names'][:].astype(str), names):
                raise ValueError('resume names mismatch')
            if 'completed_repeats' not in target:
                raise ValueError('resume file predates repeat-completion tracking')
            motions = target['motions']
            completed_repeats = target['completed_repeats']
        else:
            motions = target.create_dataset(
                'motions', shape=(len(indices), args.repeats, 150, 56 * 12),
                dtype=np.float32, chunks=(1, 1, 150, 56 * 12), compression='lzf',
            )
            target.create_dataset('lengths', data=lengths)
            target.create_dataset('names', data=np.asarray(names, dtype='S'))
            completed_repeats = target.create_dataset(
                'completed_repeats', data=np.zeros(args.repeats, dtype=np.bool_)
            )
            for key, value in expected_attributes.items():
                target.attrs[key] = value

        saved_tokens = (token_archive(target, (len(indices), args.repeats,
                                               transformer.block_size - 1, 3), resume_existing)
                        if args.save_token_ids else None)
        saved_control_tokens = (token_archive(target, saved_tokens.shape, resume_existing,
                                              name='root_replay_control_token_ids')
                                if args.root_replay_control else None)
        for repeat in range(args.repeats):
            if bool(completed_repeats[repeat]):
                print({'repeat': repeat, 'status': 'already complete'}, flush=True)
                continue
            seed_everything(args.seed + repeat)
            for start in range(0, len(indices), args.batch_size):
                end = min(start + args.batch_size, len(indices))
                batch_lengths = torch.from_numpy(lengths[start:end]).long().to(device)
                batch_gender = torch.from_numpy(gender[start:end]).long().to(device)
                batch_contact = torch.from_numpy(contact[start:end]).long().to(device)
                text_tokens = clip.tokenize(captions[start:end].tolist(), truncate=True).to(device)
                with torch.inference_mode():
                    global_text, word_text = text_encoder(text_tokens)
                    token_contact = downsample_contact_mask(batch_contact, factor=4)
                    max_tokens = transformer.block_size - 1
                    if token_contact.shape[1] < max_tokens:
                        token_contact = torch.nn.functional.pad(
                            token_contact, (0, max_tokens - token_contact.shape[1])
                        )
                    token_contact = token_contact[:, :max_tokens]
                    motion_ids = sample(
                        global_text, word_text, type='sample_intra_cas',
                        contact_mask=token_contact if args.contact_guidance else None,
                        betas=torch.from_numpy(mgt_betas[start:end]).float().to(device), gender=batch_gender,
                        m_length=batch_lengths, rand_pos=True, CFG=args.cfg_scale,
                        max_steps=args.sampling_steps,
                        vocab_contact_map=contact_probabilities,
                        contact_guidance_scale=args.contact_guidance_scale,
                        temperature=args.temperature,
                        temperature_end=args.temperature_end,
                        confidence_noise_scale=args.confidence_noise_scale,
                        confidence_noise_until_step=args.confidence_noise_until_step,
                        confidence_noise_scope=args.confidence_noise_scope,
                        confidence_ranking=args.confidence_ranking,
                        channel_schedule=args.channel_schedule,
                    )
                    if args.root_replay_control:
                        motion_ids, control_ids = motion_ids
                        saved_control_tokens[start:end, repeat] = control_ids.cpu().numpy()
                    token_lengths = torch.ceil(batch_lengths.float() / 4).long()
                    official = torch.zeros(
                        end - start, 150, 56 * 12, device=device, dtype=torch.float32
                    )
                    for local_index, token_length in enumerate(token_lengths.tolist()):
                        decoded = tokenizer(
                            motion_ids[local_index:local_index + 1, :token_length],
                            torch.from_numpy(vq_betas[start + local_index:start + local_index + 1]).float().to(device),
                            batch_gender[local_index:local_index + 1], type='decode',
                        )
                        decoded = normalizer.backward(decoded).squeeze(0)
                        valid_length = int(batch_lengths[local_index].item())
                        official[local_index, :valid_length] = decoded[:valid_length]
                if not torch.isfinite(official).all():
                    raise RuntimeError(f'non-finite HHC generation in batch {start}:{end}')
                motions[start:end, repeat] = official.cpu().numpy()
                if saved_tokens is not None:
                    saved_tokens[start:end, repeat] = motion_ids.cpu().numpy()
                print({'repeat': repeat, 'generated': end, 'total': len(indices)}, flush=True)
            # Make every generated motion durable before publishing the
            # completion marker.  A killed process must never leave a true
            # marker pointing at partially buffered motion data.
            target.flush()
            completed_repeats[repeat] = True
            target.flush()
    finally:
        target.close()


if __name__ == '__main__':
    main()
