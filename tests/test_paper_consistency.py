import ast
import os
import hashlib
import json
import inspect
import pickle
import random
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from dataset.dataset_interx import MotionDatasetV2HHI
from models.two_person_transformer import Text2Motion_Transformer
from utils.motion_masks import (
    all_valid_token_nll,
    masked_token_nll,
    sample_historical_mmm_corruption,
    sample_kinematic_cascade_mask,
)
from utils.eval_trans import downsample_contact_mask
from utils.eval_trans import calculate_R_precision
from tools.score_fair_interx_generations import (
    EVALUATOR_CHECKPOINT,
    EVALUATOR_OPTIONS,
    EVALUATOR_OPTION_SOURCE,
    EVALUATOR_WRAPPER_SOURCE,
    artifact_record as scorer_artifact_record,
    calculate_batched_retrieval,
    embedding_lengths,
    validate_generation_metadata,
    validate_generation_file,
)
from tools.compare_mgt_common50_bridge import (
    CONTACT_MAP_SHA256 as BRIDGE_CONTACT_MAP_SHA256,
    MEAN_SHA256 as BRIDGE_MEAN_SHA256,
    REQUEST_SHA256 as BRIDGE_REQUEST_SHA256,
    STD_SHA256 as BRIDGE_STD_SHA256,
    VQ_SHA256 as BRIDGE_VQ_SHA256,
    validate_evaluator as validate_bridge_evaluator,
    validate_evaluator_runtime as validate_bridge_evaluator_runtime,
    validate_metadata as validate_bridge_metadata,
)
from tools.aggregate_fair_interx_metrics import (
    validate_contact_timing_provenance,
    validate_replication_provenance,
)
from tools.compare_fair_interx_methods_paired import (
    compare_methods, load_strict_replications,
    holm_bonferroni,
)
from tools.assess_hhc_global_masked_loss_control import (
    MATCHED_ARGUMENTS as GLOBAL_LOSS_MATCHED_ARGUMENTS,
    validate_single_factor_contract,
)
from tools.assess_hhc_token_window_control import (
    MATCHED_ARGUMENTS as TOKEN_WINDOW_MATCHED_ARGUMENTS,
)
from tools.assess_hhc_historical_mask_control import (
    MASK_MATCHED_ARGUMENTS,
    validate_single_factor_contract as validate_historical_mask_contract,
)
from tools.apply_promoted_original38_shape_baseline import (
    FILES as PROMOTED38_SHAPE_FILES,
    transform as transform_promoted38_shape_sources,
)
from tools.build_review_results_report import (
    HHC_NO_TIMING_LABEL, HHC_ORACLE_TIMING_LABEL,
    sha256_file as report_sha256_file,
    validate_method_artifacts, validate_method_manifest,
    validate_method_checkpoint_bindings,
    validate_formal_optimization,
    validate_optimization_ablation,
    validate_generalization_manifest_contract,
    validate_llm_robustness_results,
    validate_shape_augmentation_results,
    resolve_review_completion_status,
    validate_method_provenance,
    validate_training_repository_matches_generation,
    validate_training_and_protocol,
    validate_coshmdm_representation_adapter,
    validate_final_ordinary_only_metric_audit,
    validate_hhc_validation_sampler_calibration,
    sampler_identity_value_matches,
)
from tools.build_shape_augmentation_results import (
    validate_backend_equivalence, volumetric_checkpoint_contract_valid,
)
from tools.freeze_fair_manifest_implementation_sources import freeze_manifest
from tools.run_fair_interx_method import report_is_current
from tools.audit_fair_interx_request_replay import validate_creation_provenance
from tools.audit_interx_codex_contact_cache import manifest_wall_clock
from tools.verify_fair_eval_requests import verify_frozen_manifest
from tools.verify_interx_protocol import protocol_alias
from tests.test_timotion_validation_checkpoint_pair import (
    test_pair_audit_requires_bitwise_state_and_metric_binding,
)
from tests.test_hybrid_shape_token_cache import (
    test_hybrid_cache_preserves_original_bytes_and_partitions_names,
    test_hybrid_cache_rejects_conflicting_original_and_scope_contamination,
    test_hybrid_partition_rejects_malformed_or_out_of_scope_augmented_names,
)
from tools.audit_mgt_common50_mm_replay import compare_mm_files
from tools.freeze_mgt_training_sources import freeze_sources
from tools.apply_post_formal_original38_token_window import (
    FILES as ORIGINAL38_TRANSFORM_FILES,
    transform as transform_original38_sources,
)
from tools.select_best_mgt_checkpoint import (
    discover_completed_run, exact_validation_fid,
)
from tools.select_coshmdm_checkpoint_on_interx_val import (
    ACCEPTED_CAP_STATUSES, CAP_POLICY, validate_budget_contract,
)
from tools.preserve_coshmdm_milestones import eligible_sources, preserve
from tools.resolve_mgt_training_state import resolve_training_state
from tools.export_optimizer_npz_to_fair_h5 import (
    pair_to_raw,
    validate_optimizer_sidecar,
)
from tools.export_fair_h5_to_optimizer_npz import RESUME_CONTRACT
from tools.run_paper_optimization_formal import validate_hhc_source_contract
from utils.rotation_conversions import (
    axis_angle_to_matrix, matrix_to_rotation_6d,
)
from utils.quantizer_state import (
    quantizer_training_state,
    restore_quantizer_training_state,
)
from utils.contact_map import build_code_contact_map
from utils.runtime import evaluating, preserving_rng_state
from tools.export_hhc_official_interx import (
    paper_model_args,
    resolve_conditioning_contracts,
    resolve_contact_timing,
    sha256_file,
    validate_contact_map_provenance,
)
from train.train_t2i_trans import (
    FixedStepValidationPairCheckpoint,
    MGT_RESUME_IDENTITY_KEYS,
    MGT_TRAINING_SOURCE_PATHS,
    MaskGITLightning,
    assert_exact_historical_cache_scope,
    import_historical_token_pairs,
    load_transformer_checkpoint,
    prepare_mgt_output_directory,
)
from optimize.optimize_full_body import optimize_full_body
from optimization.losses import DyadicInteractionLoss
import optimization.losses as losses_module
from optimization.text_2_regions import ContactExtractor
from eval.extract_val_contacts import (
    canonical_sha256 as contact_canonical_sha256,
    load_caption_requests,
)
from tools.run_paper_contact_optimization import (
    load_strict_contacts,
    validate_contact_cache,
    validate_generation_record_inputs,
)
from utils.paramUtil import CANDIDATES


def official_main_only_evaluator_provenance():
    repository = Path(__file__).resolve().parents[1]
    return {
        'contract': 'official_interx_full_672d',
        'options': scorer_artifact_record(EVALUATOR_OPTIONS),
        'checkpoint': scorer_artifact_record(EVALUATOR_CHECKPOINT),
        'wrapper_source': scorer_artifact_record(EVALUATOR_WRAPPER_SOURCE),
        'option_loader_source': scorer_artifact_record(EVALUATOR_OPTION_SOURCE),
        'scorer_source': scorer_artifact_record(
            repository / 'tools/score_fair_interx_main.py'
        ),
    }


def official_main_only_evaluator_runtime(device='cpu'):
    is_cuda = device.startswith('cuda')
    return {
        'contract': 'pytorch_official_evaluator_single_device_v1',
        'requested_device': device,
        'device_type': 'cuda' if is_cuda else 'cpu',
        'device_index': int(device.split(':')[1]) if is_cuda else None,
        'device_name': 'fixture CUDA device' if is_cuda else 'cpu',
        'torch_version': 'fixture',
        'torch_cuda_version': 'fixture' if is_cuda else None,
    }


class _Quantizer(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.codebook = nn.Embedding(size, dim)

    def dequantize(self, indices):
        return self.codebook(indices)


class _Tokenizer(nn.Module):
    code_dim = 256

    def __init__(self, codebook_size):
        super().__init__()
        self.quantizer_upper = _Quantizer(codebook_size, 96)
        self.quantizer_lower = _Quantizer(codebook_size, 96)
        self.quantizer_root = _Quantizer(codebook_size, 64)


class _EMAQuantizer(nn.Module):
    def __init__(self, size=5, dim=3):
        super().__init__()
        self.nb_code = size
        self.register_buffer("codebook", torch.randn(size, dim))
        self.init = True
        self.code_sum = torch.randn(size, dim)
        self.code_count = torch.rand(size)
        self.reset_count = 7
        self.usage = torch.randint(0, 2, (size, 1)).float()


class _EMAContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer_upper = _EMAQuantizer()
        self.quantizer_lower = _EMAQuantizer()
        self.quantizer_root = _EMAQuantizer()


class _TrainingSideEffect(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def forward(self):
        if self.training:
            self.updates += 1


def test_training_mask_enforces_root_body_hand_hierarchy():
    valid = torch.zeros(256, 38, 3, dtype=torch.bool)
    valid[:, :30] = True
    mask, ratios, taus = sample_kinematic_cascade_mask(
        valid, torch.Generator().manual_seed(7)
    )

    # Tensor channel order is Body, Hand, Root.
    assert torch.all(taus[:, 2] > taus[:, 0])
    assert torch.all(taus[:, 0] > taus[:, 1])
    assert torch.all(ratios[:, 2] < ratios[:, 0])
    assert torch.all(ratios[:, 0] < ratios[:, 1])
    assert not mask[:, 30:].any()
    expected_counts = torch.round(30 * ratios).long().clamp(min=1, max=30)
    torch.testing.assert_close(mask.sum(dim=1), expected_counts)


def test_masked_nll_ignores_out_of_vocabulary_end_and_pad_targets():
    logits = torch.randn(2, 4, 3, 8, requires_grad=True)
    target = torch.full((2, 4, 3), 9, dtype=torch.long)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0, 0, :] = True
    mask[1, :2, :] = True
    target[mask] = torch.arange(int(mask.sum())) % 8

    loss = masked_token_nll(logits, target, mask)
    expected = torch.stack([
        torch.nn.functional.cross_entropy(logits[index][mask[index]], target[index][mask[index]])
        for index in range(2)
    ]).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[~mask]) == 0


def test_historical_mmm_corruption_matches_recovered_source_logic():
    target = torch.arange(2 * 7 * 3).view(2, 7, 3) % 13
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[0, :5] = True
    valid[1, :7] = True
    seed = 81

    actual_input, actual_mask, actual_ratio = sample_historical_mmm_corruption(
        target, valid, 13, pkeep=0.5,
        generator=torch.Generator().manual_seed(seed),
    )

    generator = torch.Generator().manual_seed(seed)
    keep = torch.bernoulli(
        0.5 * torch.ones_like(target, dtype=torch.float32),
        generator=generator,
    ).bool()
    keep = keep | ~valid
    random_indices = torch.randint(
        13, target.shape, dtype=target.dtype, generator=generator,
    )
    corrupted = torch.where(keep, target, random_indices)
    rand_time = torch.rand(2, generator=generator)
    expected_ratio = torch.cos(rand_time * np.pi * 0.5).clamp(0.1, 1.0)
    counts = (valid.sum((1, 2)).float() * expected_ratio).round().clamp(min=1).long()
    scores = torch.rand(2, 7, 3, generator=generator)
    scores = scores + torch.tensor([0.5, 1.0, 0.0])
    scores = scores + (~valid).float() * 1e9
    expected_mask = (
        scores.view(2, -1).argsort(dim=-1) < counts[:, None]
    ).view_as(target)
    expected_input = torch.where(expected_mask, 15, corrupted)

    torch.testing.assert_close(actual_ratio, expected_ratio)
    torch.testing.assert_close(actual_mask, expected_mask)
    torch.testing.assert_close(actual_input, expected_input)


def test_historical_all_valid_nll_ignores_padding_but_scores_visible_tokens():
    logits = torch.randn(2, 4, 3, 8, requires_grad=True)
    target = torch.full((2, 4, 3), 9, dtype=torch.long)
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[0, :2] = True
    valid[1, :3] = True
    target[valid] = torch.arange(int(valid.sum())) % 8
    loss = all_valid_token_nll(logits, target, valid)
    weights = valid.float()
    weights = weights / valid.sum((1, 2), keepdim=True).clamp(min=1)
    expected = torch.nn.functional.cross_entropy(
        logits[valid].reshape(-1, logits.shape[-1]),
        target[valid].reshape(-1), reduction='none',
    )
    expected = (expected * weights[valid].reshape(-1)).sum() / logits.shape[0]
    assert torch.equal(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[~valid]) == 0


def test_paper_transformer_has_nine_blocks_and_cascade_resolves_masks():
    codebook_size = 32
    model = Text2Motion_Transformer(
        _Tokenizer(codebook_size), num_vq=codebook_size, embed_dim=64,
        clip_dim=32, block_size=5, num_layers=9, num_local_layer=9,
        n_head=8, drop_out_rate=0.0, fc_rate=2, shape=True,
        paper_architecture=True,
    ).eval()
    assert len(model.trans_base.paper_blocks) == 9
    assert len(model.trans_head.blocks) == 0
    assert model.trans_base.shape_mask_prob == 0.0
    assert not model.trans_base.contact
    assert model.trans_base.shared_special_emb.num_embeddings == 3

    batch_size, token_length = 1, 4
    clip_feature = torch.randn(batch_size, 32)
    word_emb = torch.randn(batch_size, 7, 32)
    betas = torch.randn(batch_size, 20)
    gender = torch.zeros(batch_size, 2, dtype=torch.long)
    with torch.no_grad():
        sampled = model(
            clip_feature, word_emb, type='sample_intra_cas',
            m_length=torch.tensor([token_length * 4]), betas=betas,
            gender=gender, CFG=-1, max_steps=2, temperature=0.5,
        )
    assert sampled.shape == (batch_size, token_length, 3)
    assert sampled.min() >= 0
    assert sampled.max() < codebook_size


def test_historical_conditioning_reproduces_clip_shape_dropout_without_word_dropout():
    codebook_size = 32
    model = Text2Motion_Transformer(
        _Tokenizer(codebook_size), num_vq=codebook_size, embed_dim=64,
        clip_dim=32, block_size=5, num_layers=2, num_local_layer=1,
        n_head=8, drop_out_rate=0.0, fc_rate=2, shape=True,
        paper_architecture=False,
        conditioning_contract='historical_mmm_shape_dropout',
    ).train()
    assert not hasattr(model.trans_base, 'shared_special_emb')
    model.trans_base.shape_mask_prob = 1.0
    captured = {}

    def capture(name):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach().clone()
        return hook

    hooks = [
        model.trans_base.word_emb.register_forward_pre_hook(capture('word')),
        model.trans_base.cond_emb.register_forward_pre_hook(capture('clip')),
        model.trans_base.shape_projection.register_forward_pre_hook(capture('shape')),
    ]
    batch_size, token_length = 2, 4
    indices = torch.randint(0, codebook_size, (batch_size, token_length, 3))
    word = torch.randn(batch_size, 7, 32)
    model(
        indices, torch.randn(batch_size, 32),
        src_mask=torch.ones(batch_size, token_length, dtype=torch.bool),
        word_emb=word, betas=torch.randn(batch_size, 20),
        gender=torch.zeros(batch_size, 2, dtype=torch.long),
    )
    for hook in hooks:
        hook.remove()
    torch.testing.assert_close(captured['word'], word)
    assert torch.count_nonzero(captured['clip']) == 0
    assert torch.count_nonzero(captured['shape']) == 0


def test_historical_training_and_common50_inference_conditioning_are_separate():
    historical_training = types.SimpleNamespace(
        mgt_contract='historical_mmm',
        inference_conditioning_contract='training_contract',
    )
    historical_common50 = types.SimpleNamespace(
        mgt_contract='historical_mmm',
        inference_conditioning_contract='paper_text_cfg',
    )
    current_common50 = types.SimpleNamespace(
        mgt_contract='current',
        inference_conditioning_contract='training_contract',
    )
    assert resolve_conditioning_contracts(historical_training) == (
        'historical_mmm_shape_dropout',
        'historical_mmm_shape_dropout',
    )
    assert resolve_conditioning_contracts(historical_common50) == (
        'historical_mmm_shape_dropout', 'paper_text_cfg',
    )
    assert resolve_conditioning_contracts(current_common50) == (
        'paper_text_cfg', 'paper_text_cfg',
    )

    driver = Path('tools/run_fair_interx_method.py').read_text(encoding='utf-8')
    finalizer = Path('tools/finalize_mgt_repair_v7.sh').read_text(
        encoding='utf-8'
    )
    assert '--inference-conditioning-contract' in driver
    assert '--hhc-inference-conditioning-contract paper_text_cfg' in finalizer


def test_paper_special_tokens_follow_autocast_projection_dtype():
    """BF16 projection outputs must accept FP32 learnable special tokens."""
    codebook_size = 32
    model = Text2Motion_Transformer(
        _Tokenizer(codebook_size), num_vq=codebook_size, embed_dim=64,
        clip_dim=32, block_size=5, num_layers=1, num_local_layer=1,
        n_head=8, drop_out_rate=0.0, fc_rate=2, shape=True,
        paper_architecture=True,
    ).eval()
    token_length = 4
    indices = torch.randint(0, codebook_size, (1, token_length, 3))
    indices[:, 0, :] = codebook_size + 2
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        logits = model(
            indices, torch.randn(1, 32),
            src_mask=torch.ones(1, token_length, dtype=torch.bool),
            word_emb=torch.randn(1, 7, 32),
            betas=torch.randn(1, 20),
            gender=torch.zeros(1, 2, dtype=torch.long),
        )
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()


def test_hhc_exporter_uses_and_validates_vq_checkpoint_arguments():
    saved_args = {
        'dataname': 'hhi', 'motion_rep': 'raw', 'nb_code': 8096,
        'code_dim': 256, 'output_emb_width': 512, 'down_t': 2,
        'stride_t': 2, 'width': 512, 'depth': 3,
        'dilation_growth_rate': 3, 'vq_act': 'relu', 'vq_norm': None,
        'mu': 0.99, 'quantizer': 'ema_reset',
    }
    state = {
        'quantizer_upper.codebook': torch.zeros(8096, 96),
        'condition_mlp.2.weight': torch.zeros(8, 512),
    }
    args = paper_model_args(
        'cpu', 'paper-vq.pth', vq_payload={'args': saved_args, 'net': state}
    )
    assert args.mu == 0.99
    assert args.nb_code == 8096

    incompatible = dict(saved_args, nb_code=12288)
    try:
        paper_model_args(
            'cpu', 'legacy-vq.pth',
            vq_payload={'args': incompatible, 'net': state},
        )
    except ValueError as error:
        assert 'nb_code' in str(error)
    else:
        raise AssertionError('incompatible VQ arguments were accepted')


def test_cascade_uses_each_part_update_immediately():
    codebook_size = 8
    model = Text2Motion_Transformer(
        _Tokenizer(codebook_size), num_vq=codebook_size, embed_dim=32,
        clip_dim=16, block_size=5, num_layers=1, num_local_layer=1,
        n_head=4, drop_out_rate=0.0, fc_rate=2, shape=False,
        paper_architecture=True,
    ).eval()
    snapshots = []

    def deterministic_forward(self, ids, *args, **kwargs):
        snapshots.append(ids.clone())
        logits = torch.full(
            (*ids.shape, codebook_size), -100.0, device=ids.device
        )
        for channel, token in enumerate((3, 5, 7)):
            logits[:, :, channel, token] = 100.0
        return logits

    model.forward_function = types.MethodType(deterministic_forward, model)
    sampled = model(
        torch.zeros(1, 16), torch.zeros(1, 2, 16),
        type='sample_intra_cas', m_length=torch.tensor([16]),
        CFG=-1, max_steps=1, rand_pos=False,
    )
    mask_id = codebook_size + 2
    assert len(snapshots) == 3
    assert torch.all(snapshots[0] == mask_id)
    assert torch.all(snapshots[1][:, :, 2] == 7)
    assert torch.all(snapshots[1][:, :, :2] == mask_id)
    assert torch.all(snapshots[2][:, :, 2] == 7)
    assert torch.all(snapshots[2][:, :, 0] == 3)
    assert torch.all(snapshots[2][:, :, 1] == mask_id)
    assert torch.all(sampled[:, :, 0] == 3)
    assert torch.all(sampled[:, :, 1] == 5)
    assert torch.all(sampled[:, :, 2] == 7)


def test_historical_cascade_repredicts_all_valid_tokens_each_round():
    codebook_size = 16
    model = Text2Motion_Transformer(
        _Tokenizer(codebook_size), num_vq=codebook_size, embed_dim=32,
        clip_dim=16, block_size=5, num_layers=1, num_local_layer=1,
        n_head=4, drop_out_rate=0.0, fc_rate=2, shape=False,
        paper_architecture=False,
    ).eval()
    calls = []

    def round_varying_forward(self, ids, *args, **kwargs):
        call_index = len(calls)
        calls.append(ids.clone())
        logits = torch.full(
            (*ids.shape, codebook_size), -100.0, device=ids.device
        )
        logits[..., call_index] = 100.0
        return logits

    model.forward_function = types.MethodType(round_varying_forward, model)
    sampled = model(
        torch.zeros(1, 16), torch.zeros(1, 2, 16),
        type='sample_intra_cas_historical', m_length=torch.tensor([16]),
        CFG=-1, max_steps=2, rand_pos=False,
    )
    assert len(calls) == 6
    # Historical call order is Root, Body, Hand per round.  Its second round
    # overwrites every valid position, including tokens fixed in round one.
    assert torch.all(sampled[:, :, 2] == 3)
    assert torch.all(sampled[:, :, 0] == 4)
    assert torch.all(sampled[:, :, 1] == 5)


def test_formal_setting_repredict_diagnostic_changes_only_update_support():
    codebook_size = 16
    model = Text2Motion_Transformer(
        _Tokenizer(codebook_size), num_vq=codebook_size, embed_dim=32,
        clip_dim=16, block_size=5, num_layers=1, num_local_layer=1,
        n_head=4, drop_out_rate=0.0, fc_rate=2, shape=False,
        paper_architecture=True,
    ).eval()
    calls = []

    def round_varying_forward(self, ids, *args, **kwargs):
        call_index = len(calls)
        calls.append(ids.clone())
        logits = torch.full(
            (*ids.shape, codebook_size), -100.0, device=ids.device
        )
        logits[..., call_index] = 100.0
        return logits

    model.forward_function = types.MethodType(round_varying_forward, model)
    sampled = model(
        torch.zeros(1, 16), torch.zeros(1, 2, 16),
        type='sample_intra_cas_repredict', m_length=torch.tensor([16]),
        CFG=-1, max_steps=2, rand_pos=False, temperature=0.5,
    )
    assert len(calls) == 6
    assert torch.all(sampled[:, :, 2] == 3)
    assert torch.all(sampled[:, :, 0] == 4)
    assert torch.all(sampled[:, :, 1] == 5)


def test_official_150_frame_contact_pool_has_38_tokens():
    contact = torch.zeros(2, 150, dtype=torch.long)
    contact[0, 149] = 1
    pooled = downsample_contact_mask(contact)
    assert pooled.shape == (2, 38)
    assert pooled[0, -1] == 1
    assert pooled[1].sum() == 0


def test_official_retrieval_is_accumulated_per_complete_batch():
    generator = np.random.default_rng(11)
    text = generator.normal(size=(64, 16))
    motion = generator.normal(size=(64, 16))
    expected_r = np.zeros(3)
    expected_matching = 0.0
    for start in (0, 32):
        batch_r, batch_matching = calculate_R_precision(
            text[start:start + 32], motion[start:start + 32], 3, sum_all=True
        )
        expected_r += batch_r
        expected_matching += batch_matching
    actual_r, actual_matching = calculate_batched_retrieval(text, motion, 32)
    np.testing.assert_allclose(actual_r, expected_r / 64)
    np.testing.assert_allclose(actual_matching, expected_matching / 64)


def test_incomplete_generation_repeats_are_rejected():
    names = np.asarray(['sample_a', 'sample_b'])
    lengths = np.asarray([24, 28], dtype=np.int64)
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as handle:
        path = handle.name
    try:
        with h5py.File(path, 'w') as target:
            target.create_dataset('motions', shape=(2, 2, 150, 672), dtype=np.float32)
            target.create_dataset('lengths', data=lengths)
            target.create_dataset('names', data=np.asarray(names, dtype='S'))
            target.create_dataset('completed_repeats', data=np.asarray([True, False]))
        with h5py.File(path, 'r') as source:
            try:
                validate_generation_file(source, names, lengths, 2, 'test')
            except ValueError as error:
                assert 'incomplete' in str(error)
            else:
                raise AssertionError('partial generation file was accepted')
    finally:
        os.unlink(path)


def test_exporters_flush_motion_data_before_repeat_completion_marker():
    exporter_markers = {
        'tools/export_hhc_official_interx.py': 'completed_repeats[repeat]',
        '/mnt/8T-HDD/wenning/InterMask/tools_export_official_interx.py':
        'completed_repeats[repeat]',
        '/mnt/8T-HDD/wenning/coshmdm-code/tools/export_official_interx.py':
        'completed_repeats[repeat]',
        'tools/export_interact2ar_official_interx.py':
        'completed_repeats[repeat]',
        'tools/export_timotion_official_interx.py': 'completed[repeat]',
    }
    for path, marker in exporter_markers.items():
        source = Path(path).read_text(encoding='utf-8')
        durable_commit = (
            f'target.flush()\n            {marker} = True\n'
            '            target.flush()'
        )
        assert durable_commit in source, path


def test_native_generation_lengths_are_explicit_and_bounded():
    requested = np.asarray([150, 116], dtype=np.int64)
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as handle:
        path = handle.name
    try:
        with h5py.File(path, 'w') as target:
            target.create_dataset(
                'generated_lengths', data=np.asarray([148, 116], dtype=np.int64)
            )
        with h5py.File(path, 'r') as source:
            np.testing.assert_array_equal(
                embedding_lengths(source, requested), np.asarray([148, 116])
            )
    finally:
        os.unlink(path)


def test_ema_quantizer_checkpoint_state_round_trips_exactly():
    source = _EMAContainer()
    saved = quantizer_training_state(source)
    restored = _EMAContainer()
    for quantizer in (
        restored.quantizer_upper,
        restored.quantizer_lower,
        restored.quantizer_root,
    ):
        quantizer.init = False
        quantizer.code_sum = None
        quantizer.code_count = None
        quantizer.reset_count = 0
        quantizer.usage.zero_()

    assert not restore_quantizer_training_state(restored, saved)
    for name in ("quantizer_upper", "quantizer_lower", "quantizer_root"):
        before = getattr(source, name)
        after = getattr(restored, name)
        assert after.init == before.init
        torch.testing.assert_close(after.code_sum, before.code_sum)
        torch.testing.assert_close(after.code_count, before.code_count)
        assert after.reset_count == before.reset_count
        torch.testing.assert_close(after.usage, before.usage)


def test_legacy_ema_checkpoint_fallback_preserves_loaded_codebooks():
    restored = _EMAContainer()
    expected = {}
    for name in ("quantizer_upper", "quantizer_lower", "quantizer_root"):
        quantizer = getattr(restored, name)
        quantizer.init = False
        quantizer.code_sum = None
        quantizer.code_count = None
        expected[name] = quantizer.codebook.clone()

    assert restore_quantizer_training_state(restored, None)
    for name, codebook in expected.items():
        quantizer = getattr(restored, name)
        assert quantizer.init
        torch.testing.assert_close(quantizer.codebook, codebook)
        torch.testing.assert_close(quantizer.code_sum, codebook)
        torch.testing.assert_close(
            quantizer.code_count, torch.ones(quantizer.nb_code)
        )


def test_vq_dataset_uses_the_requested_normalization_statistics():
    with tempfile.TemporaryDirectory() as directory:
        split = os.path.join(directory, "empty.txt")
        motion = os.path.join(directory, "empty.h5")
        with open(split, "w", encoding="utf-8"):
            pass
        with h5py.File(motion, "w"):
            pass

        raw_dataset = MotionDatasetV2HHI(
            split, motion, window_size=1, motion_rep="raw"
        )
        velocity_dataset = MotionDatasetV2HHI(
            split, motion, window_size=1, motion_rep="vel"
        )
        torch.testing.assert_close(
            raw_dataset.normalizer.motion_std,
            torch.from_numpy(np.load("dataset/interx/Interx_Std.npy")).float(),
        )
        torch.testing.assert_close(
            velocity_dataset.normalizer.motion_std,
            torch.from_numpy(np.load("dataset/interx/Interx_Std_root.npy")).float(),
        )
        assert not torch.equal(
            raw_dataset.normalizer.motion_std,
            velocity_dataset.normalizer.motion_std,
        )


def test_contact_code_map_can_be_restricted_to_training_names():
    with tempfile.TemporaryDirectory() as directory:
        indices_dir = os.path.join(directory, "indices")
        contact_dir = os.path.join(directory, "contact")
        os.makedirs(indices_dir)
        os.makedirs(contact_dir)
        np.save(os.path.join(indices_dir, "train.npy"), np.zeros((1, 3), dtype=np.int64))
        np.save(os.path.join(contact_dir, "train.npy"), np.ones(1, dtype=np.float32))
        np.save(os.path.join(indices_dir, "test.npy"), np.ones((1, 3), dtype=np.int64))
        np.save(os.path.join(contact_dir, "test.npy"), np.ones(1, dtype=np.float32))

        probabilities, totals, used_files = build_code_contact_map(
            indices_dir, contact_dir, 2, included_names={"train"}
        )
        assert used_files == 1
        np.testing.assert_allclose(totals[:, 0], 1.0)
        np.testing.assert_allclose(totals[:, 1], 0.0)
        np.testing.assert_allclose(probabilities[:, 0], 1.0)
        np.testing.assert_allclose(probabilities[:, 1], 0.5)


def test_token_cache_crop_is_sample_keyed_and_resume_stable():
    from dataset.dataset_interx_tokenize import deterministic_cache_crop

    expected = deterministic_cache_crop('G001T000A000R000', 151, 4, 123)
    # Loader order and unrelated RNG consumption cannot change a sample crop.
    np.random.seed(999)
    random.seed(999)
    np.random.rand(100)
    [random.random() for _ in range(100)]
    assert deterministic_cache_crop('G001T000A000R000', 151, 4, 123) == expected
    cropped_length, crop_start = expected
    assert cropped_length in (148, 144)
    assert cropped_length % 4 == 0
    assert 0 <= crop_start <= 151 - cropped_length


def test_shape_overlay_tokenizer_matches_recovered_historical_root_contract():
    import dataset.dataset_interx_tokenize_historical_root as historical_root
    from dataset.dataset_interx_tokenize import (
        deterministic_cache_crop as raw_deterministic_cache_crop,
    )

    assert historical_root.TOKENIZER_CONTRACT == 'deterministic_historical_root_v1'
    assert 'prepare_interx' not in inspect.getsource(historical_root)

    # The recovered contract changes only normalization. Crop selection remains
    # sample keyed and exactly shared with the audited cache builder.
    for name, length, unit_length, seed in (
        ('G001T000A000R000', 151, 4, 123),
        ('G032T004A017R003', 148, 4, 114514),
        ('G010T003A008R001', 150, 8, 7),
    ):
        assert historical_root.deterministic_cache_crop(
            name, length, unit_length, seed
        ) == raw_deterministic_cache_crop(name, length, unit_length, seed)

    with tempfile.TemporaryDirectory() as directory:
        name = 'G001T000A000R000'
        split = os.path.join(directory, 'split.txt')
        motion = os.path.join(directory, 'motion.h5')
        beta_dir = os.path.join(directory, 'betas')
        contact_dir = os.path.join(directory, 'contact')
        os.makedirs(beta_dir)
        os.makedirs(contact_dir)
        with open(split, 'w', encoding='utf-8') as handle:
            handle.write(name + '\n')
        with h5py.File(motion, 'w') as handle:
            handle.create_dataset(name, data=np.zeros((32, 56, 6), dtype=np.float32))
        np.savez(
            os.path.join(beta_dir, name + '.npz'),
            betas=np.zeros((2, 10), dtype=np.float32),
            gender=np.asarray(['male', 'female']),
        )
        np.save(
            os.path.join(contact_dir, name + '.npy'),
            np.zeros(32, dtype=np.float32),
        )

        dataset = historical_root.Text2MotionDatasetV2HHI(
            split, None, motion, unit_length=4, cache_seed=114514,
            beta_dir=beta_dir, contact_dir=contact_dir, strict_assets=True,
            stats_dir='dataset/interx', stats_prefix='Interx',
        )
        assert dataset.tokenizer_contract == historical_root.TOKENIZER_CONTRACT
        torch.testing.assert_close(
            dataset.normalizer.motion_mean,
            torch.from_numpy(np.load('dataset/interx/Interx_Mean_root.npy')).float(),
        )
        torch.testing.assert_close(
            dataset.normalizer.motion_std,
            torch.from_numpy(np.load('dataset/interx/Interx_Std_root.npy')).float(),
        )
        assert not torch.equal(
            dataset.normalizer.motion_std,
            torch.from_numpy(np.load('dataset/interx/Interx_Std.npy')).float(),
        )


def test_observational_export_disables_training_side_effects_and_restores_mode():
    module = _TrainingSideEffect().train()
    with evaluating(module), torch.no_grad():
        assert not module.training
        module()
    assert module.training
    assert module.updates.item() == 0

    module.eval()
    with evaluating(module), torch.no_grad():
        module()
    assert not module.training
    assert module.updates.item() == 0


def test_mgt_training_rejects_transformer_eval_mode_after_sanity_validation():
    import inspect

    # The training constructor must leave the transformer in its PyTorch
    # default train mode. Lightning restores the exact pre-sanity child-module
    # modes and does not repair an explicitly initialized eval child.
    init_source = inspect.getsource(MaskGITLightning.__init__)
    assert 'self.trans_encoder.eval()' not in init_source

    module = MaskGITLightning.__new__(MaskGITLightning)
    torch.nn.Module.__init__(module)
    module.trans_encoder = torch.nn.Sequential(torch.nn.Dropout(0.1)).train()
    module.net = torch.nn.Identity().eval()
    module.historical_replay = True
    module.on_train_start()

    module.trans_encoder.eval()
    try:
        module.on_train_start()
    except RuntimeError as error:
        assert 'dropout would be disabled' in str(error)
    else:
        raise AssertionError('MGT accepted a transformer in eval mode for training')


def test_mgt_restart_resolution_and_in_place_resume_are_strict():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        experiment = 'formal_restart_fixture'
        pattern = str(root / f'*_{experiment}')
        assert resolve_training_state(pattern, experiment, 300)['status'] == 'new'

        failed_experiment = 'failed_precheckpoint_fixture'
        failed_run = root / f'2026-09-01-00-00-00_{failed_experiment}'
        failed_run.mkdir()
        failed_id = 'deadbeef'
        (failed_run / 'training_manifest.json').write_text(json.dumps({
            'resolved_arguments': {'exp_name': failed_experiment},
            'observability': {'wandb_run_id': failed_id},
        }), encoding='utf-8')
        failed_wandb = root / 'wandb' / f'run-20260901_000000-{failed_id}' / 'files'
        failed_wandb.mkdir(parents=True)
        (failed_wandb / 'output.log').write_text(
            'Traceback (most recent call last):\n'
            '  File "train.py", line 1, in <module>\n'
            'UnboundLocalError: fixture failure\n',
            encoding='utf-8',
        )
        failed_pattern = str(root / f'*_{failed_experiment}')
        try:
            resolve_training_state(failed_pattern, failed_experiment, 300)
        except RuntimeError as error:
            assert 'not explicitly acknowledged as failed' in str(error)
        else:
            raise AssertionError('unacknowledged checkpoint-free run was ignored')
        acknowledged = resolve_training_state(
            failed_pattern, failed_experiment, 300,
            acknowledged_failed_run_ids=[failed_id],
            wandb_root=root / 'wandb',
        )
        assert acknowledged['status'] == 'new'
        assert acknowledged['matched_runs'] == []
        assert acknowledged['abandoned_attempts'][0]['wandb_run_id'] == failed_id
        assert acknowledged['abandoned_attempts'][0]['latest_step'] == 0
        assert acknowledged['abandoned_attempts'][0]['terminal_exception'] == (
            'UnboundLocalError: fixture failure'
        )

        run = root / f'2026-08-29-00-00-00_{experiment}'
        run.mkdir()
        checkpoint = run / 'latest-000000100.ckpt'
        checkpoint.write_bytes(b'atomic checkpoint fixture')

        values = {key: None for key in MGT_RESUME_IDENTITY_KEYS}
        values.update({
            'exp_name': experiment,
            'total_iter': 300,
            'sampling_steps': 50,
            'training_mask_contract': 'paper_kinematic',
        })
        source_hashes = {
            relative: report_sha256_file(Path(relative))
            for relative in MGT_TRAINING_SOURCE_PATHS
        }
        manifest = {
            'resolved_arguments': values,
            'sha256': source_hashes,
        }
        (run / 'training_manifest.json').write_text(
            json.dumps(manifest), encoding='utf-8'
        )

        resolved = resolve_training_state(pattern, experiment, 300)
        assert resolved['status'] == 'resume'
        assert resolved['resume_checkpoint'] == str(checkpoint.resolve())
        args = types.SimpleNamespace(**values)
        args.out_dir = str(root)
        args.resume_training_ckpt = str(checkpoint)
        args.resume_in_place = True
        resume = prepare_mgt_output_directory(args)
        assert resume['mode'] == 'strict_in_place_resume'
        assert resume['checkpoint_step'] == 100
        assert args.out_dir == str(run.resolve())

        drifted = types.SimpleNamespace(**values)
        drifted.out_dir = str(root)
        drifted.resume_training_ckpt = str(checkpoint)
        drifted.resume_in_place = True
        drifted.sampling_steps = 10
        try:
            prepare_mgt_output_directory(drifted)
        except ValueError as error:
            assert 'resume contract differs' in str(error)
        else:
            raise AssertionError('MGT resume accepted a sampling-contract drift')

        manifest['sha256']['utils/runtime.py'] = '0' * 64
        (run / 'training_manifest.json').write_text(
            json.dumps(manifest), encoding='utf-8'
        )
        source_drifted = types.SimpleNamespace(**values)
        source_drifted.out_dir = str(root)
        source_drifted.resume_training_ckpt = str(checkpoint)
        source_drifted.resume_in_place = True
        try:
            prepare_mgt_output_directory(source_drifted)
        except ValueError as error:
            assert 'resume source differs' in str(error)
        else:
            raise AssertionError('MGT resume accepted launch-source drift')

        manifest['sha256'] = source_hashes
        (run / 'training_manifest.json').write_text(
            json.dumps(manifest), encoding='utf-8'
        )

        completed = run / 'latest-000000300.ckpt'
        completed.write_bytes(b'completed checkpoint fixture')
        resolved = resolve_training_state(pattern, experiment, 300)
        assert resolved['status'] == 'complete'
        assert resolved['training_required'] is False


def test_generic_lightning_state_resolver_requires_full_resume_state():
    from tools.resolve_lightning_checkpoint_state import resolve

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'last.ckpt'
        full_state = {
            'global_step': 123,
            'epoch': 4,
            'state_dict': {'weight': torch.ones(1)},
            'optimizer_states': [{'state': {}, 'param_groups': [{}]}],
            'lr_schedulers': [{'last_epoch': 4}],
            'loops': {'fit_loop': {'epoch_progress': 4}},
        }
        torch.save(full_state, checkpoint)
        resumed = resolve(checkpoint, expected_final_step=200)
        assert resumed['status'] == 'resume'
        assert resumed['global_step'] == 123
        assert resumed['training_required'] is True
        assert resumed['resume_checkpoint'] == str(checkpoint.resolve())

        full_state['global_step'] = 200
        torch.save(full_state, checkpoint)
        completed = resolve(checkpoint, expected_final_step=200)
        assert completed['status'] == 'complete'
        assert completed['training_required'] is False
        assert completed['resume_checkpoint'] is None

        del full_state['optimizer_states']
        torch.save(full_state, checkpoint)
        try:
            resolve(checkpoint, expected_final_step=200)
        except ValueError as error:
            assert 'full stateful resume' in str(error)
        else:
            raise AssertionError('weights-only checkpoint was accepted for resume')


def test_coshmdm_queue_has_strict_os_restart_resume_gate():
    source = Path('tools/queue_gpu0_cosh_timotion.sh').read_text(encoding='utf-8')
    assert source.index('while pgrep -f') < source.index(
        'tools/resolve_lightning_checkpoint_state.py'
    )
    assert source.index('tools/audit_coshmdm_training_contract.py') < source.index(
        'tools/resolve_lightning_checkpoint_state.py'
    )
    assert source.count('tools/resolve_lightning_checkpoint_state.py') == 2
    assert source.count('--expected-final-step 568000') == 2
    assert 'if [[ "${cosh_status}" == resume ]]' in source
    assert '--resume-checkpoint "${cosh_checkpoint}"' in source
    assert source.index('--resume-checkpoint "${cosh_checkpoint}"') < source.rindex(
        'tools/resolve_lightning_checkpoint_state.py'
    )
    assert "['selected_step']" in source
    assert 'audit_step_args=(--expected-step "${cosh_selected_step}")' in source
    assert source.index("['selected_step']") < source.index(
        'audit_step_args=(--expected-step "${cosh_selected_step}")'
    )
    assert 'tools/audit_coshmdm_resume_provenance.py --require-complete' in source
    assert 'rm ' not in source
    final_source = Path(
        'tools/queue_review_report_after_all_training.sh'
    ).read_text(encoding='utf-8')
    assert final_source.index(
        'tools/audit_coshmdm_representation_adapter.py'
    ) < final_source.index('tools/audit_coshmdm_generation_contract.py')
    assert final_source.index(
        'tools/audit_coshmdm_generation_contract.py'
    ) < final_source.index('tools/build_review_results_report.py')
    assert final_source.index(
        'tools/audit_timotion_generation_contract.py'
    ) < final_source.index('tools/build_review_results_report.py')


def test_coshmdm_validation_selection_requires_finalized_two_day_cap():
    base = {
        'schema': 'hhc_coshmdm_user_training_cap_v1',
        'preferred_terminal_optimizer_step': 200000,
        'policy': CAP_POLICY,
    }
    for status in ACCEPTED_CAP_STATUSES:
        validate_budget_contract({**base, 'status': status})
    for invalid in (
        {**base, 'status': 'monitoring'},
        {**base, 'status': 'interrupt_sent'},
        {**base, 'status': 'stopped_gracefully',
         'preferred_terminal_optimizer_step': 568000},
        {**base, 'status': 'stopped_gracefully', 'policy': 'train forever'},
    ):
        try:
            validate_budget_contract(invalid)
        except ValueError as error:
            assert 'finalized valid two-day training-cap' in str(error)
        else:
            raise AssertionError(f'unfinalized/invalid cap was accepted: {invalid}')

    selector_source = Path(
        'tools/select_coshmdm_checkpoint_on_interx_val.py'
    ).read_text(encoding='utf-8')
    assert 'checkpoint_hash[:16]' in selector_source
    assert 'checkpoint_source_stable_during_export' in selector_source
    assert selector_source.index('checkpoint_hash[:16]') < selector_source.index(
        'subprocess.run(command, cwd=COSH_ROOT, env=environment, check=True)'
    ) < selector_source.index(
        'checkpoint_source_stable_during_export'
    )


def test_coshmdm_milestones_are_zero_copy_and_bounded_to_user_cap():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_20k = root / 'epoch=70-step=20000.ckpt'
        source_60k = root / 'epoch=211-step=60000.ckpt'
        source_220k = root / 'epoch=774-step=220000.ckpt'
        source_20k.write_bytes(b'20k full-state fixture')
        source_60k.write_bytes(b'60k full-state fixture')
        source_220k.write_bytes(b'outside user cap')
        assert eligible_sources(root, 200000) == [
            (20000, source_20k.resolve()),
            (60000, source_60k.resolve()),
        ]
        target = root / 'milestone-000060000.ckpt'
        assert preserve(source_60k, target) == 'created_hard_link'
        assert os.path.samefile(source_60k, target)
        assert source_60k.stat().st_ino == target.stat().st_ino
        assert preserve(source_60k, target) == 'preexisting_same_inode'

        wrong = root / 'milestone-000020000.ckpt'
        wrong.write_bytes(b'different checkpoint fixture')
        try:
            preserve(source_20k, wrong)
        except RuntimeError as error:
            assert 'not the source inode' in str(error)
        else:
            raise AssertionError('a different milestone inode was accepted')


def test_mgt_validation_pair_survives_rolling_and_topk_checkpoint_deletion():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rolling = root / 'latest-000020000.ckpt'
        rolling.write_bytes(b'pre-validation state fixture')
        callback = FixedStepValidationPairCheckpoint(root, step=20000)
        class _Trainer:
            global_step = 20000
            is_global_zero = True

            @staticmethod
            def save_checkpoint(path, weights_only=False):
                assert weights_only is False
                Path(path).write_bytes(b'post-validation state fixture')

        trainer = _Trainer()
        callback.on_train_batch_end(trainer, None, None, None, 0)
        frozen = root / 'prevalidation-000020000.ckpt'
        assert frozen.read_bytes() == b'pre-validation state fixture'
        assert frozen.stat().st_ino == rolling.stat().st_ino
        rolling.unlink()
        assert frozen.read_bytes() == b'pre-validation state fixture'
        callback.on_validation_end(trainer, None)
        post = root / 'postvalidation-000020000.ckpt'
        assert post.read_bytes() == b'post-validation state fixture'
        assert callback.state_dict() == {
            'step': 20000, 'pre_saved': True, 'post_saved': True,
        }


def test_mgt_validation_pair_audit_requires_post_validation_marker():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        before = root / 'prevalidation-000020000.ckpt'
        after = root / 'postvalidation-000020000.ckpt'
        output = root / 'audit.json'

        def checkpoint(post_saved):
            return {
                'global_step': 20000,
                'state_dict': {'weight': torch.tensor([1.0])},
                'optimizer_states': [{'state': {}}],
                'lr_schedulers': [{'last_epoch': 20000}],
                'hyper_parameters': {'contract': 'paper'},
                'loops': {},
                'callbacks': {
                    'ModelCheckpoint{monitor=val/FID}': {
                        'monitor': 'val/FID',
                        'current_score': torch.tensor(0.5),
                        'best_model_score': torch.tensor(0.5),
                        'best_model_path': 'model-000020000-0.5000.ckpt',
                        'best_k_models': {},
                    },
                    'FixedStepValidationPairCheckpoint': {
                        'step': 20000,
                        'pre_saved': post_saved,
                        'post_saved': post_saved,
                    },
                },
            }

        torch.save(checkpoint(False), before)
        torch.save(checkpoint(True), after)
        command = [
            sys.executable, 'tools/audit_lightning_checkpoint_pair.py',
            '--before', str(before), '--after', str(after),
            '--expected-step', '20000', '--require-fixed-pair-marker',
            '--output', str(output),
        ]
        result = subprocess.run(
            command, cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(output.read_text())['fixed_pair_callback']['after'][
            'post_saved'
        ] is True

        torch.save(checkpoint(False), after)
        result = subprocess.run(
            command, cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode != 0
        assert 'fixed validation-pair callback markers are invalid' in result.stderr


def test_stochastic_validation_restores_training_rng_streams():
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    with preserving_rng_state():
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        random.random(), np.random.rand(), torch.rand(1)
    actual = (random.random(), np.random.rand(), torch.rand(4))

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    expected = (random.random(), np.random.rand(), torch.rand(4))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])

def test_contact_guidance_requires_matching_train_only_manifest():
    with tempfile.TemporaryDirectory() as directory:
        vq_path = os.path.join(directory, 'net.pth')
        cache_directory = os.path.join(directory, 'augmented_cache')
        map_path = os.path.join(cache_directory, 'code_contact_probs.npy')
        manifest_directory = os.path.join(cache_directory, 'codebook')
        manifest_path = os.path.join(manifest_directory, '_cache_manifest.json')
        os.makedirs(manifest_directory)
        with open(vq_path, 'wb') as handle:
            handle.write(b'vq checkpoint fixture')
        np.save(map_path, np.zeros((3, 2), dtype=np.float32))
        manifest = {
            'status': 'complete',
            'contact_map_scope': 'train_only',
            'contact_map_estimator': 'empirical_frequency_unseen_neutral_0.5',
            'vq_checkpoint_sha256': sha256_file(vq_path),
            'contact_map_sha256': sha256_file(map_path),
            'normalization_mean_sha256': sha256_file(
                'dataset/interx/Interx_Mean.npy'
            ),
            'normalization_std_sha256': sha256_file(
                'dataset/interx/Interx_Std.npy'
            ),
        }
        with open(manifest_path, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle)
        assert validate_contact_map_provenance(vq_path, map_path) == manifest_path

        manifest['contact_map_scope'] = 'train+val'
        with open(manifest_path, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle)
        try:
            validate_contact_map_provenance(vq_path, map_path)
        except RuntimeError as error:
            assert 'contact_map_scope' in str(error)
        else:
            raise AssertionError('non-training-only contact map was accepted')


def test_historical_mmm_cache_import_is_exact_and_excludes_test_tokens():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        code_source = root / 'source_code'
        contact_source = root / 'source_contact'
        code_destination = root / 'isolated_code'
        contact_destination = root / 'isolated_contact'
        code_source.mkdir()
        contact_source.mkdir()

        payloads = {
            'train_a': (b'code-a', b'contact-a'),
            'val_b': (b'code-b', b'contact-b'),
            'test_c': (b'code-test', b'contact-test'),
        }
        for name, (code_bytes, contact_bytes) in payloads.items():
            (code_source / f'{name}.npy').write_bytes(code_bytes)
            (contact_source / f'{name}.npy').write_bytes(contact_bytes)

        imported = import_historical_token_pairs(
            code_source, contact_source, code_destination,
            contact_destination, {'train_a', 'val_b'},
        )
        assert imported == 2
        assert {path.stem for path in code_destination.glob('*.npy')} == {
            'train_a', 'val_b'
        }
        assert {path.stem for path in contact_destination.glob('*.npy')} == {
            'train_a', 'val_b'
        }
        for name in ('train_a', 'val_b'):
            assert (code_destination / f'{name}.npy').read_bytes() == payloads[name][0]
            assert (contact_destination / f'{name}.npy').read_bytes() == payloads[name][1]

        assert_exact_historical_cache_scope(
            {'train_a', 'val_b'}, {'train_a', 'val_b'}, {'train_a', 'val_b'}
        )
        try:
            assert_exact_historical_cache_scope(
                {'train_a', 'val_b', 'test_c'},
                {'train_a', 'val_b'},
                {'train_a', 'val_b'},
            )
        except RuntimeError as error:
            assert 'outside official eligible train+val' in str(error)
        else:
            raise AssertionError('historical test token was accepted in training cache')


def test_historical_mmm_cache_import_preflights_all_pairs_before_writing():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        code_source = root / 'source_code'
        contact_source = root / 'source_contact'
        code_destination = root / 'isolated_code'
        contact_destination = root / 'isolated_contact'
        code_source.mkdir()
        contact_source.mkdir()
        (code_source / 'train_a.npy').write_bytes(b'code-a')
        # Deliberately omit the matching contact token.
        try:
            import_historical_token_pairs(
                code_source, contact_source, code_destination,
                contact_destination, {'train_a'},
            )
        except FileNotFoundError as error:
            assert 'contact:train_a' in str(error)
        else:
            raise AssertionError('incomplete historical pair was imported')
        assert not code_destination.exists()
        assert not contact_destination.exists()


def test_legacy_axis_bug_contact_map_requires_explicit_audited_contract():
    vq_path = 'checkpoints/vq_precleanup_raw12288/net_175000.pth'
    legacy_map = (
        'checkpoints/vq_precleanup_raw12288/'
        'code_contact_probs_mmm_legacy_full_axis_bug.npy'
    )
    try:
        validate_contact_map_provenance(vq_path, legacy_map)
    except RuntimeError as error:
        assert 'contact_map_sha256' in str(error)
    else:
        raise AssertionError('legacy leaking/axis-bug map was accepted as fair')

    audit_path = validate_contact_map_provenance(
        vq_path, legacy_map,
        contact_map_contract='mmm_legacy_full_axis_bug',
    )
    assert Path(audit_path).resolve() == Path(
        'output/hhc_contact_guidance_map_audit.json'
    ).resolve()


def test_transformer_checkpoint_cannot_replace_requested_vq():
    def build_small_model():
        return Text2Motion_Transformer(
            _Tokenizer(8), num_vq=8, embed_dim=32, clip_dim=16,
            block_size=5, num_layers=1, num_local_layer=1, n_head=4,
            drop_out_rate=0.0, fc_rate=2, shape=False,
            paper_architecture=True,
        )

    source = build_small_model()
    target = build_small_model()
    preserved = {}
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            if name.startswith('trans_base.vqvae.'):
                parameter.fill_(42.0)
                preserved[name] = parameter.clone()

    with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as handle:
        path = handle.name
    try:
        torch.save({'trans_encoder': source.state_dict()}, path)
        load_transformer_checkpoint(target, path, preserve_vqvae=True)
        for name, expected in preserved.items():
            torch.testing.assert_close(dict(target.named_parameters())[name], expected)
        torch.testing.assert_close(
            target.trans_head.head.weight, source.trans_head.head.weight
        )
    finally:
        os.unlink(path)


def test_main_and_multimodality_generation_provenance_must_match():
    main = {
        'checkpoint': '/models/model.ckpt',
        'checkpoint_sha256': 'abc',
        'requests': '/requests/replication_00.npz',
        'requests_sha256': 'request-content-hash',
        'seed_strategy': 'base_seed + repeat_index',
        'seed': 1234,
        'sampler': 'sample_intra_cas',
        'sampler_semantics': 'paper_high_confidence_tokens_fixed',
        'cascade_order': 'root,body,hands',
        'sampling_steps': 50,
        'sampling_steps_unit': 'outer_maskgit_iterations',
        'maximum_active_channel_stages_per_outer_iteration': 3,
        'maximum_model_forward_calls_per_outer_iteration_with_cfg': 6,
        'resolved_channels_skip_forward_but_not_outer_iterations': True,
        'temperature': 0.5,
    }
    multimodality = dict(main, seed=101234)
    validate_generation_metadata(main, multimodality, 1234)

    mismatched = dict(multimodality, checkpoint_sha256='different')
    try:
        validate_generation_metadata(main, mismatched, 1234)
    except ValueError as error:
        assert 'checkpoint_sha256' in str(error)
    else:
        raise AssertionError('main/MM files from different checkpoints were accepted')

    mismatched = dict(multimodality, requests_sha256='different-request')
    try:
        validate_generation_metadata(main, mismatched, 1234)
    except ValueError as error:
        assert 'requests_sha256' in str(error)
    else:
        raise AssertionError('main/MM files from different frozen requests were accepted')

    mismatched = dict(
        multimodality, maximum_model_forward_calls_per_outer_iteration_with_cfg=3
    )
    try:
        validate_generation_metadata(main, mismatched, 1234)
    except ValueError as error:
        assert 'maximum_model_forward_calls_per_outer_iteration_with_cfg' in str(error)
    else:
        raise AssertionError('different HHC compute budgets were accepted')

    oracle_main = dict(
        main,
        contact_guidance=True,
        contact_timing_source='request_oracle',
        contact_timing_file='/requests/replication_00.npz',
        contact_timing_file_sha256='request-content-hash',
    )
    predicted_mm = dict(
        multimodality,
        contact_guidance=True,
        contact_timing_source='external_text_predictor',
        contact_timing_file='/timing/predicted.npz',
        contact_timing_file_sha256='predicted-hash',
    )
    try:
        validate_generation_metadata(oracle_main, predicted_mm, 1234)
    except ValueError as error:
        assert 'contact_timing' in str(error)
    else:
        raise AssertionError('main/MM files with different timing sources were accepted')

    conditioned_main = dict(
        main,
        sampling_process='author_autoregressive_ancestral_10_steps',
        first_frame_conditioning='none',
    )
    conditioned_mm = dict(
        multimodality,
        sampling_process='author_autoregressive_ancestral_10_steps',
        first_frame_conditioning='ground_truth',
    )
    try:
        validate_generation_metadata(conditioned_main, conditioned_mm, 1234)
    except ValueError as error:
        assert 'first_frame_conditioning' in str(error)
    else:
        raise AssertionError('main/MM files with different first-frame conditions were accepted')


def test_metric_aggregation_requires_unique_requests_and_one_sampler():
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        records = []
        for replication in range(2):
            metric_path = os.path.join(directory, f'replication_{replication:02d}.json')
            request_path = os.path.join(directory, f'replication_{replication:02d}.npz')
            main_path = os.path.join(directory, f'replication_{replication:02d}_main.h5')
            mm_path = os.path.join(directory, f'replication_{replication:02d}_mm.h5')
            with open(request_path, 'wb') as handle:
                handle.write(f'unique request {replication}'.encode())
            with open(main_path, 'wb') as handle:
                handle.write(f'main generation {replication}'.encode())
            with open(mm_path, 'wb') as handle:
                handle.write(f'mm generation {replication}'.encode())
            paths.append(metric_path)
            records.append({
                'requests': request_path,
                'requests_sha256': sha256_file(request_path),
                'main_generations': main_path,
                'main_generations_sha256': sha256_file(main_path),
                'mm_generations': mm_path,
                'mm_generations_sha256': sha256_file(mm_path),
                'generation_request_hash_verified': True,
                'main_generation_metadata': {
                    'checkpoint_sha256': 'one-model',
                    'sampling_steps': 10,
                    'maximum_active_channel_stages_per_outer_iteration': 3,
                    'maximum_model_forward_calls_per_outer_iteration_with_cfg': 6,
                    'resolved_channels_skip_forward_but_not_outer_iterations': True,
                },
            })

        provenance = validate_replication_provenance(paths, records)
        assert provenance['unique_requests_verified']
        assert provenance['replication_ids'] == [0, 1]
        assert provenance['sampler_identity'][
            'maximum_active_channel_stages_per_outer_iteration'
        ] == 3
        assert provenance['sampler_identity'][
            'maximum_model_forward_calls_per_outer_iteration_with_cfg'
        ] == 6
        assert provenance['sampler_identity'][
            'resolved_channels_skip_forward_but_not_outer_iterations'
        ] is True

        legacy_records = []
        for record in records:
            legacy = dict(record)
            legacy.pop('main_generations_sha256')
            legacy.pop('mm_generations_sha256')
            legacy_records.append(legacy)
        legacy_provenance = validate_replication_provenance(
            paths, legacy_records, allow_legacy=True
        )
        assert legacy_provenance['legacy_provenance_allowed']
        assert not legacy_provenance['generation_files_sha256_verified']

        duplicated = [records[0], dict(records[1], requests=records[0]['requests'])]
        try:
            validate_replication_provenance(paths, duplicated)
        except ValueError as error:
            assert 'expected request' in str(error) or 'unique' in str(error)
        else:
            raise AssertionError('duplicate frozen requests were accepted')

        different_sampler = [records[0], dict(records[1])]
        different_sampler[1]['main_generation_metadata'] = {
            'checkpoint_sha256': 'another-model', 'sampling_steps': 10,
        }
        try:
            validate_replication_provenance(paths, different_sampler)
        except ValueError as error:
            assert 'identity differs' in str(error)
        else:
            raise AssertionError('different model identities were pooled')

        corrupted = [records[0], dict(records[1])]
        with open(corrupted[1]['mm_generations'], 'ab') as handle:
            handle.write(b'corruption')
        try:
            validate_replication_provenance(paths, corrupted)
        except ValueError as error:
            assert 'mm_generations hash' in str(error)
        else:
            raise AssertionError('modified MM generation HDF5 was accepted')


def test_contact_optimization_defaults_match_original_mmm_baseline():
    signature = inspect.signature(optimize_full_body)
    # The user explicitly selected the current/original MMM optimizer as the
    # shape-augmentation baseline. Git 96b45bd used these same defaults and a
    # hard-coded freeze_epoch=100; VolumetricSMPL may change only distance
    # evaluation, not this schedule or the nearest/contact loss logic.
    assert signature.parameters['num_epochs'].default == 200
    assert signature.parameters['lr'].default == 0.01
    source = inspect.getsource(optimize_full_body)
    assert 'freeze_epoch=100' in source
    loss_signature = inspect.signature(DyadicInteractionLoss)
    assert loss_signature.parameters['signed_distance_backend'].default == 'nearest_vertex'
    assert signature.parameters['signed_distance_backend'].default == 'nearest_vertex'
    assert signature.parameters['volumetric_smpl_seed'].default == 1234
    assert signature.parameters['volumetric_smpl_backend'].default is None
    assert loss_signature.parameters['volumetric_smpl_backend'].default is None
    penetration_source = inspect.getsource(
        DyadicInteractionLoss._calc_penetration_loss
    )
    assert 'normalize_by_num_penetrating=True' in penetration_source
    assert 'thresh_shallow=0.2' in penetration_source
    assert 'thresh_deep=0.5' in penetration_source
    metric_source = inspect.getsource(
        DyadicInteractionLoss._calc_penetration_metric
    )
    assert 'normalize_by_num_penetrating=False' in metric_source
    assert 'thresh_shallow=10.0' in metric_source
    contact_source = inspect.getsource(
        DyadicInteractionLoss._calc_semantic_contact_loss
    )
    assert 'torch.relu(dist_signed_A)' in contact_source
    assert 'torch.relu(dist_signed_B)' in contact_source
    forward_source = inspect.getsource(DyadicInteractionLoss.forward)
    contact_index = forward_source.index('if interaction_pairs is not None:')
    freeze_index = forward_source.index('if epoch == freeze_epoch')
    penetration_index = forward_source.index(
        'loss_pene = self._calc_penetration_loss', freeze_index
    )
    assert contact_index < freeze_index < penetration_index


def test_fine_contact_freeze_is_unique_bidirectional_matching():
    loss = DyadicInteractionLoss.__new__(DyadicInteractionLoss)
    loss.is_anchored = False
    loss.fixed_pairs = {}
    loss.current_winner_regions = {
        0: (torch.arange(3), torch.arange(3))
    }
    loss.device = 'cpu'
    verts_a = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    verts_b = torch.tensor([[[0.1, 0.0, 0.0], [2.1, 0.0, 0.0], [9.0, 0.0, 0.0]]])
    normals = torch.zeros_like(verts_a)
    loss.freeze_contact_candidates(
        verts_a, verts_b, normals, normals, [{}], torch.ones(1)
    )
    indices_a = loss.fixed_pairs[0]['idx_A'].tolist()
    indices_b = loss.fixed_pairs[0]['idx_B'].tolist()
    assert len(indices_a) == len(set(indices_a))
    assert len(indices_b) == len(set(indices_b))
    assert len(indices_a) == 3
    freeze_source = inspect.getsource(
        DyadicInteractionLoss.freeze_contact_candidates
    )
    assert 'LOCK_DIST_THRESH' not in freeze_source
    assert 'LOCK_NORMAL_THRESH' not in freeze_source


def test_shared_volumetric_backend_is_identity_reused_and_contract_checked():
    original_body_model = losses_module.BodyModel
    original_backend_class = losses_module.VolumetricSMPLBackend

    class FakeBodyModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class FakeBackend(nn.Module):
        def __init__(self, device='cpu', query_chunk=512):
            super().__init__()
            self.device = torch.device(device)
            self.max_query_points_per_call = query_chunk

    try:
        losses_module.BodyModel = FakeBodyModel
        losses_module.VolumetricSMPLBackend = FakeBackend
        shared = FakeBackend(device='cpu', query_chunk=512)
        loss = DyadicInteractionLoss(
            device='cpu', volumetric_smpl_query_chunk=512,
            volumetric_smpl_backend=shared,
        )
        assert loss.volumetric_smpl is shared

        wrong_chunk = FakeBackend(device='cpu', query_chunk=256)
        try:
            DyadicInteractionLoss(
                device='cpu', volumetric_smpl_query_chunk=512,
                volumetric_smpl_backend=wrong_chunk,
            )
        except ValueError as error:
            assert 'query chunk differs' in str(error)
        else:
            raise AssertionError('shared backend with another query chunk accepted')

        try:
            DyadicInteractionLoss(
                device='cpu', volumetric_smpl_query_chunk=512,
                volumetric_smpl_backend=nn.Identity(),
            )
        except TypeError as error:
            assert 'VolumetricSMPLBackend instance' in str(error)
        else:
            raise AssertionError('wrong shared backend type accepted')
    finally:
        losses_module.BodyModel = original_body_model
        losses_module.VolumetricSMPLBackend = original_backend_class


def test_llm_contact_parser_fails_closed_on_out_of_schema_values():
    extractor = ContactExtractor.__new__(ContactExtractor)
    table = """| **Person 1 Body Part** | **Person 2 Body Part** | **Contact Area** | **Role Ratio (P1:P2)** |
|---|---|---|---|
| right hand (palm) | left shoulder | flat | 9:1 |
"""
    parsed = extractor._parse_markdown_table(table)
    mapped = extractor.map_contacts_to_ids(parsed, CANDIDATES)
    assert len(mapped) == 1
    assert mapped[0]['contact_type'] == 'flat'
    assert mapped[0]['align_type'] == 'strict'
    assert abs(mapped[0]['w_p1'] - 0.9) < 1e-6
    assert abs(mapped[0]['w_p2'] - 0.1) < 1e-6

    invalid = [dict(parsed[0], **{'contact area': 'flat, wrap'})]
    try:
        extractor.map_contacts_to_ids(invalid, CANDIDATES)
    except ValueError as error:
        assert 'Contact Area' in str(error)
    else:
        raise AssertionError('out-of-schema contact area was silently accepted')

    try:
        extractor._parse_markdown_table('There are no contacts.')
    except ValueError as error:
        assert 'Markdown table' in str(error)
    else:
        raise AssertionError('missing table was confused with a valid empty table')


def test_contact_extraction_uses_exact_frozen_request_captions():
    with tempfile.TemporaryDirectory() as directory:
        root = os.path.abspath(directory)
        split = os.path.join(root, 'test.txt')
        text_dir = os.path.join(root, 'texts')
        requests = os.path.join(root, 'replication_00.npz')
        os.makedirs(text_dir)
        with open(split, 'w', encoding='utf-8') as handle:
            handle.write('sample_a\nsample_b\nsample_dropped\n')
        for name in ('sample_a', 'sample_b', 'sample_dropped'):
            with open(
                os.path.join(text_dir, f'{name}.txt'), 'w', encoding='utf-8'
            ) as handle:
                handle.write(f'first caption for {name}#tokens#0#0\n')
        np.savez_compressed(
            requests,
            names=np.asarray(['sample_b', 'sample_a']),
            captions=np.asarray(['selected B line 2', 'selected A line 3']),
        )
        records, source = load_caption_requests(
            Path(split), Path(text_dir), Path(requests)
        )
        assert [record['sample_name'] for record in records] == [
            'sample_b', 'sample_a'
        ]
        assert [record['caption'] for record in records] == [
            'selected B line 2', 'selected A line 3'
        ]
        assert [record['request_index'] for record in records] == [0, 1]
        assert source['caption_source'] == 'frozen_fair_evaluation_requests'
        assert source['requests_sha256'] == sha256_file(requests)


def test_codex_contact_cache_binds_frozen_requests_and_prompt_contract():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        split = root / 'test.txt'
        requests = root / 'replication_00.npz'
        contact_dir = root / 'contacts'
        metadata_dir = contact_dir / '_metadata'
        metadata_dir.mkdir(parents=True)
        split.write_text('sample_a\n', encoding='utf-8')
        names = ['sample_a']
        captions = ['Person 1 taps Person 2 on the shoulder.']
        np.savez_compressed(
            requests, names=np.asarray(names), captions=np.asarray(captions)
        )
        manifest = {
            'schema': 'hhc_interx_contact_pairs_codex_strict_v3',
            'strict_schema': True,
            'status': 'complete',
            'completed_count': 1,
            'sample_count': 1,
            'split_sha256': sha256_file(split),
            'caption_source': 'frozen_fair_evaluation_requests',
            'requests': str(requests.resolve()),
            'requests_sha256': sha256_file(requests),
            'request_names_sha256': contact_canonical_sha256(names),
            'request_captions_sha256': contact_canonical_sha256(captions),
            'contract_sha256': 'synthetic-contract',
            'prompt_template_sha256': 'synthetic-prompt',
            'provider': 'openai_codex_cli_chatgpt_login',
            'requested_model': 'gpt-5.6-luna',
            'reasoning_effort': 'low',
        }
        (contact_dir / '_extraction_manifest.json').write_text(
            json.dumps(manifest), encoding='utf-8'
        )
        metadata = {
            'schema': manifest['schema'],
            'sample_name': 'sample_a',
            'status': 'success',
            'prompt_template_sha256': manifest['prompt_template_sha256'],
            'provider': manifest['provider'],
            'requested_model': manifest['requested_model'],
            'reasoning_effort': manifest['reasoning_effort'],
            'response_model_version': manifest['requested_model'],
            'caption_source': manifest['caption_source'],
            'requests_sha256': manifest['requests_sha256'],
            'request_binding_writer': {
                'path': str(
                    Path(__file__).resolve().parents[1]
                    / 'eval/extract_interx_contacts_codex_frozen_requests.py'
                ),
                'sha256': sha256_file(
                    Path(__file__).resolve().parents[1]
                    / 'eval/extract_interx_contacts_codex_frozen_requests.py'
                ),
            },
            'caption': captions[0],
            'caption_sha256': hashlib.sha256(
                captions[0].encode('utf-8')
            ).hexdigest(),
        }
        (metadata_dir / 'sample_a.json').write_text(
            json.dumps(metadata), encoding='utf-8'
        )
        with (contact_dir / 'sample_a.pkl').open('wb') as handle:
            pickle.dump([], handle)

        loaded, _, _ = validate_contact_cache(contact_dir, split)
        contacts, _, _, _ = load_strict_contacts(
            contact_dir, 'sample_a', loaded
        )
        assert contacts == []

        metadata.pop('prompt_template_sha256')
        metadata['prompt_sha256'] = manifest['prompt_template_sha256']
        (metadata_dir / 'sample_a.json').write_text(
            json.dumps(metadata), encoding='utf-8'
        )
        try:
            load_strict_contacts(contact_dir, 'sample_a', loaded)
        except ValueError as error:
            assert 'prompt_template_sha256' in str(error)
        else:
            raise AssertionError('legacy prompt field was accepted as Codex v3')


def test_full_luna_cache_reports_manifest_wall_clock_without_fake_latency():
    root = Path(__file__).resolve().parents[1]
    pairs = root / 'dataset/interx/contact_pairs_codex_luna_strict_v3_train'
    manifest = json.loads(
        (pairs / '_extraction_manifest.json').read_text(encoding='utf-8')
    )
    batch_paths = sorted((pairs / '_batches').glob('*.json'))
    recovered = manifest_wall_clock(manifest, batch_paths)
    audit = json.loads(
        (root / 'output/interx_luna_train_cache_audit.json').read_text(
            encoding='utf-8'
        )
    )
    usage = audit['resource_usage']
    assert recovered == usage['end_to_end_wall_clock']
    assert recovered['batch_files'] == 285
    assert recovered['batch_mtime_window_inside_manifest'] is True
    assert recovered['total_seconds'] == 11363.309084
    assert usage['wall_seconds'] is None
    assert usage['invocations']['wall_timed_attempts'] == 0
    assert usage['invocations']['wall_untimed_attempts'] == 294


def test_contact_timing_sources_are_explicit_and_length_safe():
    with tempfile.TemporaryDirectory() as directory:
        request_path = os.path.join(directory, 'requests.npz')
        request_contact = np.zeros((2, 150), dtype=np.int8)
        request_contact[0, 3:5] = 1
        request_contact[1, 6:8] = 1
        np.savez_compressed(
            request_path,
            names=np.asarray(['a', 'b']),
            contact=request_contact,
        )
        request = np.load(request_path, allow_pickle=False)
        common = {
            'requests': request_path,
            'contact_timing_file': None,
        }
        args = types.SimpleNamespace(
            **common, contact_guidance=True,
            contact_timing_source='request_oracle',
        )
        contact, source, path, digest = resolve_contact_timing(
            args, request, np.arange(2), np.asarray(['a', 'b']),
            np.asarray([10, 12]),
        )
        assert np.array_equal(contact, request_contact)
        assert source == 'request_oracle'
        assert path == os.path.abspath(request_path)
        assert digest == sha256_file(request_path)

        args.contact_timing_source = 'full_valid_weak'
        contact, source, path, digest = resolve_contact_timing(
            args, request, np.arange(2), np.asarray(['a', 'b']),
            np.asarray([10, 12]),
        )
        assert source == 'full_valid_weak' and not path and not digest
        assert contact[0, :10].all() and not contact[0, 10:].any()
        assert contact[1, :12].all() and not contact[1, 12:].any()

        args.contact_guidance = False
        args.contact_timing_source = 'none'
        contact, source, path, digest = resolve_contact_timing(
            args, request, np.arange(2), np.asarray(['a', 'b']),
            np.asarray([10, 12]),
        )
        assert source == 'none' and not contact.any() and not path and not digest
        request.close()


def test_aggregate_binds_oracle_timing_to_each_request():
    with tempfile.TemporaryDirectory() as directory:
        request = os.path.join(directory, 'replication_00.npz')
        np.savez_compressed(request, names=np.asarray(['a']))
        digest = sha256_file(request)
        validate_contact_timing_provenance({
            'contact_timing_source': 'request_oracle',
            'contact_timing_file': request,
            'contact_timing_file_sha256': digest,
        }, request, digest)
        try:
            validate_contact_timing_provenance({
                'contact_timing_source': 'request_oracle',
                'contact_timing_file': request,
                'contact_timing_file_sha256': 'stale',
            }, request, digest)
        except ValueError as error:
            assert 'hash' in str(error)
        else:
            raise AssertionError('stale oracle contact timing was accepted')


def test_holm_bonferroni_is_monotone_in_sorted_p_values():
    raw = [0.04, 0.001, 0.02, 0.8]
    adjusted = holm_bonferroni(raw)
    assert np.allclose(adjusted, [0.08, 0.004, 0.06, 0.8])
    ordered = np.argsort(raw)
    assert np.all(np.diff(np.asarray(adjusted)[ordered]) >= 0)


def test_paired_method_comparison_respects_metric_directions():
    reference, comparator = {}, {}
    for replication in range(3):
        common = {
            '_request_sha256_verified': f'request-{replication}',
            'diversity_gt': 10.0,
        }
        reference[replication] = {
            **common,
            'fid': 1.0,
            'top1': 0.8, 'top2': 0.9, 'top3': 0.95,
            'matching': 2.0, 'multimodality': 2.0,
            'diversity': 9.0,
        }
        comparator[replication] = {
            **common,
            'fid': 2.0,
            'top1': 0.5, 'top2': 0.6, 'top3': 0.7,
            'matching': 3.0, 'multimodality': 1.0,
            'diversity': 7.0,
        }
    result = compare_methods(
        reference, comparator, np.random.default_rng(123), 100
    )
    for record in result.values():
        assert record['reference_advantage_mean'] > 0
        assert record['reference_wins'] == 3
        assert record['reference_losses'] == 0


def test_paired_replication_zero_sensitivity_filters_completed_reference():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for replication in (0, 1):
            request = directory / f'request_{replication:02d}.npz'
            np.savez_compressed(request, names=np.asarray(['sample']))
            payload = {
                'samples': 1696, 'mm_samples': 100, 'mm_repeats': 30,
                'requests': str(request),
                'requests_sha256': sha256_file(request),
                'generation_request_hash_verified': True,
            }
            (directory / f'replication_{replication:02d}.json').write_text(
                json.dumps(payload), encoding='utf-8'
            )
        selected = load_strict_replications(directory, 1, [0])
        assert set(selected) == {0}
        try:
            load_strict_replications(directory, 1)
        except ValueError as error:
            assert 'expected replication IDs' in str(error)
        else:
            raise AssertionError('extra replications were silently ignored')


def test_generation_source_must_match_training_source():
    training = {'repository': {
        'revision': 'launch-revision',
        'tracked_diff_sha256': 'launch-diff',
    }}
    provenance = {'repositories': {'hhc': {
        'revision': 'launch-revision',
        'tracked_diff_sha256': 'launch-diff',
    }}}
    validate_training_repository_matches_generation(
        'hhc-test', training, provenance, 'hhc'
    )
    provenance['repositories']['hhc']['tracked_diff_sha256'] = 'later-diff'
    try:
        validate_training_repository_matches_generation(
            'hhc-test', training, provenance, 'hhc'
        )
    except ValueError as error:
        assert 'differs between training and generation' in str(error)
    else:
        raise AssertionError('post-training tracked source drift was accepted')


def test_generation_checkpoints_are_bound_to_training_audits():
    selected = '/tmp/hhc-selected.ckpt'
    cosh = '/tmp/cosh-last.ckpt'
    timotion = '/tmp/timotion-last.ckpt'
    manifests = {
        HHC_NO_TIMING_LABEL: {'provenance': {'artifacts': {
            'transformer': {'path': selected, 'sha256': 'hhc-hash'},
            'vq': {'path': '/tmp/hhc-vq.pth', 'sha256': 'hhc-vq-hash'},
        }}},
        HHC_ORACLE_TIMING_LABEL: {'provenance': {'artifacts': {
            'transformer': {'path': selected, 'sha256': 'hhc-hash'},
            'vq': {'path': '/tmp/hhc-vq.pth', 'sha256': 'hhc-vq-hash'},
        }}},
        'coshmdm_interx_retrained_2daycap_seed123': {
            'provenance': {'artifacts': {
                'model': {'path': cosh, 'sha256': 'cosh-hash'},
            }},
        },
        'timotion_transformer_lpa_interx_retrained_fair': {
            'provenance': {'artifacts': {
                'model': {'path': timotion, 'sha256': 'timotion-hash'},
                'validation_selection_manifest': {
                    'path': '/tmp/timotion-selection.json',
                    'sha256': 'timotion-selection-hash',
                },
            }},
        },
    }
    records = {
        'hhc_mgt_final_selection': {'payload': {
            'selected_checkpoint': selected,
            'selected_checkpoint_sha256': 'hhc-hash',
        }},
        'vq_import_manifest': {'payload': {'checkpoint': {
            'path': '/tmp/hhc-vq.pth',
            'sha256': 'hhc-vq-hash',
        }}},
        'coshmdm_final_state_audit': {'payload': {
            'checkpoint': cosh, 'checkpoint_sha256': 'cosh-hash',
        }},
        'timotion_final_state_audit': {'payload': {
            'checkpoint': timotion, 'checkpoint_sha256': 'timotion-hash',
        }},
        'timotion_validation_selection': {
            'artifact': {
                'path': '/tmp/timotion-selection.json',
                'sha256': 'timotion-selection-hash',
            },
            'payload': {
                'selected_checkpoint': timotion,
                'selected_checkpoint_sha256': 'timotion-hash',
            },
        },
    }
    generalization = {}
    for experiment in ('shape', 'action'):
        trans_path = f'/tmp/{experiment}-selected.ckpt'
        trans_hash = f'{experiment}-trans-hash'
        vq_path = f'/tmp/{experiment}-vq.pth'
        vq_hash = f'{experiment}-vq-hash'
        records[f'mgt_generalization_{experiment}_selection'] = {'payload': {
            'selected_checkpoint': trans_path,
            'selected_checkpoint_sha256': trans_hash,
        }}
        records[f'vq_generalization_{experiment}_state_audit'] = {'payload': {
            'checkpoint': vq_path, 'checkpoint_sha256': vq_hash,
        }}
        for partition in ('test_unseen', 'test_seen'):
            generalization[
                f'hhc_generalization_{experiment}_{partition}'
            ] = {'provenance': {'artifacts': {
                'transformer': {'path': trans_path, 'sha256': trans_hash},
                'vq': {'path': vq_path, 'sha256': vq_hash},
            }}}
    validate_method_checkpoint_bindings(manifests, records, generalization)
    records['coshmdm_final_state_audit']['payload'][
        'checkpoint_sha256'
    ] = 'different-checkpoint'
    try:
        validate_method_checkpoint_bindings(manifests, records, generalization)
    except ValueError as error:
        assert 'coshmdm_final_state_audit' in str(error)
    else:
        raise AssertionError('generation from an unaudited checkpoint was accepted')


def test_cached_physical_report_must_match_frozen_protocol():
    definition = (
        'LLM-free joint metrics using the frozen binary contact schedule and '
        'subject-specific Inter-X skeletons; penetration remains a separate '
        'mesh metric.'
    )
    expected_values = {
        'protocol': 'interx_official_v1_physical_joint_v1',
        'definition': definition,
        'contact_threshold_m': 0.1,
        'ground_height_threshold_m': 0.05,
        'repeat': 0,
    }
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        requests = directory / 'requests.npz'
        generations = directory / 'generations.h5'
        requests.write_bytes(b'requests')
        generations.write_bytes(b'generations')
        report_path = directory / 'physical.json'
        report = {
            **expected_values,
            'requests': str(requests.resolve()),
            'requests_sha256': report_sha256_file(requests),
            'generations': str(generations.resolve()),
            'generations_sha256': report_sha256_file(generations),
        }
        report_path.write_text(json.dumps(report), encoding='utf-8')
        assert report_is_current(
            str(report_path),
            {'requests': str(requests), 'generations': str(generations)},
            [str(requests), str(generations)],
            expected_values=expected_values,
        )
        report['contact_threshold_m'] = 0.15
        report_path.write_text(json.dumps(report), encoding='utf-8')
        assert not report_is_current(
            str(report_path),
            {'requests': str(requests), 'generations': str(generations)},
            [str(requests), str(generations)],
            expected_values=expected_values,
        )


def test_review_completion_status_never_hides_pending_scope():
    assert resolve_review_completion_status(
        optimization_complete=False,
        current_interx_pending=[],
        external_prerequisites_pending=[],
    ) == 'partial_current_interx_requirements'
    assert resolve_review_completion_status(
        optimization_complete=True,
        current_interx_pending=['figure_2_typo'],
        external_prerequisites_pending=[],
    ) == 'partial_current_interx_requirements'
    assert resolve_review_completion_status(
        optimization_complete=True,
        current_interx_pending=[],
        external_prerequisites_pending=['blinded_user_study_not_conducted'],
    ) == 'interx_complete_external_prerequisites_pending'
    assert resolve_review_completion_status(
        optimization_complete=True,
        current_interx_pending=[],
        external_prerequisites_pending=[],
    ) == 'complete'


def test_final_report_rejects_failed_or_wrong_step_training_audits():
    with tempfile.TemporaryDirectory() as directory:
        bound = Path(directory) / 'bound.bin'
        bound.write_bytes(b'checkpoint-and-manifest-smoke')
        digest = report_sha256_file(bound)
        timotion_manifest_path = Path(directory) / 'timotion_manifest.json'
        timotion_manifest_path.write_text(json.dumps({
            'observability': {'wandb_run_id': 'fixturewandb'},
        }), encoding='utf-8')
        timotion_manifest_digest = report_sha256_file(timotion_manifest_path)
        timotion_smoke_wandb_path = (
            Path(directory) / 'timotion_smoke_wandb.json'
        )
        timotion_smoke_wandb_path.write_text(json.dumps({
            'schema': 'hhc_wandb_online_metric_history_audit_v1',
            'status': 'pass',
            'run_id': 'oi4wa5qn',
            'expected_final_step': 50,
            'checks': {'all_required_metrics': True},
            'metrics': {
                'train/nonfinite_gradients': {'maximum': 0.0},
                'train/nonfinite_weights': {'maximum': 0.0},
            },
        }), encoding='utf-8')
        timotion_smoke_wandb_digest = report_sha256_file(
            timotion_smoke_wandb_path
        )
        cosh_milestone_audit = Path(
            'output/coshmdm_statsfix_20k_preserved_checkpoint_audit.json'
        ).resolve()
        cosh_milestone = Path(
            '/mnt/8T-HDD/wenning/coshmdm-code/checkpoints/'
            'CoShMDM-InterX-fair-statsfix-seed123-bs32x2/model/'
            'milestone-000020000.ckpt'
        ).resolve()
        assert cosh_milestone_audit.is_file() and cosh_milestone.is_file()
        validation_before = Path(directory) / 'prevalidation-000020000.ckpt'
        validation_after = Path(directory) / 'postvalidation-000020000.ckpt'
        validation_before.write_bytes(bound.read_bytes())
        validation_after.write_bytes(bound.read_bytes())
        generalization_counts = {
            'shape/test_unseen': 320, 'shape/test_seen': 1344,
            'action/test_unseen': 352, 'action/test_seen': 1312,
        }
        adapter_implementation_paths = (
            Path('tools/audit_coshmdm_representation_adapter.py'),
            Path('/mnt/8T-HDD/wenning/coshmdm-code/utils/interx_official_adapter.py'),
            Path('/mnt/8T-HDD/wenning/coshmdm-code/utils/interx_projection.py'),
            Path('/mnt/8T-HDD/wenning/coshmdm-code/tools/export_official_interx.py'),
            Path('tools/export_timotion_official_interx.py'),
            Path('tools/prepare_coshmdm_interx.py'),
        )
        adapter_implementation = {
            str(path.resolve()): report_sha256_file(path)
            for path in adapter_implementation_paths
        }
        timotion_candidate_epochs = list(range(100, 1501, 100))
        timotion_candidates = [
            {
                'epoch': epoch,
                'step': epoch * 71,
                'samples': 544,
                'test_set_used': False,
                'checkpoint': {'path': str(bound), 'sha256': digest},
                'generations': {'path': str(bound), 'sha256': digest},
                'metrics': {'path': str(bound), 'sha256': digest},
                'values': {
                    'val/FID': 0.5 if epoch == 300 else 1.0 + epoch / 1000,
                    'val/R_precision_top1': 0.1,
                    'val/R_precision_top2': 0.2,
                    'val/R_precision_top3': 0.3,
                    'val/Matching': 4.0,
                    'val/Diversity': 8.0,
                },
            }
            for epoch in timotion_candidate_epochs
        ]
        timotion_validation_history_path = (
            Path(directory) / 'timotion_validation_history.json'
        )
        timotion_finalizer_path = (
            Path(__file__).resolve().parents[1]
            / 'tools/finalize_timotion_validation_history.py'
        )
        timotion_validation_history_path.write_text(json.dumps({
            'schema': 'timotion_official_interx_validation_history_v1',
            'status': 'complete',
            'final_global_step': 106500,
            'completed_validation_epochs': timotion_candidate_epochs,
            'training_manifest': {
                'path': str(timotion_manifest_path),
                'sha256': timotion_manifest_digest,
            },
            'implementation': {
                'path': str(timotion_finalizer_path),
                'sha256': report_sha256_file(timotion_finalizer_path),
            },
            'records': timotion_candidates,
        }), encoding='utf-8')
        timotion_validation_history_digest = report_sha256_file(
            timotion_validation_history_path
        )

        def contact_cache_payload(samples, caption_binding, frozen=False):
            inputs = {
                'manifest': str(bound), 'manifest_sha256': digest,
                'split': str(bound), 'split_sha256': digest,
                'requests': str(bound) if frozen else None,
                'requests_sha256': digest if frozen else None,
            }
            return {
                'schema': 'hhc_interx_codex_contact_cache_audit_v1',
                'status': 'pass', 'complete': True,
                'integrity_passed': True,
                'contract_hash_matches': True,
                'implementation_resolved': True,
                'output_schema_resolved': True,
                'token_count_matches': True,
                'caption_binding': caption_binding,
                'counts': {
                    'expected_samples': samples,
                    'completed_samples': samples,
                    'structured_contact_rows': 1,
                    'manifest_total_tokens_used': 2,
                },
                'resource_usage': {
                    'end_to_end_wall_clock': {
                        'source': (
                            'extraction_manifest_created_at_utc_to_'
                            'completed_at_utc'
                        ),
                        'created_at_utc': '2026-08-28T13:00:00+00:00',
                        'completed_at_utc': '2026-08-28T14:00:00+00:00',
                        'total_seconds': 3600.0,
                        'total_hours': 1.0,
                        'batch_files': 1,
                        'batch_mtime_window_inside_manifest': True,
                    },
                    'invocations': {
                        'wall_timed_attempts': 0,
                        'wall_untimed_attempts': 1,
                    },
                    'wall_seconds': None,
                },
                'implementation': {
                    'path': str(bound), 'sha256': digest,
                },
                'inputs': inputs,
                'missing_pairs': [], 'missing_metadata': [],
                'extra_pairs': [], 'extra_metadata': [],
                'batch_errors': {}, 'payload_errors': {},
                'metadata_errors': {}, 'manifest_mismatches': {},
            }

        def contact_timing_payload(samples):
            return {
                'schema': 'hhc_interx_contact_timing_consistency_audit_v1',
                'status': 'complete', 'fusion_readiness_passed': True,
                'exact_modal_agreement_passed': False,
                'fusion_policy': {
                    'timing_positive_and_regions_empty': 'physical_losses_only',
                    'timing_zero_and_regions_nonempty': 'semantic_regions_inactive',
                },
                'counts': {
                    'expected_samples': samples,
                    'audited_samples': samples,
                    'structured_contact_rows': 1,
                    'manifest_total_tokens_used': 2,
                    'timing_covered_samples': samples,
                    'timing_positive_samples': max(samples - 1, 0),
                    'positive_timing_empty_regions': 1,
                    'zero_timing_nonempty_regions': 1,
                },
                'inputs': {
                    'extraction_manifest_sha256': digest,
                    'split_sha256': digest,
                },
                'implementation': {
                    'path': str(bound), 'sha256': digest,
                },
                'manifest_errors': [], 'payload_errors': {},
                'metadata_errors': {}, 'timing_errors': {},
                'missing_pairs': [], 'missing_metadata': [],
                'extra_pairs': [], 'extra_metadata': [],
            }

        records = {
            'hhc_paper_training_contract_audit': {
                'payload': {
                    'schema': 'hhc_paper_training_contract_audit_v1',
                    'status': 'pass_with_disclosed_exceptions',
                    'audit_script': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'paper': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'active_training_manifest': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'active_vq_manifest': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'interx_protocol_audit': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'historical_tokenizer_runtime_audit': {
                        'path': str(bound), 'sha256': digest, 'valid': True,
                    },
                    'paper_stated_numeric_mismatches': {},
                    'paper_stated_qualitative_mismatches': {},
                    'validation_metric_logging': {
                        'val/FID': 'current milestone value',
                        'val/Diversity': 'running best-so-far value',
                        'val/TOP1': 'running best-so-far value',
                        'val/TOP2': 'running best-so-far value',
                        'val/TOP3': 'running best-so-far value',
                        'val/Matching': 'running best-so-far value',
                        'checkpoint_selection':
                        'minimum current-milestone val/FID only',
                    },
                    'validation_loader_roles': {
                        'checkpoint_selection_fid': {
                            'batch_size': 32,
                            'drop_last': True,
                            'evaluated_entries': 544,
                            'intermask_contract_matches': True,
                            'official_interx_contract_matches': True,
                            'contract_mismatches': {},
                        },
                        'lightning_token_loss': {
                            'batch_size': 96,
                            'drop_last_batches': 5,
                            'evaluated_entries_after_drop_last': 480,
                            'used_for_checkpoint_fid': False,
                            'contract_mismatches': {},
                        },
                    },
                    'launch_source_stability': {
                        'status': 'pass', 'scope': 'live_source_tree',
                        'snapshot_manifest': None, 'checked_files': 14,
                        'mismatches': {},
                        'records': {
                            f'source_{index}.py': {
                                'path': str(bound), 'sha256': digest,
                                'launch_sha256': digest,
                            }
                            for index in range(14)
                        },
                    },
                    'precleanup_visible_source': {
                        'commit': '75814e5',
                        'file': 'train_t2i_trans.py',
                        'precision_markers_found': ['precision=', 'bf16'],
                        'precision': 'bf16-mixed',
                        'optimizer_file': 'utils/utils_model.py',
                        'optimizer_betas': [0.5, 0.9],
                        'exact_historical_wandb_run_source_archive_available':
                        False,
                    },
                    'legacy_vs_active_differential': {
                        'same_observed_setting': {
                            'numeric_precision': 'bf16-mixed',
                        },
                        'different_settings': {
                            'optimizer_betas': {
                                'legacy': [0.5, 0.9],
                                'active': [0.9, 0.99],
                            },
                        },
                    },
                },
            },
            'historical_mgt_tokenizer_runtime_audit': {
                'artifact': {'path': str(bound), 'sha256': digest},
                'payload': {
                    'schema': 'hhc_historical_mgt_tokenizer_runtime_audit_v1',
                    'status': 'pass',
                    'checks': {
                        'cache_completed_before_wandb_start': True,
                        'root_statistics_selected': True,
                        'prepare_interx_absent': True,
                    },
                },
            },
            'historical_mgt_launch_source_recovery': {
                'artifact': {'path': str(bound), 'sha256': digest},
                'payload': {
                    'schema': 'hhc_mgt_launch_source_recovery_v1',
                    'status': 'pass',
                    'training_manifest': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'selected_session_records': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'implementation': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'changed_sources_recovered_from_history': [
                        'options/option_transformer.py',
                        'train/train_t2i_trans.py',
                    ],
                    'reversed_patch_records': [{} for _ in range(7)],
                    'snapshots': {
                        f'source_{index}.py': {
                            'path': str(bound), 'sha256': digest,
                        }
                        for index in range(14)
                    },
                    'all_snapshot_hashes_match_launch_manifest': True,
                },
            },
            'mgt_repair_gate': {'payload': {
                'schema': 'hhc_mgt_repair_gate_v1',
                'status': 'pass',
                'tokenizer': {
                    'reconstruction_evaluator_runtime': (
                        official_main_only_evaluator_runtime()
                    ),
                },
                'historical_replay': {
                    'status': 'pass', 'completed_step': 300000,
                    'trajectory_exact_through_step': 299999,
                    'launch_source_recovery_status': 'pass',
                    'launch_source_snapshot_count': 14,
                    'reversed_patch_record_count': 7,
                    'launch_source_snapshots': {
                        f'source_{index}.py': {
                            'path': str(bound), 'sha256': digest,
                        }
                        for index in range(14)
                    },
                },
                'common50_bridge': {
                    'sampler': 'sample_intra_cas', 'sampling_steps': 50,
                    'temperature': 0.5, 'cfg_scale': 3.5,
                    'contact_guidance': True,
                    'contact_guidance_scale': 1.0,
                    'evaluator_runtime': (
                        official_main_only_evaluator_runtime()
                    ),
                },
                'evidence': [
                    {
                        'role': role, 'status': 'pass',
                        'path': str(bound), 'sha256': digest,
                    }
                    for role in (
                        'vq_reconstruction',
                        'historical_training_trajectory',
                        'historical_ordinary_fid',
                        'historical_checkpoint_pair',
                        'common50_ordinary_fid',
                        'historical_launch_sources',
                    )
                ],
            }},
            'historical_mgt_common50_mm_replay': {'payload': {
                'schema': 'hhc_mgt_common50_mm_replay_v1',
                'status': 'pass', 'samples': 100, 'repeats': 30,
                'names_equal': True, 'lengths_equal': True,
                'motions_byte_exact': True,
                'maximum_absolute_motion_difference': 0.0,
                'nonzero_motion_element_count': 0,
                'reference_motion_bytes_sha256': digest,
                'candidate_motion_bytes_sha256': digest,
                'conditioning_contracts': {
                    'training': 'historical_mmm_shape_dropout',
                    'inference': 'paper_text_cfg',
                },
                'repeat_records': [
                    {
                        'repeat': repeat, 'motions_byte_exact': True,
                        'maximum_absolute_motion_difference': 0.0,
                        'nonzero_motion_element_count': 0,
                        'reference_motion_bytes_sha256': digest,
                        'candidate_motion_bytes_sha256': digest,
                    }
                    for repeat in range(30)
                ],
                'reference_exporter': {
                    'sha256':
                    '312ee55ca12fe92f29108bf513e1da9fcde394508e119cc4b29bbbf4239dd76e',
                },
                **{
                    name: {'path': str(bound), 'sha256': digest}
                    for name in (
                        'reference_generations', 'candidate_generations',
                        'reference_manifest', 'implementation',
                    )
                },
            }},
            'hhc_formal_launch_contract_audit': {'payload': {
                'schema': 'hhc_formal_launch_contract_v1',
                'status': 'pass',
                'launcher': {'path': str(bound), 'sha256': digest},
                'state_resolver': {'path': str(bound), 'sha256': digest},
                'source_freezer': {'path': str(bound), 'sha256': digest},
                'implementation': {'path': str(bound), 'sha256': digest},
                'sampling_parity_source_scope': 'live_source_tree',
                'source_snapshot_manifest': None,
                'sampling_parity_sources': {
                    role: {'path': str(bound), 'sha256': digest}
                    for role in (
                        'training_entrypoint',
                        'validation_implementation',
                        'fair_exporter',
                    )
                },
                'required_contract_entries': ['formal fixture'],
                'missing_contract_entries': [],
                'forbidden_entries_present': [],
                'ordering': {'training_before_selection': True},
                'validation_export_sampling_parity': {
                    key: True for key in (
                        'validation_selects_configured_sampler',
                        'validation_passes_all_sampling_parameters',
                        'validation_uses_contact_schedule_only_at_sampling',
                        'validation_rng_is_fixed_and_observational',
                        'validation_uses_raw_interx_normalization',
                        'validation_downsamples_contact_by_four',
                        'validation_forwards_cascade_contact_bias',
                        'fair_export_uses_raw_interx_normalization',
                        'fair_export_forwards_same_sampling_parameters',
                        'formal_launch_uses_one_shared_train_only_contact_map',
                        'formal_launch_exports_contact_and_no_contact_rows',
                    )
                },
            }},
            'hhc_sampling_runtime_audit': {'payload': {
                'schema': 'hhc_sampling_runtime_audit_v1',
                'status': 'pass',
                'contract': {
                    'sampler': 'sample_intra_cas',
                    'sampling_steps': 50,
                    'sampling_steps_unit': 'outer_maskgit_iterations',
                    'maximum_active_channel_stages_per_outer_iteration': 3,
                    'maximum_model_forward_calls_per_outer_iteration_with_cfg': 6,
                    'resolved_channels_skip_forward_but_not_outer_iterations': True,
                },
                'full_length_runtime': {
                    'motion_length_frames': 150,
                    'motion_length_tokens': 38,
                    'observed_outer_iterations': 50,
                    'active_channel_stages': 150,
                    'total_forward_calls_with_cfg': 300,
                },
                'one_token_runtime': {
                    'observed_outer_iterations': 50,
                    'active_channel_stages': 3,
                    'total_forward_calls_with_cfg': 6,
                },
                'checks': {'runtime_contract': True},
                'source': {'path': str(bound), 'sha256': digest},
                'implementation': {'path': str(bound), 'sha256': digest},
            }},
            'hhc_contact_guidance_map_audit': {'payload': {
                'schema': 'hhc_contact_guidance_map_audit_v1',
                'status': 'pass',
                'audit_script': {
                    'path': str(bound), 'sha256': digest,
                },
                'vq_checkpoint': {
                    'path': str(bound), 'sha256': digest,
                },
                'paper_formula_fair_map': {
                    'path': str(bound), 'sha256': digest,
                    'scope': 'official_train_only', 'samples': 9095,
                    'exact_recomputation_match': True,
                    'unseen_code_probability': 0.5,
                },
                'mmm_legacy_full_axis_bug_map': {
                    'copied_path': str(bound), 'sha256': digest,
                    'copy_byte_identical': True,
                    'exact_bug_replay_match': True,
                    'uses_test_split': True, 'test_samples_used': 1706,
                    'scope_by_official_split': {
                        'train': 9095, 'val': 569, 'test': 1706,
                    },
                },
                'policy': {
                    'fair_main_table': 'paper_formula_fair_map',
                    'legacy_map_role': (
                        'author-implementation reproduction sensitivity only; '
                        'excluded from the fair main table'
                    ),
                },
            }},
            'hhc_validation_observational_pair_audit': {'payload': {
                'schema': 'hhc_lightning_same_step_pair_audit_v1',
                'status': 'pass', 'expected_step': 20000,
                'model_tensor_count': 947, 'differing_model_keys': [],
                'exact': {
                    'model_state': True, 'optimizer_state': True,
                    'lr_scheduler_state': True, 'hyper_parameters': True,
                },
                'validation_callback': {
                    'before': {
                        'current_fid': 1.2,
                        'best_fid': 1.1,
                    },
                    'after': {
                        'current_fid': 1.1,
                        'best_fid': 1.1,
                    },
                },
                'fixed_pair_callback': {
                    'before': {
                        'step': 20000,
                        'pre_saved': False,
                        'post_saved': False,
                    },
                    'after': {
                        'step': 20000,
                        'pre_saved': True,
                        'post_saved': True,
                    },
                },
                'before': {
                    'path': str(validation_before), 'sha256': digest,
                },
                'after': {
                    'path': str(validation_after), 'sha256': digest,
                },
                'implementation': {'path': str(bound), 'sha256': digest},
            }},
            'hhc_mgt_train_val_conditional_support_audit': {'payload': {
                'schema': 'hhc_mgt_train_val_conditional_support_audit_v1',
                'status': 'complete',
                'loader_contract': {
                    'training_token_length_contract': 'paper_50',
                    'tensor_window_including_end': 50,
                    'minimum_payload_tokens': 20,
                    'maximum_payload_tokens': 49,
                    'filter_then_truncate': True,
                },
                'training_split': {
                    'cached_samples': 9095,
                    'loaded_samples': 8617,
                    'excluded_below_training_bucket': ['x'] * 478,
                    'truncated_to_payload_window': 2618,
                    'sha256':
                    '47c5c9a09d7293e90e127fb49ba9dc9398afbff66c8f2b45ada3ef6c4453157a',
                },
                'validation_split': {
                    'cached_samples': 569,
                    'loaded_samples': 534,
                    'excluded_below_training_bucket': ['x'] * 35,
                    'truncated_to_payload_window': 174,
                    'sha256':
                    '6c77e85017b14b512f42f2d540da2ce05874d9777d707bebd85782d6221f28a8',
                },
                'marginal_code_support': {
                    part: {'validation_occurrence_coverage': 0.9995}
                    for part in ('body', 'hand', 'root')
                },
                'same_frame_conditional_tuple_support': {
                    'root_to_body_pair': {
                        'validation_occurrence_coverage': 0.08,
                    },
                    'root_body_to_hand_triplet': {
                        'validation_occurrence_coverage': 0.003,
                    },
                },
                'within_part_transition_support': {
                    part: {'validation_occurrence_coverage': 0.7}
                    for part in ('body', 'hand', 'root')
                },
                'training_manifest': {
                    'path': str(bound), 'sha256': digest,
                },
                'implementation': {
                    'path': str(Path(
                        'tools/audit_mgt_train_val_conditional_support.py'
                    ).resolve()),
                    'sha256': report_sha256_file(
                        'tools/audit_mgt_train_val_conditional_support.py'
                    ),
                },
            }},
            **{
                f'hhc_mgt_shape_dependence_{split_role}': {
                    'artifact': {'path': str(bound), 'sha256': digest},
                    'payload': {
                        'schema':
                        'hhc_mgt_shape_condition_dependence_audit_v1',
                        'status': 'complete',
                        'split_role': split_role,
                        'samples': 256,
                        'sample_selection': (
                            'uniform_without_replacement_over_active_split'
                        ),
                        'shape_derangement': {
                            'fixed_points': 0,
                            'same_shape_signature_pairs': 0,
                            'unique_shape_signatures': 12,
                            'mapping_sha256': 'd' * 64,
                        },
                        'checkpoint': {
                            'path': str(bound), 'sha256': digest,
                        },
                        'training_manifest': {
                            'path': str(bound), 'sha256': digest,
                        },
                        'implementation': {
                            'path': str(Path(
                                'tools/audit_mgt_shape_condition_dependence.py'
                            ).resolve()),
                            'sha256': report_sha256_file(
                                'tools/audit_mgt_shape_condition_dependence.py'
                            ),
                        },
                        'metrics': {'aggregate': {
                            condition: {
                                'tokens': 100,
                                'cross_entropy': (
                                    1.0 if condition == 'requested' else 2.0
                                ),
                                'accuracy_percent': (
                                    (20.0 if split_role == 'training' else 2.0)
                                    if condition == 'requested'
                                    else (
                                        10.0 if split_role == 'training' else 1.0
                                    )
                                ),
                            }
                            for condition in (
                                'requested', 'permuted_beta_and_gender',
                                'zero_beta_requested_gender',
                            )
                        }},
                    },
                }
                for split_role in ('training', 'validation')
            },
            'hhc_mgt_shape_dependence_pair': {
                'artifact': {'path': str(bound), 'sha256': digest},
                'payload': {
                    'schema':
                    'hhc_mgt_shape_condition_dependence_pair_v1',
                    'status': 'pass',
                    'decision':
                    'train_only_shape_identity_shortcut_supported',
                    'thresholds_preregistered_before_execution': {
                        'minimum_train_requested_shape_advantage_percent_points':
                        5.0,
                        'minimum_train_minus_validation_advantage_percent_points':
                        3.0,
                    },
                    'checkpoint': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'inputs': {
                        split_role: {'path': str(bound), 'sha256': digest}
                        for split_role in ('training', 'validation')
                    },
                    'implementation': {
                        'path': str(Path(
                            'tools/summarize_mgt_shape_condition_dependence.py'
                        ).resolve()),
                        'sha256': report_sha256_file(
                            'tools/summarize_mgt_shape_condition_dependence.py'
                        ),
                    },
                    'effects': {
                        'permuted_beta_and_gender': {
                            'training': {
                                'requested_accuracy_advantage_percent_points':
                                10.0,
                            },
                            'validation': {
                                'requested_accuracy_advantage_percent_points':
                                1.0,
                            },
                        },
                    },
                },
            },
            'hhc_global_masked_loss_control_assessment': {
                'artifact': {'path': str(bound), 'sha256': digest},
                'payload': {
                    'schema':
                    'hhc_global_masked_loss_control_assessment_v1',
                    'status': 'complete',
                    'decision': (
                        'global_loss_not_sufficient_'
                        'investigate_mask_conditioning_next'
                    ),
                    'control': {'hypothesis_supported': False},
                },
            },
            'hhc_post_50k_training_dispatch': {
                'artifact': {'path': str(bound), 'sha256': digest},
                'payload': {
                    'schema': 'hhc_post_50k_training_dispatch_v1',
                    'status': 'ready',
                    'source_decision': (
                        'global_loss_not_sufficient_'
                        'investigate_mask_conditioning_next'
                    ),
                    'action': 'continue_shape_augmentation',
                    'continue_global_control': False,
                    'shape_training_loss_contract': 'paper_masked_only',
                    'shape_training_budget': 300000,
                    'global_control_training_budget': 300000,
                    'shape_source_decision': (
                        'train_only_shape_identity_shortcut_supported'
                    ),
                    'evidence': {
                        'masked_loss_assessment': {
                            'path': str(bound), 'sha256': digest,
                        },
                        'shape_dependence_pair': {
                            'path': str(bound), 'sha256': digest,
                        },
                    },
                    'implementation': {
                        'path': str(Path(
                            'tools/dispatch_hhc_post_50k_training.py'
                        ).resolve()),
                        'sha256': report_sha256_file(
                            'tools/dispatch_hhc_post_50k_training.py'
                        ),
                    },
                },
            },
            'review_requirement_coverage': {'payload': {
                'schema': 'hhc_review_requirement_coverage_audit_v1',
                'status': 'pass',
                'coverage_complete': True,
                'requirements': 20,
                'review_sources': [
                    'Meta-review', 'Reviewer L4xb',
                    'Reviewer ZyzM', 'Reviewer NbvC',
                ],
                'source_review': {
                    'path': str(bound), 'sha256': digest,
                },
                'matrix': {'path': str(bound), 'sha256': digest},
                'plan': {'path': str(bound), 'sha256': digest},
                'records': [
                    {
                        'id': label,
                        'status': status,
                    }
                    for label, status in (
                        ('unseen_shape_quantitative', 'active_queued'),
                        ('unseen_action_quantitative', 'cancelled_no_claim'),
                        ('chi3d_quantitative_generalization',
                         'deferred_after_interx'),
                        ('second_generation_dataset',
                         'deferred_after_interx'),
                        ('user_study_values', 'external_pending'),
                        *((f'synthetic_requirement_{index}', 'complete')
                          for index in range(15)),
                    )
                ],
                'implementation_sha256': report_sha256_file(
                    'tools/audit_review_requirement_coverage.py'
                ),
            }},
            **{
                f'interx_luna_{split_label}_cache_audit': {
                    'payload': contact_cache_payload(
                        samples, 'first_caption_from_split_text_file'
                    )
                }
                for split_label, samples in (
                    ('train', 9110), ('val_full', 570),
                    ('test_full', 1708),
                )
            },
            **{
                f'interx_luna_{split_label}_timing_audit': {
                    'payload': contact_timing_payload(samples)
                }
                for split_label, samples in (
                    ('train', 9110), ('val_full', 570),
                    ('test_full', 1708),
                )
            },
            'interx_luna_test_frozen_cache_audit': {
                'payload': contact_cache_payload(
                    1696, 'frozen_fair_evaluation_requests', frozen=True
                ),
            },
            'interx_terra_robustness_cache_audit': {
                'payload': contact_cache_payload(
                    128, 'frozen_fair_evaluation_requests', frozen=True
                ),
            },
            'interx_protocol_audit': {'payload': {
                'status': 'pass',
                'split_entries': 1708,
                'valid_entries': 1706,
                'evaluated_entries': 1696,
                'batch_size': 32,
                'drop_last': True,
                'official_evaluator_weights': True,
                'official_evaluator_options': True,
                'official_evaluator_weights_sha256':
                'b1699f13276c044d8b7ae5f6063eb241b7ef96d7e92c878857abd5081e40701e',
                'intermask_source_matches': True,
                'official_interx_source_matches': True,
                'mismatches': {},
                'asset_protocol': {
                    'mmm_data_links_match': True,
                    'mmm_evaluator_link_matches': True,
                    'mmm_smplx_link_matches': True,
                    'statistics_byte_identical_to_mmm': True,
                    'evaluator_options_sha256':
                    'f4ddaa81157259ce6b795f456be98263e50cd788a7800c30aacbfc2361b2d6dd',
                    'statistics': {
                        'Interx_Mean.npy': {
                            'shape': [672], 'dtype': 'float32', 'finite': True,
                            'sha256':
                            'a54dcfd012e07742077050e3a7ee1b4a2918ba2825466ab698d0cf6f9e438446',
                        },
                        'Interx_Std.npy': {
                            'shape': [672], 'dtype': 'float32', 'finite': True,
                            'sha256':
                            '23a2cf52a19cbaab16334821f2a3183e915c671f59fac702b68a1256742bc376',
                        },
                    },
                    'motions': {
                        'train': {
                            'samples': 9110, 'feature_shapes': [[56, 6]],
                            'dtypes': ['float32'],
                            'sha256':
                            '42e242fcd267c83b095591bb3445887ffc2578c1e45eb036bbd0188f174a7d2a',
                        },
                        'val': {
                            'samples': 570, 'feature_shapes': [[56, 6]],
                            'dtypes': ['float32'],
                            'sha256':
                            '91701e2f7687bfb0985013fa2995aa9d06aad8e55340ad6c7e9881662249ffd',
                        },
                        'test': {
                            'samples': 1708, 'feature_shapes': [[56, 6]],
                            'dtypes': ['float32'],
                            'sha256':
                            'd67fd5f24342aff843ff2588314f9ffe6e932fcc5590393e9660a306ce9d6957',
                        },
                    },
                    'smplx_model_sha256': {
                        'male':
                        'ab318e3f37d2bfaae26abf4e6fab445c2a610e1d63714794d60379cc263bc2a5',
                        'female':
                        'b2a3686c9d6d218ff6822fba411c607a3c8125a70af340f384ce68bebecabe0e',
                    },
                    'volumetric_smpl_checkpoint_sha256': {
                        'male':
                        '96b22f6409e67dc950dafdd83bb8dc4779381ecc2b2c752e1c6fa3a205b2e18c',
                        'female':
                        '2b582f009c2daeec50c90b5f39689a248bac96a4cc9983548ad4be677b72ccdb',
                    },
                },
                'source_protocol': {
                    'mmm_pre_cleanup_baseline_commit':
                    '75814e57e0702789176da6713056a15ed4a192e2',
                    'mmm_cleanup_first_commit':
                    '0f58ffbb9db95a5d9ddc7f2e8a40b8602c17a325',
                    'mmm_cleanup_parent_commit':
                    '75814e57e0702789176da6713056a15ed4a192e2',
                    'mmm_pre_cleanup_is_immediate_parent': True,
                    'mmm_pre_cleanup_tree':
                    'd6911e7f8e63ce6333b1707fd587c556be450cf6',
                    'mmm_cleanup_tree':
                    'c1d3e14ecc49671516a0e0c38e6e8ec5feaf24ba',
                'mmm_pre_cleanup_source_records': {
                        path: {
                            'git_blob': blob, 'sha256': source_hash,
                            'bytes': 1,
                        }
                        for path, blob, source_hash in (
                            (
                                'models/t2i_trans.py',
                                '123d7240deb0a42639dea7fe83a2b569f771db8d',
                                '13069e9292c9f39e1271657def16b5e7cfcf7d6eeb827df3d145b96b00e36ce1',
                            ),
                            (
                                'utils/eval_trans.py',
                                '64d902cc2fe09c79823c521f29676dc797b19017',
                                '9113bc2d6d77af5ab0bb984df814f41965b837024c1c492d8916dd72a04cd139',
                            ),
                            (
                                'train_t2i_trans.py',
                                'ae6bf74f15376376c24ee64a4d5cdc8913b4fdb8',
                                '36a1d00adfe0c1a7919334029fbce134dd99e3ebfa1d9b95e1a34c3ac7eae318',
                            ),
                            (
                                'generate_mm_samples.py',
                                '43b0544d714f3fc96a51324290d97b81a0519e7a',
                                '47552dfff5fcf1f9c2ece33b290a1a4e1072cc5dcf352c7f29efafa1326b960c',
                            ),
                            (
                                'options/option_transformer.py',
                                'a1378dcbcb37a376987b587bad3bedc0d5236922',
                                '8168bc8a06df26e41da2bf4c385e1dabe58d1b0af08b6e4f30316b6ae59670d3',
                            ),
                        )
                    },
                    'historical_evidence_priority': [
                        'historical_runtime_logs',
                        'local_history_time_bracketed_source',
                        'pre_cleanup_git_tree',
                        'cleaned_main',
                    ],
                    'mmm_historical_runtime_call_records': {
                        'runtime-a.log': [{
                            'line': 107,
                            'text': 'type="sample_intra_cas"',
                            'sha256': 'a' * 64,
                        }],
                        'runtime-b.log': [{
                            'line': 60,
                            'text': 'type="sample_intra_cas"',
                            'sha256': 'b' * 64,
                        }],
                    },
                    'mmm_historical_runtime_metadata_records': {
                        'runtime-a-metadata.json': {
                            'git_commit':
                            '8c3e7ad107ed717b878779724958cf6eff00938e',
                            'git_remote':
                            'https://github.com/exitudio/MMM.git',
                            'code_path': 'train_t2i_trans.py',
                            'sha256': 'c' * 64,
                        },
                        'runtime-b-metadata.json': {
                            'git_commit':
                            '8c3e7ad107ed717b878779724958cf6eff00938e',
                            'git_remote':
                            'https://github.com/exitudio/MMM.git',
                            'code_path': 'train_t2i_trans.py',
                            'sha256': 'd' * 64,
                        },
                    },
                    'mmm_historical_runtime_source_state': (
                        'uncommitted_two_person_code_on_upstream_git_base'
                    ),
                    'mmm_historical_runtime_local_history_audit': {
                        'schema': 'historical_mmm_sampler_local_history_audit_v1',
                        'status': 'pass',
                        'scope': (
                            'read_only_recovery_of_uncommitted_runtime_source_not_git_history'
                        ),
                        'sha256': 'e' * 64,
                    },
                },
                'sampling_protocol': {
                    'hhc_common_steps': 50,
                    'hhc_sampling_steps_unit': 'outer_maskgit_iterations',
                    'hhc_maximum_active_channel_stages_per_outer_iteration': 3,
                    'hhc_maximum_model_forward_calls_per_outer_iteration_with_cfg': 6,
                    'hhc_maximum_common50_active_channel_stages': 150,
                    'hhc_maximum_common50_model_forward_calls_with_cfg': 300,
                    'hhc_resolved_channels_skip_forward_but_not_outer_iterations': True,
                    'intermask_common_steps': 50,
                    'mmm_pre_cleanup_cascade_default_outer_steps': 3,
                    'mmm_pre_cleanup_cascade_forward_passes_without_cfg': 9,
                    'mmm_pre_cleanup_parallel_default_steps': 10,
                    'mmm_historical_runtime_sampler': 'sample_intra_cas',
                    'mmm_historical_runtime_sampler_log_matches': True,
                    'mmm_historical_runtime_minimum_sampling_updates_user_recorded': 30,
                    'mmm_historical_runtime_exact_step_count_status': 'recovered_and_ast_verified',
                    'mmm_historical_runtime_outer_steps': 10,
                    'mmm_historical_runtime_channel_updates': 30,
                    'mmm_historical_runtime_local_history_audit_passes': True,
                    'mmm_historical_runtime_vfs_content_ids_verified': True,
                    'intermask_author_default_steps': 20,
                },
                'checkpoint_selection_validation': {
                    'split_entries': 570, 'eligible_entries': 569,
                    'evaluated_entries': 544, 'batch_size': 32,
                    'active_min_motion_frames': 24,
                    'active_max_motion_frames': 150,
                    'intermask_contract_matches': True,
                    'official_interx_contract_matches': True,
                    'assets_byte_identical_to_mmm': True,
                    'split_sha256': 'val-split-hash',
                    'motion_h5_sha256': 'val-motion-hash',
                },
            }},
            'official_request_manifest': {
                'artifact': {'path': str(bound), 'sha256': digest},
                'payload': {
                'status': 'pass', 'replications': 20,
                'samples_per_replication': 1696,
                'request_contract_version': 2,
                'multimodality': {
                    'prompts': 100, 'repeats': 30, 'pairs': 10,
                },
                'frozen_input_provenance': {
                    name: {
                        'path': str(bound), 'sha256': digest,
                    }
                    for name in (
                        'protocol', 'request_preparer', 'request_verifier',
                        'dataset_loader', 'normalizer', 'evaluator_wrapper',
                        'evaluator_options', 'evaluator_weights',
                        'test_split', 'test_motions', 'raw_mean', 'raw_std',
                    )
                },
                'records': [
                    {
                        'replication': replication,
                        'seed': 1234 + replication,
                        'path': str(bound), 'sha256': digest,
                        'bytes': bound.stat().st_size,
                    }
                    for replication in range(20)
                ],
                },
            },
            'fair_request_provenance_recovery': {
                'artifact': {'path': str(bound), 'sha256': digest},
                'payload': {
                'schema': 'fair_interx_request_provenance_recovery_v1',
                'status': 'pass',
                'canonical_manifest_unchanged': True,
                'request_npz_files_modified': 0,
                'request_npz_files_verified': 20,
                'frozen_manifest_snapshot': {
                    'path': str(bound), 'sha256': digest,
                },
                'frozen_source_snapshot': {
                    'path': str(bound), 'sha256': digest,
                },
                'frozen_verifier_snapshot': {
                    'path': str(bound), 'sha256': digest,
                },
                'frozen_protocol_snapshot': {
                    'path': str(bound), 'sha256': digest,
                },
                'frozen_creation_records_snapshot': {
                    'path': str(bound), 'sha256': digest,
                },
                'request_creation_runtime_recovery': {
                    'source_session_path': 'frozen-session.jsonl',
                    'launch_line_number': 2371,
                    'completion_line_number': 2522,
                    'launch_timestamp': '2026-08-25T20:15:54.402Z',
                    'completion_timestamp': '2026-08-25T20:21:20.202Z',
                    'launch_line_sha256': (
                        '4da1a2adf58b2b529963ebe8c47524a29c04c761ae31a3fb714e74a2d289e560'
                    ),
                    'completion_line_sha256': (
                        'df2a9895370b19d390b5740ec7d586c7593739a05c2a13928213d45b2e3078b1'
                    ),
                    'command': (
                        'OMP_NUM_THREADS=8 conda run --no-capture-output '
                        '-n New_MMM python -u tools/prepare_fair_eval_requests.py '
                        '--output-dir output/fair_interx_requests --device cpu '
                        '--replications 20 --seed 1234'
                    ),
                    'cwd': str(Path(__file__).resolve().parents[1]),
                    'status': 'completed',
                    'exit_code': 0,
                    'runtime': {
                        'contract': (
                            'pytorch_official_evaluator_single_device_v1'
                        ),
                        'requested_device': 'cpu',
                        'device_type': 'cpu',
                        'device_index': None,
                        'device_name': 'cpu',
                    },
                    'replications_reported_complete': 20,
                    'samples_per_replication_reported': 1696,
                },
                'implementation': {
                    'path': str(bound), 'sha256': digest,
                },
            }},
            'fair_request_replay_audit': {'payload': {
                'schema': 'fair_interx_request_replay_audit_v1',
                'status': 'pass',
                'scope': 'replication_00_seed1234_official_loader_cpu_replay',
                'sample_count': 1696,
                'embedding_absolute_tolerance': 1e-6,
                'exact_fields': [
                    'names', 'captions', 'lengths', 'betas', 'gender',
                    'contact', 'mm_indices', 'seed',
                ],
                'tolerance_fields': [
                    'text_embeddings', 'gt_embeddings',
                    'gt_body_embeddings',
                ],
                'request_creation_runtime': {
                    'contract': 'pytorch_official_evaluator_single_device_v1',
                    'requested_device': 'cpu',
                    'device_type': 'cpu',
                    'device_index': None,
                    'device_name': 'cpu',
                },
                'request_creation_provenance': {
                    'path': str(bound), 'sha256': digest,
                },
                'frozen': {'path': str(bound), 'sha256': digest},
                'replay': {'path': str(bound), 'sha256': digest},
                'implementation': {'path': str(bound), 'sha256': digest},
                'comparisons': {
                    field: {'status': 'pass', 'exact': True}
                    for field in (
                        'names', 'captions', 'lengths', 'betas', 'gender',
                        'contact', 'mm_indices', 'seed',
                    )
                } | {
                    field: {
                        'status': 'pass',
                        'within_absolute_tolerance': True,
                        'maximum_absolute_difference': 5e-7,
                    }
                    for field in (
                        'text_embeddings', 'gt_embeddings',
                        'gt_body_embeddings',
                    )
                },
                'failures': [],
            }},
            'mgt_validation_selection': {'payload': {
                'status': 'pass', 'seed': 500123, 'batch_size': 32,
                'eligible_samples': 569, 'evaluated_samples': 544,
                'batches': 17, 'repeat_deterministic': True,
                'omitted_by_drop_last': ['sample'] * 25,
                'ordered_request_records_sha256': 'a' * 64,
                'ordered_evaluator_inputs_sha256': 'b' * 64,
                'split_sha256': 'val-split-hash',
                'motions_sha256': 'val-motion-hash',
                'implementation': {
                    'path': str(bound), 'sha256': digest,
                },
                'records': [{}] * 544,
            }},
            'generalization_split_manifest': {'payload': {
                'schema': 'hhc_interx_generalization_splits_v1',
                'experiments': {'shape': {}, 'action': {}},
            }},
            'generalization_stats_manifest': {'payload': {'status': 'pass'}},
            'generalization_request_manifest': {'payload': {
                'schema': 'hhc_interx_generalization_frozen_requests_v1',
                'batch_size': 32,
                'source_manifest': str(bound),
                'source_manifest_sha256': digest,
                'implementation_sha256': report_sha256_file(
                    'tools/prepare_interx_generalization_requests.py'
                ),
                'partitions': {
                    label: {
                        'split': str(bound), 'split_sha256': digest,
                        'replications': [
                            {
                                'replication': replication,
                                'samples': samples,
                                'source': str(bound),
                                'source_sha256': digest,
                                'output': str(bound),
                                'output_sha256': digest,
                            }
                            for replication in range(20)
                        ],
                    }
                    for label, samples in generalization_counts.items()
                },
            }},
            'model_capacity_audit': {'payload': {
                'schema': 'hhc_model_capacity_audit_v2',
                'status': 'pass',
                'audit_script': {
                    'path': str(bound), 'sha256': digest,
                },
                'hhc': {
                    'shape_aware_vq_parameters': 36606748,
                    'mgt_parameters_excluding_shared_vq': 73986640,
                    'unique_motion_parameters': 110593388,
                    'frozen_clip_vit_b32_parameters': 151277313,
                    'total_inference_resident_parameters': 261870701,
                    'vq_contract': 'precleanup_raw12288',
                    'checkpoint_load': 'strict',
                    'vq_checkpoint': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'mgt_checkpoint': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'selection_manifest': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'source_files': {
                        name: {'path': str(bound), 'sha256': digest}
                        for name in (
                            'models/model_tools.py',
                            'models/vqvae_sep3_shape.py',
                            'models/encdec.py', 'models/quantize_cnn.py',
                            'models/two_person_transformer.py',
                            'train/train_t2i_trans.py',
                            'tools/export_hhc_official_interx.py',
                        )
                    },
                },
                'intermask': {
                    'vq_parameters': 45418502,
                    'transformer_total_parameters': 177333376,
                    'transformer_trainable_parameters': 54332032,
                    'unique_motion_parameters': 99750534,
                    'total_inference_resident_parameters': 222751878,
                    'checkpoint_load': 'strict',
                    'vq_checkpoint': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'transformer_checkpoint': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'source_files': {
                        name: {'path': str(bound), 'sha256': digest}
                        for name in (
                            'models/vq/model.py',
                            'models/vq/residual_vq.py',
                            'models/mask_transformer/transformer.py',
                            'models/mask_transformer/transformer_modules.py',
                            'checkpoints/interx/vq_default/opt.txt',
                            'checkpoints/interx/trans_default/opt.txt',
                            'tools_export_official_interx.py',
                        )
                    },
                },
                'coshmdm': {
                    'total_parameters': 305442310,
                    'trainable_parameters': 182440966,
                    'frozen_parameters': 123001344,
                    'checkpoint_load': 'strict',
                    'checkpoint': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'source_files': {
                        name: {'path': str(bound), 'sha256': digest}
                        for name in (
                            'configs/model.yaml', 'models/coshmdm.py',
                            'models/gaussian_diffusion.py',
                            'models/nets.py', 'models/blocks.py',
                            'tools/export_official_interx.py',
                        )
                    },
                },
            }},
            'vq_import_manifest': {
                'artifact': {'path': str(bound), 'sha256': digest},
                'payload': {
                'artifact_role': (
                    'Imported pre-cleanup mainline MMM raw Inter-X VQ checkpoint'
                ),
                'import_policy': (
                    'byte-identical copy; source checkpoint is not modified'
                ),
                'checkpoint': {
                    'path': str(bound), 'sha256': digest,
                    'bytes': bound.stat().st_size,
                },
                'frozen_architecture': {
                    'motion_representation': 'raw',
                    'codebook_size_per_part': 12288,
                    'code_dim_total': 256,
                },
                'verified_reconstruction': {
                    'fair_fid': 0.29441722725218256,
                    'mpjpe_m': 0.1362613624044445,
                },
            }},
            'vq_final_reconstruction': {'payload': {
                'checkpoint_iteration': -1, 'samples': 1696,
                'codebook_size_per_part': 12288,
                'checkpoint_sha256': digest,
                'mpjpe_m': 0.1362613624044445,
            }},
            'vq_final_reconstruction_fid': {'payload': {
                'schema': 'fair_interx_main_only_metrics_v1',
                'status': 'complete', 'samples': 1696,
                'fid': 0.29441722725218256,
                'generation_request_hash_verified': True,
                'main_generation_metadata': {'checkpoint_sha256': digest},
                'evaluator_provenance': (
                    official_main_only_evaluator_provenance()
                ),
                'evaluator_runtime': official_main_only_evaluator_runtime(),
            }},
            'historical_mmm_wandb_comparison': {'payload': {
                'status': 'pass',
                'comparability': {'same_configuration': False},
                'historical_run': {
                    'validation_history': [
                        {
                            'trainer/global_step': step,
                            'val/FID': 1.0,
                            'val/TOP1': 0.1,
                        }
                        for step in range(9999, 300000, 10000)
                    ],
                    'remote_history_artifact': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'remote_history_provenance': {
                        'server_summary_exact_match': True,
                        'source_code_archive_available': False,
                    },
                    'validation_runtime': {
                        'status': 'pass',
                        'effective_validation_runtime': {
                            'sampler': 'sample_intra_cas',
                            'outer_steps': 10,
                            'channel_updates': 30,
                            'cascade_order': 'root_body_hand',
                            'rand_pos': False,
                            'temperature': 0.0,
                            'cfg_scale': -1,
                            'contact': False,
                            'model_forward_calls_per_generation': 30,
                        },
                        'source_interval_checks': {
                            'model': True,
                            'train': True,
                            'evaluation': True,
                        },
                        'checks': {'all_exact': True},
                        'wandb_run': {
                            'metadata_path': str(bound),
                            'metadata_sha256': digest,
                        },
                        'model': {'path': str(bound), 'sha256': digest},
                        'train': {'path': str(bound), 'sha256': digest},
                        'evaluation': {
                            'path': str(bound), 'sha256': digest,
                        },
                    },
                },
                'current_run': {
                    'latest_training_metrics': {
                        'trainer/global_step': 299999,
                        'completed_updates': 300000,
                    },
                    'validation_history': [{'trainer/global_step': 10000}],
                    'wandb_file': str(bound), 'wandb_file_sha256': digest,
                },
                'implementation': {
                    'audit': {'path': str(bound), 'sha256': digest},
                    'local_history_parser': {
                        'path': str(bound), 'sha256': digest,
                    },
                },
            }},
            'intermask_native20_hdf5_protocol_audit': {'payload': {
                'schema': 'intermask_existing_native20_hdf5_protocol_audit_v1',
                'status': 'pass',
                'classification':
                'author-native-20-steps_exact-length_not-common50',
                'replications': 20, 'samples_per_replication': 1696,
                'sampler_protocol': {
                    'time_steps': 20, 'cond_scale': 2.0,
                    'temperature': 1.0, 'topk_filter_thres': 0.9,
                    'length_policy': 'exact',
                    'seed_strategy': 'base_seed + repeat_index',
                },
                'records': [
                    {
                        'replication': replication,
                        'main': {
                            'protocol': {
                                'time_steps': 20, 'cond_scale': 2.0,
                                'temperature': 1.0,
                                'topk_filter_thres': 0.9,
                                'length_policy': 'exact',
                                'seed_strategy': 'base_seed + repeat_index',
                            },
                            'exact_lengths_verified': True,
                        },
                        'multimodality': {
                            'protocol': {
                                'time_steps': 20, 'cond_scale': 2.0,
                                'temperature': 1.0,
                                'topk_filter_thres': 0.9,
                                'length_policy': 'exact',
                                'seed_strategy': 'base_seed + repeat_index',
                            },
                            'exact_lengths_verified': True,
                        },
                    }
                    for replication in range(20)
                ],
                'implementation_sha256': report_sha256_file(
                    'tools/audit_existing_intermask_native20.py'
                ),
            }},
            'optimization_ablation_readiness': {'payload': {
                'schema': 'hhc_optimization_ablation_readiness_v1',
                'status': 'pending', 'ready': False,
                'blockers': ['strict_test_contact_manifest_missing'],
                'expected_samples': 1696,
                'expected_converted_npz_files': 3392,
                'codex_chatgpt_login_available': True,
                'authentication_secret_recorded': False,
                'implementation': {
                    'path': str(bound), 'sha256': digest,
                },
                'artifacts': {
                    name: {
                        'path': str(bound), 'sha256': digest, 'exists': True,
                    }
                    for name in (
                        'requests', 'expected_split',
                        'contact_extraction_manifest', 'contact_cache_audit',
                    )
                } | {
                    name: {'path': str(bound), 'exists': False}
                    for name in ('generation_records', 'conversion_manifest')
                },
            }},
            'review_external_prerequisites': {'payload': {
                'schema': 'hhc_review_external_prerequisites_v1',
                'status': 'pass',
                'all_external_prerequisites_ready': False,
                'unresolved_external_prerequisites': [
                    'interhuman_dataset_not_installed',
                ],
                'strict_llm_contact_cache': {
                    'ready': True, 'samples': 1696,
                    'metadata_records': 1696, 'blockers': [],
                },
                'optimization_inputs': {
                    'ready': False,
                    'classification': 'internal_downstream_dependency',
                    'blockers': ['strict_test_contact_manifest_missing'],
                },
                'interhuman': {
                    'execution_scope': (
                        'after Inter-X completion, ingest and protocol-audit '
                        'the results produced on the user-designated external '
                        'server'
                    ),
                    'local_retraining_authorized': False,
                    'baseline_checkpoint_policy': (
                        'use compatible author checkpoints first; retrain a '
                        'baseline only when its released checkpoint is '
                        'unavailable or protocol/data incompatible'
                    ),
                    'required_experiment': (
                        'after Inter-X completion, verify and merge results; '
                        'do not rerun the InterHuman baseline on this machine'
                    ),
                },
                'chi3d': {'prosepose_test_named_cache_audit': {
                    'cache_detected': True,
                    'usable_as_official_chi3d_test': False,
                    'contact_subjects': ['s03'],
                    'contact_record_count': 504,
                    'classification': (
                        'internal_experiment_split_of_official_training_subject_s03; '
                        'filename_test_does_not_mean_official_chi3d_test_package'
                    ),
                    'contact_records': {
                        'path': str(bound), 'sha256': digest,
                    },
                    'processed_optimization_cache': {
                        'path': str(bound), 'sha256': digest,
                    },
                }},
            }},
            'training_wallclock_audit': {'payload': {
                'schema': 'hhc_fair_training_wallclock_v1',
                'status': 'pass',
                'hardware_observation': {
                    'source': 'nvidia-smi --query-gpu=index,name,uuid',
                    'gpus': [
                        {'index': 0, 'name': 'Synthetic GPU 0',
                         'uuid': 'GPU-synthetic-0'},
                        {'index': 1, 'name': 'Synthetic GPU 1',
                         'uuid': 'GPU-synthetic-1'},
                    ],
                },
                'records': {
                    name: {
                        'optimizer_updates': updates,
                        'effective_batch_size': batch_size,
                        'sample_exposures': updates * batch_size,
                        'assigned_gpu_index': gpu_index,
                        'gpu_name_observed_at_audit':
                        f'Synthetic GPU {gpu_index}',
                        'gpu_uuid_observed_at_audit':
                        f'GPU-synthetic-{gpu_index}',
                        'elapsed_seconds': 1.0,
                        'training_manifest': {
                            'path': str(bound), 'sha256': digest,
                        },
                        'final_checkpoint': {
                            'path': str(bound), 'sha256': digest,
                        },
                    }
                    for name, updates, batch_size, gpu_index in (
                        ('hhc_mgt', 300000, 96, 1),
                        ('coshmdm', 40000, 64, 0),
                        ('timotion', 106500, 256, 0),
                    )
                },
            }},
            'hhc_mgt_final_selection': {
                'artifact': {'path': str(bound), 'sha256': digest},
                'payload': {
                'run_complete': True, 'expected_final_step': 300000,
                'test_metrics_used_for_selection': False,
                'selection_metric': 'minimum fixed-protocol validation FID',
                'validation_inference_contract': {
                    'contact_timing_source':
                    'validation_dataset_annotation_oracle',
                    'contact_timing_is_test_data': False,
                    'contact_as_transformer_training_input': False,
                    'sampler': 'sample_intra_cas', 'sampling_steps': 50,
                    'temperature': 0.5, 'cfg_scale': 3.5,
                    'contact_guidance_scale': 1.0, 'contact': True,
                    'contact_conditioning': False, 'eval_iter': 10000,
                },
                'selected_checkpoint': str(bound),
                'selected_checkpoint_sha256': digest,
                'training_manifest': str(bound),
                'training_manifest_sha256': digest,
                'final_latest_checkpoint': str(bound),
                'final_latest_checkpoint_sha256': digest,
                'final_state_audit': {
                    'status': 'pass', 'global_step': 300000,
                    'model_state': {'all_floating_tensors_finite': True},
                    'optimizer_state': {'all_floating_tensors_finite': True},
                    'has_loop_state': True,
                },
            }},
            **{
                f'vq_generalization_{experiment}_state_audit': {'payload': {
                    'status': 'pass', 'iteration': 150000,
                    'scheduler_last_epoch': 150000,
                    'checkpoint': str(bound), 'checkpoint_sha256': digest,
                }}
                for experiment in ('shape', 'action')
            },
            **{
                f'mgt_generalization_{experiment}_selection': {'payload': {
                    'run_complete': True, 'expected_final_step': 300000,
                    'test_metrics_used_for_selection': False,
                    'selection_metric': 'minimum fixed-protocol validation FID',
                    'validation_inference_contract': {
                        'contact_timing_source':
                        'validation_dataset_annotation_oracle',
                        'contact_timing_is_test_data': False,
                        'contact_as_transformer_training_input': False,
                        'sampler': 'sample_intra_cas', 'sampling_steps': 50,
                        'temperature': 0.5, 'cfg_scale': 3.5,
                        'contact_guidance_scale': 1.0, 'contact': True,
                        'contact_conditioning': False, 'eval_iter': 10000,
                    },
                    'selected_checkpoint': str(bound),
                    'selected_checkpoint_sha256': digest,
                    'training_manifest': str(bound),
                    'training_manifest_sha256': digest,
                    'final_latest_checkpoint': str(bound),
                    'final_latest_checkpoint_sha256': digest,
                    'final_state_audit': {
                        'status': 'pass', 'global_step': 300000,
                        'model_state': {
                            'all_floating_tensors_finite': True,
                        },
                        'optimizer_state': {
                            'all_floating_tensors_finite': True,
                        },
                        'has_loop_state': True,
                    },
                }}
                for experiment in ('shape', 'action')
            },
            'coshmdm_training_contract_audit': {'payload': {
                'schema': 'hhc_coshmdm_interx_training_contract_v1',
                'status': 'pass',
                'frozen_files_independently_rehashed': True,
                'frozen_files': {
                    f'frozen-{index}': {'path': str(bound), 'sha256': digest}
                    for index in range(13)
                },
                'repository': {'matches_training_launch': True},
                'released_checkpoint_compatibility': {
                    'documented_pretrained_model_links': 1,
                    'documented_interx_checkpoint_available': False,
                    'tracked_checkpoint_paths': [],
                    'upstream_checkpoint_download_label': 'InterHuman',
                    'upstream_pretrain_download_sha256':
                    '1fa25981b86a93f3957a7a91854be0ea9a28d89066ee54626729a80d7a38eaac',
                },
                'training_manifest': {
                    'path': str(bound), 'sha256': digest,
                },
                'training': {
                    'epochs': 2000,
                    'expected_optimizer_updates': 568000,
                    'effective_batch_size': 64,
                },
                'dataset': {
                    'independent_split_counts': {
                        'train': 9110, 'val': 570, 'test': 1708,
                    },
                    'independent_split_overlap': {
                        'train_val': 0, 'train_test': 0, 'val_test': 0,
                    },
                },
                'statistics': {'manifest': {
                    'path': str(bound), 'sha256': digest,
                }},
            }},
            'coshmdm_statsfix_startup_audit': {'payload': {
                'schema': 'hhc_coshmdm_statsfix_startup_audit_v2',
                'status': 'pass',
                'runtime_contract': {
                    'dataset': 'interx',
                    'stats_prefix': 'interx_global',
                    'normalization_mean_sha256':
                    'aa281b878b8a71c6d10c001059b29374921c18545c082102bf69a404982f568d',
                    'normalization_std_sha256':
                    '30a83e92965a465301a3642eb55b2ba705d3dceaba05e45846aa433d3900b104',
                },
                'first_epoch': {
                    'optimizer_updates': 284,
                    'dropped_incomplete_accumulation_micro_batches': 1,
                    'last_logged_decade_update': 280,
                    'first_epoch_1_logged_update': 290,
                },
                'epoch_boundary_series': {
                    'recorded_completed_epochs': 140,
                    'first_epoch': 0,
                    'last_epoch': 139,
                    'steps_contiguous_from_zero': True,
                    'all_discard_values_equal_one': True,
                    'duplicate_scalar_records': 0,
                },
                'runtime_manifest': {
                    'path': str(bound), 'sha256': digest,
                },
                'event_files': [
                    {'path': str(bound), 'sha256': digest},
                ],
                'training_log': {'path': str(bound), 'sha256': digest},
                'implementation': {'path': str(bound), 'sha256': digest},
            }},
            'coshmdm_representation_adapter_audit': {'payload': {
                'schema': 'hhc_coshmdm_representation_adapter_audit_v1',
                'status': 'pass', 'samples': 1696,
                'adapter_contract': (
                    'shape-specific all-joint least-squares rigid root fit v2; '
                    'full requested length preserved by terminal velocity padding; '
                    'no GT motion is used by generation exporters'
                ),
                'thresholds': {
                    'all_rmse_mean_max': 1e-3,
                    'all_rmse_p95_max': 1e-3,
                    'all_rmse_sample_max': 1e-2,
                    'root_rotation6d_rmse_sample_max': 1e-3,
                    'translation_rmse_m_sample_max': 5e-2,
                    'body_rotation6d_rmse_sample_max': 1e-3,
                    'max_abs_difference_max': 1e-1,
                    'nonroot_body_rotation_max_abs': 0.0,
                },
                'pass_conditions': {
                    name: True for name in (
                        'sample_count', 'all_rmse_mean', 'all_rmse_p95',
                        'all_rmse_max', 'root_rotation6d_rmse_max',
                        'translation_rmse_m_max',
                        'body_rotation6d_rmse_max', 'max_abs_difference',
                        'nonroot_body_rotations_preserved_exactly',
                    )
                },
                'projection_once_vs_twice': {
                    name: {'mean': 0.0, 'p95': 0.0, 'maximum': 0.0}
                    for name in (
                        'all_rmse', 'root_rotation6d_rmse',
                        'translation_rmse_m', 'body_rotation6d_rmse',
                        'max_abs_difference',
                    )
                },
                'raw_to_projection_information_loss': {
                    name: {'rmse': 0.0, 'maximum': 0.0, 'values': 1}
                    for name in (
                        'root_rotation6d', 'nonroot_body_rotation6d',
                        'hand_rotation6d', 'root_translation_m', 'all',
                    )
                },
                'irreversible_or_unmodelled_fields': [
                    'sequence-initial global root rotation',
                    'sequence-initial global root translation and floor offset',
                    '33 hand-joint local rotations per person',
                ],
                'inputs': {
                    'requests': {'path': str(bound), 'sha256': digest},
                    'ground_truth': {'path': str(bound), 'sha256': digest},
                    'skeletons': {
                        str(group): {'path': str(bound), 'sha256': digest}
                        for group in range(1, 60)
                    },
                },
                'implementation': adapter_implementation,
            }},
            'coshmdm_validation_selection': {
                'artifact': {'path': str(bound), 'sha256': digest},
                'payload': {
                'schema': 'hhc_coshmdm_capped_validation_selection_v1',
                'status': 'complete',
                'training_cap_status': 'stopped_gracefully',
                'test_metrics_used_for_selection': False,
                'formal_hhc_validation_record_set_exact': True,
                'formal_hhc_validation_selection_sha256': 'a' * 64,
                'candidate_steps': [40000],
                'candidates': [{
                    'step': 40000,
                    'validation_fid': 1.0,
                    'checkpoint_source_stable_during_export': True,
                    'checkpoint': {'path': str(bound), 'sha256': digest},
                    'generations': {
                        'path': str(
                            bound.parent /
                            f'step_000040000_{digest[:16]}_main.h5'
                        ),
                        'sha256': digest,
                    },
                }],
                'selected_step': 40000,
                'selected_validation_fid': 1.0,
                'selected_checkpoint': str(bound),
                'selected_checkpoint_sha256': digest,
            }},
            'coshmdm_training_trajectory_diagnosis': {'payload': {
                'schema': 'hhc_coshmdm_training_trajectory_diagnosis_v1',
                'status': 'complete',
                'checks': {
                    'all_logged_losses_finite': True,
                    'all_milestone_weights_finite': True,
                    'training_total_loss_decreases_at_every_milestone': True,
                    'validation_fid_worsens_after_selected_checkpoint': True,
                    'historical_gradient_norms_available': False,
                },
                'selection_manifest': {
                    'path': str(bound), 'sha256': digest,
                },
                'milestones': [
                    {
                        'step': step,
                        'checkpoint': {
                            'nonfinite_weight_values': 0,
                            'weight_l2_norm': 400.0 + step / 10000.0,
                        },
                        'training_scalars': {
                            'total': {
                                'trailing_window': {
                                    'median': 1.0 / (step / 20000.0),
                                },
                            },
                        },
                    }
                    for step in (20000, 40000, 60000, 80000, 100000)
                ],
            }},
            'coshmdm_representation_oracle_metrics': {'payload': {
                'schema': 'fair_interx_main_only_metrics_v1',
                'status': 'complete',
                'samples': 1696,
                'fid': 15.0,
                'main_generations_sha256': digest,
                'main_generation_metadata': {
                    'diagnostic_only': True,
                    'not_a_body_controlled_fid': True,
                },
            }},
            'coshmdm_embedding_gap_diagnosis': {'payload': {
                'schema': 'hhc_coshmdm_embedding_gap_diagnosis_v1',
                'status': 'complete',
                'not_a_body_controlled_fid': True,
                'not_a_reported_comparison_row': True,
                'gaps': {
                    'official_ground_truth_to_representation_oracle': {
                        'fid': 15.0,
                    },
                },
                'representation_oracle': {
                    'path': str(bound), 'sha256': digest,
                },
                'requests': {'path': str(bound), 'sha256': digest},
                'model_generations': {
                    'path': str(bound), 'sha256': digest,
                },
                'evaluator_provenance': {'checkpoint': {
                    'sha256': (
                        'b1699f13276c044d8b7ae5f6063eb241b7ef96d7e92c878857abd5081e40701e'
                    ),
                }},
                'implementation': {
                    'path': str(bound), 'sha256': digest,
                },
            }},
            'coshmdm_final_state_audit': {
                'artifact': {'path': str(bound), 'sha256': digest},
                'payload': {
                    'status': 'pass', 'global_step': 40000,
                    'checkpoint': str(bound), 'checkpoint_sha256': digest,
                },
            },
            'coshmdm_resume_provenance': {'payload': {
                'schema': 'hhc_coshmdm_training_provenance_v2',
                'status': 'pass_user_capped',
                'training_classification': (
                    'fresh Inter-X retraining with checkpoint-bound '
                    'interx_global normalization; user-capped to two '
                    'additional days; validation-selected milestone; '
                    'mixed-stat predecessor excluded'
                ),
                'initial_global_step': 0,
                'expected_final_global_step': 568000,
                'original_author_budget_completed': False,
                'user_capped_run_complete': True,
                'selected_global_step': 40000,
                'maximum_logged_update': 40000,
                'runtime_contract': {
                    'dataset': 'interx',
                    'stats_prefix': 'interx_global',
                    'normalization_mean_sha256':
                    'aa281b878b8a71c6d10c001059b29374921c18545c082102bf69a404982f568d',
                    'normalization_std_sha256':
                    '30a83e92965a465301a3642eb55b2ba705d3dceaba05e45846aa433d3900b104',
                },
                'training_manifest': {
                    'path': str(bound), 'sha256': digest,
                },
                'runtime_manifest': {
                    'path': str(bound), 'sha256': digest,
                },
                'budget_cap_manifest': {
                    'path': str(bound), 'sha256': digest,
                },
                'validation_selection_manifest': {
                    'path': str(bound), 'sha256': digest,
                },
                'excluded_mixed_stats_run': {
                    'path': str(bound), 'sha256': digest,
                },
                'milestone_20k_checkpoint_audit': {
                    'path': str(cosh_milestone_audit),
                    'sha256': report_sha256_file(cosh_milestone_audit),
                },
                'milestone_20k_checkpoint': {
                    'path': str(cosh_milestone),
                    'sha256': report_sha256_file(cosh_milestone),
                },
                'queue_source': {
                    'path': str(bound), 'sha256': digest,
                },
                'implementation': {
                    'path': str(bound), 'sha256': digest,
                },
                'training_attempts': [
                    {
                        'artifact': {
                            'path': str(bound), 'sha256': digest,
                        },
                        'segmentation_fault': False,
                        'maximum_logged_update': 40000,
                        'trainer_finished': True,
                    },
                ],
                'resume_policy': (
                    'A resume is accepted only when the checkpoint-embedded '
                    'runtime contract exactly equals the current statistics.'
                ),
                'reproducibility_limitation': (
                    'The worker streams are distributionally but not bitwise '
                    'equivalent to an uninterrupted run.'
                ),
                'final_checkpoint_audit': {
                    'path': str(bound), 'sha256': digest,
                },
            }},
            'timotion_training_contract_audit': {'payload': {
                'schema': 'hhc_timotion_interx_training_contract_v1',
                'status': 'pass',
                'tree_contents_independently_rehashed': True,
                'public_repository_revision':
                '172580fd7f2551a7cc10fde1f15ff73315742e65',
                'manifest': {
                    'path': str(timotion_manifest_path),
                    'sha256': timotion_manifest_digest,
                },
                'gradient_update_contract': {
                    'accumulated_loss_divisor': 8,
                    'telemetry_is_pre_clip': True,
                    'gradient_clip_global_norm': 0.5,
                    'clip_precedes_optimizer_step': True,
                    'matches_released_tools_train_py': True,
                },
                'artifacts': {
                    'losses': {'path': str(bound), 'sha256': digest},
                    'runner': {'path': str(bound), 'sha256': digest},
                    'scheduler_and_model_utils': {
                        'path': str(bound), 'sha256': digest,
                    },
                },
                'paper_budget': {
                    'epochs': 1500, 'global_batch': 256,
                    'optimizer_updates': 106500,
                    'sample_exposures': 27264000,
                    'public_readme_default_1900_overridden_by_paper': True,
                },
                'released_checkpoint_compatibility': {
                    'documented_pretrained_model_links': 2,
                    'documented_interx_checkpoint_available': False,
                    'tracked_interx_paths': [],
                    'tracked_checkpoint_paths': [],
                    'upstream_readme_sha256':
                    '48377e5b2031acd18ba8bd30b97a6cbcc692bc3514b960ad18e2bee2421a22eb',
                    'upstream_dataset_config_sha256':
                    'ff441a5d6c577196f1a5c0975c9dd32e1acd00fb2cac1be8fc575c21db56a4e7',
                    'upstream_evaluator_download_sha256':
                    'fe36d172784a55f37a418ea6896515d12694a51339439ef360f92367201a1df0',
                },
                'dataset': {
                    'union_count': 11388,
                    'converted_pair_count': 11388,
                    'all_h5_exact_id_match': True,
                    'all_train_ids_pass_loader_minimum': True,
                    'normalization_train_only_recompute': True,
                    'below_15_frames_by_split': {
                        'train': {'count': 0, 'names': []},
                    },
                },
            }},
            'timotion_cosh_adapter_equivalence_audit': {'payload': {
                'schema': 'timotion_cosh_adapter_equivalence_audit_v1',
                'status': 'pass',
                'checks': {'all_cases_equal': True},
                'contract': {
                    'input': 'released two-person 262-D representation',
                    'output': 'official raw Inter-X 672-D representation',
                    'formal_path': (
                        'shape-specific all-joint root fit; identity hands'
                    ),
                    'ground_truth_motion_consumed': False,
                },
                'cases': {
                    f'case_{index}': {
                        'bitwise_equal': True,
                        'max_abs': 0.0,
                        'finite': True,
                    }
                    for index in range(8)
                },
                'sources': {},
                'implementation': {
                    'path': str(bound), 'sha256': digest,
                },
            }},
            'timotion_padding_ro_stabilization_audit': {'payload': {
                'schema': 'timotion_padding_ro_stabilization_audit_v1',
                'status': 'pass',
                'checks': {'finite_replacement': True},
                'failed_smoke_nonfinite_motion_values': 18324708,
                'synthetic_runtime': {
                    'old_masked_loss_finite': False,
                    'new_nonfinite_target_quaternions': 0,
                },
                'source': {'path': str(bound), 'sha256': digest},
                'remote_wandb_audit': {
                    'path': str(timotion_smoke_wandb_path),
                    'sha256': timotion_smoke_wandb_digest,
                },
            }},
            'timotion_interrupted_run_exclusion': {'payload': {
                'schema': 'timotion_training_run_exclusion_v1',
                'status': 'excluded',
                'run_id': 'jqh616am',
                'remote_state': 'failed',
                'result_eligible': False,
                'checkpoint_retained': False,
                'important_correction': (
                    'the run already executed the released global gradient '
                    'clip norm 0.5 before interruption'
                ),
                'replacement_policy': 'fresh replacement from optimizer step 0',
            }},
            'timotion_gradient_accumulation_audit': {'payload': {
                'schema': 'timotion_gradient_accumulation_audit_v1',
                'status': 'pass',
                'contract': {
                    'paper_global_batch': 256,
                    'accumulation_steps': 8,
                    'gradient_clip_global_norm': 0.5,
                },
                'runtime_equivalence': {
                    'pre_clip_gradient_max_abs_difference': 0.0,
                    'post_clip_gradient_max_abs_difference': 0.0,
                    'post_adamw_weight_max_abs_difference': 0.0,
                },
                'sources': {
                    'fair_runner': {
                        'path': str(bound), 'sha256': digest,
                    },
                },
            }},
            'timotion_warmup_stability_audit': {'payload': {
                'schema': 'hhc_wandb_warmup_stability_audit_v1',
                'status': 'pass',
                'warmup_epochs': 10,
                'completed_epochs': 12,
                'optimizer_updates_per_epoch': 71,
                'telemetry_interval_optimizer_steps': 50,
                'minimum_expected_telemetry_events': 14,
                'max_weight_relative_span': 0.05,
                'checks': {'all_finite': True},
                'run_path': 'entity/project/fixturewandb',
                'metrics': {
                    'train/nonfinite_gradients': {'maximum': 0.0},
                    'train/nonfinite_weights': {'maximum': 0.0},
                    'train/weight_l2_norm': {
                        'relative_span_from_first': 0.01,
                    },
                    'train/weight_max_abs': {
                        'relative_span_from_first': 0.01,
                    },
                },
            }},
            'timotion_transient_velocity_spike_audit': {'payload': {
                'schema': 'timotion_transient_velocity_spike_audit_v1',
                'status': 'pass',
                'checks': {'all_histories_finite': True},
                'interpretation_policy': (
                    'post-hoc operational diagnosis only; not a checkpoint-'
                    'selection criterion and not evidence for changing the '
                    'released loss'
                ),
                'run_path': 'entity/project/fixturewandb',
                'peak': {
                    'completed_epoch': 66,
                    'train_loss_mean': 0.74,
                    'validation_loss': 1.25,
                    'train_simple_mean': 0.107,
                },
                'decomposition': {
                    'velocity_excess_fraction_of_total_excess': 1.0,
                },
                'latest_complete_epoch': {
                    'completed_epoch': 69,
                    'train_loss_mean': 0.174,
                    'validation_loss': 0.208,
                },
                'telemetry_latest': {
                    'train/nonfinite_gradients': 0.0,
                    'train/nonfinite_weights': 0.0,
                },
            }},
            'timotion_scheduler_boundary_audit': {'payload': {
                'schema': 'timotion_released_scheduler_boundary_audit_v1',
                'status': 'pass',
                'checks': {'boundary_matches': True},
                'contract': {
                    'base_learning_rate': 0.0001,
                    'warmup_parameter': 10,
                    'cosine_horizon_epochs': 2000,
                    'optimizer_updates_per_epoch': 71,
                    'port_policy': 'preserve released scheduler behavior exactly',
                },
                'transitions': [
                    {'epoch_zero_based': 10, 'observed_learning_rate': 0.00011},
                    {'epoch_zero_based': 11, 'observed_learning_rate': 0.000099},
                ],
                'source': {'path': str(bound), 'sha256': digest},
            }},
            'timotion_formal_wandb_history_audit': {'payload': {
                'schema': 'hhc_wandb_online_metric_history_audit_v1',
                'status': 'pass',
                'expected_final_step': 106500,
                'metric_profile': 'diffusion_regression_20260831_launch',
                'remote_state': 'finished',
                'run_id': 'fixturewandb',
                'optimizer_step_axis': 'trainer/global_step',
                'checks': {
                    'all_required_metrics': True,
                    'all_training_metrics_reach_expected_step': True,
                    'all_validation_metrics_follow_final_training_event': True,
                    'nonfinite_counts_zero': True,
                },
                'implementation': {
                    'sha256': report_sha256_file(
                        Path(__file__).resolve().parents[1]
                        / 'tools/audit_wandb_online_metric_history.py'
                    ),
                },
                'metrics': {
                    metric: {
                        'count': 2130,
                        'last_step': 106500,
                        'minimum': 0.0 if 'nonfinite' in metric else 1.0,
                        'maximum': 0.0 if 'nonfinite' in metric else 2.0,
                    }
                    for metric in (
                        'train/gradient_l2_norm',
                        'train/gradient_max_abs',
                        'train/weight_l2_norm',
                        'train/weight_max_abs',
                        'train/nonfinite_gradients',
                        'train/nonfinite_weights',
                    )
                },
            }},
            'timotion_validation_selection': {'payload': {
                'schema': 'hhc_timotion_interx_validation_selection_v1',
                'status': 'complete',
                'selection_metric': 'minimum official validation FID',
                'test_metrics_used_for_selection': False,
                'validation_samples': 544,
                'validation_seed': 500123,
                'candidate_epochs': timotion_candidate_epochs,
                'candidate_steps': [
                    epoch * 71 for epoch in timotion_candidate_epochs
                ],
                'candidates': timotion_candidates,
                'selected_epoch': 300,
                'selected_step': 21300,
                'selected_validation_fid': 0.5,
                'selected_checkpoint': str(bound),
                'selected_checkpoint_sha256': digest,
                'training_manifest': {
                    'path': str(timotion_manifest_path),
                    'sha256': timotion_manifest_digest,
                },
                'validation_history': {
                    'path': str(timotion_validation_history_path),
                    'sha256': timotion_validation_history_digest,
                },
                'final_full_state_checkpoint': {
                    'path': str(bound), 'sha256': digest,
                },
                'implementation': {
                    'path': str(bound), 'sha256': digest,
                },
            }},
            'timotion_final_state_audit': {'payload': {
                'status': 'pass', 'global_step': 106500,
                'checkpoint': str(bound), 'checkpoint_sha256': digest,
            }},
        }
        records['training_wallclock_audit']['payload']['records'][
            'coshmdm'
        ].update({
            'original_author_target_optimizer_updates': 568000,
            'training_budget_contract': (
                'user-capped to a two-day wall-clock window with validation '
                'selection among completed 20k milestones'
            ),
        })
        validate_training_and_protocol(records)
        formal_norms = records[
            'timotion_formal_wandb_history_audit'
        ]['payload']['metrics']
        formal_norms['train/gradient_l2_norm']['last_step'] = 106450
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'full online-W&B history through optimizer step' in str(error)
        else:
            raise AssertionError(
                'truncated TiMotion gradient telemetry was accepted'
            )
        formal_norms['train/gradient_l2_norm']['last_step'] = 106500
        validate_training_and_protocol(records)
        validation_history_record = records[
            'timotion_validation_selection'
        ]['payload']['validation_history']
        validation_history_payload = json.loads(
            timotion_validation_history_path.read_text(encoding='utf-8')
        )
        validation_history_payload['status'] = 'running'
        timotion_validation_history_path.write_text(
            json.dumps(validation_history_payload), encoding='utf-8'
        )
        validation_history_record['sha256'] = report_sha256_file(
            timotion_validation_history_path
        )
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'validation ledger was not finalized' in str(error)
        else:
            raise AssertionError(
                'an unfinished TiMotion validation ledger was accepted'
            )
        validation_history_payload['status'] = 'complete'
        timotion_validation_history_path.write_text(
            json.dumps(validation_history_payload), encoding='utf-8'
        )
        validation_history_record['sha256'] = report_sha256_file(
            timotion_validation_history_path
        )
        validate_training_and_protocol(records)
        shape_pair = records['hhc_mgt_shape_dependence_pair']['payload']
        dispatch = records['hhc_post_50k_training_dispatch']['payload']
        shape_pair['decision'] = (
            'train_only_shape_identity_shortcut_not_established'
        )
        dispatch['shape_source_decision'] = shape_pair['decision']
        dispatch['mask_conditioning_status'] = (
            'unresolved_after_two_negative_diagnostics'
        )
        dispatch['diagnostic_followup_required'] = True
        validate_training_and_protocol(records)
        dispatch.pop('mask_conditioning_status')
        dispatch.pop('diagnostic_followup_required')
        dispatch['shape_source_decision'] = (
            'train_only_shape_identity_shortcut_supported'
        )
        shape_pair['decision'] = dispatch['shape_source_decision']
        cosh_candidate = records[
            'coshmdm_validation_selection'
        ]['payload']['candidates'][0]
        cosh_candidate['checkpoint_source_stable_during_export'] = False
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'minimum-FID candidate' in str(error)
        else:
            raise AssertionError(
                'CoShMDM validation output without a stable checkpoint was accepted'
            )
        cosh_candidate['checkpoint_source_stable_during_export'] = True
        launch_sources = records[
            'hhc_paper_training_contract_audit'
        ]['payload']['launch_source_stability']
        launch_sources['scope'] = 'immutable_launch_snapshot'
        launch_sources['snapshot_manifest'] = {
            'path': str(bound), 'sha256': digest,
        }
        validate_training_and_protocol(records)
        launch_sources['snapshot_manifest']['sha256'] = '0' * 64
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'source_snapshot_manifest' in str(error)
        else:
            raise AssertionError('tampered formal source snapshot was accepted')
        launch_sources['scope'] = 'live_source_tree'
        launch_sources['snapshot_manifest'] = None
        formal_launch = records[
            'hhc_formal_launch_contract_audit'
        ]['payload']
        formal_launch['sampling_parity_source_scope'] = (
            'immutable_launch_snapshot'
        )
        formal_launch['source_snapshot_manifest'] = {
            'path': str(bound), 'sha256': digest,
        }
        validate_training_and_protocol(records)
        formal_launch['source_snapshot_manifest']['sha256'] = '0' * 64
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'source_snapshot_manifest' in str(error)
        else:
            raise AssertionError('tampered launch-contract snapshot was accepted')
        formal_launch['sampling_parity_source_scope'] = 'live_source_tree'
        formal_launch['source_snapshot_manifest'] = None
        mm_repeat = records[
            'historical_mgt_common50_mm_replay'
        ]['payload']['repeat_records'][17]
        mm_repeat['motions_byte_exact'] = False
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'exactly replay all 30' in str(error)
        else:
            raise AssertionError('one divergent MGT MM repeat was accepted')
        mm_repeat['motions_byte_exact'] = True
        cosh_epoch_series = records[
            'coshmdm_statsfix_startup_audit'
        ]['payload']['epoch_boundary_series']
        cosh_epoch_series['recorded_completed_epochs'] = 139
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'one discarded half-batch at every completed epoch' in str(error)
        else:
            raise AssertionError('incomplete CoShMDM epoch-boundary proof was accepted')
        cosh_epoch_series['recorded_completed_epochs'] = 140
        tokenizer_checks = records[
            'historical_mgt_tokenizer_runtime_audit'
        ]['payload']['checks']
        tokenizer_checks['root_statistics_selected'] = False
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'historical tokenizer runtime evidence' in str(error)
        else:
            raise AssertionError('failed historical tokenizer audit was accepted')
        tokenizer_checks['root_statistics_selected'] = True
        limitation = records['coshmdm_resume_provenance']['payload'][
            'reproducibility_limitation'
        ]
        records['coshmdm_resume_provenance']['payload'][
            'reproducibility_limitation'
        ] = 'bitwise identical to an uninterrupted run'
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'corrected training provenance' in str(error)
        else:
            raise AssertionError('overstated CoShMDM restart claim was accepted')
        records['coshmdm_resume_provenance']['payload'][
            'reproducibility_limitation'
        ] = limitation
        request_protocol = records['official_request_manifest']['payload'][
            'frozen_input_provenance'
        ]['protocol']
        request_protocol['sha256'] = '0' * 64
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'official_request_manifest/frozen_input_provenance/protocol' in str(error)
        else:
            raise AssertionError('stale frozen-request provenance was accepted')
        request_protocol['sha256'] = digest

        replay_names = records['fair_request_replay_audit']['payload'][
            'comparisons'
        ]['names']
        replay_names['exact'] = False
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'fair_request_replay_audit' in str(error)
        else:
            raise AssertionError('non-reproducible frozen request was accepted')
        replay_names['exact'] = True

        # Held-out shape/action request manifests are archived diagnostics, not
        # active final-report inputs.  Shape generalization is now tested by
        # train-only optimized random-shape augmentation on the unchanged
        # official validation/test protocol (see test_shape_augmentation_pipeline).

        historical_run = records[
            'historical_mmm_wandb_comparison'
        ]['payload']['historical_run']
        removed_milestone = historical_run['validation_history'].pop()
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert '30-milestone remote trajectory' in str(error)
        else:
            raise AssertionError('truncated historical W&B trajectory was accepted')
        historical_run['validation_history'].append(removed_milestone)
        historical_run['remote_history_provenance'][
            'source_code_archive_available'
        ] = True
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert '30-milestone remote trajectory' in str(error)
        else:
            raise AssertionError('false historical source archive was accepted')
        historical_run['remote_history_provenance'][
            'source_code_archive_available'
        ] = False

        historical_runtime = historical_run[
            'validation_runtime'
        ]['effective_validation_runtime']
        historical_runtime['outer_steps'] = 3
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'completed historical MGT validation runtime' in str(error)
        else:
            raise AssertionError(
                'wrong historical validation sampler budget was accepted'
            )
        historical_runtime['outer_steps'] = 10

        paper_mismatches = records[
            'hhc_paper_training_contract_audit'
        ]['payload']['paper_stated_numeric_mismatches']
        paper_mismatches['attention_heads'] = {'paper': 8, 'active': 16}
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'paper-stated MGT configuration' in str(error)
        else:
            raise AssertionError('mismatched paper training contract was accepted')
        paper_mismatches.clear()

        contact_map_policy = records[
            'hhc_contact_guidance_map_audit'
        ]['payload']['policy']
        contact_map_policy['fair_main_table'] = 'mmm_legacy_full_axis_bug_map'
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'hhc_contact_guidance_map_audit' in str(error)
        else:
            raise AssertionError('leaking legacy map was accepted for main table')
        contact_map_policy['fair_main_table'] = 'paper_formula_fair_map'

        coshmdm_release = records[
            'coshmdm_training_contract_audit'
        ]['payload']['released_checkpoint_compatibility']
        coshmdm_release['upstream_checkpoint_download_label'] = 'Inter-X'
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'CoShMDM source/data/budget contract' in str(error)
        else:
            raise AssertionError(
                'undocumented CoShMDM Inter-X checkpoint was accepted'
            )
        coshmdm_release['upstream_checkpoint_download_label'] = 'InterHuman'

        timotion_release = records[
            'timotion_training_contract_audit'
        ]['payload']['released_checkpoint_compatibility']
        timotion_release['documented_interx_checkpoint_available'] = True
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'TiMotion paper/data/source/restart contract' in str(error)
        else:
            raise AssertionError(
                'undocumented TiMotion Inter-X checkpoint was accepted'
            )
        timotion_release['documented_interx_checkpoint_available'] = False

        validation_exact = records[
            'hhc_validation_observational_pair_audit'
        ]['payload']['exact']
        validation_exact['optimizer_state'] = False
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'hhc_validation_observational_pair_audit' in str(error)
        else:
            raise AssertionError('validation-side optimizer mutation was accepted')
        validation_exact['optimizer_state'] = True

        fixed_validation_after = records[
            'hhc_validation_observational_pair_audit'
        ]['payload']['fixed_pair_callback']['after']
        fixed_validation_after['post_saved'] = False
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'hhc_validation_observational_pair_audit' in str(error)
        else:
            raise AssertionError(
                'checkpoint captured before post-validation save was accepted'
            )
        fixed_validation_after['post_saved'] = True

        test_full_counts = records[
            'interx_luna_test_full_cache_audit'
        ]['payload']['counts']
        test_full_counts['completed_samples'] = 1707
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'full-split Luna cache' in str(error)
        else:
            raise AssertionError('incomplete full-test Luna cache was accepted')
        test_full_counts['completed_samples'] = 1708

        records['interx_protocol_audit']['payload']['evaluated_entries'] = 1664
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'test GT count' in str(error)
        else:
            raise AssertionError('smaller evaluator GT set was accepted')
        records['interx_protocol_audit']['payload']['evaluated_entries'] = 1696

        coshmdm_contract = records[
            'coshmdm_training_contract_audit'
        ]['payload']
        coshmdm_contract['frozen_files_independently_rehashed'] = False
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'coshmdm_training_contract_audit' in str(error)
        else:
            raise AssertionError('unhashed CoShMDM source/data were accepted')
        coshmdm_contract['frozen_files_independently_rehashed'] = True

        adapter_audit = records[
            'coshmdm_representation_adapter_audit'
        ]['payload']
        adapter_audit['pass_conditions']['sample_count'] = False
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'coshmdm_representation_adapter_audit' in str(error)
        else:
            raise AssertionError('failed CoShMDM adapter audit was accepted')
        adapter_audit['pass_conditions']['sample_count'] = True
        implementation_path = next(iter(adapter_audit['implementation']))
        implementation_hash = adapter_audit['implementation'][implementation_path]
        adapter_audit['implementation'][implementation_path] = '0' * 64
        try:
            validate_coshmdm_representation_adapter(adapter_audit)
        except ValueError as error:
            assert 'coshmdm_representation_adapter_audit' in str(error)
        else:
            raise AssertionError('stale CoShMDM adapter code was accepted')
        adapter_audit['implementation'][implementation_path] = implementation_hash

        records['coshmdm_final_state_audit']['payload']['status'] = 'fail'
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'coshmdm_final_state_audit' in str(error)
        else:
            raise AssertionError('failed CoShMDM audit was accepted')
        records['coshmdm_final_state_audit']['payload']['status'] = 'pass'
        records['coshmdm_final_state_audit']['payload']['checkpoint_sha256'] = '0' * 64
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'coshmdm_final_state_audit' in str(error)
        else:
            raise AssertionError('stale CoShMDM checkpoint hash was accepted')
        records['coshmdm_final_state_audit']['payload']['checkpoint_sha256'] = digest
        timotion_contract = records[
            'timotion_training_contract_audit'
        ]['payload']
        timotion_contract['tree_contents_independently_rehashed'] = False
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'timotion_training_contract_audit' in str(error)
        else:
            raise AssertionError('unhashed TiMotion data tree was accepted')
        timotion_contract['tree_contents_independently_rehashed'] = True
        records['timotion_final_state_audit']['payload']['global_step'] = 106499
        try:
            validate_training_and_protocol(records)
        except ValueError as error:
            assert 'expected global step 106500' in str(error)
        else:
            raise AssertionError('wrong TiMotion final step was accepted')


        metric_names = (
            'top1', 'top2', 'top3', 'fid',
            'matching', 'diversity', 'multimodality',
        )
        summary = {
            'replications': 20, 'samples_per_replication': 1696,
            'inputs': ['metric.json'] * 20,
            'metrics': {
                name: {'values': [1.0] * 20, 'mean': 1.0, 'ci95': 0.0}
                for name in metric_names
            },
        }
        multimodality = {
            'replications': 20,
            'legacy_provenance_allowed': False,
            'per_replication': [
                {'replication': replication} for replication in range(20)
            ],
            'official_10_pairs': {'values': [1.0] * 20},
            'exhaustive_all_pairs': {
                'mean': 1.0, 'ci95': 0.0, 'values': [1.0] * 20,
            },
        }
        physical = {
            'samples': 1696, 'repeat': 0,
            'protocol': 'interx_official_v1_physical_joint_v1',
            'definition': (
                'LLM-free joint metrics using the frozen binary contact '
                'schedule and subject-specific Inter-X skeletons; '
                'penetration remains a separate mesh metric.'
            ),
            'contact_threshold_m': 0.1,
            'ground_height_threshold_m': 0.05,
            'generation_request_hash_verified': True,
            'requests': str(bound), 'requests_sha256': digest,
            'generations': str(bound), 'generations_sha256': digest,
            'contact_f1': 0.5, 'contact_temporal_iou': 0.4,
            'false_contact_rate': 0.2,
            'minimum_interperson_joint_distance_m': 0.01,
            'foot_sliding_mps': 0.1,
        }
        timing = {
            'schema': 'fair_interx_generation_wall_timing_summary_v1',
            'replications': 20, 'label': 'synthetic',
            'inputs': [
                {'path': str(bound), 'sha256': digest} for _ in range(20)
            ],
            'main_elapsed_seconds': {
                'mean': 1.0, 'ci95_half_width': 0.1,
                'median': 1.0, 'p95': 1.1,
            },
            'main_seconds_per_sample_amortized': {
                'mean': 0.1, 'ci95_half_width': 0.01,
                'median': 0.1, 'p95': 0.11,
            },
        }
        validate_method_artifacts(
            'synthetic', summary, multimodality, physical, timing
        )
        physical['generation_request_hash_verified'] = False
        try:
            validate_method_artifacts(
                'synthetic', summary, multimodality, physical, timing
            )
        except ValueError as error:
            assert 'request hash' in str(error)
        else:
            raise AssertionError('unverified method generation was accepted')
        physical['generation_request_hash_verified'] = True
        physical['contact_threshold_m'] = 0.15
        try:
            validate_method_artifacts(
                'synthetic', summary, multimodality, physical, timing
            )
        except ValueError as error:
            assert 'frozen LLM-free Inter-X contract' in str(error)
        else:
            raise AssertionError('mismatched physical protocol was accepted')
        physical['contact_threshold_m'] = 0.1

        # A native InterMask 20-step artifact must not pass merely because its
        # directory/run-manifest was renamed to the common-50 label.
        summary['provenance'] = {
            'legacy_provenance_allowed': False,
            'legacy_provenance_reasons': [],
            'generation_files_sha256_verified': True,
            'unique_requests_verified': True,
            'sampler_identity': {
                'time_steps': 50, 'cond_scale': 2.0,
                'temperature': 1.0, 'topk_filter_thres': 0.9,
                'length_policy': 'exact',
            },
        }
        timing['label'] = 'intermask_official_exact_common50'
        validate_method_artifacts(
            'intermask_official_exact_common50', summary, multimodality,
            physical, timing,
        )
        summary['provenance']['sampler_identity']['time_steps'] = 20
        try:
            validate_method_artifacts(
                'intermask_official_exact_common50', summary, multimodality,
                physical, timing,
            )
        except ValueError as error:
            assert 'time_steps=20' in str(error)
        else:
            raise AssertionError('native InterMask 20-step HDF5 was accepted as common50')

        summary['provenance']['sampler_identity'] = {
            'sampler': 'sample_intra_cas',
            'sampler_semantics': 'paper_high_confidence_tokens_fixed',
            'cascade_order': 'root,body,hands',
            'sampling_steps': 50,
            'sampling_steps_unit': 'outer_maskgit_iterations',
            'maximum_active_channel_stages_per_outer_iteration': 3,
            'maximum_model_forward_calls_per_outer_iteration_with_cfg': 6,
            'resolved_channels_skip_forward_but_not_outer_iterations': True,
            'temperature': 0.5, 'cfg_scale': 3.5,
            'contact_guidance': True, 'contact_guidance_scale': 1.0,
            'contact_map_contract': 'paper_formula_train_only',
            'generation_length_policy': 'exact_request_length',
        }
        timing['label'] = 'hhc_generalization_shape_test_unseen'
        validate_method_artifacts(
            'hhc_generalization_shape_test_unseen', summary, multimodality,
            physical, timing,
        )
        summary['provenance']['sampler_identity']['sampling_steps'] = 20
        try:
            validate_method_artifacts(
                'hhc_generalization_shape_test_unseen', summary,
                multimodality, physical, timing,
            )
        except ValueError as error:
            assert 'sampling_steps=20' in str(error)
        else:
            raise AssertionError('non-common50 generalization HDF5 was accepted')

        repository = Path(directory) / 'provenance_repo'
        repository.mkdir()
        subprocess.run(['git', 'init', '-q', str(repository)], check=True)
        subprocess.run(
            ['git', '-C', str(repository), 'config', 'user.email',
             'paper-consistency@example.invalid'], check=True,
        )
        subprocess.run(
            ['git', '-C', str(repository), 'config', 'user.name',
             'Paper Consistency Test'], check=True,
        )
        source = repository / 'model.py'
        repository_note = repository / 'README.md'
        source.write_text('sampler = "common50"\n')
        repository_note.write_text('frozen source tree\n')
        subprocess.run(
            ['git', '-C', str(repository), 'add', 'model.py', 'README.md'],
            check=True,
        )
        subprocess.run(
            ['git', '-C', str(repository), 'commit', '-q', '-m', 'fixture'],
            check=True,
        )
        revision = subprocess.check_output(
            ['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True,
        ).strip()
        clean_diff_hash = hashlib.sha256(subprocess.check_output(
            ['git', '-C', str(repository), 'diff', '--binary', 'HEAD']
        )).hexdigest()
        strict_provenance = {
            'artifacts': {
                'checkpoint': {'path': str(bound), 'sha256': digest},
            },
            'implementation': {
                'model': {
                    'path': str(source),
                    'sha256': report_sha256_file(source),
                },
            },
            'repositories': {
                'fixture': {
                    'path': str(repository), 'revision': revision,
                    'tracked_diff_sha256': clean_diff_hash,
                },
            },
        }
        validate_method_provenance('synthetic', {'provenance': strict_provenance})
        repository_note.write_text('source tree drifted after generation\n')
        # Repository records describe the historical launch state.  Unrelated
        # later worktree edits must not invalidate hash-bound executable bytes.
        validate_method_provenance('synthetic', {'provenance': strict_provenance})
        source.write_text('sampler = "changed_after_generation"\n')
        try:
            validate_method_provenance(
                'synthetic', {'provenance': strict_provenance}
            )
        except ValueError as error:
            assert 'bound artifact SHA-256' in str(error)
        else:
            raise AssertionError('post-generation bound source drift was accepted')

        intermask_manifest = {
            'method': 'intermask', 'label': 'intermask_test',
            'replication_range': [0, 20], 'samples_per_replication': 1696,
            'sampling_steps': 50, 'intermask_length_policy': 'exact',
        }
        validate_method_manifest('intermask_test', intermask_manifest)
        intermask_manifest['sampling_steps'] = 20
        try:
            validate_method_manifest('intermask_test', intermask_manifest)
        except ValueError as error:
            assert 'not 50' in str(error)
        else:
            raise AssertionError('non-common InterMask sampling budget was accepted')

        generalization_manifest = {
            'method': 'intermask',
            'label': 'intermask_generalization_count_smoke',
            'replication_range': [0, 20],
            'samples_per_replication': 320,
            'sampling_steps': 50,
            'intermask_length_policy': 'exact',
        }
        validate_method_manifest(
            generalization_manifest['label'], generalization_manifest,
            expected_samples=320,
        )
        generalization_manifest['samples_per_replication'] = 1696
        try:
            validate_method_manifest(
                generalization_manifest['label'], generalization_manifest,
                expected_samples=320,
            )
        except ValueError as error:
            assert 'does not bind 320 samples' in str(error)
        else:
            raise AssertionError(
                'wrong generalization partition sample count was accepted'
            )

        held_out_contract = {
            'samples_per_replication': 320,
            'requests_dir':
            'output/fair_interx_generalization_requests/shape/test_unseen',
            'requests_manifest':
            'output/fair_interx_generalization_requests/manifest.json',
            'hhc_stats_dir': 'configs/interx_generalization/shape/stats',
            'hhc_stats_prefix': 'generalization_shape',
            'selection_manifest':
            'output/hhc_mgt_generalization_shape_selection.json',
            'sampling_steps': 50,
            'contact_guidance': True,
        }
        validate_generalization_manifest_contract(
            'hhc_generalization_shape_test_unseen', held_out_contract
        )
        held_out_contract['hhc_stats_prefix'] = 'Interx'
        try:
            validate_generalization_manifest_contract(
                'hhc_generalization_shape_test_unseen', held_out_contract
            )
        except ValueError as error:
            assert 'request/stats/selection/common50 contract' in str(error)
        else:
            raise AssertionError('wrong held-out normalization was accepted')

        cosh_training_path = Path(directory) / 'cosh_training.json'
        cosh_training = {
            'training': {'epochs': 2000, 'expected_optimizer_updates': 568000},
            'model': {'sampling_strategy': 'ddim50', 'cfg_weight': 3.5},
        }
        cosh_training_path.write_text(json.dumps(cosh_training))
        cosh_cap_path = Path(directory) / 'cosh_training_cap.json'
        cosh_cap_path.write_text(json.dumps({
            'schema': 'hhc_coshmdm_user_training_cap_v1',
            'status': 'stopped_gracefully',
            'preferred_terminal_optimizer_step': 200000,
        }))
        cosh_preservation_path = Path(directory) / 'cosh_preservation.json'
        cosh_preservation_path.write_text(json.dumps({
            'schema': 'hhc_coshmdm_milestone_preservation_v1',
            'status': 'trainer_exited',
            'maximum_preserved_step': 200000,
            'interval_steps': 20000,
            'preserved': [{
                'step': 40000, 'same_inode': True,
            }],
        }))
        cosh_selection_path = Path(directory) / 'cosh_selection.json'
        cosh_selection_path.write_text(json.dumps({
            'schema': 'hhc_coshmdm_capped_validation_selection_v1',
            'status': 'complete',
            'training_cap_status': 'stopped_gracefully',
            'test_metrics_used_for_selection': False,
            'formal_hhc_validation_record_set_exact': True,
            'formal_hhc_validation_selection_sha256': 'a' * 64,
            'selected_checkpoint': str(bound),
            'selected_checkpoint_sha256': digest,
            'artifacts': {
                'formal_hhc_validation_selection': {
                    'path': str(bound), 'sha256': digest,
                },
            },
        }))
        cosh_source_names = (
            'driver', 'scorer', 'exporter', 'adapter', 'model', 'diffusion',
            'gaussian_diffusion', 'normalizer', 'model_config',
            'normalization_mean', 'normalization_std', 'requests_manifest',
        )
        cosh_generation_audit = {
            'schema': 'hhc_coshmdm_generation_contract_audit_v1',
            'status': 'pass',
            'checks': {'all_contract_checks': True},
            'sampling': {
                'process': 'DDIM50_eta0_CFG3.5',
                'runtime_sampling_steps': 50, 'ddim_eta': 0.0,
            },
            'requests': {
                'replications': 20, 'samples_per_replication': 1696,
                'verified_file_hashes': 20,
            },
            'representation': {'gt_motion_consumed_by_exporter': False},
            'sources': {
                name: {'sha256': digest} for name in cosh_source_names
            },
            'implementation': {'sha256': digest},
        }
        cosh_generation_audit_path = Path(directory) / 'cosh_generation.json'
        cosh_generation_audit_path.write_text(json.dumps(cosh_generation_audit))
        bound_record = {'path': str(bound), 'sha256': digest}
        cosh_implementation = {
            name: {'sha256': digest} for name in (
                'driver', 'semantic_scorer', 'exporter', 'official_adapter',
                'official_model', 'official_diffusion', 'gaussian_diffusion',
                'normalizer', 'generation_contract_auditor',
            )
        }
        cosh_manifest = {
            'method': 'coshmdm', 'label': 'cosh_test',
            'replication_range': [0, 20], 'samples_per_replication': 1696,
            'coshmdm_length_policy': 'exact_request_length_grouped',
            'coshmdm_sampling_process': 'DDIM50_eta0_CFG3.5',
            'provenance': {
                'artifacts': {
                    'training_manifest': {
                        'path': str(cosh_training_path),
                        'sha256': report_sha256_file(cosh_training_path),
                    },
                    'user_training_budget_cap': {
                        'path': str(cosh_cap_path),
                        'sha256': report_sha256_file(cosh_cap_path),
                    },
                    'milestone_preservation_manifest': {
                        'path': str(cosh_preservation_path),
                        'sha256': report_sha256_file(cosh_preservation_path),
                    },
                    'validation_selection_manifest': {
                        'path': str(cosh_selection_path),
                        'sha256': report_sha256_file(cosh_selection_path),
                    },
                    'model': bound_record,
                    'generation_contract_audit': {
                        'path': str(cosh_generation_audit_path),
                        'sha256': report_sha256_file(cosh_generation_audit_path),
                    },
                    'model_config': bound_record,
                    'normalization_mean': bound_record,
                    'normalization_std': bound_record,
                    'requests_manifest': bound_record,
                },
                'implementation': cosh_implementation,
            },
        }
        validate_method_manifest('cosh_test', cosh_manifest)
        cosh_training['model']['sampling_strategy'] = 'ddim2000'
        cosh_training_path.write_text(json.dumps(cosh_training))
        cosh_manifest['provenance']['artifacts']['training_manifest'][
            'sha256'
        ] = report_sha256_file(cosh_training_path)
        try:
            validate_method_manifest('cosh_test', cosh_manifest)
        except ValueError as error:
            assert 'DDIM50' in str(error)
        else:
            raise AssertionError('CoShMDM non-DDIM50 inference was accepted')


def test_timotion_generation_contract_is_bound_to_final_manifest():
    label = 'timotion_transformer_lpa_interx_retrained_fair'
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        bound = directory / 'bound.bin'
        bound.write_bytes(b'timotion-generation-contract-fixture')
        digest = report_sha256_file(bound)
        snapshot = directory / f'{digest}__bound.bin'
        snapshot.write_bytes(bound.read_bytes())
        repository = Path('/mnt/8T-HDD/wenning/TIMotion')
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repository, text=True
        ).strip()
        tracked_diff = subprocess.check_output(
            ['git', 'diff', '--binary', 'HEAD'], cwd=repository
        )
        tracked_diff_sha256 = hashlib.sha256(tracked_diff).hexdigest()
        training_path = directory / 'training.json'
        training_path.write_text(json.dumps({'repository': {
            'revision': revision,
            'tracked_diff_sha256': tracked_diff_sha256,
        }}))
        source_names = (
            'driver', 'scorer', 'exporter', 'model', 'diffusion',
            'interaction_block', 'gaussian_diffusion', 'model_config',
            'normalization_mean', 'normalization_std', 'requests_manifest',
        )
        audit = {
            'schema': 'hhc_timotion_generation_contract_audit_v1',
            'status': 'pass', 'checks': {'all_contract_checks': True},
            'sampling': {
                'process': 'DDIM50_eta0_CFG3.5',
                'runtime_sampling_steps': 50, 'ddim_eta': 0.0,
            },
            'requests': {
                'replications': 20, 'samples_per_replication': 1696,
                'verified_file_hashes': 20,
            },
            'representation': {
                'gt_motion_consumed_by_exporter': False,
                'gt_first_frame_consumed_by_exporter': False,
            },
            'sources': {
                name: {'sha256': digest} for name in source_names
            },
            'implementation': {'sha256': digest},
        }
        audit_path = directory / 'generation_audit.json'
        audit_path.write_text(json.dumps(audit))
        bound_record = {'path': str(bound), 'sha256': digest}
        implementation_record = {
            'path': str(snapshot), 'sha256': digest,
            'content_addressed_snapshot': True,
            'source_path_at_execution': str(bound),
        }
        implementation = {
            name: dict(implementation_record) for name in (
                'driver', 'semantic_scorer', 'exporter', 'official_model',
                'official_denoiser', 'official_interaction_block',
                'gaussian_diffusion', 'generation_contract_auditor',
            )
        }
        manifest = {
            'method': 'timotion', 'label': label,
            'replication_range': [0, 20], 'samples_per_replication': 1696,
            'timotion_first_frame_conditioning': 'none',
            'timotion_sampling_process': 'DDIM50_eta0_CFG3.5',
            'timotion_length_policy': 'exact_request_length_grouped',
            'implementation_source_stability_verified_at_completion': True,
            'provenance': {
                'source_validation_contract': (
                    'fair_interx_content_addressed_implementation_snapshots_v1'
                ),
                'repositories': {'timotion': {
                    'path': str(repository), 'revision': revision,
                    'tracked_diff_sha256': tracked_diff_sha256,
                }},
                'artifacts': {
                    'training_manifest': {
                        'path': str(training_path),
                        'sha256': report_sha256_file(training_path),
                    },
                    'generation_contract_audit': {
                        'path': str(audit_path),
                        'sha256': report_sha256_file(audit_path),
                    },
                    'model_config': bound_record,
                    'normalization_mean': bound_record,
                    'normalization_std': bound_record,
                    'requests_manifest': bound_record,
                },
                'implementation': implementation,
            },
        }
        validate_method_manifest(label, manifest)
        audit['sampling']['runtime_sampling_steps'] = 49
        audit_path.write_text(json.dumps(audit))
        manifest['provenance']['artifacts']['generation_contract_audit'][
            'sha256'
        ] = report_sha256_file(audit_path)
        try:
            validate_method_manifest(label, manifest)
        except ValueError as error:
            assert 'generation contract audit is incomplete' in str(error)
        else:
            raise AssertionError('TiMotion non-DDIM50 generation was accepted')


def test_protocol_compute_field_aliases_reject_conflicts():
    assert protocol_alias(
        {'channel_updates_per_outer_step': 3},
        'maximum_active_channel_stages_per_outer_iteration',
        'channel_updates_per_outer_step',
    ) == 3
    assert protocol_alias(
        {
            'maximum_active_channel_stages_per_outer_iteration': 3,
            'channel_updates_per_outer_step': 3,
        },
        'maximum_active_channel_stages_per_outer_iteration',
        'channel_updates_per_outer_step',
    ) == 3
    try:
        protocol_alias(
            {
                'maximum_active_channel_stages_per_outer_iteration': 2,
                'channel_updates_per_outer_step': 3,
            },
            'maximum_active_channel_stages_per_outer_iteration',
            'channel_updates_per_outer_step',
        )
    except ValueError as error:
        assert 'conflicting protocol aliases' in str(error)
    else:
        raise AssertionError('conflicting protocol aliases were accepted')


def test_sampler_skeleton_identity_accepts_only_same_canonical_directory():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        skeletons = root / 'skeletons'
        skeletons.mkdir()
        alias = root / 'workspace_skeletons'
        alias.symlink_to(skeletons, target_is_directory=True)
        other = root / 'other_skeletons'
        other.mkdir()
        assert sampler_identity_value_matches(
            'skeleton_root', str(alias), str(skeletons)
        )
        assert not sampler_identity_value_matches(
            'skeleton_root', str(alias), str(other)
        )
        assert sampler_identity_value_matches('sampling_steps', 50, 50)
        assert not sampler_identity_value_matches('sampling_steps', 20, 50)


def test_official_scorer_and_report_exclude_secondary_fid_protocols():
    root = Path(__file__).resolve().parents[1]
    scorer = root / 'tools/score_fair_interx_generations.py'
    report = root / 'tools/build_review_results_report.py'
    result = subprocess.run(
        [
            sys.executable, str(scorer), '--requests', 'missing-request.npz',
            '--main-generations', 'missing-main.h5', '--mm-generations',
            'missing-mm.h5', '--output', 'unused.json', '--gt-generations',
            'forbidden-projected-gt.h5',
        ],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert 'unrecognized arguments: --gt-generations' in result.stderr

    scorer_source = scorer.read_text(encoding='utf-8')
    report_source = report.read_text(encoding='utf-8')
    for forbidden in (
        'fid_body_controlled', 'representation_controlled',
        "parser.add_argument('--gt-generations'",
    ):
        assert forbidden not in scorer_source
    for forbidden in (
        'coshmdm_representation_controlled_rep0',
        'timotion_representation_controlled_rep0',
        'coshmdm_representation_ceiling_rep0',
    ):
        assert forbidden not in report_source


def test_mgt_common50_bridge_binds_contact_vq_and_evaluator():
    metadata = {
        'sampler': 'sample_intra_cas',
        'sampler_semantics': 'paper_high_confidence_tokens_fixed',
        'sampling_steps': 50,
        'sampling_steps_unit': 'outer_maskgit_iterations',
        'temperature': 0.5,
        'cfg_scale': 3.5,
        'cascade_order': 'root,body,hands',
        'maximum_active_channel_stages_per_outer_iteration': 3,
        'maximum_model_forward_calls_per_outer_iteration_with_cfg': 6,
        'resolved_channels_skip_forward_but_not_outer_iterations': True,
        'contact_guidance': True,
        'contact_guidance_scale': 1.0,
        'contact_timing_source': 'request_oracle',
        'contact_timing_file_sha256': BRIDGE_REQUEST_SHA256,
        'requests_sha256': BRIDGE_REQUEST_SHA256,
        'seed_strategy': 'base_seed + repeat_index',
        'seed': 1234,
        'generation_length_policy': 'exact_request_length',
        'mgt_contract': 'historical_mmm',
        'vq_contract': 'precleanup_raw12288',
        'vq_checkpoint_sha256': BRIDGE_VQ_SHA256,
        'contact_map_sha256': BRIDGE_CONTACT_MAP_SHA256,
        'contact_map_contract': 'paper_formula_train_only',
        'normalization_mean_sha256': BRIDGE_MEAN_SHA256,
        'normalization_std_sha256': BRIDGE_STD_SHA256,
        'training_conditioning_contract': 'historical_mmm_shape_dropout',
        'inference_conditioning_contract': 'paper_text_cfg',
    }
    assert validate_bridge_metadata(metadata, 1234, 'fixture') == (False, False)
    archived = dict(metadata)
    archived.pop('maximum_active_channel_stages_per_outer_iteration')
    archived.pop('maximum_model_forward_calls_per_outer_iteration_with_cfg')
    archived.pop('resolved_channels_skip_forward_but_not_outer_iterations')
    archived['channel_updates_per_outer_step'] = 3
    archived['model_forward_calls_per_outer_step_with_cfg'] = 6
    archived.pop('training_conditioning_contract')
    archived.pop('inference_conditioning_contract')
    assert validate_bridge_metadata(
        archived, 1234, 'archived fixture', allow_legacy_compute_names=True
    ) == (True, True)
    try:
        validate_bridge_metadata(archived, 1234, 'new fixture')
    except RuntimeError as error:
        assert 'generation contract mismatch' in str(error)
    else:
        raise AssertionError('new generation accepted legacy compute field names')
    invalid = dict(metadata, contact_guidance_scale=0.0)
    try:
        validate_bridge_metadata(invalid, 1234, 'fixture')
    except RuntimeError as error:
        assert 'contact_guidance_scale' in str(error)
    else:
        raise AssertionError('wrong contact-guidance scale was accepted')

    scorer = Path(__file__).resolve().parents[1] / 'tools/score_fair_interx_generations.py'
    record = {
        'evaluator_provenance': {
            'contract': 'official_interx_full_672d',
            'options': scorer_artifact_record(EVALUATOR_OPTIONS),
            'checkpoint': scorer_artifact_record(EVALUATOR_CHECKPOINT),
            'wrapper_source': scorer_artifact_record(EVALUATOR_WRAPPER_SOURCE),
            'option_loader_source': scorer_artifact_record(EVALUATOR_OPTION_SOURCE),
            'scorer_source': scorer_artifact_record(scorer),
        },
        'evaluator_runtime': official_main_only_evaluator_runtime(),
    }
    validate_bridge_evaluator(record, 'fixture')
    validate_bridge_evaluator_runtime(record, 'fixture')
    invalid_runtime = json.loads(json.dumps(record))
    invalid_runtime['evaluator_runtime'] = (
        official_main_only_evaluator_runtime('cuda:0')
    )
    try:
        validate_bridge_evaluator_runtime(invalid_runtime, 'fixture')
    except RuntimeError as error:
        assert 'frozen CPU contract' in str(error)
    else:
        raise AssertionError('CUDA common-50 bridge runtime was accepted')
    record['evaluator_provenance']['options']['sha256'] = '0' * 64
    try:
        validate_bridge_evaluator(record, 'fixture')
    except RuntimeError as error:
        assert 'recorded hash changed' in str(error)
    else:
        raise AssertionError('wrong evaluator-options hash was accepted')


def test_mgt_repair_gate_rejects_contact_contract_drift():
    repository = Path(__file__).resolve().parents[1]
    finalizer_source = (
        repository / 'tools/finalize_mgt_repair_v7.sh'
    ).read_text(encoding='utf-8')
    assert finalizer_source.count('tools/score_fair_interx_generations.py') == 2
    assert 'reference_generation_root=' in finalizer_source
    assert '--contact-guidance-scale 1.0 --hhc-contact-timing-source request_oracle' in finalizer_source
    assert 'tools/compare_mgt_common50_bridge.py' in finalizer_source
    assert '--launch-source-recovery output/historical_mgt_v7_launch_source_recovery.json' in finalizer_source
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint_hash = '1' * 64
        records = {
            'vq.json': {
                'schema': 'fair_interx_main_only_metrics_v1',
                'status': 'complete', 'samples': 1696, 'fid': 0.294,
                'main_generation_metadata': {
                    'checkpoint_sha256': BRIDGE_VQ_SHA256,
                },
                'evaluator_provenance': (
                    official_main_only_evaluator_provenance()
                ),
                'evaluator_runtime': official_main_only_evaluator_runtime(),
            },
            'trajectory.json': {
                'schema': 'hhc_mgt_wandb_training_trajectory_comparison_v1',
                'status': 'pass',
                'comparison_scope':
                'all_candidate_training_rows_present_in_reference',
                'compared_step_count': 6001,
                'requested_steps': [299999],
                'maximum_absolute_loss_difference': 0,
                'threshold_failures': [], 'missing': [],
            },
            'validation.json': {
                'schema': 'hhc_mgt_wandb_validation_comparison_v1',
                'status': 'pass',
                'requested_steps': list(range(9999, 300000, 10000)),
                'maximum_absolute_difference': 0,
                'threshold_failures': [], 'missing': [],
            },
            'checkpoint.json': {
                'schema': 'hhc_mgt_historical_checkpoint_pair_v1',
                'status': 'pass', 'expected_step': 280000,
                'exact': {'model': True, 'optimizer': True, 'scheduler': True},
                'candidate': {'sha256': checkpoint_hash},
            },
            'bridge.json': {
                'schema': 'hhc_mgt_common50_bridge_v1', 'status': 'pass',
                'evaluated_samples': 1696, 'sampling_steps': 50,
                'sampler': 'sample_intra_cas', 'temperature': 0.5,
                'cfg_scale': 3.5, 'contact_guidance': True,
                'training_conditioning_contract':
                'historical_mmm_shape_dropout',
                'inference_conditioning_contract': 'paper_text_cfg',
                'contact_guidance_scale': 1.0,
                'contact_timing_source': 'request_oracle',
                'contact_timing_file_sha256': BRIDGE_REQUEST_SHA256,
                'contact_map_contract':
                'paper_formula_train_only_by_frozen_hash',
                'contact_map_sha256': BRIDGE_CONTACT_MAP_SHA256,
                'vq_contract': 'precleanup_raw12288',
                'vq_checkpoint_sha256': BRIDGE_VQ_SHA256,
                'requests_sha256': BRIDGE_REQUEST_SHA256,
                'reference': {'checkpoint_step': 280000, 'fid': 0.18},
                'candidate': {
                    'checkpoint_step': 280000,
                    'checkpoint_sha256': checkpoint_hash,
                    'fid': 0.18,
                },
                'differences': {'fid': {'absolute': 0.0}},
                'maximum_absolute_fid_difference': 0.02,
                'evaluator_provenance': {
                    'contract': 'official_interx_full_672d',
                    'options': scorer_artifact_record(EVALUATOR_OPTIONS),
                    'checkpoint': scorer_artifact_record(EVALUATOR_CHECKPOINT),
                },
                'evaluator_runtime': official_main_only_evaluator_runtime(),
            },
        }
        for name, payload in records.items():
            (root / name).write_text(json.dumps(payload), encoding='utf-8')

        def local_artifact(path):
            return {
                'path': str(path.resolve()),
                'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        launch_source_names = [
            'options/option_transformer.py',
            'train/train_t2i_trans.py',
            *[f'models/launch_source_{index}.py' for index in range(12)],
        ]
        launch_snapshots = {}
        launch_hashes = {}
        for index, name in enumerate(launch_source_names):
            source = root / f'launch_source_{index}.py'
            source.write_text(f'# launch source {name}\n', encoding='utf-8')
            launch_snapshots[name] = local_artifact(source)
            launch_hashes[name] = launch_snapshots[name]['sha256']
        launch_manifest = root / 'training_manifest.json'
        launch_manifest.write_text(
            json.dumps({'sha256': launch_hashes}), encoding='utf-8'
        )
        patch_lines = [68502, 68508, 68514, 68572, 68683, 68689, 68854]
        selected_records = root / 'selected_patch_records.jsonl'
        selected_text = ''.join(
            json.dumps({'line': line}) + '\n' for line in patch_lines
        )
        selected_records.write_text(selected_text, encoding='utf-8')
        selected_rows = selected_text.splitlines(keepends=True)
        launch_recovery = root / 'launch_recovery.json'
        launch_recovery.write_text(json.dumps({
            'schema': 'hhc_mgt_launch_source_recovery_v1',
            'status': 'pass',
            'training_manifest': local_artifact(launch_manifest),
            'selected_session_records': local_artifact(selected_records),
            'reversed_patch_records': [
                {
                    'line': line,
                    'patch_sha256': hashlib.sha256(
                        f'patch-{line}'.encode()
                    ).hexdigest(),
                    'session_line_sha256': hashlib.sha256(
                        selected_rows[index].encode()
                    ).hexdigest(),
                }
                for index, line in enumerate(patch_lines)
            ],
            'changed_sources_recovered_from_history': [
                'options/option_transformer.py',
                'train/train_t2i_trans.py',
            ],
            'snapshots': launch_snapshots,
            'all_snapshot_hashes_match_launch_manifest': True,
        }), encoding='utf-8')
        gate = root / 'gate.json'
        result = subprocess.run([
            sys.executable, str(repository / 'tools/build_mgt_repair_gate.py'),
            '--vq-reconstruction', str(root / 'vq.json'),
            '--trajectory-audit', str(root / 'trajectory.json'),
            '--validation-audit', str(root / 'validation.json'),
            '--checkpoint-pair-audit', str(root / 'checkpoint.json'),
            '--common50-bridge', str(root / 'bridge.json'),
            '--launch-source-recovery', str(launch_recovery),
            '--output', str(gate),
        ], cwd=repository, capture_output=True, text=True)
        assert result.returncode == 0, (result.stdout, result.stderr)
        require_command = [
            sys.executable, str(repository / 'tools/require_mgt_repair_gate.py'),
            '--gate', str(gate),
        ]
        result = subprocess.run(
            require_command, cwd=repository, capture_output=True, text=True,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)

        pristine_gate = json.loads(gate.read_text(encoding='utf-8'))
        payload = json.loads(json.dumps(pristine_gate))
        payload['common50_bridge']['contact_guidance_scale'] = 0.0
        gate.write_text(json.dumps(payload), encoding='utf-8')
        result = subprocess.run(
            require_command, cwd=repository, capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert 'contact-guidance scale' in result.stderr

        payload = json.loads(json.dumps(pristine_gate))
        payload['tokenizer']['reconstruction_evaluator_provenance'][
            'options'
        ]['sha256'] = '0' * 64
        gate.write_text(json.dumps(payload), encoding='utf-8')
        result = subprocess.run(
            require_command, cwd=repository, capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert 'reconstruction evaluator options' in result.stderr

        payload = json.loads(json.dumps(pristine_gate))
        payload['tokenizer']['reconstruction_evaluator_runtime'][
            'requested_device'
        ] = 'cuda:0'
        gate.write_text(json.dumps(payload), encoding='utf-8')
        result = subprocess.run(
            require_command, cwd=repository, capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert 'reconstruction evaluator runtime' in result.stderr

        payload = json.loads(json.dumps(pristine_gate))
        payload['common50_bridge']['evaluator_runtime'][
            'requested_device'
        ] = 'cuda:0'
        gate.write_text(json.dumps(payload), encoding='utf-8')
        result = subprocess.run(
            require_command, cwd=repository, capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert 'common-50 evaluator runtime' in result.stderr

        gate.write_text(json.dumps(pristine_gate), encoding='utf-8')
        Path(next(iter(
            pristine_gate['historical_replay'][
                'launch_source_snapshots'
            ].values()
        ))['path']).write_text('# tampered\n', encoding='utf-8')
        result = subprocess.run(
            require_command, cwd=repository, capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert 'historical launch source' in result.stderr


def test_vq_reconstruction_scorer_migration_is_exact_and_source_bound():
    repository = Path(__file__).resolve().parents[1]
    scorer = repository / 'tools/score_fair_interx_main.py'
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provenance = official_main_only_evaluator_provenance()
        payload = {
            'schema': 'fair_interx_main_only_metrics_v1',
            'status': 'complete',
            'samples': 1696,
            'requests_sha256': '1' * 64,
            'main_generations_sha256': '2' * 64,
            'generation_length_policy': 'requested_lengths',
            'main_generation_metadata': {
                'generation_length_policy': 'exact_request_length',
                'checkpoint_sha256': BRIDGE_VQ_SHA256,
            },
            'evaluator_provenance': provenance,
            'evaluator_runtime': official_main_only_evaluator_runtime(),
            'fid': 0.29441722725218256,
            'diversity_gt': 9.167706489562988,
            'diversity': 8.851842880249023,
            'top1': 0.4027122641509434,
            'top2': 0.5902122641509434,
            'top3': 0.6904481132075472,
            'matching': 3.7767529487609863,
            'multimodality': None,
        }
        before_payload = json.loads(json.dumps(payload))
        before_payload['evaluator_provenance']['scorer_source'] = {
            'path': str(scorer.resolve()),
            'bytes': scorer.stat().st_size - 1,
            'sha256': '0' * 64,
        }
        before = root / 'before.json'
        after = root / 'after.json'
        output = root / 'audit.json'
        before.write_text(json.dumps(before_payload), encoding='utf-8')
        after.write_text(json.dumps(payload), encoding='utf-8')
        command = [
            sys.executable,
            str(repository / 'tools/audit_vq_reconstruction_scorer_migration.py'),
            '--before', str(before), '--after', str(after),
            '--output', str(output),
        ]
        result = subprocess.run(
            command, cwd=repository, capture_output=True, text=True,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        audit = json.loads(output.read_text(encoding='utf-8'))
        assert audit['status'] == 'pass'
        assert audit['checks']['all_metrics_exact'] is True
        assert audit['scorer_source']['after'] == scorer_artifact_record(scorer)

        changed = json.loads(json.dumps(payload))
        changed['fid'] += 1e-12
        after.write_text(json.dumps(changed), encoding='utf-8')
        result = subprocess.run(
            command, cwd=repository, capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert 'metric changed: fid' in result.stderr


def test_final_ordinary_only_audit_binds_every_metric_record():
    with tempfile.TemporaryDirectory() as directory:
        bound = Path(directory) / 'bound.json'
        bound.write_text('{}', encoding='utf-8')
        record = {
            'path': str(bound.resolve()),
            'sha256': report_sha256_file(bound),
        }
        repository = Path(__file__).resolve().parents[1]
        implementation_paths = {
            'audit_and_rescore':
            repository / 'tools/rescore_final_interx_ordinary.py',
            'semantic_scorer':
            repository / 'tools/score_fair_interx_generations.py',
            'aggregator':
            repository / 'tools/aggregate_fair_interx_metrics.py',
            'evaluator_options':
            repository / 'checkpoints/hhi/Comp_v6_KLD01/opt.txt',
            'evaluator_checkpoint':
            repository / 'checkpoints/hhi/text_mot_match/model/finest.tar',
            'evaluator_wrapper':
            repository / 'models/evaluator_wrapper.py',
            'evaluator_option_loader':
            repository / 'options/get_eval_option.py',
        }
        implementation = {
            key: {
                'path': str(path.resolve()),
                'sha256': report_sha256_file(path),
            }
            for key, path in implementation_paths.items()
        }
        payload = {
            'schema': 'hhc_final_interx_ordinary_only_metric_audit_v1',
            'status': 'pass',
            'metric_protocol': 'official_interx_full_672d_ordinary_only',
            'metrics': [
                'fid', 'diversity_gt', 'diversity', 'top1', 'top2',
                'top3', 'matching', 'multimodality',
            ],
            'forbidden_metrics': [
                'fid_body_controlled', 'diversity_body_controlled',
                'diversity_gt_body_controlled',
            ],
            'rescored_labels': [],
            'implementation': implementation,
            'evaluator_runtime': official_main_only_evaluator_runtime(),
            'per_record_evaluator_provenance': {
                'contract': 'official_interx_full_672d',
                'options': implementation['evaluator_options'],
                'checkpoint': implementation['evaluator_checkpoint'],
                'wrapper_source': implementation['evaluator_wrapper'],
                'option_loader_source':
                implementation['evaluator_option_loader'],
                'scorer_source': implementation['semantic_scorer'],
            },
            'methods': {
                'fixture': {
                    'replications': 2,
                    'summary': dict(record),
                    'records': [
                        {
                            'replication': replication,
                            'metrics': dict(record),
                            'evaluator_provenance_verified': True,
                            'evaluator_runtime_verified': True,
                        }
                        for replication in range(2)
                    ],
                },
            },
        }
        validate_final_ordinary_only_metric_audit(
            payload, expected_labels={'fixture'}, expected_replications=2,
            expected_rescored_labels=[],
        )
        payload['methods']['fixture']['records'][1][
            'evaluator_provenance_verified'
        ] = False
        try:
            validate_final_ordinary_only_metric_audit(
                payload, expected_labels={'fixture'}, expected_replications=2,
                expected_rescored_labels=[],
            )
        except ValueError as error:
            assert 'final_ordinary_only_metric_audit' in str(error)
        else:
            raise AssertionError('unverified per-record evaluator was accepted')
        payload['methods']['fixture']['records'][1][
            'evaluator_provenance_verified'
        ] = True
        payload['methods']['fixture']['records'][1][
            'evaluator_runtime_verified'
        ] = False
        try:
            validate_final_ordinary_only_metric_audit(
                payload, expected_labels={'fixture'}, expected_replications=2,
                expected_rescored_labels=[],
            )
        except ValueError as error:
            assert 'final_ordinary_only_metric_audit' in str(error)
        else:
            raise AssertionError('mixed evaluator runtime was accepted')
        payload['methods']['fixture']['records'][1][
            'evaluator_runtime_verified'
        ] = True
        payload['methods']['fixture']['records'][1]['metrics']['sha256'] = (
            '0' * 64
        )
        try:
            validate_final_ordinary_only_metric_audit(
                payload, expected_labels={'fixture'}, expected_replications=2,
                expected_rescored_labels=[],
            )
        except ValueError as error:
            assert 'final_ordinary_only_metric_audit' in str(error)
        else:
            raise AssertionError('stale ordinary-only metric record was accepted')


def test_active_report_and_postprocessing_bind_formal_repaired_hhc():
    root = Path(__file__).resolve().parents[1]
    active_paths = (
        root / 'tools/build_review_results_report.py',
        root / 'tools/run_final_interx_mesh_physical.py',
        root / 'tools/queue_review_report_after_all_training.sh',
        root / 'tools/queue_post_hhc_gpu1_experiments.sh',
        root / 'tools/queue_hhc_timing_after_post_gpu1.sh',
        root / 'tools/queue_post_shape_review_experiments_gpu1.sh',
        root / 'tools/apply_post_formal_shape_root_tokenizer_wiring.sh',
        root / 'tools/run_paper_optimization_formal.py',
        root / 'tools/audit_optimization_ablation_readiness.py',
        root / 'tools/run_hhc_contact_timing_ablation.py',
        root / 'tools/summarize_hhc_contact_timing_ablation.py',
        root / 'tools/render_fair_interx_qualitative.py',
        root / 'tools/prepare_qualitative_comparison_selection.py',
        root / 'tools/queue_paired_report_finalize.sh',
        root / 'tools/queue_paired_report_finalize_v2.sh',
        root / 'tools/queue_interx_shape_augmentation_gpu1.sh',
        root / 'tools/build_shape_augmentation_results.py',
        root / 'tools/build_hhc_global_masked_loss_promotion_results.py',
        root / 'tools/continue_hhc_global_masked_loss_control_300k.sh',
        root / 'tools/queue_final_report_after_v3.sh',
    )
    combined = '\n'.join(
        path.read_text(encoding='utf-8') for path in active_paths
    )
    active_orchestrators = (
        root / 'tools/queue_repaired_interx_end_to_end.sh',
        root / 'tools/run_formal_hhc_after_mgt_repair.sh',
        root / 'tools/queue_post_hhc_gpu1_experiments.sh',
        root / 'tools/queue_hhc_timing_after_post_gpu1.sh',
        root / 'tools/queue_post_shape_review_experiments_gpu1.sh',
        root / 'tools/apply_post_formal_shape_root_tokenizer_wiring.sh',
        root / 'tools/queue_interx_shape_augmentation_gpu1.sh',
        root / 'tools/queue_review_report_after_all_training.sh',
    )
    orchestrator_source = '\n'.join(
        path.read_text(encoding='utf-8') for path in active_orchestrators
    )
    for cancelled_action_entrypoint in (
        'queue_interx_generalization',
        'prepare_interx_generalization',
        'generalization_action',
        'unseen_action',
    ):
        assert cancelled_action_entrypoint not in orchestrator_source
    assert 'hhc_precleanup_raw12288_current_mgt_common50' not in combined
    assert 'output/hhc_mgt_precleanup_raw12288_selection.json' not in combined
    assert 'hhc_formal_raw12288_repaired_mgt_common50_no_contact' in combined
    assert 'hhc_formal_raw12288_repaired_mgt_common50_contact' in combined
    assert 'output/hhc_formal_raw12288_repaired_mgt_selection.json' in combined
    assert 'tools/run_final_interx_mesh_physical.py' in combined
    assert 'final_mesh_physical_main_table' in combined
    assert 'penetrating_vertex_fraction' in combined
    assert 'paper_eq6_squared_sum_m2_per_frame' in combined
    assert 'deterministic replication-0 descriptive main-table geometry' in combined
    post_hhc_source = (
        root / 'tools/queue_post_hhc_gpu1_experiments.sh'
    ).read_text(encoding='utf-8')
    post_shape_review_source = (
        root / 'tools/queue_post_shape_review_experiments_gpu1.sh'
    ).read_text(encoding='utf-8')
    formal_optimization_source = (
        root / 'tools/run_paper_optimization_formal.py'
    ).read_text(encoding='utf-8')
    assert '--repeat-id 0 --resume' in post_shape_review_source
    assert 'convert_gt.append("--resume")' in formal_optimization_source
    assert 'manifest-only shortcut is insufficient' in formal_optimization_source
    assert 'output/hhc_shapeaug_random_current_mgt_selection.json' not in combined
    assert 'output/hhc_shapeaug_optimized_current_mgt_selection.json' not in combined
    assert 'hhc_mgt_shapeaug_random_common50' not in combined
    assert 'hhc_mgt_shapeaug_optimized_common50' not in combined
    assert (
        'output/hhc_formal_raw12288_repaired_mgt_shapeaug_random_selection.json'
        in combined
    )
    assert (
        'output/hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_selection.json'
        in combined
    )
    assert (
        'hhc_formal_raw12288_repaired_mgt_shapeaug_random_common50_no_contact'
        in combined
    )
    assert (
        'hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_common50_contact'
        in combined
    )
    assert (
        'hhc_formal_raw12288_repaired_mgt_shapeaug_random_training_contract_audit.json'
        in combined
    )
    assert (
        'hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_20k_pre_post_validation_pair_audit.json'
        in combined
    )
    for explicit_paper_argument in (
        '--embed-dim-gpt 512', '--n-head-gpt 8', '--num-layers 9',
        '--num-local-layer 9', '--training-mask-contract paper_kinematic',
        '--training-loss-contract paper_masked_only',
        '--optimizer-contract paper_warmup_cosine',
        '--conditioning-contract paper_text_cfg',
        '--training-token-length-contract paper_50',
        '--training-validation-contract paper_isolated_common',
        '--resume-in-place',
    ):
        assert explicit_paper_argument in combined
    assert 'tools/audit_shape_root_tokenizer_wiring.py' in combined
    assert 'tools/build_hhc_global_masked_loss_promotion_results.py' in combined
    assert 'hhc_global_masked_loss_control_300k_results.json' in combined
    assert 'paper_masked_global_token_mean' in combined
    assert 'deterministic_historical_root_v1' in combined
    shape_timing_queue_source = (
        root / 'tools/queue_hhc_timing_after_post_gpu1.sh'
    ).read_text(encoding='utf-8')
    assert 'bash tools/apply_post_formal_shape_root_tokenizer_wiring.sh' in (
        shape_timing_queue_source
    )
    assert shape_timing_queue_source.index(
        'bash tools/apply_post_formal_shape_root_tokenizer_wiring.sh'
    ) < shape_timing_queue_source.index(
        'tools/audit_shape_root_tokenizer_wiring.py'
    )
    assert shape_timing_queue_source.index(
        'tools/audit_shape_root_tokenizer_wiring.py'
    ) < shape_timing_queue_source.index(
        'bash tools/queue_interx_shape_augmentation_gpu1.sh'
    )
    assert shape_timing_queue_source.index(
        'bash tools/run_hhc_global_masked_loss_control.sh'
    ) < shape_timing_queue_source.index(
        'bash tools/queue_interx_shape_augmentation_gpu1.sh'
    ) < shape_timing_queue_source.index(
        'tools/queue_post_shape_review_experiments_gpu1.sh'
    )
    assert 'tools/run_hhc_contact_timing_ablation.py' in (
        post_shape_review_source
    )
    assert 'sleep 300' not in shape_timing_queue_source
    assert 'until conda run' not in shape_timing_queue_source

    shape_launch = json.loads((
        root / 'output/interx_shape_augmentation_last_stage_contract.json'
    ).read_text(encoding='utf-8'))
    assert shape_launch['status'] == 'pass'
    assert shape_launch['artifacts']['launcher']['sha256'] == report_sha256_file(
        root / 'tools/run_interx_shape_augmentation_nearest_last_stage.sh'
    )
    assert shape_launch['checks']['launcher_has_no_wait_or_sleep_loop'] is True
    assert shape_launch['checks']['nearest_vertex_is_explicit'] is True
    assert shape_launch['checks']['disk_capacity_gate_precedes_full'] is True
    assert shape_launch['checks']['legacy_waiter_entrypoints_fail_closed'] is True
    formal_launch = json.loads((
        root / 'output/hhc_formal_launch_contract_audit.json'
    ).read_text(encoding='utf-8'))
    assert formal_launch['status'] == 'pass'
    assert formal_launch['ordering']['selection_before_validation_pair'] is True
    assert formal_launch['launcher']['sha256'] == report_sha256_file(
        root / 'tools/run_formal_hhc_after_mgt_repair.sh'
    )
    assert formal_launch['state_resolver']['sha256'] == report_sha256_file(
        root / 'tools/resolve_mgt_training_state.py'
    )

    report_queue_source = (
        root / 'tools/queue_review_report_after_all_training.sh'
    ).read_text(encoding='utf-8')
    assert 'prevalidation-000020000.ckpt' in report_queue_source
    assert 'postvalidation-000020000.ckpt' in report_queue_source
    assert '--require-fixed-pair-marker' in report_queue_source
    assert 'latest-000020000.ckpt' not in report_queue_source
    assert "-name 'model-000020000-*.ckpt'" not in report_queue_source
    assert 'tools/rescore_final_interx_ordinary.py' in report_queue_source
    assert 'rescore every final method on the same' in report_queue_source
    assert 'rescore_final_interx_ordinary.py --device cpu' in (
        report_queue_source
    )
    assert "DEFAULT_RESCORE_LABELS = FINAL_LABELS" in (
        root / 'tools/rescore_final_interx_ordinary.py'
    ).read_text(encoding='utf-8')

    report_source = active_paths[0].read_text(encoding='utf-8')
    assert 'root_tokenizer_wiring' in report_source
    assert 'official_interx_full_672d_ordinary_only' in report_source
    assert 'pytorch_official_evaluator_single_device_v1' in report_source
    assert 'final_ordinary_only_metric_audit.json' in report_source
    assert 'net_175000_reconstruction_ordinary_fid.json' in report_source
    assert "'fid_body_controlled' not in reconstruction_fid" in report_source
    vq_manifest = json.loads((
        root / 'checkpoints/vq_precleanup_raw12288/training_manifest.json'
    ).read_text(encoding='utf-8'))
    assert vq_manifest['verified_reconstruction']['fair_fid'] == (
        0.29441722725218256
    )
    assert vq_manifest['verified_reconstruction']['fid_report_path'].endswith(
        'net_175000_reconstruction_ordinary_fid.json'
    )


def test_coshmdm_representation_adapter_audit_is_current():
    payload = json.loads(Path(
        'output/coshmdm_representation_adapter_audit.json'
    ).read_text(encoding='utf-8'))
    validate_coshmdm_representation_adapter(payload)

    invalid = json.loads(json.dumps(payload))
    invalid['raw_to_projection_information_loss'][
        'nonroot_body_rotation6d'
    ]['maximum'] = 1e-6
    try:
        validate_coshmdm_representation_adapter(invalid)
    except ValueError as error:
        assert 'information-loss decomposition' in str(error)
    else:
        raise AssertionError(
            'lossy non-root body rotation adapter was accepted'
        )


def test_mgt_selection_ignores_interrupted_same_name_runs():
    with tempfile.TemporaryDirectory() as directory:
        interrupted = Path(directory) / 'timestamp_a_same_exp'
        completed = Path(directory) / 'timestamp_b_same_exp'
        interrupted.mkdir()
        completed.mkdir()
        (interrupted / 'latest-5000.ckpt').write_bytes(b'interrupted')
        final = completed / 'latest-300000.ckpt'
        final.write_bytes(b'complete')
        selected_run, selected_checkpoint, latest_by_run = (
            discover_completed_run(
                [str(interrupted.resolve()), str(completed.resolve())], 300000
            )
        )
        assert selected_run == str(completed.resolve())
        assert selected_checkpoint == str(final.resolve())
        assert len(latest_by_run) == 2


def test_mgt_selection_uses_exact_checkpoint_fid_not_rounded_filename():
    with tempfile.NamedTemporaryFile(suffix='.ckpt') as handle:
        torch.save({
            'global_step': 10000,
            'callbacks': {
                'fid': {
                    'monitor': 'val/FID',
                    'current_score': torch.tensor(1.1268210411071777),
                },
            },
        }, handle.name)
        score = exact_validation_fid(handle.name, 10000, 1.1268)
        assert score == 1.1268210411071777
        try:
            exact_validation_fid(handle.name, 10000, 1.2)
        except ValueError as error:
            assert 'does not round to filename value' in str(error)
        else:
            raise AssertionError('stale rounded validation FID was accepted')


def test_optimizer_npz_inverse_preserves_two_person_raw_layout():
    generator = torch.Generator().manual_seed(19)
    length = 7
    poses = [
        torch.randn(length, 55, 3, generator=generator) * 0.25
        for _ in range(2)
    ]
    translations = [
        torch.randn(length, 3, generator=generator) * 0.1
        for _ in range(2)
    ]
    payloads = [
        {
            'poses': pose.numpy(), 'trans': translation.numpy(),
            'betas': np.zeros(10, dtype=np.float32),
            'gender': np.asarray('female'),
        }
        for pose, translation in zip(poses, translations)
    ]
    raw = pair_to_raw(*payloads, length)
    expected = [
        matrix_to_rotation_6d(axis_angle_to_matrix(pose.double())).float()
        for pose in poses
    ]
    assert raw.shape == (length, 56, 12)
    assert np.max(np.abs(raw[:, :55, :6] - expected[0].numpy())) < 1e-6
    assert np.max(np.abs(raw[:, :55, 6:12] - expected[1].numpy())) < 1e-6
    assert np.array_equal(raw[:, 55, :3], translations[0].numpy())
    assert np.array_equal(raw[:, 55, 6:9], translations[1].numpy())
    assert np.count_nonzero(raw[:, 55, 3:6]) == 0
    assert np.count_nonzero(raw[:, 55, 9:12]) == 0


def test_fair_h5_converter_records_bind_actual_optimizer_npz():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        requests = root / 'requests.npz'
        name = 'sample'
        caption = 'two people shake hands'
        length = 3
        np.savez_compressed(
            requests,
            names=np.asarray([name]),
            captions=np.asarray([caption]),
            lengths=np.asarray([length], dtype=np.int64),
            betas=np.arange(20, dtype=np.float32).reshape(1, 20) / 100,
            gender=np.asarray([[0, 1]], dtype=np.int64),
        )
        generations = root / 'generations.h5'
        raw = np.zeros((1, 1, 150, 672), dtype=np.float32)
        rotations = raw.reshape(1, 1, 150, 56, 12)[:, :, :, :55]
        rotations[..., 0] = 1.0
        rotations[..., 4] = 1.0
        rotations[..., 6] = 1.0
        rotations[..., 10] = 1.0
        with h5py.File(generations, 'w') as handle:
            handle.create_dataset(
                'names', data=np.asarray([name], dtype=h5py.string_dtype()),
            )
            handle.create_dataset(
                'lengths', data=np.asarray([length], dtype=np.int64),
            )
            handle.create_dataset('motions', data=raw)
            handle.create_dataset(
                'completed_repeats', data=np.asarray([True]),
            )
            handle.attrs['requests_sha256'] = sha256_file(requests)
        output = root / 'optimizer_inputs'
        repository = Path(__file__).resolve().parents[1]
        subprocess.run([
            sys.executable,
            str(repository / 'tools/export_fair_h5_to_optimizer_npz.py'),
            '--requests', str(requests), '--generations', str(generations),
            '--output-dir', str(output), '--repeat-id', '0',
        ], cwd=repository, check=True, capture_output=True, text=True)

        record = json.loads(
            (output / 'generation_records.jsonl').read_text().strip()
        )
        p1 = output / 'sample_r000_pred_p1.npz'
        p2 = output / 'sample_r000_pred_p2.npz'
        hashes = validate_generation_record_inputs(
            record, p1, p2, 'sample_r000', name, 0,
        )
        assert hashes == (record['p1_sha256'], record['p2_sha256'])
        assert record['text'] == caption
        assert record['length'] == length
        manifest = json.loads(
            (output / '_conversion_manifest.json').read_text()
        )
        assert manifest['samples'] == 1
        assert manifest['generation_records_sha256'] == sha256_file(
            output / 'generation_records.jsonl'
        )
        original_hashes = (sha256_file(p1), sha256_file(p2))
        resumed = subprocess.run([
            sys.executable,
            str(repository / 'tools/export_fair_h5_to_optimizer_npz.py'),
            '--requests', str(requests), '--generations', str(generations),
            '--output-dir', str(output), '--repeat-id', '0', '--resume',
        ], cwd=repository, check=False, capture_output=True, text=True)
        assert resumed.returncode == 0, (resumed.stdout, resumed.stderr)
        assert (sha256_file(p1), sha256_file(p2)) == original_hashes

        with np.load(p1, allow_pickle=False) as stored:
            stale_payload = {key: stored[key] for key in stored.files}
        stale_payload['betas'] = stale_payload['betas'].copy()
        stale_payload['betas'][0] += 1.0
        np.savez_compressed(p1, **stale_payload)
        stale = subprocess.run([
            sys.executable,
            str(repository / 'tools/export_fair_h5_to_optimizer_npz.py'),
            '--requests', str(requests), '--generations', str(generations),
            '--output-dir', str(output), '--repeat-id', '0', '--resume',
        ], cwd=repository, check=False, capture_output=True, text=True)
        assert stale.returncode != 0
        assert 'values differ' in stale.stderr


def test_inverse_converter_binds_optimizer_sidecar_identity_and_contract():
    p1 = {'sha256': 'p1-hash'}
    p2 = {'sha256': 'p2-hash'}
    sidecar = {
        'schema': 'hhc_paper_contact_optimization_sample_v1',
        'sample_key': 'sample_r000',
        'sample_name': 'sample',
        'repeat_id': 0,
        'run_contract_sha256': 'contract-hash',
        'output_p1_sha256': p1['sha256'],
        'output_p2_sha256': p2['sha256'],
    }
    validate_optimizer_sidecar(
        sidecar, 'sample_r000', 'sample', 0, 'contract-hash', p1, p2,
    )
    sidecar['sample_name'] = 'another-sample'
    try:
        validate_optimizer_sidecar(
            sidecar, 'sample_r000', 'sample', 0, 'contract-hash', p1, p2,
        )
    except ValueError as error:
        assert 'optimizer sidecar contract mismatch' in str(error)
        assert 'sample_name' in str(error)
    else:
        raise AssertionError('cross-sample optimizer sidecar was accepted')


def test_formal_optimization_report_branch_is_strict_before_cache_exists():
    """Exercise the future cache-ready branch without fabricating results."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output_artifacts = {}
        for condition in ('hhc', 'gt'):
            payloads = {
                'optimization_completion_manifest': json.dumps({
                    'schema': 'hhc_paper_contact_optimization_v1',
                    'complete': True,
                    'completed_samples': 1696,
                    'requested_samples': 1696,
                    'runtime': {
                        'samples': 1696,
                        'optimizer_wall_seconds_mean_per_sample': 1.0,
                        'optimizer_wall_seconds_p95_per_sample': 1.5,
                    },
                }).encode(),
                'optimized_h5': b'synthetic-h5',
                'optimized_h5_manifest': json.dumps({
                    'status': 'complete', 'samples': 1696,
                }).encode(),
                'official_main_metrics': json.dumps({
                    'schema': 'fair_interx_main_only_metrics_v1',
                    'status': 'complete', 'samples': 1696,
                    'multimodality': None,
                    'generation_request_hash_verified': True,
                    'evaluator_provenance': (
                        official_main_only_evaluator_provenance()
                    ),
                    'evaluator_runtime': (
                        official_main_only_evaluator_runtime()
                    ),
                }).encode(),
                'annotation_physical': json.dumps({
                    'samples': 1696,
                }).encode(),
                'paired_mesh_physical': json.dumps({
                    'schema': 'hhc_paired_physical_comparison_v1',
                    'sample_count': 1696,
                }).encode(),
            }
            output_artifacts[condition] = {}
            for name, content in payloads.items():
                path = root / f'{condition}_{name}'
                path.write_bytes(content)
                output_artifacts[condition][name] = {
                    'path': str(path),
                    'sha256': report_sha256_file(path),
                }
        manifest = {
            'schema': 'hhc_paper_optimization_formal_v1',
            'status': 'complete', 'samples': 1696,
            'semantic_evaluator_device': 'cpu',
            'hhc_source_condition': {
                'contact_guidance': False,
                'contact_timing_source': 'none',
                'sampler': 'sample_intra_cas',
                'sampling_steps': 50,
                'temporal_mask': (
                    'derived_from_initial_geometry_not_test_annotation'
                ),
                'verified_hdf5_attributes': {
                    'contact_guidance': False,
                    'contact_timing_source': 'none',
                    'contact_timing_file': '',
                    'contact_timing_file_sha256': '',
                    'contact_map': '',
                    'contact_map_sha256': '',
                    'sampler': 'sample_intra_cas',
                    'sampling_steps': 50,
                    'requests_sha256': 'synthetic-request-hash',
                },
            },
            'conditions': {'hhc': {}, 'gt': {}},
            'output_artifacts': output_artifacts,
        }
        loaded = validate_formal_optimization(manifest)
        assert set(loaded) == {'hhc', 'gt'}

        gt_metrics_path = Path(output_artifacts['gt'][
            'official_main_metrics']['path'])
        gt_original = gt_metrics_path.read_bytes()
        invalid_runtime = json.loads(gt_original)
        invalid_runtime['evaluator_runtime'] = (
            official_main_only_evaluator_runtime('cuda:0')
        )
        gt_metrics_path.write_text(json.dumps(invalid_runtime))
        output_artifacts['gt']['official_main_metrics'][
            'sha256'] = report_sha256_file(gt_metrics_path)
        try:
            validate_formal_optimization(manifest)
        except ValueError as error:
            assert 'one CPU runtime' in str(error)
        else:
            raise AssertionError('mixed main-only evaluator runtimes accepted')
        gt_metrics_path.write_bytes(gt_original)
        output_artifacts['gt']['official_main_metrics'][
            'sha256'] = report_sha256_file(gt_metrics_path)

        metrics_path = Path(output_artifacts['hhc'][
            'official_main_metrics']['path'])
        invalid = json.loads(metrics_path.read_text())
        invalid['multimodality'] = 0.0
        metrics_path.write_text(json.dumps(invalid))
        output_artifacts['hhc']['official_main_metrics'][
            'sha256'] = report_sha256_file(metrics_path)
        try:
            validate_formal_optimization(manifest)
        except ValueError as error:
            assert 'main-only metrics' in str(error)
        else:
            raise AssertionError('formal optimizer MM result was accepted')


def test_formal_optimizer_rejects_oracle_timing_source_provenance():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        requests = root / 'requests.npz'
        requests.write_bytes(b'frozen-requests')
        generations = root / 'generations.h5'
        request_hash = sha256_file(requests)
        expected_attrs = {
            'contact_guidance': False,
            'contact_timing_source': 'none',
            'contact_timing_file': '',
            'contact_timing_file_sha256': '',
            'contact_map': '',
            'contact_map_sha256': '',
            'sampler': 'sample_intra_cas',
            'sampling_steps': 50,
            'requests_sha256': request_hash,
        }
        with h5py.File(generations, 'w') as handle:
            for key, value in expected_attrs.items():
                handle.attrs[key] = value
        input_dir = root / 'optimizer_inputs'
        input_dir.mkdir()
        records = input_dir / 'generation_records.jsonl'
        records.write_text('{}\n')
        conversion = {
            'schema': 'hhc_fair_h5_to_optimizer_npz_v1',
            'status': 'complete',
            'samples': 1696,
            'repeat_id': 0,
            'requests_sha256': request_hash,
            'generations_sha256': sha256_file(generations),
            'generation_records': str(records.resolve()),
            'generation_records_sha256': sha256_file(records),
            'generation_hdf5_attributes': expected_attrs,
            'resume_contract': RESUME_CONTRACT,
            'implementation': {
                name: {
                    'path': str(path.resolve()),
                    'bytes': path.stat().st_size,
                    'sha256': sha256_file(path),
                }
                for name, path in {
                    'converter': Path(
                        'tools/export_fair_h5_to_optimizer_npz.py'
                    ),
                    'rotation_conversions': Path(
                        'utils/rotation_conversions.py'
                    ),
                }.items()
            },
        }
        (input_dir / '_conversion_manifest.json').write_text(
            json.dumps(conversion)
        )
        validated = validate_hhc_source_contract(
            requests, generations, input_dir,
        )
        assert validated == expected_attrs

        with h5py.File(generations, 'r+') as handle:
            handle.attrs['contact_guidance'] = True
            handle.attrs['contact_timing_source'] = 'request_oracle'
        try:
            validate_hhc_source_contract(requests, generations, input_dir)
        except ValueError as error:
            assert 'no-timing common50' in str(error)
        else:
            raise AssertionError('oracle-timing optimization source was accepted')


def test_optimizer_inputs_are_byte_bound_to_generation_record():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        p1 = root / 'sample_r000_pred_p1.npz'
        p2 = root / 'sample_r000_pred_p2.npz'
        p1.write_bytes(b'person-one')
        p2.write_bytes(b'person-two')
        record = {
            'sample_key': 'sample_r000',
            'sample_name': 'sample',
            'repeat_id': 0,
            'p1': str(p1.resolve()),
            'p1_sha256': sha256_file(p1),
            'p2': str(p2.resolve()),
            'p2_sha256': sha256_file(p2),
        }
        hashes = validate_generation_record_inputs(
            record, p1, p2, 'sample_r000', 'sample', 0,
        )
        assert hashes == (record['p1_sha256'], record['p2_sha256'])

        p1.write_bytes(b'replaced-person-one')
        try:
            validate_generation_record_inputs(
                record, p1, p2, 'sample_r000', 'sample', 0,
            )
        except ValueError as error:
            assert 'does not bind optimizer inputs' in str(error)
            assert 'p1_sha256' in str(error)
        else:
            raise AssertionError('replaced optimizer NPZ was accepted')


def test_optimizer_ablation_report_binds_outputs_statistics_and_runtime():
    from tools.build_review_results_report import (
        EXPECTED_OPTIMIZATION_ABLATIONS,
    )
    metric_names = (
        'penetrating_vertex_fraction', 'collision_frame_rate',
        'penetration_depth_sum_m_per_frame',
        'paper_eq6_squared_sum_m2_per_frame',
        'mean_depth_m_on_penetrating_vertices', 'max_depth_m',
        'mean_foot_sliding_m_per_frame',
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def write_artifact(name, payload):
            path = root / name
            path.write_text(json.dumps(payload), encoding='utf-8')
            return {'path': str(path), 'sha256': report_sha256_file(path)}

        before = write_artifact('before.json', {
            'schema': 'hhc_volumetric_smpl_physical_metrics_v1',
            'sample_count': 25,
        })
        repository_root = Path(__file__).resolve().parents[1]
        runner = {'path': str(
                      repository_root / 'tools/run_paper_optimization_ablation.py'
                  ),
                  'sha256': report_sha256_file(
                      repository_root / 'tools/run_paper_optimization_ablation.py'
                  )}
        outputs = {}
        comparisons = {
            metric_name: {
                'direction': 'lower_is_better',
                'before_mean': 1.0, 'after_mean': 0.9,
                'paired_delta_after_minus_before_mean': -0.1,
                'paired_delta_median': -0.1,
                'paired_delta_bootstrap_95ci': [-0.2, -0.01],
                'wilcoxon_two_sided_p': 0.03,
            }
            for metric_name in metric_names
        }
        for index, ablation in enumerate(EXPECTED_OPTIMIZATION_ABLATIONS):
            name = ablation['label']
            outputs[name] = {
                'optimizer_completion_artifact': write_artifact(
                    f'{index}_completion.json', {
                        'schema': 'hhc_paper_contact_optimization_v1',
                        'complete': True, 'completed_samples': 25,
                        'requested_samples': 25,
                        'runtime': {
                            'samples': 25,
                            'optimizer_wall_seconds_mean_per_sample': 1.0,
                        },
                    }),
                'after_metrics_artifact': write_artifact(
                    f'{index}_after.json', {
                        'schema': 'hhc_volumetric_smpl_physical_metrics_v1',
                        'sample_count': 25,
                    }),
                'paired_comparison_artifact': write_artifact(
                    f'{index}_paired.json', {
                        'schema': 'hhc_paired_physical_comparison_v1',
                        'sample_count': 25, 'bootstrap_samples': 10000,
                        'comparisons': comparisons,
                    }),
            }
        payload = {
            'schema': 'hhc_paper_optimizer_ablation_pilot_v1',
            'status': 'complete', 'pilot_limit': 25,
            'ablations': list(EXPECTED_OPTIMIZATION_ABLATIONS),
            'outputs': outputs,
            'before_metrics_artifact': before,
            'artifacts': {'ablation_runner': runner},
        }
        summaries = validate_optimization_ablation(payload)
        assert len(summaries) == 13
        paired_path = Path(outputs['paper_full'][
            'paired_comparison_artifact']['path'])
        invalid = json.loads(paired_path.read_text(encoding='utf-8'))
        invalid['comparisons']['max_depth_m'][
            'wilcoxon_two_sided_p'] = float('nan')
        paired_path.write_text(json.dumps(invalid), encoding='utf-8')
        outputs['paper_full']['paired_comparison_artifact'][
            'sha256'] = report_sha256_file(paired_path)
        try:
            validate_optimization_ablation(payload)
        except ValueError as error:
            assert 'paired statistics' in str(error)
        else:
            raise AssertionError('non-finite optimizer statistics were accepted')


def test_shape_augmentation_report_requires_bound_matched_arms():
    with tempfile.TemporaryDirectory() as directory:
        bound = Path(directory) / 'bound.json'
        bound.write_text('{"status":"complete"}', encoding='utf-8')
        implementation = {}
        for index in range(5):
            path = Path(directory) / f'implementation_{index}.py'
            path.write_text(f'# fixture {index}\n', encoding='utf-8')
            implementation[str(path)] = report_sha256_file(path)
        checkpoint_root = Path(torch.hub.get_dir()) / 'checkpoints'
        volume_hashes = {
            'male': (
                '96b22f6409e67dc950dafdd83bb8dc4779381ecc2b2c752e1c6fa3a205b2e18c'
            ),
            'female': (
                '2b582f009c2daeec50c90b5f39689a248bac96a4cc9983548ad4be677b72ccdb'
            ),
        }
        equivalence = Path(directory) / 'backend_equivalence.json'
        equivalence_payload = {
            'schema': 'hhc_shared_volumetric_backend_equivalence_v1',
            'status': 'pass',
            'contract': {
                'device': 'cpu', 'frames': 3, 'epochs': 1,
                'query_chunk': 512, 'seed': 1234, 'threads': 1,
                'deterministic_algorithms': True,
            },
            'checkpoint_hashes_valid': True,
            'finite_loss_histories': {
                'fresh': True, 'shared_first': True, 'shared_second': True,
            },
            'diagnostic_flags': {
                'fresh_backend_reused': False,
                'shared_first_backend_reused': True,
                'shared_second_backend_reused': True,
            },
            'fresh_vs_shared': {
                'bitwise_equal': True, 'finite': True,
                'max_abs_difference': 0.0,
            },
            'shared_first_vs_second': {
                'bitwise_equal': True, 'finite': True,
                'max_abs_difference': 0.0,
            },
            'volumetric_smpl_checkpoints': {
                gender: {
                    'path': str(
                        checkpoint_root
                        / f'VolumetricSMPL_smplx_{gender}.ckpt'
                    ),
                    'sha256': digest,
                }
                for gender, digest in volume_hashes.items()
            },
            'implementation': implementation,
        }
        equivalence.write_text(
            json.dumps(equivalence_payload), encoding='utf-8'
        )
        wiring = Path(directory) / 'root_tokenizer_wiring.json'
        wiring_payload = {
            'schema': 'hhc_shape_root_tokenizer_wiring_audit_v1',
            'status': 'pass',
            'contract': 'deterministic_historical_root_v1',
            'checks': {'fixture_contract': True},
            'missing_checks': [],
            'artifacts': {
                'fixture': {
                    'path': str(bound),
                    'sha256': report_sha256_file(bound),
                },
            },
        }
        wiring.write_text(json.dumps(wiring_payload), encoding='utf-8')
        payload = {
            'schema': 'hhc_interx_shape_augmentation_results_v1',
            'status': 'complete',
            'official_split_unchanged': True,
            'pilot_samples': 25,
            'augmented_samples_per_arm': 6238,
            'input_preflight': {
                'selected_samples': 6238,
                'positive_timing_samples': 5868,
                'semantic_contact_active_samples': 5666,
                'physical_timing_only_samples': 202,
                'regions_inactive_because_timing_is_zero': 137,
                'structured_contact_rows': 7434,
                'boundary_whitespace_caption_matches': 1,
            },
            'results': {
                'random': {
                    'backend_lifetime': (
                        'not_instantiated_for_random_shape_only_control'
                    ),
                },
                'optimized': {
                    'backend_lifetime': (
                        'one frozen male/female backend per serial process; '
                        'fresh loss state and optimizer per sample'
                    ),
                },
            },
            'artifacts': {
                name: {
                    'path': str(bound),
                    'sha256': report_sha256_file(bound),
                }
                for name in (
                    'augmentation_input_preflight', 'interx_protocol_audit',
                )
            },
        }
        payload['artifacts']['shared_backend_equivalence'] = {
            'path': str(equivalence),
            'sha256': report_sha256_file(equivalence),
        }
        payload['artifacts']['root_tokenizer_wiring'] = {
            'path': str(wiring),
            'sha256': report_sha256_file(wiring),
        }
        validate_shape_augmentation_results(payload)
        missing_preflight = dict(payload)
        missing_preflight['artifacts'] = {'bound': payload['artifacts'][
            'augmentation_input_preflight'
        ]}
        try:
            validate_shape_augmentation_results(missing_preflight)
        except ValueError as error:
            assert 'input/protocol provenance' in str(error)
        else:
            raise AssertionError('missing augmentation preflight was accepted')
        invalid_lifetime = json.loads(json.dumps(payload))
        invalid_lifetime['results']['optimized']['backend_lifetime'] = (
            'reloaded_per_sample'
        )
        try:
            validate_shape_augmentation_results(invalid_lifetime)
        except ValueError as error:
            assert 'backend lifecycle' in str(error)
        else:
            raise AssertionError('wrong augmentation backend lifetime accepted')
        wiring_payload['status'] = 'pending'
        wiring.write_text(json.dumps(wiring_payload), encoding='utf-8')
        incomplete_wiring = json.loads(json.dumps(payload))
        incomplete_wiring['artifacts']['root_tokenizer_wiring']['sha256'] = (
            report_sha256_file(wiring)
        )
        try:
            validate_shape_augmentation_results(incomplete_wiring)
        except ValueError as error:
            assert 'root-tokenizer entrypoint wiring' in str(error)
        else:
            raise AssertionError('pending root-tokenizer wiring was accepted')
        wiring_payload['status'] = 'pass'
        wiring.write_text(json.dumps(wiring_payload), encoding='utf-8')
        payload['artifacts']['root_tokenizer_wiring']['sha256'] = (
            report_sha256_file(wiring)
        )
        equivalence_payload['fresh_vs_shared']['bitwise_equal'] = False
        equivalence.write_text(
            json.dumps(equivalence_payload), encoding='utf-8'
        )
        invalid_equivalence = json.loads(json.dumps(payload))
        invalid_equivalence['artifacts']['shared_backend_equivalence'][
            'sha256'
        ] = report_sha256_file(equivalence)
        try:
            validate_shape_augmentation_results(invalid_equivalence)
        except ValueError as error:
            assert 'reuse equivalence' in str(error)
        else:
            raise AssertionError('failed backend equivalence was accepted')
        equivalence_payload['fresh_vs_shared']['bitwise_equal'] = True
        equivalence.write_text(
            json.dumps(equivalence_payload), encoding='utf-8'
        )
        payload['artifacts']['shared_backend_equivalence'][
            'sha256'
        ] = report_sha256_file(equivalence)
        payload['augmented_samples_per_arm'] = 6237
        try:
            validate_shape_augmentation_results(payload)
        except ValueError as error:
            assert 'matched random/optimized' in str(error)
        else:
            raise AssertionError('wrong augmentation sample count was accepted')


def test_shape_augmentation_volume_contract_binds_real_weights():
    protocol = json.loads(Path('output/interx_protocol_audit.json').read_text())
    checkpoint_root = Path(torch.hub.get_dir()) / 'checkpoints'
    records = {
        gender: {
            'path': str(
                checkpoint_root / f'VolumetricSMPL_smplx_{gender}.ckpt'
            ),
            'sha256': protocol['asset_protocol'][
                'volumetric_smpl_checkpoint_sha256'
            ][gender],
        }
        for gender in ('male', 'female')
    }
    optimized = {'optimizer': {'volumetric_smpl_checkpoints': records}}
    assert volumetric_checkpoint_contract_valid(
        'optimized', optimized, protocol
    )
    assert volumetric_checkpoint_contract_valid(
        'random_shape_only',
        {'optimizer': {'volumetric_smpl_checkpoints': {}}}, protocol,
    )
    invalid = json.loads(json.dumps(optimized))
    invalid['optimizer']['volumetric_smpl_checkpoints']['male']['sha256'] = (
        '0' * 64
    )
    assert not volumetric_checkpoint_contract_valid(
        'optimized', invalid, protocol
    )


def test_shape_augmentation_backend_equivalence_audit_is_current():
    # VolumetricSMPL equivalence is retained as a historical diagnostic only.
    # The current augmentation and paper-optimization contract reproduces the
    # original MMM nearest-target-vertex signed distance backend instead.
    audit = json.loads(Path(
        'output/current_interx_optimization_backend_contract.json'
    ).read_text())
    assert audit['status'] == 'pass'
    assert all(audit['checks'].values())
    assert (
        audit['contract']['optimization_signed_distance_backend']
        == 'nearest_vertex'
    )
    launcher = Path(
        'tools/run_interx_shape_augmentation_nearest_last_stage.sh'
    )
    assert audit['artifacts']['shape_launcher']['sha256'] == (
        report_sha256_file(launcher)
    )


def test_frozen_request_verifier_resolves_only_exact_content_snapshot():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        live = root / 'dataset_loader.py'
        snapshot_root = root / 'snapshots'
        snapshot_root.mkdir()
        frozen_bytes = b'original evaluator loader\n'
        live.write_bytes(frozen_bytes)
        frozen_hash = hashlib.sha256(frozen_bytes).hexdigest()
        record = {
            'path': str(live),
            'bytes': len(frozen_bytes),
            'sha256': frozen_hash,
        }
        core = {
            'protocol': 'interx_official_v1',
            'request_contract_version': 2,
            'replications': 20,
            'samples_per_replication': 1696,
            'multimodality': {'prompts': 100, 'repeats': 30, 'pairs': 10},
            'records': [{'replication': i} for i in range(20)],
            'status': 'pass',
        }
        frozen = {
            **core,
            'frozen_input_provenance': {'dataset_loader': record},
        }
        live.write_bytes(b'later train-only extensions\n')
        current_record = {
            'path': str(live),
            'bytes': live.stat().st_size,
            'sha256': hashlib.sha256(live.read_bytes()).hexdigest(),
        }
        current = {
            **core,
            'frozen_input_provenance': {
                'dataset_loader': current_record,
            },
        }
        snapshot = snapshot_root / f'{frozen_hash}__{live.name}'
        snapshot.write_bytes(frozen_bytes)
        report = verify_frozen_manifest(frozen, current, snapshot_root)
        assert report['status'] == 'pass'
        assert report['canonical_manifest_unchanged'] is True
        resolution = report['provenance_resolution']['dataset_loader']
        assert resolution['resolution'] == 'content_addressed_snapshot'
        assert resolution['live_matches_recorded'] is False

        snapshot.write_bytes(b'corrupt\n')
        try:
            verify_frozen_manifest(frozen, current, snapshot_root)
        except ValueError as error:
            assert 'frozen provenance artifact is absent or changed' in str(error)
        else:
            raise AssertionError('corrupt frozen request snapshot was accepted')


def test_frozen_interx_request_manifest_resolves_every_provenance_record():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / 'output/fair_interx_requests/manifest.json').read_text(
            encoding='utf-8'
        )
    )
    current = json.loads(json.dumps(manifest))
    for label, record in current['frozen_input_provenance'].items():
        path = Path(record['path'])
        assert path.is_file(), f'{label}: live provenance path is missing'
        record['bytes'] = path.stat().st_size
        record['sha256'] = report_sha256_file(path)

    report = verify_frozen_manifest(
        manifest, current, root / 'tools/provenance_snapshots'
    )
    assert report['status'] == 'pass'
    assert report['request_npz_records_match'] is True
    assert set(report['provenance_resolution']) == set(
        manifest['frozen_input_provenance']
    )
    assert (
        report['provenance_resolution']['protocol']['resolution']
        == 'content_addressed_snapshot'
    )
    assert (
        report['provenance_resolution']['protocol']['recorded']['sha256']
        == '1db8e9389cdc70c0a875383b1be64dba32ca9fdff92115e04916992a0c5ad11f'
    )


def test_frozen_request_creation_runtime_is_bound_to_successful_cpu_run():
    root = Path(__file__).resolve().parents[1]
    audit_path = root / 'output/fair_interx_requests/provenance_recovery.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['status'] == 'pass'
    assert audit['request_npz_files_modified'] == 0
    assert audit['request_npz_files_verified'] == 20
    record = audit['frozen_creation_records_snapshot']
    snapshot = Path(record['path'])
    assert snapshot.is_file()
    assert snapshot.stat().st_size == record['bytes'] == 16063
    assert report_sha256_file(snapshot) == record['sha256'] == (
        'e2ae556fb8774e2aefb9ddd2166bc53e27d1922ea35039c81c4ce1da10f450d1'
    )
    creation = audit['request_creation_runtime_recovery']
    assert creation['status'] == 'completed'
    assert creation['exit_code'] == 0
    assert creation['runtime'] == {
        'contract': 'pytorch_official_evaluator_single_device_v1',
        'requested_device': 'cpu',
        'device_type': 'cpu',
        'device_index': None,
        'device_name': 'cpu',
    }
    assert creation['replications_reported_complete'] == 20
    assert creation['samples_per_replication_reported'] == 1696
    assert '--device cpu --replications 20 --seed 1234' in creation['command']
    implementation = audit['implementation']
    assert report_sha256_file(implementation['path']) == implementation['sha256']
    assert validate_creation_provenance(audit_path) == creation['runtime']

    with tempfile.TemporaryDirectory() as directory:
        invalid = json.loads(json.dumps(audit))
        invalid['request_creation_runtime_recovery']['runtime'].update({
            'requested_device': 'cuda:0',
            'device_type': 'cuda',
            'device_index': 0,
            'device_name': 'NVIDIA GPU',
        })
        invalid_path = Path(directory) / 'invalid_creation_provenance.json'
        invalid_path.write_text(json.dumps(invalid), encoding='utf-8')
        try:
            validate_creation_provenance(invalid_path)
        except ValueError as error:
            assert 'CPU/20-replication contract' in str(error)
        else:
            raise AssertionError('CUDA request creation provenance was accepted')


def test_fair_manifest_freezer_requires_exact_recorded_source_bytes():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / 'driver.py'
        source.write_text('print("generation version")\n', encoding='utf-8')
        source_hash = report_sha256_file(source)
        manifest = root / 'run_manifest.json'
        manifest.write_text(json.dumps({
            'provenance': {'implementation': {'driver': {
                'path': str(source), 'sha256': source_hash,
                'bytes': source.stat().st_size,
            }}},
        }), encoding='utf-8')
        report = freeze_manifest(manifest, root / 'snapshots')
        assert report['implementation_files'] == 1
        frozen = json.loads(manifest.read_text(encoding='utf-8'))
        record = frozen['provenance']['implementation']['driver']
        assert frozen['provenance']['implementation_snapshot_freeze_mode'] \
            == 'posthoc_exact_recorded_hash_only'
        assert record['content_addressed_snapshot'] is True
        assert Path(record['path']).name.startswith(source_hash + '__')
        assert report_sha256_file(record['path']) == source_hash
        first_snapshot_path = record['path']
        freeze_manifest(manifest, root / 'snapshots')
        frozen_again = json.loads(manifest.read_text(encoding='utf-8'))
        assert frozen_again['provenance']['implementation']['driver'][
            'path'
        ] == first_snapshot_path

        stale_manifest = root / 'stale.json'
        stale_manifest.write_text(json.dumps({
            'provenance': {'implementation': {'driver': {
                'path': str(source), 'sha256': '0' * 64,
                'bytes': source.stat().st_size,
            }}},
        }), encoding='utf-8')
        try:
            freeze_manifest(stale_manifest, root / 'snapshots')
        except ValueError as error:
            assert 'differs from the generation hash' in str(error)
        else:
            raise AssertionError('freezer accepted source bytes from another run')


def test_llm_robustness_report_does_not_claim_region_oracle():
    with tempfile.TemporaryDirectory() as directory:
        bound = Path(directory) / 'bound.json'
        bound.write_text('{"status":"complete"}', encoding='utf-8')
        strata = {
            'timing_negative': 32,
            'laterality_positive': 32,
            'multiple_contact_cues_positive': 32,
            'role_reference_positive': 32,
        }

        def source(include_agreement, include_tokens):
            result = {
                'contact_presence_vs_manual_timing': {
                    'samples': 128, 'precision': 0.8, 'recall': 0.7,
                    'f1': 0.746, 'accuracy': 0.75,
                },
                'by_stratum': {
                    name: {'samples': 32} for name in strata
                },
                'resource_usage': {
                    'wall_seconds': {
                        'samples': 128, 'total': 128.0,
                        'mean_per_sample': 1.0,
                        'median_per_sample': 0.9, 'p95_per_sample': 1.2,
                    },
                    'tokens': ({
                        'samples': 128, 'total': 12800.0,
                        'mean_per_sample': 100.0,
                        'median_per_sample': 95.0, 'p95_per_sample': 130.0,
                    } if include_tokens else None),
                    'wall_untimed_attempts': 0,
                },
            }
            if include_agreement:
                result['part_pair_agreement_vs_terra_reference'] = {
                    role: {
                        'samples': 128, 'precision': 0.6, 'recall': 0.5,
                        'f1': 0.545, 'sample_exact_match_rate': 0.4,
                    }
                    for role in ('ordered', 'unordered')
                }
            return result

        payload = {
            'schema': 'hhc_interx_llm_robustness_evaluation_v1',
            'status': 'complete', 'sample_count': 128,
            'stratum_counts': strata,
            'reference_interpretation': (
                'GPT-5.6 Terra is not human region ground truth'
            ),
            'oracle_scope': (
                'manual timing only; Inter-X has no human 75-region labels'
            ),
            'results': {
                'luna': source(True, True),
                'terra_reference': source(False, True),
                'rule': source(True, False),
            },
            'artifacts': {'bound': {
                'path': str(bound),
                'sha256': report_sha256_file(bound),
            }},
        }
        validate_llm_robustness_results(payload)
        payload['oracle_scope'] = 'human 75-region oracle labels'
        try:
            validate_llm_robustness_results(payload)
        except ValueError as error:
            assert 'scope is overstated' in str(error)
        else:
            raise AssertionError('fake human region oracle was accepted')


def test_final_queue_direct_entrypoints_resolve_repo_imports():
    repository = Path(__file__).resolve().parents[1]
    entrypoints = {
        'tools/audit_fair_interx_request_replay.py': (
            'Compare a fresh official-loader request replay'
        ),
        'tools/audit_mgt_validation_selection.py': (
            'Freeze the exact stochastic validation inputs'
        ),
        'tools/audit_hhc_sampling_runtime.py': (
            "Runtime-audit HHC's cascaded MaskGIT step"
        ),
        'tools/audit_hhc_training_mask_runtime.py': (
            'Runtime-audit the paper HHC training mask'
        ),
        'tools/audit_hhc_model_structure_runtime.py': (
            'Runtime-audit the paper HHC MGT topology'
        ),
        'tools/render_fair_interx_qualitative.py': (
            'Render deterministic same-request qualitative grids'
        ),
        'tools/rescore_final_interx_ordinary.py': (
            'Rescore reused baselines and freeze an ordinary-only final metric audit'
        ),
        'tools/run_final_interx_mesh_physical.py': (
            'Run and audit one common rep-0 VolumetricSMPL protocol'
        ),
        'tools/run_paper_optimization_formal.py': (
            'Run the full paired HHC/GT paper-optimization evaluation'
        ),
        'tools/resolve_lightning_checkpoint_state.py': (
            'Resolve whether one full Lightning checkpoint needs a strict resume'
        ),
        'tools/audit_coshmdm_resume_provenance.py': (
            'Audit the corrected fresh CoShMDM Inter-X training'
        ),
        'tools/audit_coshmdm_generation_contract.py': (
            "Audit CoShMDM's frozen Inter-X generation"
        ),
        'tools/audit_timotion_generation_contract.py': (
            "Audit TiMotion's frozen Inter-X generation"
        ),
        'tools/run_hhc_validation_sampler_calibration.py': (
            'Diagnose HHC sampler settings on the frozen 544-record validation set'
        ),
        'tools/prepare_hhc_validation_bridge_requests.py': (
            "Freeze the exact sample order used by HHC's in-training validation"
        ),
    }
    for relative_path, expected_help in entrypoints.items():
        result = subprocess.run([
            sys.executable, str(repository / relative_path), '--help',
        ], cwd=repository, capture_output=True, text=True)
        assert result.returncode == 0, (
            relative_path, result.stdout, result.stderr
        )
        assert expected_help in result.stdout


def test_hhc_sampler_calibration_is_validation_only_and_common50():
    source = Path(
        'tools/run_hhc_validation_sampler_calibration.py'
    ).read_text(encoding='utf-8')
    queue = Path('tools/queue_post_hhc_gpu1_experiments.sh').read_text(
        encoding='utf-8'
    )
    post_formal = Path('tools/queue_post_formal_after_v3.sh').read_text(
        encoding='utf-8'
    )
    timing_queue = Path('tools/queue_hhc_timing_after_post_gpu1.sh').read_text(
        encoding='utf-8'
    )
    review_queue = Path(
        'tools/queue_post_shape_review_experiments_gpu1.sh'
    ).read_text(encoding='utf-8')
    assert 'output/hhc_mgt_validation_bridge_requests/manifest.json' in source
    assert 'hhc_mgt_validation_bridge_request_v1' in source
    assert 'ordered_sampling_records_match_training_selection' in source
    assert 'exact_hhc_training_validation_loader_batch_and_within_batch_order' in source
    assert 'request_manifest.get("samples") != 544' in source
    assert 'request_manifest.get("seed") != 500123' in source
    assert '"test_split_used": False' in source
    assert '"sampling_steps": 50' in source
    assert '"--sampling-steps", "50"' in source
    assert 'selection["selected_validation_fid"]' in source
    assert '"failed_formal_row_bridge"' in source
    assert 'BRIDGE_MAX_ABSOLUTE_DIFFERENCE = 1e-4' in source
    assert 'fair_interx_requests/replication_' not in source
    assert 'dataset/interx/splits/test' not in source
    assert 'tools/run_hhc_validation_sampler_calibration.py' in queue
    assert 'tools/prepare_hhc_validation_bridge_requests.py' in queue
    assert queue.index(
        'tools/prepare_hhc_validation_bridge_requests.py'
    ) < queue.index('tools/run_hhc_validation_sampler_calibration.py')
    assert 'tools/export_fair_h5_to_optimizer_npz.py' not in queue
    assert 'tools/export_fair_h5_to_optimizer_npz.py' in review_queue
    assert post_formal.index(
        'tools/queue_post_hhc_gpu1_experiments.sh none'
    ) < post_formal.index('tools/queue_hhc_timing_after_post_gpu1.sh none')
    assert timing_queue.index(
        'tools/run_hhc_global_masked_loss_control.sh'
    ) < timing_queue.index(
        'tools/queue_post_shape_review_experiments_gpu1.sh'
    )


def test_hhc_sampler_calibration_report_gate_binds_validation_artifacts():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)

        def record(path):
            path = Path(path).resolve()
            return {
                'path': str(path), 'bytes': path.stat().st_size,
                'sha256': report_sha256_file(path),
            }

        bound = directory / 'bound.bin'
        bound.write_bytes(b'calibration-bound-artifact')
        requests = directory / 'validation.npz'
        requests.write_bytes(b'frozen-validation-request')
        formal_records = directory / 'formal_validation_records.json'
        formal_records.write_text('{"status":"pass"}', encoding='utf-8')
        request_manifest = directory / 'request_manifest.json'
        request_manifest.write_text(json.dumps({
            'schema': 'hhc_mgt_validation_bridge_request_v1',
            'status': 'complete', 'samples': 544, 'seed': 500123,
            'test_split_used': False,
            'sampling_order_contract': (
                'exact_hhc_training_validation_loader_batch_and_within_batch_order'
            ),
            'ordered_sampling_records_match_training_selection': True,
            'same_record_set_as_common_validation_request': True,
            'source_common_request': record(requests),
            'source_common_request_manifest': record(bound),
            'request': record(requests),
            'formal_hhc_validation_selection': record(formal_records),
            'implementation': record(bound),
        }), encoding='utf-8')

        evaluator_provenance = {
            'contract': 'official_interx_full_672d',
            'options': scorer_artifact_record(EVALUATOR_OPTIONS),
            'checkpoint': scorer_artifact_record(EVALUATOR_CHECKPOINT),
            'wrapper_source': scorer_artifact_record(EVALUATOR_WRAPPER_SOURCE),
            'option_loader_source': scorer_artifact_record(
                EVALUATOR_OPTION_SOURCE
            ),
            'scorer_source': scorer_artifact_record(
                Path('tools/score_fair_interx_main.py')
            ),
        }
        evaluator_runtime = {
            'contract': 'pytorch_official_evaluator_single_device_v1',
            'requested_device': 'cpu', 'device_type': 'cpu',
            'device_index': None, 'device_name': 'cpu',
            'torch_version': torch.__version__, 'torch_cuda_version': None,
        }
        rows = (
            ('temp0p5_cfg3p5', 0.5, 3.5),
            ('temp0p5_cfg1p0', 0.5, 1.0),
            ('temp0p5_cfg2p5', 0.5, 2.5),
            ('temp0p5_cfg4p5', 0.5, 4.5),
            ('temp0p3_cfg3p5', 0.3, 3.5),
            ('temp0p8_cfg3p5', 0.8, 3.5),
        )
        results = {}
        for index, (name, temperature, cfg_scale) in enumerate(rows):
            generation = directory / f'{name}.h5'
            generation.write_bytes(f'generation-{name}'.encode())
            values = {
                'fid': 1.0 + index, 'diversity': 8.0 + index / 10,
                'top1': 0.4, 'top2': 0.6, 'top3': 0.7,
                'matching': 3.5 + index / 10,
            }
            metrics = directory / f'{name}.json'
            metrics.write_text(json.dumps({
                'schema': 'fair_interx_main_only_metrics_v1',
                'status': 'complete', 'samples': 544,
                'multimodality': None,
                'requests_sha256': report_sha256_file(requests),
                'main_generations': str(generation.resolve()),
                'main_generations_sha256': report_sha256_file(generation),
                **values,
                'evaluator_provenance': evaluator_provenance,
                'evaluator_runtime': evaluator_runtime,
            }), encoding='utf-8')
            results[name] = {
                'configuration': {
                    'sampling_steps': 50, 'temperature': temperature,
                    'cfg_scale': cfg_scale,
                },
                **values,
                'generation': record(generation), 'metrics': record(metrics),
            }
        payload = {
            'schema': 'hhc_validation_sampler_calibration_v1',
            'status': 'complete', 'test_split_used': False,
            'sampling_order_contract': (
                'exact_hhc_training_validation_loader_batch_and_within_batch_order'
            ),
            'formal_row': {
                'sampling_steps': 50, 'temperature': 0.5, 'cfg_scale': 3.5,
            },
            'grid': [
                {'temperature': temperature, 'cfg_scale': cfg_scale}
                for _, temperature, cfg_scale in rows
            ],
            'formal_row_bridge': {
                'training_validation_fid': 1.0,
                'offline_replay_fid': 1.0,
                'absolute_difference': 0.0,
                'maximum_absolute_difference': 1e-4,
                'tolerance_basis': (
                    'checkpoint selection records FID to four decimals; '
                    'standalone HDF5 re-embedding must reproduce within one '
                    '0.0001 unit'
                ),
                'bitwise_equivalence_claimed': False,
                'pass': True,
            },
            'lowest_validation_fid_row': 'temp0p5_cfg3p5',
            'results': results,
            'artifacts': {
                'selection_manifest': record(bound),
                'checkpoint': record(bound),
                'request_manifest': record(request_manifest),
                'requests': record(requests),
                'formal_hhc_validation_selection': record(formal_records),
                'vq_checkpoint': record(bound), 'contact_map': record(bound),
                'exporter': record(Path('tools/export_hhc_official_interx.py')),
                'scorer': record(Path('tools/score_fair_interx_main.py')),
                'implementation': record(Path(
                    'tools/run_hhc_validation_sampler_calibration.py'
                )),
            },
        }
        validate_hhc_validation_sampler_calibration(payload)
        payload['test_split_used'] = True
        try:
            validate_hhc_validation_sampler_calibration(payload)
        except ValueError as error:
            assert 'validation-only common-50 calibration' in str(error)
        else:
            raise AssertionError('test-set sampler calibration was accepted')


def test_systemd_post_formal_chain_is_fail_closed_and_path_independent():
    repository = Path(__file__).resolve().parents[1]
    chain = (repository / 'tools/queue_post_formal_after_v3.sh').read_text()
    assert 'while systemctl --user is-active --quiet "${formal_service}"' in chain
    assert 'if [[ "${formal_result}" != success ]]' in chain
    assert chain.index('formal_result=') < chain.index(
        'bash tools/queue_post_hhc_gpu1_experiments.sh none'
    )
    assert chain.index(
        'bash tools/queue_post_hhc_gpu1_experiments.sh none'
    ) < chain.index('tools/queue_hhc_timing_after_post_gpu1.sh none')
    for relative_path in (
        'tools/queue_post_hhc_gpu1_experiments.sh',
        'tools/queue_hhc_timing_after_post_gpu1.sh',
        'tools/queue_post_shape_review_experiments_gpu1.sh',
        'tools/apply_post_formal_shape_root_tokenizer_wiring.sh',
        'tools/queue_interx_shape_augmentation_gpu1.sh',
    ):
        source = (repository / relative_path).read_text()
        assert '/home/wenning/miniconda3/bin/conda "$@"' in source
    formal_manifest = json.loads((
        repository / 'output/t2i_trans/'
        '2026-08-30-16-08-07_'
        'hhc_formal_raw12288_repaired_mgt_300k_seed123_v1/'
        'training_manifest.json'
    ).read_text(encoding='utf-8'))
    applier = (repository / (
        'tools/apply_post_formal_shape_root_tokenizer_wiring.sh'
    )).read_text(encoding='utf-8')
    formal_source_hashes = formal_manifest['sha256']
    expected_options = formal_source_hashes['options/option_transformer.py']
    expected_train = formal_source_hashes['train/train_t2i_trans.py']
    assert f'expected_options_sha="{expected_options}"' in applier
    assert f'expected_train_sha="{expected_train}"' in applier
    wiring = (
        repository / 'tools/apply_post_formal_shape_root_tokenizer_wiring.sh'
    ).read_text()
    assert wiring.index('tools/freeze_mgt_training_sources.py') < wiring.index(
        'git apply --check'
    )
    assert wiring.index('tools/freeze_mgt_training_sources.py') < wiring.index(
        'tools/audit_hhc_formal_launch_contract.py'
    ) < wiring.index('git apply --check')
    final_chain = (repository / 'tools/queue_final_report_after_v3.sh').read_text()
    assert 'if [[ "${post_service}" != none ]]' in final_chain
    assert final_chain.index(
        'if [[ "${post_service}" != none ]]'
    ) < final_chain.index(
        'while systemctl --user is-active --quiet "${post_service}"'
    )
    assert 'if [[ "${post_result}" != success ]]' in final_chain
    assert final_chain.index('post_result=') < final_chain.index(
        "while pgrep -f 'bash tools/queue_gpu0_cosh_timotion.sh'"
    )
    assert final_chain.index(
        "while pgrep -f 'bash tools/queue_gpu0_cosh_timotion.sh'"
    ) < final_chain.index(
        'exec bash tools/queue_review_report_after_all_training.sh none'
    )
    report_queue = (
        repository / 'tools/queue_review_report_after_all_training.sh'
    ).read_text()
    assert '/home/wenning/miniconda3/bin/conda "$@"' in report_queue
    assert '\npython tools/audit_historical_mmm_local_history.py' not in report_queue


def test_mgt_mm_replay_audit_checks_every_completed_repeat():
    with tempfile.TemporaryDirectory() as temporary:
        reference_path = Path(temporary) / 'reference.h5'
        candidate_path = Path(temporary) / 'candidate.h5'
        motions = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        names = np.asarray([b'a', b'b'])
        lengths = np.asarray([4, 3], dtype=np.int64)
        for path in (reference_path, candidate_path):
            with h5py.File(path, 'w') as handle:
                handle.create_dataset('motions', data=motions)
                handle.create_dataset('names', data=names)
                handle.create_dataset('lengths', data=lengths)
                handle.create_dataset(
                    'completed_repeats', data=np.ones(3, dtype=np.bool_)
                )
                if path == candidate_path:
                    handle.attrs['training_conditioning_contract'] = (
                        'historical_mmm_shape_dropout'
                    )
                    handle.attrs['inference_conditioning_contract'] = (
                        'paper_text_cfg'
                    )
        result = compare_mm_files(reference_path, candidate_path, expected_repeats=3)
        assert result['status'] == 'pass'
        assert result['motions_byte_exact'] is True
        assert len(result['repeat_records']) == 3
        with h5py.File(candidate_path, 'r+') as handle:
            handle['motions'][0, 2, 0, 0] += 1
        result = compare_mm_files(reference_path, candidate_path, expected_repeats=3)
        assert result['status'] == 'fail'
        assert result['repeat_records'][2]['motions_byte_exact'] is False


def test_mgt_launch_source_freezer_materializes_all_manifest_sources():
    repository = Path(__file__).resolve().parents[1]
    sources = (
        'train/train_t2i_trans.py', 'options/option_transformer.py',
        'dataset/dataset_interx_TM.py',
        'dataset/dataset_interx_tokenize.py', 'models/model_tools.py',
        'models/motion_transformer.py', 'models/two_person_transformer.py',
        'models/vqvae_sep3_shape.py', 'utils/contact_map.py',
        'utils/interx_shape_augmentation.py', 'utils/eval_trans.py',
        'utils/motion_masks.py', 'utils/motion_process.py', 'utils/runtime.py',
    )
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        training_manifest = temporary / 'training_manifest.json'
        training_manifest.write_text(json.dumps({
            'sha256': {
                name: report_sha256_file(repository / name) for name in sources
            }
        }))
        output_dir = temporary / 'launch_sources'
        output_manifest = temporary / 'launch_source_snapshot_manifest.json'
        payload = freeze_sources(training_manifest, output_dir, output_manifest)
        assert payload['status'] == 'pass'
        assert payload['source_count'] == 14
        assert set(payload['sources']) == set(sources)
        for name in sources:
            snapshot = payload['sources'][name]
            assert Path(snapshot['path']).is_file()
            assert snapshot['sha256'] == report_sha256_file(repository / name)
        assert freeze_sources(
            training_manifest, output_dir, output_manifest
        ) == payload
        first_snapshot = Path(payload['sources'][sources[0]]['path'])
        first_snapshot.write_text('tampered', encoding='utf-8')
        repaired = freeze_sources(training_manifest, output_dir, output_manifest)
        assert repaired['sources'][sources[0]]['sha256'] == report_sha256_file(
            repository / sources[0]
        )

        observable_sources = sources + (
            'dataset/dataset_interx_tokenize_historical_root.py',
            'utils/training_telemetry.py',
        )
        observable_manifest = temporary / 'observable_training_manifest.json'
        observable_manifest.write_text(json.dumps({
            'sha256': {
                name: report_sha256_file(repository / name)
                for name in observable_sources
            }
        }))
        observable = freeze_sources(
            observable_manifest,
            temporary / 'observable_launch_sources',
            temporary / 'observable_launch_source_snapshot_manifest.json',
        )
        assert observable['status'] == 'pass'
        assert observable['source_count'] == 16
        assert set(observable['sources']) == set(observable_sources)


def test_original38_window_audit_and_post_formal_transform_are_explicit():
    repository = Path(__file__).resolve().parents[1]
    required_matched_fields = {
        'paper_architecture', 'training_mask_contract',
        'training_loss_contract', 'optimizer_contract', 'seed',
        'vq_checkpoint', 'cache_root', 'codebook_dir', 'train_motion_file',
        'train_split_file', 'val_motion_file', 'val_split_file',
        'sampling_steps', 'temperature', 'cfg_scale',
    }
    assert required_matched_fields <= set(TOKEN_WINDOW_MATCHED_ARGUMENTS)
    audit = json.loads((
        repository / 'output/hhc_formal_mgt_token_window_alignment_audit.json'
    ).read_text())
    assert audit['status'] == 'pass'
    assert audit['contracts']['original_mmm_tensor_window_including_end'] == 38
    assert audit['contracts']['current_formal_tensor_window_including_end'] == 50
    assert audit['all_cached_token_lengths']['maximum'] == 247
    assert audit['window_effects']['original_mmm_38']['samples_truncated'] == 5006
    assert audit['window_effects']['current_formal_50']['samples_truncated'] == 2792
    assert audit['fair_requests']['frame_lengths']['maximum'] == 150
    assert audit['fair_requests']['motion_token_lengths_ceil']['maximum'] == 38

    current = {
        key: path.read_text(encoding='utf-8')
        for key, path in ORIGINAL38_TRANSFORM_FILES.items()
    }
    before = dict(current)
    if 'choices=["original_mmm_38", "paper_50", "historical_38"]' in current['options']:
        # The guarded transformation is intentionally one-shot.  Once the
        # working tree has advanced, exercise its preimage against the frozen
        # formal-launch sources instead of trying to transform the live source
        # for a second time.
        frozen = (
            repository
            / 'output/t2i_trans/2026-08-30-16-08-07_hhc_formal_raw12288_repaired_mgt_300k_seed123_v1'
            / 'launch_sources'
        )
        before['options'] = (
            frozen / 'options/option_transformer.py'
        ).read_text(encoding='utf-8')
        before['train'] = (
            frozen / 'train/train_t2i_trans.py'
        ).read_text(encoding='utf-8')
        before['paper_audit'] = before['paper_audit'].replace(
            "    formal_expected = {\n",
            "    formal_expected = {\n"
            "        'token_length_contract': 'paper_50',\n",
            1,
        ).replace(
            "    allowed_token_windows = {'original_mmm_38', 'paper_50'}\n"
            "    if configured.get('token_length_contract') not in allowed_token_windows:\n"
            "        formal_mismatches['token_length_contract'] = {\n"
            "            'expected_one_of': sorted(allowed_token_windows),\n"
            "            'active': configured.get('token_length_contract'),\n"
            "        }\n",
            "",
            1,
        )
    after = transform_original38_sources(before)
    assert 'choices=["original_mmm_38", "paper_50", "historical_38"]' in after['options']
    assert 'default="original_mmm_38"' in after['options']
    assert '37 motion tokens/148 frames' in after['options']
    assert 'fair-evaluation cap of 150 frames' in after['options']
    assert after['train'].count("'original_mmm_38', 'historical_38'") == 2
    assert "allowed_token_windows = {'original_mmm_38', 'paper_50'}" in after['paper_audit']
    assessor = (
        repository / 'tools/assess_hhc_token_window_control.py'
    ).read_text()
    launcher = (
        repository / 'tools/run_hhc_original38_token_window_control.sh'
    ).read_text()
    assert '--training-token-length-contract original_mmm_38' in launcher
    assert '!= "original_mmm_38"' in assessor
    assert '!= "historical_38"' not in assessor

    queue = (repository / 'tools/queue_hhc_timing_after_post_gpu1.sh').read_text()
    assert queue.index('run_hhc_original38_token_window_control.sh') < queue.index(
        'run_hhc_global_masked_loss_control.sh'
    )
    assert 'selected_38_fid' in queue
    assert 'float(\'${selected_38_fid}\') <= 0.4' in queue


def test_global_masked_loss_control_is_strictly_single_factor():
    assert {
        'paper_architecture', 'training_token_length_contract',
        'training_mask_contract', 'optimizer_contract', 'seed',
        'vq_checkpoint', 'cache_root', 'codebook_dir', 'train_motion_file',
        'train_split_file', 'val_motion_file', 'val_split_file',
        'sampling_steps', 'temperature', 'cfg_scale',
    } <= set(GLOBAL_LOSS_MATCHED_ARGUMENTS)
    formal = {key: f'value:{key}' for key in GLOBAL_LOSS_MATCHED_ARGUMENTS}
    formal.update({
        'training_loss_contract': 'paper_masked_only',
        'training_token_length_contract': 'paper_50',
    })
    control = dict(formal)
    control['training_loss_contract'] = 'paper_masked_global_token_mean'
    validate_single_factor_contract(formal, control)

    drifted = dict(control)
    drifted['seed'] = 'different-seed'
    try:
        validate_single_factor_contract(formal, drifted)
    except ValueError as error:
        assert 'not single-factor matched' in str(error)
        assert 'seed' in str(error)
    else:
        raise AssertionError('confounded global-loss control was accepted')

    wrong_window = dict(formal)
    wrong_window['training_token_length_contract'] = 'original_mmm_38'
    try:
        validate_single_factor_contract(wrong_window, control)
    except ValueError as error:
        assert '50-token window' in str(error)
    else:
        raise AssertionError('non-50-token formal reference was accepted')


def test_historical_mask_diagnostic_is_gated_and_single_factor():
    repository = Path(__file__).resolve().parents[1]
    assessor = (
        repository / 'tools/assess_hhc_historical_mask_control.py'
    ).read_text(encoding='utf-8')
    launcher = (
        repository / 'tools/run_hhc_historical_mask_control.sh'
    ).read_text(encoding='utf-8')
    ast.parse(assessor, filename='assess_hhc_historical_mask_control.py')
    assert 'key != "training_mask_contract"' in assessor
    assert '"training_token_length_contract",' in assessor
    assert '"training_mask_contract": "paper_kinematic"' in assessor
    assert 'historical_flat_random_replace' in assessor
    assert 'historical corruption cannot replace the formal paper mask' in assessor
    assert (
        'global_loss_not_sufficient_investigate_mask_conditioning_next'
        in launcher
    )
    assert '--training-mask-contract historical_flat_random_replace' in launcher
    assert '--training-loss-contract paper_masked_only' in launcher
    assert '--training-token-length-contract paper_50' in launcher
    assert '--optimizer-contract paper_warmup_cosine' in launcher
    assert '--conditioning-contract paper_text_cfg' in launcher
    assert '--training-validation-contract paper_isolated_common' in launcher
    assert '--sampler sample_intra_cas' in launcher
    assert '--sampling-steps 50' in launcher
    assert '--wandb-mode online' in launcher

    formal = {key: f'value:{key}' for key in MASK_MATCHED_ARGUMENTS}
    formal.update({
        'training_mask_contract': 'paper_kinematic',
        'training_loss_contract': 'paper_masked_only',
        'training_token_length_contract': 'paper_50',
        'optimizer_contract': 'paper_warmup_cosine',
        'conditioning_contract': 'paper_text_cfg',
        'training_validation_contract': 'paper_isolated_common',
        'paper_architecture': True,
    })
    control = dict(formal)
    control['training_mask_contract'] = 'historical_flat_random_replace'
    validate_historical_mask_contract(formal, control)

    drifted = dict(control)
    drifted['training_token_length_contract'] = 'original_mmm_38'
    try:
        validate_historical_mask_contract(formal, drifted)
    except ValueError as error:
        assert 'not single-factor' in str(error)
        assert 'training_token_length_contract' in str(error)
    else:
        raise AssertionError('confounded historical-mask control was accepted')


def test_promoted_original38_shape_transform_is_context_scoped():
    before = {
        key: path.read_text(encoding='utf-8')
        for key, path in PROMOTED38_SHAPE_FILES.items()
    }
    after = transform_promoted38_shape_sources(before)
    for key in ('audit', 'results', 'tests'):
        ast.parse(after[key], filename=str(PROMOTED38_SHAPE_FILES[key]))
    assert '--training-token-length-contract original_mmm_38' in after['queue']
    assert (
        "'--training-token-length-contract original_mmm_38',"
        in after['tests']
    )
    # Independent provenance and loss-control tests must retain the completed
    # formal paper-50 reference after only the shape baseline is promoted.
    assert 'choices=["original_mmm_38", "paper_50", "historical_38"]' in after['tests']
    assert "'training_token_length_contract': 'paper_50'" in after['tests']
    applier = (
        Path(__file__).resolve().parents[1]
        / 'tools/apply_promoted_original38_shape_baseline.py'
    ).read_text(encoding='utf-8')
    for guard in (
        'selected_validation_fid', 'selected_fid > 0.4',
        'math.isfinite(selected_fid)',
        'test_metrics_used_for_selection',
        'hhc_wandb_online_metric_history_audit_v1',
        'expected_final_step") != 300000',
        'pass_with_disclosed_exceptions',
    ):
        assert guard in applier


def run_gpu_training_smoke(device='cuda:1', precision='bf16'):
    """One full-size paper-architecture forward/backward for manual CI smoke."""
    codebook_size = 8096
    model = Text2Motion_Transformer(
        _Tokenizer(codebook_size), num_vq=codebook_size, embed_dim=512,
        clip_dim=512, block_size=51, num_layers=9, num_local_layer=9,
        n_head=8, drop_out_rate=0.1, fc_rate=4, shape=True,
        paper_architecture=True,
    ).to(device).train()
    batch_size, token_length = 2, 38
    target = torch.randint(0, codebook_size, (batch_size, token_length, 3), device=device)
    valid = torch.ones_like(target, dtype=torch.bool)
    mask, _, _ = sample_kinematic_cascade_mask(valid)
    masked = torch.where(mask, codebook_size + 2, target)
    clip_feature = torch.randn(batch_size, 512, device=device)
    word_emb = torch.randn(batch_size, 77, 512, device=device)
    betas = torch.randn(batch_size, 20, device=device)
    gender = torch.zeros(batch_size, 2, dtype=torch.long, device=device)
    src_mask = torch.ones(batch_size, token_length, dtype=torch.bool, device=device)
    base_mask = model.get_attn_mask(
        src_mask.repeat_interleave(3, dim=1)
    )
    assert base_mask.shape[-2:] == (token_length * 3, token_length * 3)
    autocast_enabled = precision == 'bf16'
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        logits = model(
            masked, clip_feature, src_mask=src_mask, word_emb=word_emb,
            betas=betas, gender=gender,
        )
        loss = torch.nn.functional.cross_entropy(logits[mask], target[mask])
    loss.backward()
    return {
        'loss': float(loss.detach()),
        'logits_dtype': str(logits.dtype),
        'max_memory_bytes': int(torch.cuda.max_memory_allocated(device)),
    }


if __name__ == '__main__':
    tests = (
        test_training_mask_enforces_root_body_hand_hierarchy,
        test_masked_nll_ignores_out_of_vocabulary_end_and_pad_targets,
        test_historical_mmm_corruption_matches_recovered_source_logic,
        test_historical_all_valid_nll_ignores_padding_but_scores_visible_tokens,
        test_paper_transformer_has_nine_blocks_and_cascade_resolves_masks,
        test_historical_conditioning_reproduces_clip_shape_dropout_without_word_dropout,
        test_historical_training_and_common50_inference_conditioning_are_separate,
        test_paper_special_tokens_follow_autocast_projection_dtype,
        test_hhc_exporter_uses_and_validates_vq_checkpoint_arguments,
        test_cascade_uses_each_part_update_immediately,
        test_historical_cascade_repredicts_all_valid_tokens_each_round,
        test_official_150_frame_contact_pool_has_38_tokens,
        test_official_retrieval_is_accumulated_per_complete_batch,
        test_incomplete_generation_repeats_are_rejected,
        test_exporters_flush_motion_data_before_repeat_completion_marker,
        test_native_generation_lengths_are_explicit_and_bounded,
        test_ema_quantizer_checkpoint_state_round_trips_exactly,
        test_legacy_ema_checkpoint_fallback_preserves_loaded_codebooks,
        test_vq_dataset_uses_the_requested_normalization_statistics,
        test_contact_code_map_can_be_restricted_to_training_names,
        test_token_cache_crop_is_sample_keyed_and_resume_stable,
        test_shape_overlay_tokenizer_matches_recovered_historical_root_contract,
        test_observational_export_disables_training_side_effects_and_restores_mode,
        test_mgt_training_rejects_transformer_eval_mode_after_sanity_validation,
        test_mgt_restart_resolution_and_in_place_resume_are_strict,
        test_generic_lightning_state_resolver_requires_full_resume_state,
        test_coshmdm_queue_has_strict_os_restart_resume_gate,
        test_coshmdm_validation_selection_requires_finalized_two_day_cap,
        test_coshmdm_milestones_are_zero_copy_and_bounded_to_user_cap,
        test_mgt_validation_pair_survives_rolling_and_topk_checkpoint_deletion,
        test_mgt_validation_pair_audit_requires_post_validation_marker,
        test_stochastic_validation_restores_training_rng_streams,
        test_contact_guidance_requires_matching_train_only_manifest,
        test_historical_mmm_cache_import_is_exact_and_excludes_test_tokens,
        test_historical_mmm_cache_import_preflights_all_pairs_before_writing,
        test_legacy_axis_bug_contact_map_requires_explicit_audited_contract,
        test_transformer_checkpoint_cannot_replace_requested_vq,
        test_main_and_multimodality_generation_provenance_must_match,
        test_metric_aggregation_requires_unique_requests_and_one_sampler,
        test_contact_optimization_defaults_match_paper,
        test_fine_contact_freeze_is_unique_bidirectional_matching,
        test_shared_volumetric_backend_is_identity_reused_and_contract_checked,
        test_llm_contact_parser_fails_closed_on_out_of_schema_values,
        test_contact_extraction_uses_exact_frozen_request_captions,
        test_codex_contact_cache_binds_frozen_requests_and_prompt_contract,
        test_full_luna_cache_reports_manifest_wall_clock_without_fake_latency,
        test_contact_timing_sources_are_explicit_and_length_safe,
        test_aggregate_binds_oracle_timing_to_each_request,
        test_holm_bonferroni_is_monotone_in_sorted_p_values,
        test_paired_method_comparison_respects_metric_directions,
        test_paired_replication_zero_sensitivity_filters_completed_reference,
        test_generation_source_must_match_training_source,
        test_generation_checkpoints_are_bound_to_training_audits,
        test_cached_physical_report_must_match_frozen_protocol,
        test_review_completion_status_never_hides_pending_scope,
        test_final_report_rejects_failed_or_wrong_step_training_audits,
        test_timotion_generation_contract_is_bound_to_final_manifest,
        test_coshmdm_representation_adapter_audit_is_current,
        test_protocol_compute_field_aliases_reject_conflicts,
        test_official_scorer_and_report_exclude_secondary_fid_protocols,
        test_mgt_common50_bridge_binds_contact_vq_and_evaluator,
        test_mgt_repair_gate_rejects_contact_contract_drift,
        test_final_ordinary_only_audit_binds_every_metric_record,
        test_active_report_and_postprocessing_bind_formal_repaired_hhc,
        test_mgt_selection_ignores_interrupted_same_name_runs,
        test_mgt_selection_uses_exact_checkpoint_fid_not_rounded_filename,
        test_optimizer_npz_inverse_preserves_two_person_raw_layout,
        test_fair_h5_converter_records_bind_actual_optimizer_npz,
        test_inverse_converter_binds_optimizer_sidecar_identity_and_contract,
        test_formal_optimization_report_branch_is_strict_before_cache_exists,
        test_formal_optimizer_rejects_oracle_timing_source_provenance,
        test_optimizer_inputs_are_byte_bound_to_generation_record,
        test_optimizer_ablation_report_binds_outputs_statistics_and_runtime,
        test_shape_augmentation_report_requires_bound_matched_arms,
        test_shape_augmentation_volume_contract_binds_real_weights,
        test_shape_augmentation_backend_equivalence_audit_is_current,
        test_frozen_request_verifier_resolves_only_exact_content_snapshot,
        test_frozen_interx_request_manifest_resolves_every_provenance_record,
        test_frozen_request_creation_runtime_is_bound_to_successful_cpu_run,
        test_fair_manifest_freezer_requires_exact_recorded_source_bytes,
        test_llm_robustness_report_does_not_claim_region_oracle,
        test_final_queue_direct_entrypoints_resolve_repo_imports,
        test_hhc_sampler_calibration_is_validation_only_and_common50,
        test_hhc_sampler_calibration_report_gate_binds_validation_artifacts,
        test_systemd_post_formal_chain_is_fail_closed_and_path_independent,
        test_mgt_mm_replay_audit_checks_every_completed_repeat,
        test_mgt_launch_source_freezer_materializes_all_manifest_sources,
        test_original38_window_audit_and_post_formal_transform_are_explicit,
        test_global_masked_loss_control_is_strictly_single_factor,
        test_promoted_original38_shape_transform_is_context_scoped,
        test_pair_audit_requires_bitwise_state_and_metric_binding,
        test_hybrid_cache_preserves_original_bytes_and_partitions_names,
        test_hybrid_cache_rejects_conflicting_original_and_scope_contamination,
        test_hybrid_partition_rejects_malformed_or_out_of_scope_augmented_names,
    )
    for test in tests:
        test()
        print({'test': test.__name__, 'status': 'pass'})
