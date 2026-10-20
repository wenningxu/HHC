#!/usr/bin/env python
"""Build final machine-readable and Markdown review tables from strict outputs."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path


HHC_NO_TIMING_LABEL = (
    'hhc_formal_raw12288_repaired_mgt_common50_no_contact'
)
HHC_ORACLE_TIMING_LABEL = (
    'hhc_formal_raw12288_repaired_mgt_common50_contact'
)
HHC_FORMAL_LABELS = {HHC_NO_TIMING_LABEL, HHC_ORACLE_TIMING_LABEL}
COSHMDM_LABEL = 'coshmdm_interx_retrained_2daycap_seed123'
EXPECTED_OPTIMIZATION_ABLATIONS = (
    {'label': 'paper_full', 'iterations': 100, 'fine_start': 50, 'weights': {}},
    {'label': 'iterations_50', 'iterations': 50, 'fine_start': 25, 'weights': {}},
    {'label': 'iterations_200', 'iterations': 200, 'fine_start': 100, 'weights': {}},
    {'label': 'no_attract', 'iterations': 100, 'fine_start': 50,
     'weights': {'attract': 0.0}},
    {'label': 'no_penetration', 'iterations': 100, 'fine_start': 50,
     'weights': {'pene': 0.0}},
    {'label': 'no_regularization', 'iterations': 100, 'fine_start': 50,
     'weights': {'reg': 0.0}},
    {'label': 'no_smoothness', 'iterations': 100, 'fine_start': 50,
     'weights': {'smooth': 0.0}},
    {'label': 'no_foot_skate', 'iterations': 100, 'fine_start': 50,
     'weights': {'skate': 0.0}},
    {'label': 'no_ground', 'iterations': 100, 'fine_start': 50,
     'weights': {'ground': 0.0}},
    {'label': 'pene_weight_1', 'iterations': 100, 'fine_start': 50,
     'weights': {'pene': 1.0}},
    {'label': 'pene_weight_10', 'iterations': 100, 'fine_start': 50,
     'weights': {'pene': 10.0}},
    {'label': 'attract_weight_0p5', 'iterations': 100, 'fine_start': 50,
     'weights': {'attract': 0.5}},
    {'label': 'attract_weight_2', 'iterations': 100, 'fine_start': 50,
     'weights': {'attract': 2.0}},
)

METHODS = (
    ('HHC-M raw12288 (no timing, common 50 steps)', HHC_NO_TIMING_LABEL),
    ('HHC-M raw12288 (oracle timing, common 50 steps)', HHC_ORACLE_TIMING_LABEL),
    ('InterMask (common 50 steps)', 'intermask_official_exact_common50'),
    ('Interact2AR (no GT first frame)', 'interact2ar_official_fair_no_gt_first_frame'),
    ('CoShMDM (Inter-X retrained, user-capped at 2 additional days; DDIM50)',
     COSHMDM_LABEL),
    ('TiMotion public Transformer+LPA (Inter-X retrained)',
     'timotion_transformer_lpa_interx_retrained_fair'),
)
FINAL_METHOD_LABELS = {label for _, label in METHODS}
MESH_PHYSICAL_METRIC_KEYS = (
    'penetrating_vertex_fraction',
    'collision_frame_rate',
    'penetration_depth_sum_m_per_frame',
    'paper_eq6_squared_sum_m2_per_frame',
    'mean_depth_m_on_penetrating_vertices',
    'max_depth_m',
    'mean_foot_sliding_m_per_frame',
)
VOLUMETRIC_CHECKPOINT_SHA256 = {
    'male': '96b22f6409e67dc950dafdd83bb8dc4779381ecc2b2c752e1c6fa3a205b2e18c',
    'female': '2b582f009c2daeec50c90b5f39689a248bac96a4cc9983548ad4be677b72ccdb',
}
CONVERSION_RESUME_CONTRACT = (
    'existing NPZ arrays must exactly equal the deterministic current '
    'request/generation conversion; missing files are atomically created; '
    'stale files fail closed'
)
# Retained only so archived held-out manifests can still be audited explicitly.
# These experiments are not loaded, required, or reported by main().
GENERALIZATION_SAMPLES = {
    'shape/test_unseen': 320,
    'shape/test_seen': 1344,
    'action/test_unseen': 352,
    'action/test_seen': 1312,
}

TRAINING_AND_PROTOCOL_ARTIFACTS = (
    ('interx_protocol_audit', Path('output/interx_protocol_audit.json')),
    ('hhc_paper_training_contract_audit', Path(
        'output/hhc_formal_raw12288_repaired_mgt_training_contract_audit.json'
    )),
    ('hhc_formal_launch_contract_audit', Path(
        'output/hhc_formal_launch_contract_audit.json'
    )),
    ('hhc_sampling_runtime_audit', Path(
        'output/hhc_sampling_runtime_audit.json'
    )),
    ('historical_mgt_tokenizer_runtime_audit', Path(
        'output/historical_mgt_tokenizer_runtime_audit.json'
    )),
    ('historical_mgt_launch_source_recovery', Path(
        'output/historical_mgt_v7_launch_source_recovery.json'
    )),
    ('mgt_repair_gate', Path('output/mgt_repair_gate.json')),
    ('historical_mgt_common50_mm_replay', Path(
        'output/historical_mgt_v7_step280k_common50_mm_full30_audit.json'
    )),
    ('hhc_contact_guidance_map_audit', Path(
        'output/hhc_contact_guidance_map_audit.json'
    )),
    ('hhc_validation_observational_pair_audit', Path(
        'output/hhc_formal_raw12288_repaired_mgt_20k_'
        'pre_post_validation_pair_audit.json'
    )),
    ('hhc_mgt_train_val_conditional_support_audit', Path(
        'output/hhc_formal_mgt_train_val_conditional_support_audit.json'
    )),
    ('hhc_mgt_shape_dependence_training', Path(
        'output/hhc_formal_mgt_shape_dependence_training.json'
    )),
    ('hhc_mgt_shape_dependence_validation', Path(
        'output/hhc_formal_mgt_shape_dependence_validation.json'
    )),
    ('hhc_mgt_shape_dependence_pair', Path(
        'output/hhc_formal_mgt_shape_dependence_pair.json'
    )),
    ('hhc_global_masked_loss_control_assessment', Path(
        'output/hhc_global_masked_loss_control_assessment.json'
    )),
    ('hhc_post_50k_training_dispatch', Path(
        'output/hhc_post_50k_training_dispatch.json'
    )),
    ('review_requirement_coverage', Path(
        'output/review_requirement_coverage.json'
    )),
    ('interx_luna_train_cache_audit', Path(
        'output/interx_luna_train_cache_audit.json'
    )),
    ('interx_luna_train_timing_audit', Path(
        'output/interx_luna_train_timing_consistency_audit.json'
    )),
    ('interx_luna_val_full_cache_audit', Path(
        'output/interx_luna_val_full_cache_audit.json'
    )),
    ('interx_luna_val_full_timing_audit', Path(
        'output/interx_luna_val_full_timing_audit.json'
    )),
    ('interx_luna_test_full_cache_audit', Path(
        'output/interx_luna_test_full_cache_audit.json'
    )),
    ('interx_luna_test_full_timing_audit', Path(
        'output/interx_luna_test_full_timing_audit.json'
    )),
    ('interx_luna_test_frozen_cache_audit', Path(
        'output/interx_luna_test_frozen_cache_audit.json'
    )),
    ('interx_terra_robustness_cache_audit', Path(
        'output/interx_terra_robustness_cache_audit.json'
    )),
    ('mgt_validation_selection', Path('output/mgt_validation_selection_manifest.json')),
    ('official_request_manifest', Path('output/fair_interx_requests/manifest.json')),
    ('fair_request_provenance_recovery', Path(
        'output/fair_interx_requests/provenance_recovery.json'
    )),
    ('fair_request_replay_audit', Path(
        'output/fair_interx_request_replay_audit.json'
    )),
    ('model_capacity_audit', Path('output/model_capacity_audit.json')),
    ('vq_import_manifest', Path(
        'checkpoints/vq_precleanup_raw12288/training_manifest.json'
    )),
    ('vq_final_reconstruction', Path(
        'output/vq_historical_eval/net_175000_reconstruction.json'
    )),
    ('vq_final_reconstruction_fid', Path(
        'output/vq_historical_eval/net_175000_reconstruction_ordinary_fid.json'
    )),
    ('historical_mmm_wandb_comparison', Path('output/historical_mmm_wandb_comparison.json')),
    ('intermask_native20_hdf5_protocol_audit', Path(
        'output/fair_eval/intermask_official_exact/'
        'native20_hdf5_protocol_audit.json'
    )),
    ('optimization_ablation_readiness', Path('output/optimization_ablation_readiness.json')),
    ('review_external_prerequisites', Path('output/review_external_prerequisites.json')),
    ('training_wallclock_audit', Path('output/training_wallclock_audit.json')),
    ('hhc_mgt_final_selection', Path(
        'output/hhc_formal_raw12288_repaired_mgt_selection.json'
    )),
    ('coshmdm_training_contract_audit', Path(
        'output/coshmdm_training_contract_audit.json'
    )),
    ('coshmdm_statsfix_startup_audit', Path(
        'output/coshmdm_statsfix_startup_audit.json'
    )),
    ('coshmdm_representation_adapter_audit', Path(
        'output/coshmdm_representation_adapter_audit.json'
    )),
    ('timotion_cosh_adapter_equivalence_audit', Path(
        'output/timotion_cosh_adapter_equivalence_audit.json'
    )),
    ('coshmdm_final_state_audit', Path('output/coshmdm_final_checkpoint_audit.json')),
    ('coshmdm_resume_provenance', Path(
        'output/coshmdm_resume_provenance.json'
    )),
    ('coshmdm_training_budget_cap', Path(
        'output/coshmdm_training_budget_cap.json'
    )),
    ('coshmdm_validation_selection', Path(
        'output/coshmdm_interx_val_selection.json'
    )),
    ('coshmdm_training_trajectory_diagnosis', Path(
        'output/coshmdm_training_trajectory_diagnosis.json'
    )),
    ('coshmdm_representation_oracle_metrics', Path(
        'output/coshmdm_representation_oracle/ordinary_fid.json'
    )),
    ('coshmdm_embedding_gap_diagnosis', Path(
        'output/coshmdm_embedding_gap_diagnosis.json'
    )),
    ('timotion_training_contract_audit', Path(
        'output/timotion_interx_training_contract_audit.json'
    )),
    ('timotion_padding_ro_stabilization_audit', Path(
        'output/timotion_padding_ro_stabilization_audit.json'
    )),
    ('timotion_interrupted_run_exclusion', Path(
        'output/timotion_formal_jqh616am_exclusion.json'
    )),
    ('timotion_gradient_accumulation_audit', Path(
        'output/timotion_gradient_accumulation_audit.json'
    )),
    ('timotion_warmup_stability_audit', Path(
        'output/timotion_warmup_stability_audit.json'
    )),
    ('timotion_transient_velocity_spike_audit', Path(
        'output/timotion_transient_velocity_spike_audit.json'
    )),
    ('timotion_scheduler_boundary_audit', Path(
        'output/timotion_scheduler_boundary_audit.json'
    )),
    ('timotion_validation_selection', Path(
        'output/timotion_interx_val_selection.json'
    )),
    ('timotion_final_state_audit', Path('output/timotion_interx_final_checkpoint_audit.json')),
    ('timotion_formal_wandb_history_audit', Path(
        'output/timotion_interx_formal_wandb_history_audit.json'
    )),
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_artifact(path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8')), {
        'path': str(path), 'sha256': sha256_file(path)
    }


def metric(summary, key):
    record = summary['metrics'][key]
    return {'mean': record['mean'], 'ci95': record['ci95']}


def format_metric(record, digits=3):
    return f"{record['mean']:.{digits}f} ± {record['ci95']:.{digits}f}"


def require_artifact(condition, label, message):
    if not condition:
        raise ValueError(f'{label}: {message}')


def resolve_review_completion_status(
        *, optimization_complete, current_interx_pending,
        external_prerequisites_pending):
    """Report scope completion without hiding manuscript/external gaps."""
    if not optimization_complete or current_interx_pending:
        return 'partial_current_interx_requirements'
    if external_prerequisites_pending:
        return 'interx_complete_external_prerequisites_pending'
    return 'complete'


def validate_bound_file(label, payload, path_key='checkpoint', hash_key='checkpoint_sha256'):
    path = Path(payload.get(path_key, ''))
    require_artifact(path.is_file(), label, f'missing bound file {path}')
    require_artifact(
        payload.get(hash_key) == sha256_file(path), label,
        f'{path_key} SHA-256 is absent or stale',
    )


def validate_artifact_record(label, record):
    path = Path(record.get('path', ''))
    expected_hash = record.get('sha256')
    if path.is_file() and expected_hash == sha256_file(path):
        return path
    snapshot = Path('tools/provenance_snapshots') / (
        f'{expected_hash}__{path.name}'
    )
    require_artifact(
        snapshot.is_file() and expected_hash == sha256_file(snapshot),
        label,
        f'bound artifact SHA-256 is absent or stale: {path}; '
        f'content-addressed snapshot is also missing or stale: {snapshot}',
    )
    return snapshot.resolve()


def validate_main_only_evaluator_provenance(label, payload):
    provenance = payload.get('evaluator_provenance', {})
    require_artifact(
        provenance.get('contract') == 'official_interx_full_672d'
        and provenance.get('options', {}).get('sha256')
        == 'f4ddaa81157259ce6b795f456be98263e50cd788a7800c30aacbfc2361b2d6dd'
        and provenance.get('checkpoint', {}).get('sha256')
        == 'b1699f13276c044d8b7ae5f6063eb241b7ef96d7e92c878857abd5081e40701e'
        and Path(provenance.get('wrapper_source', {}).get('path', '')).name
        == 'evaluator_wrapper.py'
        and Path(provenance.get('option_loader_source', {}).get('path', '')).name
        == 'get_eval_option.py'
        and Path(provenance.get('scorer_source', {}).get('path', '')).name
        == 'score_fair_interx_main.py',
        label,
        'main-only metrics are not bound to the official Inter-X evaluator',
    )
    for name in (
        'options', 'checkpoint', 'wrapper_source', 'option_loader_source',
        'scorer_source',
    ):
        validate_artifact_record(
            f'{label}/evaluator_provenance/{name}', provenance.get(name, {})
        )
    runtime = payload.get('evaluator_runtime', {})
    require_artifact(
        runtime.get('contract')
        == 'pytorch_official_evaluator_single_device_v1'
        and runtime.get('device_type') in {'cpu', 'cuda'}
        and isinstance(runtime.get('requested_device'), str)
        and runtime.get('requested_device')
        and isinstance(runtime.get('device_name'), str)
        and runtime.get('device_name')
        and isinstance(runtime.get('torch_version'), str)
        and runtime.get('torch_version')
        and (
            runtime.get('device_index') is None
            if runtime.get('device_type') == 'cpu'
            else isinstance(runtime.get('device_index'), int)
        ),
        label,
        'main-only metrics do not record a valid evaluator runtime',
    )
    return provenance


def validate_final_ordinary_only_metric_audit(
        payload, expected_labels=None, expected_replications=20,
        expected_rescored_labels=None):
    label = 'final_ordinary_only_metric_audit'
    expected_labels = set(expected_labels or FINAL_METHOD_LABELS)
    if expected_rescored_labels is None:
        expected_rescored_labels = [method_label for _, method_label in METHODS]
    require_artifact(
        payload.get('schema')
        == 'hhc_final_interx_ordinary_only_metric_audit_v1'
        and payload.get('status') == 'pass'
        and payload.get('metric_protocol')
        == 'official_interx_full_672d_ordinary_only'
        and set(payload.get('metrics', ())) == {
            'fid', 'diversity_gt', 'diversity', 'top1', 'top2', 'top3',
            'matching', 'multimodality',
        }
        and set(payload.get('forbidden_metrics', ())) == {
            'fid_body_controlled', 'diversity_body_controlled',
            'diversity_gt_body_controlled',
        }
        and payload.get('rescored_labels') == expected_rescored_labels
        and set(payload.get('methods', {})) == expected_labels,
        label,
        'final methods are not bound to the ordinary-only evaluator contract',
    )
    implementation = payload.get('implementation', {})
    require_artifact(
        set(implementation) == {
            'audit_and_rescore', 'semantic_scorer', 'aggregator',
            'evaluator_options', 'evaluator_checkpoint',
            'evaluator_wrapper', 'evaluator_option_loader',
        },
        label,
        'ordinary-only implementation set changed',
    )
    repository = Path(__file__).resolve().parents[1]
    expected_implementation_paths = {
        'audit_and_rescore': repository / 'tools/rescore_final_interx_ordinary.py',
        'semantic_scorer': repository / 'tools/score_fair_interx_generations.py',
        'aggregator': repository / 'tools/aggregate_fair_interx_metrics.py',
        'evaluator_options': repository / 'checkpoints/hhi/Comp_v6_KLD01/opt.txt',
        'evaluator_checkpoint': (
            repository / 'checkpoints/hhi/text_mot_match/model/finest.tar'
        ),
        'evaluator_wrapper': repository / 'models/evaluator_wrapper.py',
        'evaluator_option_loader': repository / 'options/get_eval_option.py',
    }
    require_artifact(
        all(
            Path(implementation[name].get('path', '')).resolve()
            == expected_path.resolve()
            for name, expected_path in expected_implementation_paths.items()
        )
        and implementation['evaluator_options'].get('sha256')
        == 'f4ddaa81157259ce6b795f456be98263e50cd788a7800c30aacbfc2361b2d6dd'
        and implementation['evaluator_checkpoint'].get('sha256')
        == 'b1699f13276c044d8b7ae5f6063eb241b7ef96d7e92c878857abd5081e40701e',
        label,
        'ordinary-only evaluator implementation or official weights changed',
    )
    for implementation_name, record in implementation.items():
        validate_artifact_record(
            f'{label}/implementation/{implementation_name}', record
        )
    expected_per_record_evaluator = {
        'contract': 'official_interx_full_672d',
        'options': implementation['evaluator_options'],
        'checkpoint': implementation['evaluator_checkpoint'],
        'wrapper_source': implementation['evaluator_wrapper'],
        'option_loader_source': implementation['evaluator_option_loader'],
        'scorer_source': implementation['semantic_scorer'],
    }
    require_artifact(
        payload.get('per_record_evaluator_provenance')
        == expected_per_record_evaluator,
        label,
        'per-record evaluator provenance is absent or differs from the '
        'audited official evaluator implementation',
    )
    evaluator_runtime = payload.get('evaluator_runtime', {})
    require_artifact(
        evaluator_runtime.get('contract')
        == 'pytorch_official_evaluator_single_device_v1'
        and evaluator_runtime.get('requested_device') == 'cpu'
        and evaluator_runtime.get('device_type') == 'cpu'
        and evaluator_runtime.get('device_index') is None
        and evaluator_runtime.get('device_name') == 'cpu'
        and isinstance(evaluator_runtime.get('torch_version'), str)
        and evaluator_runtime.get('torch_version'),
        label,
        'final methods were not rescored on the frozen-request CPU runtime',
    )
    for method_label, method in payload['methods'].items():
        records = method.get('records', [])
        require_artifact(
            method.get('replications') == expected_replications
            and len(records) == expected_replications
            and [record.get('replication') for record in records]
            == list(range(expected_replications)),
            label,
            f'{method_label} ordinary-only replication set is incomplete',
        )
        validate_artifact_record(
            f'{label}/{method_label}/summary', method.get('summary', {})
        )
        for record in records:
            require_artifact(
                record.get('evaluator_provenance_verified') is True
                and record.get('evaluator_runtime_verified') is True,
                label,
                f'{method_label} replication {record.get("replication")} '
                'did not verify its own evaluator provenance/runtime',
            )
            validate_artifact_record(
                f'{label}/{method_label}/replication_'
                f'{record["replication"]:02d}',
                record.get('metrics', {}),
            )


def validate_coshmdm_representation_adapter(payload):
    """Bind the shape-aware CoShMDM adapter to its full Inter-X audit."""
    label = 'coshmdm_representation_adapter_audit'
    expected_thresholds = {
        'all_rmse_mean_max': 1e-3,
        'all_rmse_p95_max': 1e-3,
        'all_rmse_sample_max': 1e-2,
        'root_rotation6d_rmse_sample_max': 1e-3,
        'translation_rmse_m_sample_max': 5e-2,
        'body_rotation6d_rmse_sample_max': 1e-3,
        'max_abs_difference_max': 1e-1,
        'nonroot_body_rotation_max_abs': 0.0,
    }
    expected_conditions = {
        'sample_count', 'all_rmse_mean', 'all_rmse_p95',
        'all_rmse_max', 'root_rotation6d_rmse_max',
        'translation_rmse_m_max', 'body_rotation6d_rmse_max',
        'max_abs_difference', 'nonroot_body_rotations_preserved_exactly',
    }
    pass_conditions = payload.get('pass_conditions', {})
    require_artifact(
        payload.get('schema')
        == 'hhc_coshmdm_representation_adapter_audit_v1'
        and payload.get('status') == 'pass'
        and payload.get('samples') == 1696
        and payload.get('adapter_contract') == (
            'shape-specific all-joint least-squares rigid root fit v2; '
            'full requested length preserved by terminal velocity padding; '
            'no GT motion is used by generation exporters'
        )
        and payload.get('thresholds') == expected_thresholds
        and set(pass_conditions) == expected_conditions
        and all(pass_conditions.values()),
        label,
        'full-test shape-specific adapter audit is incomplete or changed',
    )

    inputs = payload.get('inputs', {})
    for name in ('requests', 'ground_truth'):
        validate_artifact_record(f'{label}/{name}', inputs.get(name, {}))
    skeletons = inputs.get('skeletons', {})
    require_artifact(
        set(skeletons) == {str(group) for group in range(1, 60)},
        label, 'audit does not bind the complete 59-shape skeleton bank',
    )
    for group, record in skeletons.items():
        validate_artifact_record(f'{label}/skeleton/{group}', record)

    summary = payload.get('projection_once_vs_twice', {})
    expected_summary_keys = {
        'all_rmse', 'root_rotation6d_rmse', 'translation_rmse_m',
        'body_rotation6d_rmse', 'max_abs_difference',
    }
    require_artifact(set(summary) == expected_summary_keys, label,
                     'adapter idempotence metric set changed')
    for metric_name, statistics in summary.items():
        require_artifact(
            set(statistics) == {'mean', 'p95', 'maximum'}
            and all(
                isinstance(value, (int, float))
                and math.isfinite(value) and value >= 0
                for value in statistics.values()
            ),
            label, f'{metric_name} statistics are incomplete or non-finite',
        )
    require_artifact(
        summary['all_rmse']['mean']
        <= expected_thresholds['all_rmse_mean_max']
        and summary['all_rmse']['p95']
        <= expected_thresholds['all_rmse_p95_max']
        and summary['all_rmse']['maximum']
        <= expected_thresholds['all_rmse_sample_max']
        and summary['root_rotation6d_rmse']['maximum']
        <= expected_thresholds['root_rotation6d_rmse_sample_max']
        and summary['translation_rmse_m']['maximum']
        <= expected_thresholds['translation_rmse_m_sample_max']
        and summary['body_rotation6d_rmse']['maximum']
        <= expected_thresholds['body_rotation6d_rmse_sample_max']
        and summary['max_abs_difference']['maximum']
        <= expected_thresholds['max_abs_difference_max'],
        label, 'recorded idempotence statistics exceed the frozen thresholds',
    )
    raw_projection = payload.get('raw_to_projection_information_loss', {})
    require_artifact(
        set(raw_projection) == {
            'root_rotation6d', 'nonroot_body_rotation6d',
            'hand_rotation6d', 'root_translation_m', 'all',
        }
        and all(
            set(statistics) == {'rmse', 'maximum', 'values'}
            and statistics['values'] > 0
            and math.isfinite(statistics['rmse'])
            and math.isfinite(statistics['maximum'])
            and statistics['rmse'] >= 0
            and statistics['maximum'] >= 0
            for statistics in raw_projection.values()
        )
        and raw_projection['nonroot_body_rotation6d']['maximum'] == 0.0
        and payload.get('irreversible_or_unmodelled_fields') == [
            'sequence-initial global root rotation',
            'sequence-initial global root translation and floor offset',
            '33 hand-joint local rotations per person',
        ],
        label,
        'adapter information-loss decomposition is incomplete or changed',
    )

    expected_implementation = {
        (Path(__file__).resolve().parents[1] /
         'tools/audit_coshmdm_representation_adapter.py').resolve(),
        Path('/mnt/8T-HDD/wenning/coshmdm-code/utils/interx_official_adapter.py'),
        Path('/mnt/8T-HDD/wenning/coshmdm-code/utils/interx_projection.py'),
        Path('/mnt/8T-HDD/wenning/coshmdm-code/tools/export_official_interx.py'),
        (Path(__file__).resolve().parents[1] /
         'tools/export_timotion_official_interx.py').resolve(),
        (Path(__file__).resolve().parents[1] /
         'tools/prepare_coshmdm_interx.py').resolve(),
    }
    implementation = payload.get('implementation', {})
    recorded_paths = {Path(path).resolve() for path in implementation}
    require_artifact(recorded_paths == expected_implementation, label,
                     'audited adapter implementation set changed')
    for path_text, digest in implementation.items():
        validate_artifact_record(
            f'{label}/implementation/{Path(path_text).name}',
            {'path': path_text, 'sha256': digest},
        )


def interx_skeleton_bank_sha256(root):
    digest = hashlib.sha256()
    for group in range(1, 60):
        path = Path(root) / f'{group}.npy'
        require_artifact(path.is_file(), 'coshmdm_representation_adapter_audit',
                         f'missing skeleton {path}')
        digest.update(f'{group}:'.encode('ascii'))
        digest.update(sha256_file(path).encode('ascii'))
    return digest.hexdigest()

def validate_formal_optimization(payload, expected_samples=1696):
    """Validate the cache-gated full HHC/GT optimizer result end to end.

    This is deliberately a standalone validator so the branch can be tested
    before the externally supplied strict contact cache becomes available.
    """
    label = 'optimization_formal'
    expected_hhc_source = {
        'contact_guidance': False,
        'contact_timing_source': 'none',
        'sampler': 'sample_intra_cas',
        'sampling_steps': 50,
        'temporal_mask': 'derived_from_initial_geometry_not_test_annotation',
    }
    source_condition = payload.get('hhc_source_condition', {})
    verified_source = source_condition.get('verified_hdf5_attributes', {})
    require_artifact(
        payload.get('schema') == 'hhc_paper_optimization_formal_v1'
        and payload.get('status') == 'complete'
        and payload.get('samples') == expected_samples
        and payload.get('semantic_evaluator_device') == 'cpu'
        and set(payload.get('conditions', {})) == {'hhc', 'gt'}
        and set(payload.get('output_artifacts', {})) == {'hhc', 'gt'},
        label,
        'full HHC/GT optimization comparison is incomplete',
    )
    require_artifact(
        all(source_condition.get(key) == value
            for key, value in expected_hhc_source.items())
        and all(verified_source.get(key) == value for key, value in {
            'contact_guidance': False,
            'contact_timing_source': 'none',
            'contact_timing_file': '',
            'contact_timing_file_sha256': '',
            'contact_map': '',
            'contact_map_sha256': '',
            'sampler': 'sample_intra_cas',
            'sampling_steps': 50,
        }.items())
        and verified_source.get('requests_sha256'),
        label,
        'formal optimization is not bound to no-timing common50 HHC',
    )
    loaded = {}
    for condition, condition_artifacts in payload['output_artifacts'].items():
        require_artifact(
            set(condition_artifacts) == {
                'optimization_completion_manifest',
                'optimized_h5', 'optimized_h5_manifest',
                'official_main_metrics', 'annotation_physical',
                'paired_mesh_physical',
            },
            label,
            f'{condition} formal output set changed',
        )
        for name, record in condition_artifacts.items():
            validate_artifact_record(
                f'{label}/{condition}/{name}', record
            )
        metrics = json.loads(Path(
            condition_artifacts['official_main_metrics']['path']
        ).read_text(encoding='utf-8'))
        completion = json.loads(Path(
            condition_artifacts['optimization_completion_manifest']['path']
        ).read_text(encoding='utf-8'))
        physical = json.loads(Path(
            condition_artifacts['annotation_physical']['path']
        ).read_text(encoding='utf-8'))
        paired = json.loads(Path(
            condition_artifacts['paired_mesh_physical']['path']
        ).read_text(encoding='utf-8'))
        require_artifact(
            metrics.get('schema') == 'fair_interx_main_only_metrics_v1'
            and metrics.get('status') == 'complete'
            and metrics.get('samples') == expected_samples
            and metrics.get('multimodality') is None
            and metrics.get('generation_request_hash_verified') is True,
            label,
            f'{condition} official main-only metrics are invalid',
        )
        validate_main_only_evaluator_provenance(
            f'{label}/{condition}', metrics
        )
        require_artifact(
            completion.get('schema') == 'hhc_paper_contact_optimization_v1'
            and completion.get('complete') is True
            and completion.get('completed_samples') == expected_samples
            and completion.get('requested_samples') == expected_samples
            and completion.get('runtime', {}).get('samples') == expected_samples
            and math.isfinite(completion.get('runtime', {}).get(
                'optimizer_wall_seconds_mean_per_sample', math.nan
            ))
            and math.isfinite(completion.get('runtime', {}).get(
                'optimizer_wall_seconds_p95_per_sample', math.nan
            )),
            label,
            f'{condition} optimizer runtime/completion is invalid',
        )
        require_artifact(
            physical.get('samples') == expected_samples
            and paired.get('schema')
            == 'hhc_paired_physical_comparison_v1'
            and paired.get('sample_count') == expected_samples,
            label,
            f'{condition} physical comparison is incomplete',
        )
        loaded[condition] = {
            'optimization_completion_manifest': completion,
            'official_main_metrics': metrics,
            'annotation_physical': physical,
            'paired_mesh_physical': paired,
        }
    require_artifact(
        loaded['hhc']['official_main_metrics']['evaluator_runtime']
        == loaded['gt']['official_main_metrics']['evaluator_runtime']
        and loaded['hhc']['official_main_metrics'][
            'evaluator_runtime'
        ].get('requested_device') == 'cpu'
        and loaded['hhc']['official_main_metrics'][
            'evaluator_runtime'
        ].get('device_type') == 'cpu',
        label,
        'HHC and GT optimization rows were not scored on one CPU runtime',
    )
    return loaded


def validate_optimization_ablation(payload, expected_samples=25):
    label = 'optimization_ablation_pilot'
    require_artifact(
        payload.get('schema') == 'hhc_paper_optimizer_ablation_pilot_v1'
        and payload.get('status') == 'complete'
        and payload.get('pilot_limit') == expected_samples
        and payload.get('ablations') == list(EXPECTED_OPTIMIZATION_ABLATIONS)
        and len(payload.get('outputs', {}))
        == len(EXPECTED_OPTIMIZATION_ABLATIONS),
        label, 'expected all 13 paired optimizer ablations',
    )
    validate_artifact_record(
        label, payload.get('artifacts', {}).get('ablation_runner', {})
    )
    validate_artifact_record(label, payload.get('before_metrics_artifact', {}))
    summaries = {}
    for ablation in payload['ablations']:
        name = ablation['label']
        record = payload['outputs'].get(name, {})
        for artifact_name in (
            'optimizer_completion_artifact', 'after_metrics_artifact',
            'paired_comparison_artifact',
        ):
            validate_artifact_record(
                f'{label}/{name}/{artifact_name}',
                record.get(artifact_name, {}),
            )
        completion = json.loads(Path(
            record['optimizer_completion_artifact']['path']
        ).read_text(encoding='utf-8'))
        after_metrics = json.loads(Path(
            record['after_metrics_artifact']['path']
        ).read_text(encoding='utf-8'))
        paired = json.loads(Path(
            record['paired_comparison_artifact']['path']
        ).read_text(encoding='utf-8'))
        runtime = completion.get('runtime', {})
        require_artifact(
            completion.get('schema') == 'hhc_paper_contact_optimization_v1'
            and completion.get('complete') is True
            and completion.get('completed_samples') == expected_samples
            and completion.get('requested_samples') == expected_samples
            and runtime.get('samples') == expected_samples
            and math.isfinite(runtime.get(
                'optimizer_wall_seconds_mean_per_sample', math.nan
            )),
            label, f'{name} completion/runtime is invalid',
        )
        require_artifact(
            after_metrics.get('schema')
            == 'hhc_volumetric_smpl_physical_metrics_v1'
            and after_metrics.get('sample_count') == expected_samples
            and paired.get('schema') == 'hhc_paired_physical_comparison_v1'
            and paired.get('sample_count') == expected_samples
            and paired.get('bootstrap_samples') == 10000,
            label, f'{name} physical comparison is invalid',
        )
        expected_metrics = {
            'penetrating_vertex_fraction', 'collision_frame_rate',
            'penetration_depth_sum_m_per_frame',
            'paper_eq6_squared_sum_m2_per_frame',
            'mean_depth_m_on_penetrating_vertices', 'max_depth_m',
            'mean_foot_sliding_m_per_frame',
        }
        require_artifact(
            set(paired.get('comparisons', {})) == expected_metrics,
            label, f'{name} paired metric set changed',
        )
        for metric_name, metric in paired['comparisons'].items():
            interval = metric.get('paired_delta_bootstrap_95ci', [])
            numeric_values = [
                metric.get('before_mean'), metric.get('after_mean'),
                metric.get('paired_delta_after_minus_before_mean'),
                metric.get('paired_delta_median'),
                metric.get('wilcoxon_two_sided_p'), *interval,
            ]
            require_artifact(
                metric.get('direction') == 'lower_is_better'
                and len(interval) == 2
                and all(isinstance(value, (int, float))
                        and math.isfinite(value) for value in numeric_values)
                and 0 <= metric['wilcoxon_two_sided_p'] <= 1,
                label, f'{name}/{metric_name} paired statistics are invalid',
            )
        summaries[name] = {
            'configuration': ablation,
            'runtime': runtime,
            'comparisons': paired['comparisons'],
        }
    return summaries


def validate_repository_record(label, name, record):
    """Validate recorded Git identity without tying it to today's worktree.

    Dirty-tree hashes are historical provenance and remain the equality key
    between training and generation.  Requiring the live worktree to stay
    frozen would invalidate an immutable result after any unrelated edit.
    Exact executable bytes are enforced separately through content-addressed
    implementation snapshots.
    """
    repository = Path(record.get('path', ''))
    require_artifact(repository.is_dir(), label,
                     f'missing provenance repository {name}: {repository}')
    revision = record.get('revision', '')
    require_artifact(
        re.fullmatch(r'[0-9a-f]{40}', revision) is not None,
        label, f'{name} repository revision is malformed',
    )
    commit_exists = subprocess.run(
        ['git', '-C', str(repository), 'cat-file', '-e', f'{revision}^{{commit}}'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    require_artifact(commit_exists, label,
                     f'{name} recorded revision is unavailable')
    require_artifact(
        re.fullmatch(
            r'[0-9a-f]{64}', record.get('tracked_diff_sha256', '')
        ) is not None,
        label, f'{name} historical tracked-diff hash is malformed',
    )


def validate_method_provenance(label, manifest):
    provenance = manifest.get('provenance')
    require_artifact(isinstance(provenance, dict), label,
                     'run manifest has no strict provenance block')
    for group in ('artifacts', 'implementation'):
        records = provenance.get(group)
        require_artifact(isinstance(records, dict) and records, label,
                         f'run provenance has no {group} records')
        for name, record in records.items():
            require_artifact(isinstance(record, dict), label,
                             f'invalid {group} record {name}')
            validate_artifact_record(label, record)
    if label in FINAL_METHOD_LABELS:
        require_artifact(
            provenance.get('source_validation_contract')
            == 'fair_interx_content_addressed_implementation_snapshots_v1',
            label, 'final method has no immutable implementation snapshots',
        )
        for name, record in provenance['implementation'].items():
            snapshot_path = Path(record.get('path', ''))
            require_artifact(
                record.get('content_addressed_snapshot') is True
                and bool(record.get('source_path_at_execution'))
                and snapshot_path.name.startswith(
                    record.get('sha256', '') + '__'
                ),
                label, f'implementation {name} is not content-addressed',
            )
        require_artifact(
            manifest.get(
                'implementation_source_stability_verified_at_completion'
            ) is True
            or provenance.get('implementation_snapshot_freeze_mode')
            == 'posthoc_exact_recorded_hash_only',
            label,
            'implementation sources were neither completion-verified nor '
            'honestly marked as an exact-hash posthoc freeze',
        )
    repositories = provenance.get('repositories')
    require_artifact(isinstance(repositories, dict) and repositories, label,
                     'run provenance has no repository records')
    for name, record in repositories.items():
        require_artifact(isinstance(record, dict), label,
                         f'invalid repository record {name}')
        validate_repository_record(label, name, record)


def validate_training_repository_matches_generation(
        label, training_manifest, provenance, repository_name):
    """Reject generation from a tracked source tree different from training."""
    training_repository = training_manifest.get('repository', {})
    generation_repository = provenance.get('repositories', {}).get(
        repository_name, {}
    )
    require_artifact(
        training_repository.get('revision')
        == generation_repository.get('revision')
        and training_repository.get('tracked_diff_sha256')
        == generation_repository.get('tracked_diff_sha256'),
        label,
        f'{repository_name} tracked source differs between training and generation',
    )


def same_resolved_path(left, right):
    return bool(left and right) and Path(left).resolve() == Path(right).resolve()


def complete_wandb_norm_telemetry(payload, expected_final_step):
    """Require the actual gradient/weight streams, not only an audit flag."""
    metrics = payload.get('metrics', {})
    required = (
        'train/gradient_l2_norm', 'train/gradient_max_abs',
        'train/weight_l2_norm', 'train/weight_max_abs',
        'train/nonfinite_gradients', 'train/nonfinite_weights',
    )
    for name in required:
        summary = metrics.get(name, {})
        values = (summary.get('minimum'), summary.get('maximum'))
        if (
            not isinstance(summary.get('count'), int)
            or summary['count'] <= 0
            or not isinstance(summary.get('last_step'), int)
            or summary['last_step'] < expected_final_step
            or not all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in values
            )
        ):
            return False
    return (
        metrics['train/nonfinite_gradients']['minimum'] == 0
        and metrics['train/nonfinite_gradients']['maximum'] == 0
        and metrics['train/nonfinite_weights']['minimum'] == 0
        and metrics['train/nonfinite_weights']['maximum'] == 0
    )


def validate_generalization_manifest_contract(label, manifest):
    match = re.fullmatch(
        r'hhc_generalization_(shape|action)_(test_unseen|test_seen)', label
    )
    require_artifact(match is not None, label,
                     'invalid held-out generalization label')
    experiment, partition = match.groups()
    require_artifact(
        manifest.get('samples_per_replication')
        == GENERALIZATION_SAMPLES[f'{experiment}/{partition}']
        and same_resolved_path(
            manifest.get('requests_dir'),
            f'output/fair_interx_generalization_requests/'
            f'{experiment}/{partition}',
        )
        and same_resolved_path(
            manifest.get('requests_manifest'),
            'output/fair_interx_generalization_requests/manifest.json',
        )
        and same_resolved_path(
            manifest.get('hhc_stats_dir'),
            f'configs/interx_generalization/{experiment}/stats',
        )
        and manifest.get('hhc_stats_prefix')
        == f'generalization_{experiment}'
        and same_resolved_path(
            manifest.get('selection_manifest'),
            f'output/hhc_mgt_generalization_{experiment}_selection.json',
        )
        and manifest.get('sampling_steps') == 50
        and manifest.get('contact_guidance') is True,
        label,
        'held-out request/stats/selection/common50 contract changed',
    )


def validate_method_checkpoint_bindings(
        method_manifests, records, generalization_manifests=None):
    """Bind every retrained method's generation to its audited checkpoint."""
    selection = records['hhc_mgt_final_selection']['payload']
    for label in HHC_FORMAL_LABELS:
        transformer = method_manifests[label]['provenance']['artifacts'][
            'transformer'
        ]
        vq = method_manifests[label]['provenance']['artifacts']['vq']
        require_artifact(
            same_resolved_path(
                transformer.get('path'), selection.get('selected_checkpoint')
            )
            and transformer.get('sha256')
            == selection.get('selected_checkpoint_sha256'),
            label,
            'generation checkpoint differs from validation-FID selection',
        )
        vq_audit = records['vq_import_manifest']['payload']['checkpoint']
        require_artifact(
            same_resolved_path(vq.get('path'), vq_audit.get('path'))
            and vq.get('sha256') == vq_audit.get('sha256'),
            label, 'generation VQ differs from imported raw12288 checkpoint',
        )

    for label, artifact_name, audit_label in (
        (
            COSHMDM_LABEL, 'model',
            'coshmdm_final_state_audit',
        ),
    ):
        generated_with = method_manifests[label]['provenance']['artifacts'][
            artifact_name
        ]
        audit = records[audit_label]['payload']
        require_artifact(
            same_resolved_path(
                generated_with.get('path'), audit.get('checkpoint')
            )
            and generated_with.get('sha256') == audit.get('checkpoint_sha256'),
            label,
            f'generation checkpoint differs from {audit_label}',
        )

    timotion_label = 'timotion_transformer_lpa_interx_retrained_fair'
    timotion_generated_with = method_manifests[timotion_label][
        'provenance'
    ]['artifacts']['model']
    timotion_generated_selection = method_manifests[timotion_label][
        'provenance'
    ]['artifacts'].get('validation_selection_manifest', {})
    timotion_selection = records['timotion_validation_selection']['payload']
    timotion_selection_artifact = records[
        'timotion_validation_selection'
    ].get('artifact', {})
    require_artifact(
        same_resolved_path(
            timotion_generated_with.get('path'),
            timotion_selection.get('selected_checkpoint'),
        )
        and timotion_generated_with.get('sha256')
        == timotion_selection.get('selected_checkpoint_sha256'),
        timotion_label,
        'generation checkpoint differs from held-out validation-FID selection',
    )
    require_artifact(
        same_resolved_path(
            timotion_generated_selection.get('path'),
            timotion_selection_artifact.get('path'),
        )
        and timotion_generated_selection.get('sha256')
        == timotion_selection_artifact.get('sha256'),
        timotion_label,
        'generation manifest does not bind the accepted validation selection',
    )

    if generalization_manifests is not None:
        for experiment in ('shape', 'action'):
            selection = records[
                f'mgt_generalization_{experiment}_selection'
            ]['payload']
            vq_audit = records[
                f'vq_generalization_{experiment}_state_audit'
            ]['payload']
            for partition in ('test_unseen', 'test_seen'):
                label = f'hhc_generalization_{experiment}_{partition}'
                artifacts = generalization_manifests[label]['provenance'][
                    'artifacts'
                ]
                require_artifact(
                    same_resolved_path(
                        artifacts['transformer'].get('path'),
                        selection.get('selected_checkpoint'),
                    )
                    and artifacts['transformer'].get('sha256')
                    == selection.get('selected_checkpoint_sha256'),
                    label,
                    'generation checkpoint differs from held-out validation '
                    'selection',
                )
                require_artifact(
                    same_resolved_path(
                        artifacts['vq'].get('path'),
                        vq_audit.get('checkpoint'),
                    )
                    and artifacts['vq'].get('sha256')
                    == vq_audit.get('checkpoint_sha256'),
                    label,
                    'generation VQ differs from held-out 150k VQ audit',
                )
def validate_method_manifest(label, manifest, expected_samples=1696):
    require_artifact(manifest.get('label') == label, label,
                     'run-manifest label mismatch')
    require_artifact(manifest.get('replication_range') == [0, 20], label,
                     'run manifest does not cover replications 0..19')
    require_artifact(
        manifest.get('samples_per_replication') == expected_samples, label,
        f'run manifest does not bind {expected_samples} samples per replication',
    )
    method = manifest.get('method')
    if method in {'hhc', 'intermask'}:
        require_artifact(manifest.get('sampling_steps') == 50, label,
                         'common iterative sampling budget is not 50')
    if method == 'intermask':
        require_artifact(manifest.get('intermask_length_policy') == 'exact', label,
                         'InterMask fair-table length policy is not exact')
        if label in FINAL_METHOD_LABELS:
            artifacts = manifest.get('provenance', {}).get('artifacts', {})
            require_artifact(
                artifacts.get('transformer', {}).get('sha256')
                == '0e2cfdf84cd10f9e2303c041d16915840a30c2a952b8872a5b90d7bd13d7df44'
                and artifacts.get('vq', {}).get('sha256')
                == 'a26060f49f3c905ef0bf51e168aef09382285a656855b977e7f81acf91cd5e97',
                label, 'InterMask author checkpoint hashes changed',
            )
    elif method == 'hhc':
        require_artifact(manifest.get('selection_manifest'), label,
                         'HHC run is not bound to checkpoint selection')
        formal_hhc = label in HHC_FORMAL_LABELS
        generalization_hhc = label.startswith('hhc_generalization_')
        if generalization_hhc:
            validate_generalization_manifest_contract(label, manifest)
        if formal_hhc:
            require_artifact(
                manifest.get('hhc_vq_contract') == 'precleanup_raw12288'
                and manifest.get('hhc_contact_map_contract')
                == 'paper_formula_train_only'
                and 'vq_training_manifest'
                in manifest.get('provenance', {}).get('artifacts', {}),
                label,
                'formal HHC run does not bind the imported raw12288 VQ manifest',
            )
            if label == HHC_NO_TIMING_LABEL:
                require_artifact(
                    manifest.get('contact_guidance') is False
                    and manifest.get('contact_timing_source') == 'none'
                    and manifest.get('contact_timing_source_requested') == 'none'
                    and manifest.get('contact_timing_file') is None,
                    label,
                    'no-timing HHC row requested or consumed contact timing',
                )
            else:
                require_artifact(
                    manifest.get('contact_guidance') is True
                    and manifest.get('contact_timing_source') == 'request_oracle'
                    and manifest.get('contact_timing_source_requested')
                    == 'request_oracle',
                    label,
                    'oracle-timing HHC row does not disclose frozen GT timing',
                )
        if formal_hhc or generalization_hhc:
            training_record = manifest.get('provenance', {}).get(
                'artifacts', {}
            ).get('mgt_training_manifest', {})
            training_path = validate_artifact_record(label, training_record)
            training = json.loads(training_path.read_text(encoding='utf-8'))
            validate_training_repository_matches_generation(
                label, training, manifest.get('provenance', {}), 'hhc'
            )
    elif method == 'coshmdm':
        cosh_artifacts = manifest.get('provenance', {}).get('artifacts', {})
        require_artifact(
            manifest.get('coshmdm_length_policy')
            == 'exact_request_length_grouped'
            and manifest.get('coshmdm_sampling_process')
            == 'DDIM50_eta0_CFG3.5',
            label, 'CoShMDM request-length or inference policy changed',
        )
        training_record = manifest.get('provenance', {}).get(
            'artifacts', {}
        ).get('training_manifest', {})
        training_path = validate_artifact_record(label, training_record)
        training = json.loads(training_path.read_text(encoding='utf-8'))
        require_artifact(
            training.get('training', {}).get('epochs') == 2000
            and training.get('training', {}).get(
                'expected_optimizer_updates'
            ) == 568000
            and training.get('model', {}).get('sampling_strategy') == 'ddim50'
            and training.get('model', {}).get('cfg_weight') == 3.5,
            label,
            'CoShMDM source manifest must retain the original 2000-epoch/'
            '568k author target and DDIM50/CFG3.5 inference contract',
        )
        cap_path = validate_artifact_record(
            label, cosh_artifacts.get('user_training_budget_cap', {})
        )
        preservation_path = validate_artifact_record(
            label, cosh_artifacts.get('milestone_preservation_manifest', {})
        )
        selection_path = validate_artifact_record(
            label, cosh_artifacts.get('validation_selection_manifest', {})
        )
        cap = json.loads(cap_path.read_text(encoding='utf-8'))
        preservation = json.loads(
            preservation_path.read_text(encoding='utf-8')
        )
        capped_selection = json.loads(
            selection_path.read_text(encoding='utf-8')
        )
        generated_model = cosh_artifacts.get('model', {})
        require_artifact(
            cap.get('schema') == 'hhc_coshmdm_user_training_cap_v1'
            and cap.get('status') in {
                'stopped_gracefully', 'sigterm_sent_after_180s_grace',
                'trainer_exited_before_monitor_signal',
            }
            and cap.get('preferred_terminal_optimizer_step') == 200000
            and preservation.get('schema')
            == 'hhc_coshmdm_milestone_preservation_v1'
            and preservation.get('status') == 'trainer_exited'
            and preservation.get('maximum_preserved_step') == 200000
            and preservation.get('interval_steps') == 20000
            and all(
                record.get('same_inode') is True
                and int(record.get('step', 0)) % 20000 == 0
                and 0 < int(record.get('step', 0)) <= 200000
                for record in preservation.get('preserved', [])
            )
            and capped_selection.get('schema')
            == 'hhc_coshmdm_capped_validation_selection_v1'
            and capped_selection.get('status') == 'complete'
            and capped_selection.get('test_metrics_used_for_selection') is False
            and capped_selection.get(
                'formal_hhc_validation_record_set_exact'
            ) is True
            and len(capped_selection.get(
                'formal_hhc_validation_selection_sha256', ''
            )) == 64
            and same_resolved_path(
                generated_model.get('path'),
                capped_selection.get('selected_checkpoint'),
            )
            and generated_model.get('sha256')
            == capped_selection.get('selected_checkpoint_sha256'),
            label,
            'CoShMDM generation is not bound to the disclosed two-day cap '
            'and held-out-validation checkpoint selection',
        )
        validate_artifact_record(
            f'{label}/formal_hhc_validation_selection',
            capped_selection.get('artifacts', {}).get(
                'formal_hhc_validation_selection', {}
            ),
        )
        generation_audit_path = validate_artifact_record(
            label, cosh_artifacts.get('generation_contract_audit', {})
        )
        generation_audit = json.loads(
            generation_audit_path.read_text(encoding='utf-8')
        )
        require_artifact(
            generation_audit.get('schema')
            == 'hhc_coshmdm_generation_contract_audit_v1'
            and generation_audit.get('status') == 'pass'
            and all(generation_audit.get('checks', {}).values())
            and generation_audit.get('sampling', {}).get('process')
            == 'DDIM50_eta0_CFG3.5'
            and generation_audit.get('sampling', {}).get(
                'runtime_sampling_steps'
            ) == 50
            and generation_audit.get('sampling', {}).get('ddim_eta') == 0.0
            and generation_audit.get('requests', {}).get('replications') == 20
            and generation_audit.get('requests', {}).get(
                'samples_per_replication'
            ) == 1696
            and generation_audit.get('requests', {}).get(
                'verified_file_hashes'
            ) == 20
            and generation_audit.get('representation', {}).get(
                'gt_motion_consumed_by_exporter'
            ) is False,
            label, 'CoShMDM generation contract audit is incomplete',
        )
        cosh_implementation = manifest.get('provenance', {}).get(
            'implementation', {}
        )
        source_mapping = {
            'driver': 'driver',
            'scorer': 'semantic_scorer',
            'exporter': 'exporter',
            'adapter': 'official_adapter',
            'model': 'official_model',
            'diffusion': 'official_diffusion',
            'gaussian_diffusion': 'gaussian_diffusion',
            'normalizer': 'normalizer',
        }
        require_artifact(
            all(
                generation_audit.get('sources', {}).get(source, {}).get('sha256')
                == cosh_implementation.get(implementation, {}).get('sha256')
                for source, implementation in source_mapping.items()
            )
            and generation_audit.get('implementation', {}).get('sha256')
            == cosh_implementation.get(
                'generation_contract_auditor', {}
            ).get('sha256')
            and generation_audit.get('sources', {}).get(
                'model_config', {}
            ).get('sha256') == cosh_artifacts.get('model_config', {}).get('sha256')
            and generation_audit.get('sources', {}).get(
                'normalization_mean', {}
            ).get('sha256')
            == cosh_artifacts.get('normalization_mean', {}).get('sha256')
            and generation_audit.get('sources', {}).get(
                'normalization_std', {}
            ).get('sha256')
            == cosh_artifacts.get('normalization_std', {}).get('sha256')
            and generation_audit.get('sources', {}).get(
                'requests_manifest', {}
            ).get('sha256')
            == cosh_artifacts.get('requests_manifest', {}).get('sha256'),
            label,
            'CoShMDM generation audit does not bind the executed sources/assets',
        )
        if label in FINAL_METHOD_LABELS:
            validate_training_repository_matches_generation(
                label, training, manifest.get('provenance', {}), 'coshmdm'
            )
    elif method == 'interact2ar':
        require_artifact(
            manifest.get('interact2ar_first_frame_conditioning') == 'none'
            and manifest.get('interact2ar_length_policy')
            == 'exact_request_length_grouped',
            label, 'Interact2AR fair path must disable GT first frame',
        )
        if label in FINAL_METHOD_LABELS:
            artifacts = manifest.get('provenance', {}).get('artifacts', {})
            checkpoint_manifest_path = validate_artifact_record(
                label, artifacts.get('author_checkpoint_manifest', {})
            )
            checkpoint_manifest = json.loads(
                checkpoint_manifest_path.read_text(encoding='utf-8')
            )
            models = checkpoint_manifest.get('models', [])
            author_model = next(
                (record for record in models
                 if record.get('name') == 'interact2ar_mixed_memory'), None
            )
            require_artifact(
                author_model is not None
                and author_model.get('sha256')
                == artifacts.get('model', {}).get('sha256')
                and author_model.get('metadata', {}).get(
                    'checkpoint_global_step'
                ) == 350000,
                label,
                'Interact2AR generation does not use the pinned 350k '
                'mixed-memory author checkpoint',
            )
    elif method == 'timotion':
        require_artifact(
            manifest.get('timotion_first_frame_conditioning') == 'none'
            and manifest.get('timotion_sampling_process')
            == 'DDIM50_eta0_CFG3.5'
            and manifest.get('timotion_length_policy')
            == 'exact_request_length_grouped',
            label, 'TiMotion fair inference contract changed',
        )
        if label in FINAL_METHOD_LABELS:
            timotion_artifacts = manifest.get('provenance', {}).get(
                'artifacts', {}
            )
            training_path = validate_artifact_record(
                label, timotion_artifacts.get('training_manifest', {})
            )
            training = json.loads(training_path.read_text(encoding='utf-8'))
            validate_training_repository_matches_generation(
                label, training, manifest.get('provenance', {}), 'timotion'
            )
            generation_audit_path = validate_artifact_record(
                label, timotion_artifacts.get(
                    'generation_contract_audit', {}
                )
            )
            generation_audit = json.loads(
                generation_audit_path.read_text(encoding='utf-8')
            )
            require_artifact(
                generation_audit.get('schema')
                == 'hhc_timotion_generation_contract_audit_v1'
                and generation_audit.get('status') == 'pass'
                and all(generation_audit.get('checks', {}).values())
                and generation_audit.get('sampling', {}).get('process')
                == 'DDIM50_eta0_CFG3.5'
                and generation_audit.get('sampling', {}).get(
                    'runtime_sampling_steps'
                ) == 50
                and generation_audit.get('sampling', {}).get('ddim_eta') == 0.0
                and generation_audit.get('requests', {}).get(
                    'replications'
                ) == 20
                and generation_audit.get('requests', {}).get(
                    'samples_per_replication'
                ) == 1696
                and generation_audit.get('requests', {}).get(
                    'verified_file_hashes'
                ) == 20
                and generation_audit.get('representation', {}).get(
                    'gt_motion_consumed_by_exporter'
                ) is False
                and generation_audit.get('representation', {}).get(
                    'gt_first_frame_consumed_by_exporter'
                ) is False,
                label, 'TiMotion generation contract audit is incomplete',
            )
            timotion_implementation = manifest.get('provenance', {}).get(
                'implementation', {}
            )
            source_mapping = {
                'driver': 'driver',
                'scorer': 'semantic_scorer',
                'exporter': 'exporter',
                'model': 'official_model',
                'diffusion': 'official_denoiser',
                'interaction_block': 'official_interaction_block',
                'gaussian_diffusion': 'gaussian_diffusion',
            }
            require_artifact(
                all(
                    generation_audit.get('sources', {}).get(
                        source, {}
                    ).get('sha256')
                    == timotion_implementation.get(
                        implementation, {}
                    ).get('sha256')
                    for source, implementation in source_mapping.items()
                )
                and generation_audit.get('implementation', {}).get('sha256')
                == timotion_implementation.get(
                    'generation_contract_auditor', {}
                ).get('sha256')
                and generation_audit.get('sources', {}).get(
                    'model_config', {}
                ).get('sha256')
                == timotion_artifacts.get('model_config', {}).get('sha256')
                and generation_audit.get('sources', {}).get(
                    'normalization_mean', {}
                ).get('sha256')
                == timotion_artifacts.get(
                    'normalization_mean', {}
                ).get('sha256')
                and generation_audit.get('sources', {}).get(
                    'normalization_std', {}
                ).get('sha256')
                == timotion_artifacts.get(
                    'normalization_std', {}
                ).get('sha256')
                and generation_audit.get('sources', {}).get(
                    'requests_manifest', {}
                ).get('sha256')
                == timotion_artifacts.get(
                    'requests_manifest', {}
                ).get('sha256'),
                label,
                'TiMotion generation audit does not bind executed sources/assets',
            )
    else:
        raise ValueError(f'{label}: unsupported method in run manifest: {method}')
    if label in FINAL_METHOD_LABELS:
        validate_method_provenance(label, manifest)


def validate_mgt_shape_dependence(records):
    """Validate paired train/validation shape-shortcut evidence."""
    expected_shape_conditions = {
        'requested', 'permuted_beta_and_gender',
        'zero_beta_requested_gender',
    }
    selection = records['hhc_mgt_final_selection']['payload']
    for split_role in ('training', 'validation'):
        label = f'hhc_mgt_shape_dependence_{split_role}'
        payload = records[label]['payload']
        aggregate = payload.get('metrics', {}).get('aggregate', {})
        require_artifact(
            payload.get('schema')
            == 'hhc_mgt_shape_condition_dependence_audit_v1'
            and payload.get('status') == 'complete'
            and payload.get('split_role') == split_role
            and payload.get('samples') == 256
            and payload.get('sample_selection')
            == 'uniform_without_replacement_over_active_split'
            and payload.get('shape_derangement', {}).get('fixed_points') == 0
            and payload.get('shape_derangement', {}).get(
                'same_shape_signature_pairs'
            ) == 0
            and payload.get('shape_derangement', {}).get(
                'unique_shape_signatures', 0
            ) > 1
            and payload.get('shape_derangement', {}).get('mapping_sha256')
            and payload.get('checkpoint', {}).get('sha256')
            == selection.get('selected_checkpoint_sha256')
            and payload.get('implementation', {}).get('sha256')
            == sha256_file('tools/audit_mgt_shape_condition_dependence.py')
            and set(aggregate) == expected_shape_conditions
            and all(
                isinstance(row.get('tokens'), int) and row['tokens'] > 0
                and math.isfinite(float(row.get('cross_entropy', math.nan)))
                and math.isfinite(float(row.get('accuracy_percent', math.nan)))
                for row in aggregate.values()
            ),
            label,
            'MGT shape-dependence audit is incomplete or not bound to the '
            'selected paper checkpoint',
        )
        for artifact_name in ('checkpoint', 'training_manifest', 'implementation'):
            validate_artifact_record(
                f'{label}/{artifact_name}', payload.get(artifact_name, {})
            )

    shape_pair = records['hhc_mgt_shape_dependence_pair']['payload']
    require_artifact(
        shape_pair.get('schema')
        == 'hhc_mgt_shape_condition_dependence_pair_v1'
        and shape_pair.get('status') == 'pass'
        and shape_pair.get('decision') in {
            'train_only_shape_identity_shortcut_supported',
            'train_only_shape_identity_shortcut_not_established',
        }
        and shape_pair.get('thresholds_preregistered_before_execution') == {
            'minimum_train_requested_shape_advantage_percent_points': 5.0,
            'minimum_train_minus_validation_advantage_percent_points': 3.0,
        }
        and shape_pair.get('checkpoint', {}).get('sha256')
        == selection.get('selected_checkpoint_sha256')
        and shape_pair.get('implementation', {}).get('sha256')
        == sha256_file('tools/summarize_mgt_shape_condition_dependence.py'),
        'hhc_mgt_shape_dependence_pair',
        'paired train/validation shape-dependence decision is incomplete or '
        'uses a different checkpoint/threshold',
    )
    validate_artifact_record(
        'hhc_mgt_shape_dependence_pair/implementation',
        shape_pair.get('implementation', {}),
    )
    for split_role in ('training', 'validation'):
        bound = shape_pair.get('inputs', {}).get(split_role, {})
        validate_artifact_record(
            f'hhc_mgt_shape_dependence_pair/{split_role}', bound
        )
        require_artifact(
            bound.get('sha256')
            == records[f'hhc_mgt_shape_dependence_{split_role}'][
                'artifact'
            ]['sha256'],
            'hhc_mgt_shape_dependence_pair',
            f'paired shape diagnosis does not bind the {split_role} audit',
        )


def validate_post_50k_training_dispatch(records):
    """Require a final, evidence-bound action after the weak-FID diagnosis."""
    assessment_record = records['hhc_global_masked_loss_control_assessment']
    assessment = assessment_record['payload']
    dispatch = records['hhc_post_50k_training_dispatch']['payload']
    allowed = {
        'retain_formal_contract_and_continue_shape_training': (
            'continue_shape_augmentation', False
        ),
        'promote_global_masked_loss_to_full_retrain_before_shape': (
            'continue_global_control_to_300k_then_shape', True
        ),
        'global_loss_not_sufficient_investigate_mask_conditioning_next': (
            'continue_shape_augmentation', False
        ),
    }
    source_decision = dispatch.get('source_decision')
    expected = allowed.get(source_decision)
    require_artifact(
        assessment.get('schema')
        == 'hhc_global_masked_loss_control_assessment_v1'
        and assessment.get('status') == 'complete'
        and assessment.get('decision') == source_decision
        and expected is not None
        and dispatch.get('schema') == 'hhc_post_50k_training_dispatch_v1'
        and dispatch.get('status') == 'ready'
        and dispatch.get('action') == expected[0]
        and dispatch.get('continue_global_control') is expected[1]
        and dispatch.get('shape_training_loss_contract')
        == 'paper_masked_only'
        and dispatch.get('shape_training_budget') == 300000
        and dispatch.get('global_control_training_budget') == 300000,
        'hhc_post_50k_training_dispatch',
        'weak-FID diagnosis has no final evidence-driven training action',
    )
    for label in ('implementation',):
        validate_artifact_record(
            f'hhc_post_50k_training_dispatch/{label}', dispatch.get(label, {})
        )
    require_artifact(
        dispatch.get('implementation', {}).get('sha256')
        == sha256_file('tools/dispatch_hhc_post_50k_training.py')
        and dispatch.get('evidence', {}).get(
            'masked_loss_assessment', {}
        ).get('sha256') == assessment_record['artifact']['sha256'],
        'hhc_post_50k_training_dispatch',
        'dispatch implementation or loss-assessment binding changed',
    )
    if source_decision == (
        'global_loss_not_sufficient_investigate_mask_conditioning_next'
    ):
        shape_source_decision = dispatch.get('shape_source_decision')
        shortcut_supported = (
            shape_source_decision
            == 'train_only_shape_identity_shortcut_supported'
        )
        unresolved_but_continuing = (
            shape_source_decision
            == 'train_only_shape_identity_shortcut_not_established'
            and dispatch.get('mask_conditioning_status')
            == 'unresolved_after_two_negative_diagnostics'
            and dispatch.get('diagnostic_followup_required') is True
        )
        require_artifact(
            (shortcut_supported or unresolved_but_continuing)
            and dispatch.get('evidence', {}).get(
                'shape_dependence_pair', {}
            ).get('sha256')
            == records['hhc_mgt_shape_dependence_pair']['artifact']['sha256'],
            'hhc_post_50k_training_dispatch',
            'negative loss control was released without a bound shape '
            'diagnosis or an honest unresolved-status marker',
        )


def validate_promoted_global_masked_loss_results(records):
    """Require the actual 300k/W&B/common-50 evidence when 50k is promoted."""
    dispatch_record = records['hhc_post_50k_training_dispatch']
    dispatch = dispatch_record['payload']
    if dispatch.get('action') != 'continue_global_control_to_300k_then_shape':
        return
    label = 'hhc_global_masked_loss_control_300k_results'
    result = records[label]['payload']
    require_artifact(
        result.get('schema')
        == 'hhc_global_masked_loss_control_300k_results_v1'
        and result.get('status') == 'complete'
        and result.get('training_loss_contract')
        == 'paper_masked_global_token_mean'
        and result.get('training_budget') == 300000
        and result.get('dispatch', {}).get('sha256')
        == dispatch_record['artifact']['sha256']
        and result.get('implementation', {}).get('sha256')
        == sha256_file(
            'tools/build_hhc_global_masked_loss_promotion_results.py'
        ),
        label,
        'promoted 300k result is absent, stale, or not dispatch-bound',
    )
    for artifact_name in (
        'dispatch', 'selection', 'training_manifest',
        'wandb_history_audit', 'implementation',
    ):
        validate_artifact_record(
            f'{label}/{artifact_name}', result.get(artifact_name, {})
        )
    selection = json.loads(Path(
        result['selection']['path']
    ).read_text(encoding='utf-8'))
    training = json.loads(Path(
        result['training_manifest']['path']
    ).read_text(encoding='utf-8'))
    wandb = json.loads(Path(
        result['wandb_history_audit']['path']
    ).read_text(encoding='utf-8'))
    observability = training.get('observability', {})
    require_artifact(
        selection.get('run_complete') is True
        and selection.get('expected_final_step') == 300000
        and selection.get('test_metrics_used_for_selection') is False
        and selection.get('training_manifest_sha256')
        == result['training_manifest']['sha256']
        and training.get('resolved_arguments', {}).get(
            'training_loss_contract'
        ) == 'paper_masked_global_token_mean'
        and observability.get('wandb_mode') == 'online'
        and wandb.get('status') == 'pass'
        and wandb.get('remote_state') == 'finished'
        and wandb.get('expected_final_step') == 300000
        and wandb.get('run_id') == observability.get('wandb_run_id')
        and all(wandb.get('checks', {}).values())
        and wandb.get('checks', {}).get(
            'all_training_metrics_reach_expected_step'
        ) is True
        and wandb.get('checks', {}).get(
            'all_validation_metrics_follow_final_training_event'
        ) is True
        and wandb.get('checks', {}).get('nonfinite_counts_zero') is True
        and wandb.get('optimizer_step_axis') == 'train/optimizer_step'
        and wandb.get('implementation', {}).get('sha256') == sha256_file(
            'tools/audit_wandb_online_metric_history.py'
        )
        and complete_wandb_norm_telemetry(wandb, 300000)
        and wandb.get('metrics', {}).get(
            'train/nonfinite_gradients', {}
        ).get('maximum') == 0
        and wandb.get('metrics', {}).get(
            'train/nonfinite_weights', {}
        ).get('maximum') == 0,
        label,
        'selection, loss contract, or online W&B completion changed',
    )
    expected_labels = {
        'no_contact':
            'hhc_global_masked_loss_control_300k_common50_no_contact',
        'contact': 'hhc_global_masked_loss_control_300k_common50_contact',
    }
    require_artifact(
        set(result.get('methods', {})) == set(expected_labels),
        label, 'promoted method pair is incomplete',
    )
    for guidance, expected_label in expected_labels.items():
        method = result['methods'][guidance]
        artifacts = method.get('artifacts', {})
        require_artifact(
            method.get('label') == expected_label
            and set(artifacts) == {
                'run_manifest', 'summary', 'multimodality',
                'physical_replication_0', 'generation_timing',
            },
            label, f'{guidance} artifact set changed',
        )
        loaded = {}
        for artifact_name, artifact_record in artifacts.items():
            path = validate_artifact_record(
                f'{label}/{guidance}/{artifact_name}', artifact_record
            )
            loaded[artifact_name] = json.loads(
                path.read_text(encoding='utf-8')
            )
        validate_method_manifest(expected_label, loaded['run_manifest'])
        validate_method_artifacts(
            expected_label, loaded['summary'], loaded['multimodality'],
            loaded['physical_replication_0'], loaded['generation_timing'],
        )
        generated = loaded['run_manifest'].get(
            'provenance', {}
        ).get('artifacts', {})
        require_artifact(
            same_resolved_path(
                generated.get('transformer', {}).get('path'),
                selection.get('selected_checkpoint'),
            )
            and generated.get('transformer', {}).get('sha256')
            == selection.get('selected_checkpoint_sha256')
            and generated.get('mgt_selection_manifest', {}).get('sha256')
            == result['selection']['sha256']
            and generated.get('mgt_training_manifest', {}).get('sha256')
            == result['training_manifest']['sha256']
            and loaded['run_manifest'].get('contact_guidance')
            is (guidance == 'contact'),
            label, f'{guidance} generation is not selection-bound',
        )
        for metric_name, metric_record in method.get('metrics', {}).items():
            summary_metric = loaded['summary']['metrics'][metric_name]
            require_artifact(
                float(metric_record.get('mean', math.nan))
                == float(summary_metric['mean'])
                and float(metric_record.get('ci95', math.nan))
                == float(summary_metric['ci95']),
                label, f'{guidance}/{metric_name} aggregate changed',
            )


def validate_training_and_protocol(records):
    mm_replay = records['historical_mgt_common50_mm_replay']['payload']
    repeat_records = mm_replay.get('repeat_records', [])
    require_artifact(
        mm_replay.get('schema') == 'hhc_mgt_common50_mm_replay_v1'
        and mm_replay.get('status') == 'pass'
        and mm_replay.get('samples') == 100
        and mm_replay.get('repeats') == 30
        and mm_replay.get('names_equal') is True
        and mm_replay.get('lengths_equal') is True
        and mm_replay.get('motions_byte_exact') is True
        and mm_replay.get('maximum_absolute_motion_difference') == 0
        and mm_replay.get('nonzero_motion_element_count') == 0
        and mm_replay.get('reference_motion_bytes_sha256')
        == mm_replay.get('candidate_motion_bytes_sha256')
        and mm_replay.get('conditioning_contracts') == {
            'training': 'historical_mmm_shape_dropout',
            'inference': 'paper_text_cfg',
        }
        and len(repeat_records) == 30
        and [record.get('repeat') for record in repeat_records] == list(range(30))
        and all(
            record.get('motions_byte_exact') is True
            and record.get('maximum_absolute_motion_difference') == 0
            and record.get('nonzero_motion_element_count') == 0
            and record.get('reference_motion_bytes_sha256')
            == record.get('candidate_motion_bytes_sha256')
            for record in repeat_records
        )
        and mm_replay.get('reference_exporter', {}).get('sha256')
        == '312ee55ca12fe92f29108bf513e1da9fcde394508e119cc4b29bbbf4239dd76e',
        'historical_mgt_common50_mm_replay',
        'repaired MGT does not exactly replay all 30 frozen MM repeats',
    )
    for name in (
        'reference_generations', 'candidate_generations',
        'reference_manifest', 'implementation',
    ):
        validate_artifact_record(
            f'historical_mgt_common50_mm_replay/{name}',
            mm_replay.get(name, {}),
        )

    conditional_support = records[
        'hhc_mgt_train_val_conditional_support_audit'
    ]['payload']
    conditional_marginals = conditional_support.get(
        'marginal_code_support', {}
    )
    conditional_tuples = conditional_support.get(
        'same_frame_conditional_tuple_support', {}
    )
    conditional_transitions = conditional_support.get(
        'within_part_transition_support', {}
    )
    require_artifact(
        conditional_support.get('schema')
        == 'hhc_mgt_train_val_conditional_support_audit_v1'
        and conditional_support.get('status') == 'complete'
        and conditional_support.get('loader_contract') == {
            'training_token_length_contract': 'paper_50',
            'tensor_window_including_end': 50,
            'minimum_payload_tokens': 20,
            'maximum_payload_tokens': 49,
            'filter_then_truncate': True,
        }
        and conditional_support.get('training_split', {}).get(
            'cached_samples'
        ) == 9095
        and conditional_support.get('validation_split', {}).get(
            'cached_samples'
        ) == 569
        and conditional_support.get('training_split', {}).get(
            'loaded_samples'
        ) == 8617
        and conditional_support.get('validation_split', {}).get(
            'loaded_samples'
        ) == 534
        and len(conditional_support.get('training_split', {}).get(
            'excluded_below_training_bucket', []
        )) == 478
        and len(conditional_support.get('validation_split', {}).get(
            'excluded_below_training_bucket', []
        )) == 35
        and conditional_support.get('training_split', {}).get(
            'truncated_to_payload_window'
        ) == 2618
        and conditional_support.get('validation_split', {}).get(
            'truncated_to_payload_window'
        ) == 174
        and conditional_support.get('training_split', {}).get('sha256')
        == '47c5c9a09d7293e90e127fb49ba9dc9398afbff66c8f2b45ada3ef6c4453157a'
        and conditional_support.get('validation_split', {}).get('sha256')
        == '6c77e85017b14b512f42f2d540da2ce05874d9777d707bebd85782d6221f28a8'
        and all(
            conditional_marginals.get(part, {}).get(
                'validation_occurrence_coverage', 0.0
            ) >= 0.99
            for part in ('body', 'hand', 'root')
        )
        and conditional_tuples.get('root_to_body_pair', {}).get(
            'validation_occurrence_coverage', 1.0
        ) < 0.1
        and conditional_tuples.get(
            'root_body_to_hand_triplet', {}
        ).get('validation_occurrence_coverage', 1.0) < 0.01
        and all(
            0.5 <= conditional_transitions.get(part, {}).get(
                'validation_occurrence_coverage', 0.0
            ) <= 0.8
            for part in ('body', 'hand', 'root')
        ),
        'hhc_mgt_train_val_conditional_support_audit',
        'MGT conditional-support diagnosis is incomplete or uses different data',
    )
    validate_artifact_record(
        'hhc_mgt_train_val_conditional_support_audit/implementation',
        conditional_support.get('implementation', {}),
    )

    validate_mgt_shape_dependence(records)
    validate_post_50k_training_dispatch(records)
    validate_promoted_global_masked_loss_results(records)

    coverage = records['review_requirement_coverage']['payload']
    require_artifact(
        coverage.get('schema')
        == 'hhc_review_requirement_coverage_audit_v1'
        and coverage.get('status') == 'pass'
        and coverage.get('coverage_complete') is True
        and coverage.get('requirements') == 20
        and set(coverage.get('review_sources', [])) == {
            'Meta-review', 'Reviewer L4xb',
            'Reviewer ZyzM', 'Reviewer NbvC',
        }
        and len(coverage.get('records', [])) == 20,
        'review_requirement_coverage',
        'review concerns are not completely mapped to experiments/statuses',
    )
    for name in ('source_review', 'matrix', 'plan'):
        validate_artifact_record(
            f'review_requirement_coverage/{name}', coverage.get(name, {})
        )
    require_artifact(
        coverage.get('implementation_sha256')
        == sha256_file('tools/audit_review_requirement_coverage.py'),
        'review_requirement_coverage',
        'coverage auditor implementation changed after the audit',
    )
    coverage_statuses = {
        record.get('id'): record.get('status')
        for record in coverage['records']
    }
    require_artifact(
        coverage_statuses.get('unseen_shape_quantitative')
        in {'active_queued', 'complete'}
        and coverage_statuses.get('unseen_action_quantitative')
        == 'cancelled_no_claim'
        and coverage_statuses.get('chi3d_quantitative_generalization')
        == 'deferred_after_interx'
        and coverage_statuses.get('second_generation_dataset')
        == 'deferred_after_interx'
        and coverage_statuses.get('user_study_values') == 'external_pending',
        'review_requirement_coverage',
        'user-directed experiment scope or no-claim policy changed',
    )

    launch_recovery_record = records['historical_mgt_launch_source_recovery']
    launch_recovery = launch_recovery_record['payload']
    require_artifact(
        launch_recovery.get('schema')
        == 'hhc_mgt_launch_source_recovery_v1'
        and launch_recovery.get('status') == 'pass'
        and launch_recovery.get(
            'all_snapshot_hashes_match_launch_manifest'
        ) is True
        and set(launch_recovery.get(
            'changed_sources_recovered_from_history', []
        )) == {
            'options/option_transformer.py',
            'train/train_t2i_trans.py',
        }
        and len(launch_recovery.get('reversed_patch_records', [])) == 7
        and len(launch_recovery.get('snapshots', {})) == 14,
        'historical_mgt_launch_source_recovery',
        'historical MGT launch-source recovery is incomplete',
    )
    for name in ('training_manifest', 'selected_session_records', 'implementation'):
        validate_artifact_record(
            f'historical_mgt_launch_source_recovery/{name}',
            launch_recovery.get(name, {}),
        )
    for name, record in launch_recovery['snapshots'].items():
        validate_artifact_record(
            f'historical_mgt_launch_source_recovery/source/{name}', record
        )

    repair_gate = records['mgt_repair_gate']['payload']
    repair_tokenizer = repair_gate.get('tokenizer', {})
    repair_replay = repair_gate.get('historical_replay', {})
    repair_bridge = repair_gate.get('common50_bridge', {})
    repair_bridge_runtime = repair_bridge.get('evaluator_runtime', {})
    required_repair_roles = {
        'vq_reconstruction', 'historical_training_trajectory',
        'historical_ordinary_fid', 'historical_checkpoint_pair',
        'common50_ordinary_fid', 'historical_launch_sources',
    }
    repair_evidence = repair_gate.get('evidence', [])
    repair_roles = {
        record.get('role') for record in repair_evidence
        if isinstance(record, dict)
    }
    require_artifact(
        repair_gate.get('schema') == 'hhc_mgt_repair_gate_v1'
        and repair_gate.get('status') == 'pass'
        and repair_replay.get('status') == 'pass'
        and repair_replay.get('completed_step') == 300000
        and repair_replay.get('trajectory_exact_through_step') == 299999
        and repair_replay.get('launch_source_recovery_status') == 'pass'
        and repair_replay.get('launch_source_snapshot_count') == 14
        and repair_replay.get('reversed_patch_record_count') == 7
        and len(repair_replay.get('launch_source_snapshots', {})) == 14
        and repair_bridge.get('sampler') == 'sample_intra_cas'
        and repair_bridge.get('sampling_steps') == 50
        and repair_bridge.get('temperature') == 0.5
        and repair_bridge.get('cfg_scale') == 3.5
        and repair_bridge.get('contact_guidance') is True
        and repair_bridge.get('contact_guidance_scale') == 1.0
        and repair_bridge_runtime.get('contract')
        == 'pytorch_official_evaluator_single_device_v1'
        and repair_bridge_runtime.get('requested_device') == 'cpu'
        and repair_bridge_runtime.get('device_type') == 'cpu'
        and repair_bridge_runtime.get('device_index') is None
        and repair_bridge_runtime.get('device_name') == 'cpu'
        and repair_bridge_runtime
        == repair_tokenizer.get('reconstruction_evaluator_runtime')
        and required_repair_roles <= repair_roles,
        'mgt_repair_gate',
        'MGT repair gate or its common-50 bridge is incomplete',
    )
    for record in repair_evidence:
        validate_artifact_record(
            f'mgt_repair_gate/evidence/{record.get("role")}', record
        )
    launch_evidence = next(
        record for record in repair_evidence
        if record.get('role') == 'historical_launch_sources'
    )
    require_artifact(
        launch_evidence.get('sha256')
        == launch_recovery_record['artifact']['sha256']
        and Path(launch_evidence.get('path', '')).resolve()
        == Path(launch_recovery_record['artifact']['path']).resolve(),
        'mgt_repair_gate/historical_mgt_launch_source_recovery',
        'repair gate is not bound to the report launch-source recovery',
    )
    for name, record in repair_replay['launch_source_snapshots'].items():
        validate_artifact_record(
            f'mgt_repair_gate/launch_source/{name}', record
        )
        require_artifact(
            name in launch_recovery['snapshots']
            and record.get('sha256')
            == launch_recovery['snapshots'][name].get('sha256'),
            'mgt_repair_gate',
            f'repair gate launch source differs from recovery: {name}',
        )

    paper_contract = records['hhc_paper_training_contract_audit']['payload']
    require_artifact(
        paper_contract.get('schema') == 'hhc_paper_training_contract_audit_v1'
        and paper_contract.get('status') == 'pass_with_disclosed_exceptions'
        and not paper_contract.get('paper_stated_numeric_mismatches')
        and not paper_contract.get('paper_stated_qualitative_mismatches'),
        'hhc_paper_training_contract_audit',
        'paper-stated MGT configuration is incomplete or mismatched',
    )
    for name in (
        'audit_script', 'paper', 'active_training_manifest',
        'active_vq_manifest', 'interx_protocol_audit',
    ):
        validate_artifact_record(
            f'hhc_paper_training_contract_audit/{name}',
            paper_contract.get(name, {}),
        )
    launch_sources = paper_contract.get('launch_source_stability', {})
    require_artifact(
        launch_sources.get('status') == 'pass'
        and launch_sources.get('scope') in {
            'live_source_tree', 'immutable_launch_snapshot'
        }
        and launch_sources.get('checked_files') == 14
        and not launch_sources.get('mismatches')
        and len(launch_sources.get('records', {})) == 14,
        'hhc_paper_training_contract_audit',
        'active source tree no longer matches the launched training sources',
    )
    if launch_sources.get('scope') == 'immutable_launch_snapshot':
        validate_artifact_record(
            'hhc_paper_training_contract_audit/source_snapshot_manifest',
            launch_sources.get('snapshot_manifest', {}),
        )
    for name, record in launch_sources['records'].items():
        require_artifact(
            record.get('launch_sha256') == record.get('sha256'),
            'hhc_paper_training_contract_audit',
            f'launch/current source hash differs for {name}',
        )
        validate_artifact_record(
            f'hhc_paper_training_contract_audit/source/{name}', record
        )
    historical_source = paper_contract.get('precleanup_visible_source', {})
    require_artifact(
        historical_source.get('commit') == '75814e5'
        and historical_source.get('file') == 'train_t2i_trans.py'
        and historical_source.get('precision') == 'bf16-mixed'
        and 'precision=' in historical_source.get('precision_markers_found', [])
        and 'bf16' in historical_source.get('precision_markers_found', [])
        and historical_source.get('optimizer_file') == 'utils/utils_model.py'
        and historical_source.get('optimizer_betas') == [0.5, 0.9]
        and historical_source.get(
            'exact_historical_wandb_run_source_archive_available'
        ) is False,
        'hhc_paper_training_contract_audit',
        'pre-cleanup precision provenance changed or is overstated',
    )
    differential = paper_contract.get('legacy_vs_active_differential', {})
    require_artifact(
        differential.get('same_observed_setting', {}).get(
            'numeric_precision'
        ) == 'bf16-mixed'
        and differential.get('different_settings', {}).get(
            'optimizer_betas', {}
        ).get('legacy') == [0.5, 0.9]
        and differential.get('different_settings', {}).get(
            'optimizer_betas', {}
        ).get('active') == [0.9, 0.99],
        'hhc_paper_training_contract_audit',
        'legacy-vs-active diagnostic differential is incomplete',
    )
    metric_logging = paper_contract.get('validation_metric_logging', {})
    require_artifact(
        metric_logging.get('val/FID') == 'current milestone value'
        and all(
            metric_logging.get(key) == 'running best-so-far value'
            for key in (
                'val/Diversity', 'val/TOP1', 'val/TOP2', 'val/TOP3',
                'val/Matching',
            )
        )
        and metric_logging.get('checkpoint_selection')
        == 'minimum current-milestone val/FID only',
        'hhc_paper_training_contract_audit',
        'validation metric logging/selection semantics changed',
    )
    loader_roles = paper_contract.get('validation_loader_roles', {})
    checkpoint_loader = loader_roles.get('checkpoint_selection_fid', {})
    loss_loader = loader_roles.get('lightning_token_loss', {})
    require_artifact(
        checkpoint_loader.get('batch_size') == 32
        and checkpoint_loader.get('drop_last') is True
        and checkpoint_loader.get('evaluated_entries') == 544
        and checkpoint_loader.get('intermask_contract_matches') is True
        and checkpoint_loader.get('official_interx_contract_matches') is True
        and not checkpoint_loader.get('contract_mismatches')
        and loss_loader.get('batch_size') == 96
        and loss_loader.get('drop_last_batches') == 5
        and loss_loader.get('evaluated_entries_after_drop_last') == 480
        and loss_loader.get('used_for_checkpoint_fid') is False
        and not loss_loader.get('contract_mismatches'),
        'hhc_paper_training_contract_audit',
        'checkpoint-selection evaluator was confused with token-loss loader',
    )

    formal_launch = records['hhc_formal_launch_contract_audit']['payload']
    expected_sampling_parity_checks = {
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
    }
    actual_sampling_parity = formal_launch.get(
        'validation_export_sampling_parity', {}
    )
    require_artifact(
        formal_launch.get('schema') == 'hhc_formal_launch_contract_v1'
        and formal_launch.get('status') == 'pass'
        and not formal_launch.get('missing_contract_entries')
        and not formal_launch.get('forbidden_entries_present')
        and all(formal_launch.get('ordering', {}).values())
        and set(actual_sampling_parity) == expected_sampling_parity_checks
        and all(actual_sampling_parity.values()),
        'hhc_formal_launch_contract_audit',
        'formal launch or validation/export sampler parity changed',
    )
    for role in ('launcher', 'state_resolver', 'source_freezer', 'implementation'):
        validate_artifact_record(
            f'hhc_formal_launch_contract_audit/{role}',
            formal_launch.get(role, {}),
        )
    parity_sources = formal_launch.get('sampling_parity_sources', {})
    parity_source_scope = formal_launch.get('sampling_parity_source_scope')
    require_artifact(
        set(parity_sources) == {
            'training_entrypoint', 'validation_implementation', 'fair_exporter'
        }
        and parity_source_scope in {
            'live_source_tree', 'immutable_launch_snapshot'
        },
        'hhc_formal_launch_contract_audit',
        'validation/export parity source set changed',
    )
    if parity_source_scope == 'immutable_launch_snapshot':
        validate_artifact_record(
            'hhc_formal_launch_contract_audit/source_snapshot_manifest',
            formal_launch.get('source_snapshot_manifest', {}),
        )
    for role, record in parity_sources.items():
        validate_artifact_record(
            f'hhc_formal_launch_contract_audit/{role}', record
        )

    sampling_runtime = records['hhc_sampling_runtime_audit']['payload']
    sampling_contract = sampling_runtime.get('contract', {})
    full_runtime = sampling_runtime.get('full_length_runtime', {})
    short_runtime = sampling_runtime.get('one_token_runtime', {})
    require_artifact(
        sampling_runtime.get('schema') == 'hhc_sampling_runtime_audit_v1'
        and sampling_runtime.get('status') == 'pass'
        and all(sampling_runtime.get('checks', {}).values())
        and sampling_contract.get('sampler') == 'sample_intra_cas'
        and sampling_contract.get('sampling_steps') == 50
        and sampling_contract.get('sampling_steps_unit')
        == 'outer_maskgit_iterations'
        and sampling_contract.get(
            'maximum_active_channel_stages_per_outer_iteration'
        ) == 3
        and sampling_contract.get(
            'maximum_model_forward_calls_per_outer_iteration_with_cfg'
        ) == 6
        and sampling_contract.get(
            'resolved_channels_skip_forward_but_not_outer_iterations'
        ) is True
        and full_runtime.get('motion_length_frames') == 150
        and full_runtime.get('motion_length_tokens') == 38
        and full_runtime.get('observed_outer_iterations') == 50
        and full_runtime.get('active_channel_stages') == 150
        and full_runtime.get('total_forward_calls_with_cfg') == 300
        and short_runtime.get('observed_outer_iterations') == 50
        and short_runtime.get('active_channel_stages') < 150
        and short_runtime.get('total_forward_calls_with_cfg') < 300,
        'hhc_sampling_runtime_audit',
        '50-step runtime or bounded compute semantics changed',
    )
    validate_artifact_record(
        'hhc_sampling_runtime_audit/source', sampling_runtime.get('source', {})
    )
    validate_artifact_record(
        'hhc_sampling_runtime_audit/implementation',
        sampling_runtime.get('implementation', {}),
    )

    contact_map_audit = records['hhc_contact_guidance_map_audit']['payload']
    fair_map = contact_map_audit.get('paper_formula_fair_map', {})
    legacy_map = contact_map_audit.get('mmm_legacy_full_axis_bug_map', {})
    map_policy = contact_map_audit.get('policy', {})
    require_artifact(
        contact_map_audit.get('schema')
        == 'hhc_contact_guidance_map_audit_v1'
        and contact_map_audit.get('status') == 'pass'
        and fair_map.get('scope') == 'official_train_only'
        and fair_map.get('samples') == 9095
        and fair_map.get('exact_recomputation_match') is True
        and fair_map.get('unseen_code_probability') == 0.5
        and legacy_map.get('copy_byte_identical') is True
        and legacy_map.get('exact_bug_replay_match') is True
        and legacy_map.get('uses_test_split') is True
        and legacy_map.get('test_samples_used') == 1706
        and legacy_map.get('scope_by_official_split')
        == {'train': 9095, 'val': 569, 'test': 1706}
        and map_policy.get('fair_main_table') == 'paper_formula_fair_map'
        and 'author-implementation reproduction sensitivity only'
        in map_policy.get('legacy_map_role', ''),
        'hhc_contact_guidance_map_audit',
        'fair/legacy contact-map separation is incomplete or changed',
    )
    validate_artifact_record(
        'hhc_contact_guidance_map_audit/audit_script',
        contact_map_audit.get('audit_script', {}),
    )
    validate_artifact_record(
        'hhc_contact_guidance_map_audit/vq_checkpoint',
        contact_map_audit.get('vq_checkpoint', {}),
    )
    for name, record in (
        ('paper_formula_fair_map', fair_map),
        ('mmm_legacy_full_axis_bug_map', {
            'path': legacy_map.get('copied_path'),
            'sha256': legacy_map.get('sha256'),
        }),
    ):
        validate_artifact_record(
            f'hhc_contact_guidance_map_audit/{name}', record
        )

    validation_pair = records[
        'hhc_validation_observational_pair_audit'
    ]['payload']
    validation_callback = validation_pair.get('validation_callback', {})
    fixed_pair_callback = validation_pair.get('fixed_pair_callback', {})
    fixed_pair_before = fixed_pair_callback.get('before', {})
    fixed_pair_after = fixed_pair_callback.get('after', {})
    validation_before = validation_pair.get('before', {})
    validation_after = validation_pair.get('after', {})
    require_artifact(
        validation_pair.get('schema')
        == 'hhc_lightning_same_step_pair_audit_v1'
        and validation_pair.get('status') == 'pass'
        and validation_pair.get('expected_step') == 20000
        and validation_pair.get('model_tensor_count') == 947
        and not validation_pair.get('differing_model_keys')
        and validation_pair.get('exact') == {
            'model_state': True,
            'optimizer_state': True,
            'lr_scheduler_state': True,
            'hyper_parameters': True,
        }
        and Path(validation_before.get('path', '')).name
        == 'prevalidation-000020000.ckpt'
        and Path(validation_after.get('path', '')).name
        == 'postvalidation-000020000.ckpt'
        and fixed_pair_before.get('step') == 20000
        and isinstance(fixed_pair_before.get('pre_saved'), bool)
        and fixed_pair_before.get('post_saved') is False
        and fixed_pair_after == {
            'step': 20000,
            'pre_saved': True,
            'post_saved': True,
        }
        and isinstance(
            validation_callback.get('after', {}).get('current_fid'),
            (int, float),
        )
        and math.isfinite(validation_callback['after']['current_fid'])
        and isinstance(
            validation_callback.get('after', {}).get('best_fid'),
            (int, float),
        )
        and math.isfinite(validation_callback['after']['best_fid']),
        'hhc_validation_observational_pair_audit',
        'real pre/post-validation training state is not bitwise identical',
    )
    for name in ('before', 'after', 'implementation'):
        validate_artifact_record(
            f'hhc_validation_observational_pair_audit/{name}',
            validation_pair.get(name, {}),
        )

    contact_cache_contracts = (
        ('train', 9110, 'first_caption_from_split_text_file'),
        ('val_full', 570, 'first_caption_from_split_text_file'),
        ('test_full', 1708, 'first_caption_from_split_text_file'),
    )
    for split_label, expected_samples, expected_caption_binding in (
        contact_cache_contracts
    ):
        cache_label = f'interx_luna_{split_label}_cache_audit'
        timing_label = f'interx_luna_{split_label}_timing_audit'
        cache = records[cache_label]['payload']
        timing = records[timing_label]['payload']
        cache_counts = cache.get('counts', {})
        timing_counts = timing.get('counts', {})
        require_artifact(
            cache.get('schema') == 'hhc_interx_codex_contact_cache_audit_v1'
            and cache.get('status') == 'pass'
            and cache.get('complete') is True
            and cache.get('integrity_passed') is True
            and cache.get('contract_hash_matches') is True
            and cache.get('implementation_resolved') is True
            and cache.get('output_schema_resolved') is True
            and cache.get('token_count_matches') is True
            and cache.get('caption_binding') == expected_caption_binding
            and cache_counts.get('expected_samples') == expected_samples
            and cache_counts.get('completed_samples') == expected_samples
            and not cache.get('missing_pairs')
            and not cache.get('missing_metadata')
            and not cache.get('extra_pairs')
            and not cache.get('extra_metadata')
            and not cache.get('batch_errors')
            and not cache.get('payload_errors')
            and not cache.get('metadata_errors')
            and not cache.get('manifest_mismatches'),
            cache_label,
            'full-split Luna cache is incomplete or fails integrity checks',
        )
        for name, hash_name in (
            ('manifest', 'manifest_sha256'), ('split', 'split_sha256')
        ):
            validate_bound_file(
                cache_label, cache['inputs'], name, hash_name
            )
        validate_artifact_record(
            f'{cache_label}/implementation', cache.get('implementation', {})
        )
        require_artifact(
            timing.get('schema')
            == 'hhc_interx_contact_timing_consistency_audit_v1'
            and timing.get('status') == 'complete'
            and timing.get('fusion_readiness_passed') is True
            and timing_counts.get('expected_samples') == expected_samples
            and timing_counts.get('audited_samples') == expected_samples
            and timing_counts.get('structured_contact_rows')
            == cache_counts.get('structured_contact_rows')
            and timing_counts.get('manifest_total_tokens_used')
            == cache_counts.get('manifest_total_tokens_used')
            and all(
                isinstance(timing_counts.get(key), int)
                and 0 <= timing_counts[key] <= expected_samples
                for key in (
                    'timing_covered_samples', 'timing_positive_samples',
                    'positive_timing_empty_regions',
                    'zero_timing_nonempty_regions',
                )
            )
            and timing.get('fusion_policy', {}).get(
                'timing_positive_and_regions_empty'
            ) == 'physical_losses_only'
            and timing.get('fusion_policy', {}).get(
                'timing_zero_and_regions_nonempty'
            ) == 'semantic_regions_inactive'
            and not timing.get('manifest_errors')
            and not timing.get('payload_errors')
            and not timing.get('metadata_errors')
            and not timing.get('timing_errors')
            and not timing.get('missing_pairs')
            and not timing.get('missing_metadata')
            and not timing.get('extra_pairs')
            and not timing.get('extra_metadata')
            and timing.get('inputs', {}).get('extraction_manifest_sha256')
            == cache.get('inputs', {}).get('manifest_sha256')
            and timing.get('inputs', {}).get('split_sha256')
            == cache.get('inputs', {}).get('split_sha256'),
            timing_label,
            'manual-timing/LLM-region fusion audit is incomplete or unbound',
        )
        validate_artifact_record(
            f'{timing_label}/implementation',
            timing.get('implementation', {}),
        )

    train_usage = records['interx_luna_train_cache_audit']['payload'].get(
        'resource_usage', {}
    )
    train_wall_clock = train_usage.get('end_to_end_wall_clock', {})
    train_invocations = train_usage.get('invocations', {})
    require_artifact(
        train_wall_clock.get('source')
        == 'extraction_manifest_created_at_utc_to_completed_at_utc'
        and isinstance(train_wall_clock.get('created_at_utc'), str)
        and isinstance(train_wall_clock.get('completed_at_utc'), str)
        and isinstance(train_wall_clock.get('total_seconds'), (int, float))
        and math.isfinite(float(train_wall_clock['total_seconds']))
        and train_wall_clock['total_seconds'] > 0
        and train_wall_clock.get('batch_files', 0) > 0
        and train_wall_clock.get('batch_mtime_window_inside_manifest') is True
        and train_usage.get('wall_seconds') is None
        and train_invocations.get('wall_timed_attempts') == 0
        and train_invocations.get('wall_untimed_attempts', 0) > 0,
        'interx_luna_train_cache_audit',
        'full-train annotation wall-clock disclosure is missing or overstated',
    )

    for cache_label, expected_samples in (
        ('interx_luna_test_frozen_cache_audit', 1696),
        ('interx_terra_robustness_cache_audit', 128),
    ):
        cache = records[cache_label]['payload']
        counts = cache.get('counts', {})
        require_artifact(
            cache.get('schema') == 'hhc_interx_codex_contact_cache_audit_v1'
            and cache.get('status') == 'pass'
            and cache.get('complete') is True
            and cache.get('integrity_passed') is True
            and cache.get('contract_hash_matches') is True
            and cache.get('caption_binding') == 'frozen_fair_evaluation_requests'
            and counts.get('expected_samples') == expected_samples
            and counts.get('completed_samples') == expected_samples
            and not cache.get('missing_pairs')
            and not cache.get('payload_errors')
            and not cache.get('metadata_errors'),
            cache_label,
            'frozen-request contact cache is incomplete or invalid',
        )
        for name, hash_name in (
            ('manifest', 'manifest_sha256'), ('split', 'split_sha256'),
            ('requests', 'requests_sha256'),
        ):
            validate_bound_file(
                cache_label, cache['inputs'], name, hash_name
            )
        validate_artifact_record(
            f'{cache_label}/implementation', cache.get('implementation', {})
        )

    protocol = records['interx_protocol_audit']['payload']
    require_artifact(protocol.get('status') == 'pass', 'interx_protocol_audit',
                     'status is not pass')
    require_artifact(
        protocol.get('split_entries') == 1708
        and protocol.get('valid_entries') == 1706
        and protocol.get('evaluated_entries') == 1696
        and protocol.get('batch_size') == 32
        and protocol.get('drop_last') is True
        and protocol.get('official_evaluator_weights') is True
        and protocol.get('official_evaluator_options') is True
        and protocol.get('official_evaluator_weights_sha256')
        == 'b1699f13276c044d8b7ae5f6063eb241b7ef96d7e92c878857abd5081e40701e'
        and protocol.get('intermask_source_matches') is True
        and protocol.get('official_interx_source_matches') is True
        and not protocol.get('mismatches'),
        'interx_protocol_audit',
        'test GT count or official evaluator/InterMask source contract changed',
    )
    assets = protocol.get('asset_protocol', {})
    expected_statistics = {
        'Interx_Mean.npy': (
            [672], 'float32',
            'a54dcfd012e07742077050e3a7ee1b4a2918ba2825466ab698d0cf6f9e438446',
        ),
        'Interx_Std.npy': (
            [672], 'float32',
            '23a2cf52a19cbaab16334821f2a3183e915c671f59fac702b68a1256742bc376',
        ),
    }
    require_artifact(
        assets.get('mmm_data_links_match') is True
        and assets.get('mmm_evaluator_link_matches') is True
        and assets.get('mmm_smplx_link_matches') is True
        and assets.get('statistics_byte_identical_to_mmm') is True
        and assets.get('evaluator_options_sha256')
        == 'f4ddaa81157259ce6b795f456be98263e50cd788a7800c30aacbfc2361b2d6dd'
        and all(
            assets.get('statistics', {}).get(name, {}).get('shape') == shape
            and assets['statistics'][name].get('dtype') == dtype
            and assets['statistics'][name].get('finite') is True
            and assets['statistics'][name].get('sha256') == digest
            for name, (shape, dtype, digest) in expected_statistics.items()
        )
        and assets.get('motions') == {
            'train': {
                'samples': 9110, 'feature_shapes': [[56, 6]],
                'dtypes': ['float32'],
                'sha256': '42e242fcd267c83b095591bb3445887ffc2578c1e45eb036bbd0188f174a7d2a',
            },
            'val': {
                'samples': 570, 'feature_shapes': [[56, 6]],
                'dtypes': ['float32'],
                'sha256': '91701e2f7687bfb0985013fa2995aa9d06aad8e55340ad6c7e9881662249ffd',
            },
            'test': {
                'samples': 1708, 'feature_shapes': [[56, 6]],
                'dtypes': ['float32'],
                'sha256': 'd67fd5f24342aff843ff2588314f9ffe6e932fcc5590393e9660a306ce9d6957',
            },
        }
        and assets.get('smplx_model_sha256') == {
            'male': 'ab318e3f37d2bfaae26abf4e6fab445c2a610e1d63714794d60379cc263bc2a5',
            'female': 'b2a3686c9d6d218ff6822fba411c607a3c8125a70af340f384ce68bebecabe0e',
        }
        and assets.get('volumetric_smpl_checkpoint_sha256') == {
            'male': '96b22f6409e67dc950dafdd83bb8dc4779381ecc2b2c752e1c6fa3a205b2e18c',
            'female': '2b582f009c2daeec50c90b5f39689a248bac96a4cc9983548ad4be677b72ccdb',
        },
        'interx_protocol_audit',
        'MMM data/statistics/evaluator/SMPL asset identity changed',
    )
    source = protocol.get('source_protocol', {})
    require_artifact(
        source.get('mmm_pre_cleanup_baseline_commit')
        == '75814e57e0702789176da6713056a15ed4a192e2'
        and source.get('mmm_cleanup_first_commit')
        == '0f58ffbb9db95a5d9ddc7f2e8a40b8602c17a325'
        and source.get('mmm_cleanup_parent_commit')
        == source.get('mmm_pre_cleanup_baseline_commit')
        and source.get('mmm_pre_cleanup_is_immediate_parent') is True
        and source.get('mmm_pre_cleanup_tree')
        == 'd6911e7f8e63ce6333b1707fd587c556be450cf6'
        and source.get('mmm_cleanup_tree')
        == 'c1d3e14ecc49671516a0e0c38e6e8ec5feaf24ba',
        'interx_protocol_audit',
        'MMM pre-cleanup snapshot is not the verified parent of cleanup',
    )
    expected_pre_cleanup_sources = {
        'models/t2i_trans.py': (
            '123d7240deb0a42639dea7fe83a2b569f771db8d',
            '13069e9292c9f39e1271657def16b5e7cfcf7d6eeb827df3d145b96b00e36ce1',
        ),
        'utils/eval_trans.py': (
            '64d902cc2fe09c79823c521f29676dc797b19017',
            '9113bc2d6d77af5ab0bb984df814f41965b837024c1c492d8916dd72a04cd139',
        ),
        'train_t2i_trans.py': (
            'ae6bf74f15376376c24ee64a4d5cdc8913b4fdb8',
            '36a1d00adfe0c1a7919334029fbce134dd99e3ebfa1d9b95e1a34c3ac7eae318',
        ),
        'generate_mm_samples.py': (
            '43b0544d714f3fc96a51324290d97b81a0519e7a',
            '47552dfff5fcf1f9c2ece33b290a1a4e1072cc5dcf352c7f29efafa1326b960c',
        ),
        'options/option_transformer.py': (
            'a1378dcbcb37a376987b587bad3bedc0d5236922',
            '8168bc8a06df26e41da2bf4c385e1dabe58d1b0af08b6e4f30316b6ae59670d3',
        ),
    }
    source_records = source.get('mmm_pre_cleanup_source_records', {})
    require_artifact(
        set(source_records) == set(expected_pre_cleanup_sources)
        and all(
            source_records[path].get('git_blob') == expected[0]
            and source_records[path].get('sha256') == expected[1]
            and source_records[path].get('bytes', 0) > 0
            for path, expected in expected_pre_cleanup_sources.items()
        ),
        'interx_protocol_audit',
        'MMM pre-cleanup generation/evaluation source blobs changed',
    )
    runtime_records = source.get('mmm_historical_runtime_call_records', {})
    runtime_metadata = source.get(
        'mmm_historical_runtime_metadata_records', {}
    )
    require_artifact(
        source.get('historical_evidence_priority') == [
            'historical_runtime_logs',
            'local_history_time_bracketed_source',
            'pre_cleanup_git_tree',
            'cleaned_main',
        ]
        and len(runtime_records) == 2
        and all(
            records
            and all(
                record.get('line', 0) > 0
                and 'type="sample_intra_cas"' in record.get('text', '')
                and len(record.get('sha256', '')) == 64
                for record in records
            )
            for records in runtime_records.values()
        )
        and len(runtime_metadata) == 2
        and all(
            record.get('git_commit')
            == '8c3e7ad107ed717b878779724958cf6eff00938e'
            and record.get('git_remote')
            == 'https://github.com/exitudio/MMM.git'
            and record.get('code_path') == 'train_t2i_trans.py'
            and len(record.get('sha256', '')) == 64
            for record in runtime_metadata.values()
        )
        and source.get('mmm_historical_runtime_source_state')
        == 'uncommitted_two_person_code_on_upstream_git_base'
        and source.get(
            'mmm_historical_runtime_local_history_audit', {}
        ).get('schema') == 'historical_mmm_sampler_local_history_audit_v1'
        and source.get(
            'mmm_historical_runtime_local_history_audit', {}
        ).get('status') == 'pass'
        and source.get(
            'mmm_historical_runtime_local_history_audit', {}
        ).get('scope')
        == 'read_only_recovery_of_uncommitted_runtime_source_not_git_history'
        and len(source.get(
            'mmm_historical_runtime_local_history_audit', {}
        ).get('sha256', '')) == 64,
        'interx_protocol_audit',
        'historical evidence order or runtime sampler records changed',
    )
    sampling = protocol.get('sampling_protocol', {})
    require_artifact(
        sampling.get('hhc_common_steps') == 50
        and sampling.get('intermask_common_steps') == 50
        and sampling.get('hhc_sampling_steps_unit') == 'outer_maskgit_iterations'
        and sampling.get(
            'hhc_maximum_active_channel_stages_per_outer_iteration'
        ) == 3
        and sampling.get(
            'hhc_maximum_model_forward_calls_per_outer_iteration_with_cfg'
        ) == 6
        and sampling.get('hhc_maximum_common50_active_channel_stages') == 150
        and sampling.get(
            'hhc_maximum_common50_model_forward_calls_with_cfg'
        ) == 300
        and sampling.get(
            'hhc_resolved_channels_skip_forward_but_not_outer_iterations'
        ) is True,
        'interx_protocol_audit', 'common 50-step sampler contract changed',
    )
    require_artifact(
        sampling.get('mmm_pre_cleanup_cascade_default_outer_steps') == 3
        and sampling.get('mmm_pre_cleanup_cascade_forward_passes_without_cfg') == 9
        and sampling.get('mmm_pre_cleanup_parallel_default_steps') == 10
        and sampling.get('mmm_historical_runtime_sampler') == 'sample_intra_cas'
        and sampling.get('mmm_historical_runtime_sampler_log_matches') is True
        and sampling.get('mmm_historical_runtime_minimum_sampling_updates_user_recorded') == 30
        and sampling.get('mmm_historical_runtime_exact_step_count_status') == 'recovered_and_ast_verified'
        and sampling.get('mmm_historical_runtime_outer_steps') == 10
        and sampling.get('mmm_historical_runtime_channel_updates') == 30
        and sampling.get('mmm_historical_runtime_local_history_audit_passes') is True
        and sampling.get('mmm_historical_runtime_vfs_content_ids_verified') is True
        and sampling.get('intermask_author_default_steps') == 20,
        'interx_protocol_audit', 'historical sampler audit changed',
    )
    selection_validation = protocol.get('checkpoint_selection_validation', {})
    require_artifact(
        selection_validation.get('split_entries') == 570
        and selection_validation.get('eligible_entries') == 569
        and selection_validation.get('evaluated_entries') == 544
        and selection_validation.get('batch_size') == 32
        and selection_validation.get('active_min_motion_frames') == 24
        and selection_validation.get('active_max_motion_frames') == 150
        and selection_validation.get('intermask_contract_matches') is True
        and selection_validation.get('official_interx_contract_matches') is True
        and selection_validation.get('assets_byte_identical_to_mmm') is True,
        'interx_protocol_audit',
        'checkpoint-selection validation differs from InterMask/Inter-X',
    )
    selection_manifest = records['mgt_validation_selection']['payload']
    require_artifact(
        selection_manifest.get('status') == 'pass'
        and selection_manifest.get('seed') == 500123
        and selection_manifest.get('batch_size') == 32
        and selection_manifest.get('eligible_samples') == 569
        and selection_manifest.get('evaluated_samples') == 544
        and selection_manifest.get('batches') == 17
        and selection_manifest.get('repeat_deterministic') is True
        and len(selection_manifest.get('omitted_by_drop_last', [])) == 25
        and len(selection_manifest.get('ordered_request_records_sha256', '')) == 64
        and len(selection_manifest.get('ordered_evaluator_inputs_sha256', '')) == 64
        and len(selection_manifest.get('records', [])) == 544,
        'mgt_validation_selection',
        'fixed checkpoint-selection inputs are incomplete or changed',
    )
    require_artifact(
        selection_manifest.get('split_sha256')
        == selection_validation.get('split_sha256')
        and selection_manifest.get('motions_sha256')
        == selection_validation.get('motion_h5_sha256'),
        'mgt_validation_selection',
        'fixed selection inputs are bound to different val assets',
    )
    validate_artifact_record(
        'mgt_validation_selection/implementation',
        selection_manifest.get('implementation', {}),
    )
    requests = records['official_request_manifest']['payload']
    require_artifact(requests.get('status') == 'pass', 'official_request_manifest',
                     'status is not pass')
    require_artifact(requests.get('replications') == 20,
                     'official_request_manifest', 'expected 20 replications')
    require_artifact(requests.get('samples_per_replication') == 1696,
                     'official_request_manifest', 'expected 1696 samples')
    require_artifact(len(requests.get('records', [])) == 20,
                     'official_request_manifest', 'expected 20 request records')
    require_artifact(
        requests.get('request_contract_version') == 2
        and requests.get('multimodality')
        == {'prompts': 100, 'repeats': 30, 'pairs': 10},
        'official_request_manifest',
        'request contract version or multimodality protocol changed',
    )
    expected_request_provenance = {
        'protocol', 'request_preparer', 'request_verifier', 'dataset_loader',
        'normalizer', 'evaluator_wrapper', 'evaluator_options',
        'evaluator_weights', 'test_split', 'test_motions', 'raw_mean',
        'raw_std',
    }
    request_provenance = requests.get('frozen_input_provenance', {})
    require_artifact(
        set(request_provenance) == expected_request_provenance,
        'official_request_manifest',
        'frozen input provenance is incomplete',
    )
    for name, record in request_provenance.items():
        validate_artifact_record(
            f'official_request_manifest/frozen_input_provenance/{name}',
            record,
        )
    for replication, record in enumerate(requests['records']):
        require_artifact(
            record.get('replication') == replication
            and record.get('seed') == 1234 + replication,
            'official_request_manifest',
            f'replication {replication} ID or seed changed',
        )
        request_path = validate_artifact_record(
            f'official_request_manifest/replication_{replication:02d}',
            record,
        )
        require_artifact(
            record.get('bytes') == request_path.stat().st_size,
            'official_request_manifest',
            f'replication {replication} byte count is stale',
        )

    request_recovery = records['fair_request_provenance_recovery']['payload']
    require_artifact(
        request_recovery.get('schema')
        == 'fair_interx_request_provenance_recovery_v1'
        and request_recovery.get('status') == 'pass'
        and request_recovery.get('canonical_manifest_unchanged') is True
        and request_recovery.get('request_npz_files_modified') == 0
        and request_recovery.get('request_npz_files_verified') == 20,
        'fair_request_provenance_recovery',
        'request provenance recovery did not preserve frozen requests',
    )
    recovered_manifest = validate_artifact_record(
        'fair_request_provenance_recovery/manifest',
        request_recovery.get('frozen_manifest_snapshot', {}),
    )
    recovered_loader = validate_artifact_record(
        'fair_request_provenance_recovery/dataset_loader',
        request_recovery.get('frozen_source_snapshot', {}),
    )
    validate_artifact_record(
        'fair_request_provenance_recovery/request_verifier',
        request_recovery.get('frozen_verifier_snapshot', {}),
    )
    recovered_protocol = validate_artifact_record(
        'fair_request_provenance_recovery/protocol',
        request_recovery.get('frozen_protocol_snapshot', {}),
    )
    validate_artifact_record(
        'fair_request_provenance_recovery/request_creation_records',
        request_recovery.get('frozen_creation_records_snapshot', {}),
    )
    validate_artifact_record(
        'fair_request_provenance_recovery/implementation',
        request_recovery.get('implementation', {}),
    )
    require_artifact(
        sha256_file(recovered_manifest)
        == records['official_request_manifest']['artifact']['sha256']
        and sha256_file(recovered_loader)
        == request_provenance['dataset_loader']['sha256']
        and sha256_file(recovered_protocol)
        == request_provenance['protocol']['sha256'],
        'fair_request_provenance_recovery',
        'recovered manifest/source/protocol do not match the original frozen records',
    )
    creation = request_recovery.get('request_creation_runtime_recovery', {})
    expected_creation_command = (
        'OMP_NUM_THREADS=8 conda run --no-capture-output -n New_MMM python -u '
        'tools/prepare_fair_eval_requests.py '
        '--output-dir output/fair_interx_requests --device cpu '
        '--replications 20 --seed 1234'
    )
    expected_creation_runtime = {
        'contract': 'pytorch_official_evaluator_single_device_v1',
        'requested_device': 'cpu',
        'device_type': 'cpu',
        'device_index': None,
        'device_name': 'cpu',
    }
    require_artifact(
        creation.get('command') == expected_creation_command
        and creation.get('cwd') == str(Path(__file__).resolve().parents[1])
        and creation.get('status') == 'completed'
        and creation.get('exit_code') == 0
        and creation.get('runtime') == expected_creation_runtime
        and creation.get('replications_reported_complete') == 20
        and creation.get('samples_per_replication_reported') == 1696
        and creation.get('launch_line_number') == 2371
        and creation.get('completion_line_number') == 2522
        and creation.get('launch_line_sha256')
        == '4da1a2adf58b2b529963ebe8c47524a29c04c761ae31a3fb714e74a2d289e560'
        and creation.get('completion_line_sha256')
        == 'df2a9895370b19d390b5740ec7d586c7593739a05c2a13928213d45b2e3078b1',
        'fair_request_provenance_recovery',
        'request creation CPU runtime or successful completion proof changed',
    )

    request_replay = records['fair_request_replay_audit']['payload']
    require_artifact(
        request_replay.get('schema')
        == 'fair_interx_request_replay_audit_v1'
        and request_replay.get('status') == 'pass'
        and request_replay.get('sample_count') == 1696
        and request_replay.get('scope')
        == 'replication_00_seed1234_official_loader_cpu_replay'
        and request_replay.get('failures') == []
        and request_replay.get('embedding_absolute_tolerance') == 1e-6,
        'fair_request_replay_audit',
        'official-loader replication-0 replay contract failed or changed',
    )
    replay_creation_provenance = validate_artifact_record(
        'fair_request_replay_audit/request_creation_provenance',
        request_replay.get('request_creation_provenance', {}),
    )
    require_artifact(
        request_replay.get('request_creation_runtime')
        == expected_creation_runtime
        and sha256_file(replay_creation_provenance)
        == records['fair_request_provenance_recovery']['artifact']['sha256'],
        'fair_request_replay_audit',
        'replay audit is not bound to the recovered CPU creation runtime',
    )
    replay_frozen = validate_artifact_record(
        'fair_request_replay_audit/frozen', request_replay.get('frozen', {})
    )
    validate_artifact_record(
        'fair_request_replay_audit/replay', request_replay.get('replay', {})
    )
    validate_artifact_record(
        'fair_request_replay_audit/implementation',
        request_replay.get('implementation', {}),
    )
    require_artifact(
        sha256_file(replay_frozen) == requests['records'][0]['sha256'],
        'fair_request_replay_audit',
        'replayed reference is not frozen replication 0',
    )
    exact_fields = set(request_replay.get('exact_fields', []))
    tolerance_fields = set(request_replay.get('tolerance_fields', []))
    require_artifact(
        exact_fields == {
            'names', 'captions', 'lengths', 'betas', 'gender', 'contact',
            'mm_indices', 'seed',
        }
        and tolerance_fields == {
            'text_embeddings', 'gt_embeddings', 'gt_body_embeddings',
        },
        'fair_request_replay_audit',
        'exact/tolerance field partition changed',
    )
    replay_comparisons = request_replay.get('comparisons', {})
    require_artifact(
        all(
            replay_comparisons.get(field, {}).get('status') == 'pass'
            and replay_comparisons[field].get('exact') is True
            for field in exact_fields
        )
        and all(
            replay_comparisons.get(field, {}).get('status') == 'pass'
            and replay_comparisons[field].get('within_absolute_tolerance') is True
            and replay_comparisons[field].get(
                'maximum_absolute_difference', float('inf')
            ) <= 1e-6
            for field in tolerance_fields
        ),
        'fair_request_replay_audit',
        'request fields or evaluator embeddings do not replay within contract',
    )

    capacity = records['model_capacity_audit']['payload']
    require_artifact(
        capacity.get('schema') == 'hhc_model_capacity_audit_v2'
        and capacity.get('status') == 'pass',
        'model_capacity_audit', 'schema/status is not the executable v2 audit',
    )
    expected_capacity = {
        'hhc': {
            'shape_aware_vq_parameters': 36606748,
            'mgt_parameters_excluding_shared_vq': 73986640,
            'unique_motion_parameters': 110593388,
            'frozen_clip_vit_b32_parameters': 151277313,
            'total_inference_resident_parameters': 261870701,
        },
        'intermask': {
            'vq_parameters': 45418502,
            'transformer_total_parameters': 177333376,
            'transformer_trainable_parameters': 54332032,
            'unique_motion_parameters': 99750534,
            'total_inference_resident_parameters': 222751878,
        },
        'coshmdm': {
            'total_parameters': 305442310,
            'trainable_parameters': 182440966,
            'frozen_parameters': 123001344,
        },
    }
    expected_capacity_sources = {
        'hhc': {
            'models/model_tools.py', 'models/vqvae_sep3_shape.py',
            'models/encdec.py', 'models/quantize_cnn.py',
            'models/two_person_transformer.py',
            'train/train_t2i_trans.py',
            'tools/export_hhc_official_interx.py',
        },
        'intermask': {
            'models/vq/model.py', 'models/vq/residual_vq.py',
            'models/mask_transformer/transformer.py',
            'models/mask_transformer/transformer_modules.py',
            'checkpoints/interx/vq_default/opt.txt',
            'checkpoints/interx/trans_default/opt.txt',
            'tools_export_official_interx.py',
        },
        'coshmdm': {
            'configs/model.yaml', 'models/coshmdm.py',
            'models/gaussian_diffusion.py', 'models/nets.py',
            'models/blocks.py', 'tools/export_official_interx.py',
        },
    }
    validate_artifact_record(
        'model_capacity_audit/script', capacity.get('audit_script', {})
    )
    for method, expected_counts in expected_capacity.items():
        method_record = capacity.get(method, {})
        require_artifact(
            all(method_record.get(key) == value
                for key, value in expected_counts.items()),
            'model_capacity_audit', f'{method} parameter counts changed',
        )
        require_artifact(
            bool(method_record.get('checkpoint_load')),
            'model_capacity_audit', f'{method} checkpoint load is unproven',
        )
        if method == 'hhc':
            require_artifact(
                method_record.get('vq_contract') == 'precleanup_raw12288',
                'model_capacity_audit',
                'HHC capacity was not audited with the raw12288 contract',
            )
        source_files = method_record.get('source_files', {})
        require_artifact(
            set(source_files) == expected_capacity_sources[method],
            'model_capacity_audit',
            f'{method} constructor/config source set changed',
        )
        for source_name, source_record in source_files.items():
            validate_artifact_record(
                f'model_capacity_audit/{method}/source/{source_name}',
                source_record,
            )
    for method, names in {
        'hhc': ('vq_checkpoint', 'mgt_checkpoint', 'selection_manifest'),
        'intermask': ('vq_checkpoint', 'transformer_checkpoint'),
        'coshmdm': ('checkpoint',),
    }.items():
        for name in names:
            validate_artifact_record(
                f'model_capacity_audit/{method}/{name}',
                capacity.get(method, {}).get(name, {}),
            )
    capacity_selection = records['hhc_mgt_final_selection']['payload']
    capacity_vq = records['vq_import_manifest']['payload']['checkpoint']
    capacity_cosh = records['coshmdm_final_state_audit']['payload']
    require_artifact(
        same_resolved_path(
            capacity['hhc']['selection_manifest']['path'],
            records['hhc_mgt_final_selection']['artifact']['path'],
        )
        and capacity['hhc']['selection_manifest']['sha256']
        == records['hhc_mgt_final_selection']['artifact']['sha256']
        and same_resolved_path(
            capacity['hhc']['mgt_checkpoint']['path'],
            capacity_selection.get('selected_checkpoint'),
        )
        and capacity['hhc']['mgt_checkpoint']['sha256']
        == capacity_selection.get('selected_checkpoint_sha256'),
        'hhc_mgt_final_selection/model_capacity_audit',
        'capacity HHC model differs from the formally selected checkpoint',
    )
    require_artifact(
        same_resolved_path(
            capacity['hhc']['vq_checkpoint']['path'],
            capacity_vq.get('path'),
        )
        and capacity['hhc']['vq_checkpoint']['sha256']
        == capacity_vq.get('sha256'),
        'vq_import_manifest/model_capacity_audit',
        'capacity HHC VQ differs from the formally audited checkpoint',
    )
    require_artifact(
        same_resolved_path(
            capacity['coshmdm']['checkpoint']['path'],
            capacity_cosh.get('checkpoint'),
        )
        and capacity['coshmdm']['checkpoint']['sha256']
        == capacity_cosh.get('checkpoint_sha256'),
        'coshmdm_final_state_audit/model_capacity_audit',
        'capacity CoShMDM differs from the formally audited checkpoint',
    )

    vq_import = records['vq_import_manifest']['payload']
    vq = vq_import.get('checkpoint', {})
    require_artifact(
        vq_import.get('artifact_role')
        == 'Imported pre-cleanup mainline MMM raw Inter-X VQ checkpoint'
        and vq_import.get('import_policy')
        == 'byte-identical copy; source checkpoint is not modified'
        and vq_import.get('frozen_architecture', {}).get(
            'motion_representation') == 'raw'
        and vq_import.get('frozen_architecture', {}).get(
            'codebook_size_per_part') == 12288
        and vq_import.get('frozen_architecture', {}).get(
            'code_dim_total') == 256
        and vq_import.get('verified_reconstruction', {}).get(
            'fair_fid') == 0.29441722725218256
        and vq_import.get('verified_reconstruction', {}).get(
            'mpjpe_m') == 0.1362613624044445,
        'vq_import_manifest',
        'pre-cleanup raw12288 checkpoint contract is incomplete or changed',
    )
    require_artifact(
        Path(vq.get('path', '')).is_file()
        and vq.get('sha256') == sha256_file(vq['path'])
        and vq.get('bytes') == Path(vq['path']).stat().st_size,
        'vq_import_manifest', 'imported checkpoint path/hash/size is stale',
    )

    reconstruction = records['vq_final_reconstruction']['payload']
    require_artifact(
        reconstruction.get('checkpoint_iteration') == -1
        and reconstruction.get('samples') == 1696
        and reconstruction.get('codebook_size_per_part') == 12288
        and reconstruction.get('checkpoint_sha256') == vq.get('sha256')
        and reconstruction.get('mpjpe_m') == 0.1362613624044445,
        'vq_final_reconstruction',
        'raw12288 reconstruction contract or checkpoint binding changed',
    )
    reconstruction_fid = records['vq_final_reconstruction_fid']['payload']
    require_artifact(
        reconstruction_fid.get('schema') == 'fair_interx_main_only_metrics_v1'
        and reconstruction_fid.get('status') == 'complete'
        and reconstruction_fid.get('samples') == 1696
        and reconstruction_fid.get('fid') == 0.29441722725218256
        and reconstruction_fid.get('generation_request_hash_verified') is True
        and 'fid_body_controlled' not in reconstruction_fid
        and 'diversity_body_controlled' not in reconstruction_fid
        and reconstruction_fid.get('main_generation_metadata', {}).get(
            'checkpoint_sha256') == vq.get('sha256'),
        'vq_final_reconstruction_fid',
        'raw12288 reconstruction FID contract or checkpoint binding changed',
    )
    validate_main_only_evaluator_provenance(
        'vq_final_reconstruction_fid', reconstruction_fid
    )
    historical = records['historical_mmm_wandb_comparison']['payload']
    require_artifact(
        historical.get('status') == 'pass'
        and historical.get('comparability', {}).get('same_configuration') is False,
        'historical_mmm_wandb_comparison',
        'historical W&B run must remain labelled non-comparable',
    )
    current_wandb = historical.get('current_run', {})
    historical_wandb = historical.get('historical_run', {})
    historical_validation = historical_wandb.get('validation_history', [])
    require_artifact(
        [record.get('trainer/global_step') for record in historical_validation]
        == list(range(9999, 300000, 10000))
        and historical_wandb.get('remote_history_provenance', {}).get(
            'server_summary_exact_match'
        ) is True
        and historical_wandb.get('remote_history_provenance', {}).get(
            'source_code_archive_available'
        ) is False,
        'historical_mmm_wandb_comparison',
        'historical W&B 30-milestone remote trajectory is incomplete',
    )
    validate_artifact_record(
        'historical_mmm_wandb_comparison',
        historical_wandb.get('remote_history_artifact', {}),
    )
    historical_runtime = historical_wandb.get('validation_runtime', {})
    historical_effective = historical_runtime.get(
        'effective_validation_runtime', {}
    )
    require_artifact(
        historical_runtime.get('status') == 'pass'
        and historical_effective == {
            'sampler': 'sample_intra_cas',
            'outer_steps': 10,
            'channel_updates': 30,
            'cascade_order': 'root_body_hand',
            'rand_pos': False,
            'temperature': 0.0,
            'cfg_scale': -1,
            'contact': False,
            'model_forward_calls_per_generation': 30,
        }
        and historical_runtime.get('source_interval_checks') == {
            'model': True, 'train': True, 'evaluation': True,
        }
        and historical_runtime.get('checks')
        and all(historical_runtime['checks'].values()),
        'historical_mmm_wandb_comparison',
        'completed historical MGT validation runtime is not exactly bound',
    )
    for source_name in ('model', 'train', 'evaluation'):
        validate_artifact_record(
            f'historical_mmm_wandb_comparison/validation_runtime/{source_name}',
            historical_runtime.get(source_name, {}),
        )
    validate_bound_file(
        'historical_mmm_wandb_comparison/validation_runtime/wandb_run',
        historical_runtime.get('wandb_run', {}),
        'metadata_path', 'metadata_sha256',
    )
    for implementation_name, implementation_record in historical.get(
        'implementation', {}
    ).items():
        validate_artifact_record(
            f'historical_mmm_wandb_comparison/implementation/{implementation_name}',
            implementation_record,
        )
    require_artifact(
        set(historical.get('implementation', {}))
        == {'audit', 'local_history_parser'},
        'historical_mmm_wandb_comparison',
        'historical runtime audit implementation closure is incomplete',
    )
    require_artifact(
        current_wandb.get('latest_training_metrics', {}).get(
            'completed_updates'
        ) == 300000,
        'historical_mmm_wandb_comparison',
        'current W&B trajectory is not complete at 300000 steps',
    )
    require_artifact(
        bool(current_wandb.get('validation_history')),
        'historical_mmm_wandb_comparison',
        'current W&B trajectory has no validation history',
    )
    validate_bound_file(
        'historical_mmm_wandb_comparison', current_wandb,
        'wandb_file', 'wandb_file_sha256',
    )
    native20 = records['intermask_native20_hdf5_protocol_audit']['payload']
    native20_protocol = {
        'time_steps': 20, 'cond_scale': 2.0, 'temperature': 1.0,
        'topk_filter_thres': 0.9, 'length_policy': 'exact',
        'seed_strategy': 'base_seed + repeat_index',
    }
    require_artifact(
        native20.get('schema')
        == 'intermask_existing_native20_hdf5_protocol_audit_v1'
        and native20.get('status') == 'pass'
        and native20.get('classification')
        == 'author-native-20-steps_exact-length_not-common50'
        and native20.get('replications') == 20
        and native20.get('samples_per_replication') == 1696
        and native20.get('sampler_protocol') == native20_protocol
        and [record.get('replication') for record in native20.get('records', [])]
        == list(range(20))
        and all(
            record.get(kind, {}).get('protocol') == native20_protocol
            and record.get(kind, {}).get('exact_lengths_verified') is True
            for record in native20.get('records', [])
            for kind in ('main', 'multimodality')
        )
        and native20.get('implementation_sha256')
        == sha256_file('tools/audit_existing_intermask_native20.py'),
        'intermask_native20_hdf5_protocol_audit',
        'legacy InterMask artifact is not proven native-20/exact-length',
    )
    require_artifact(reconstruction.get('checkpoint_sha256') == vq.get('sha256'),
                     'vq_final_reconstruction',
                     'reconstruction and import manifest use different VQ checkpoints')

    selection = records['hhc_mgt_final_selection']['payload']
    require_artifact(
        paper_contract['active_training_manifest']['sha256']
        == selection.get('training_manifest_sha256')
        and Path(paper_contract['active_training_manifest']['path']).resolve()
        == Path(selection.get('training_manifest', '')).resolve(),
        'hhc_paper_training_contract_audit/hhc_mgt_final_selection',
        'paper-contract audit and selected MGT use different training manifests',
    )
    require_artifact(
        conditional_support.get('training_manifest', {}).get('sha256')
        == selection.get('training_manifest_sha256')
        and same_resolved_path(
            conditional_support.get('training_manifest', {}).get('path'),
            selection.get('training_manifest'),
        ),
        'hhc_mgt_train_val_conditional_support_audit/hhc_mgt_final_selection',
        'conditional-support audit is not bound to the selected MGT training manifest',
    )
    vq_manifest_artifact = records['vq_import_manifest']['artifact']
    require_artifact(
        paper_contract['active_vq_manifest']['sha256']
        == vq_manifest_artifact['sha256']
        and Path(paper_contract['active_vq_manifest']['path']).resolve()
        == Path(vq_manifest_artifact['path']).resolve(),
        'hhc_paper_training_contract_audit/vq_import_manifest',
        'paper-contract audit and selected VQ use different manifests',
    )
    tokenizer_runtime_artifact = records[
        'historical_mgt_tokenizer_runtime_audit'
    ]['artifact']
    tokenizer_runtime = records[
        'historical_mgt_tokenizer_runtime_audit'
    ]['payload']
    require_artifact(
        tokenizer_runtime.get('status') == 'pass'
        and all(tokenizer_runtime.get('checks', {}).values())
        and paper_contract.get('historical_tokenizer_runtime_audit', {}).get(
            'valid'
        ) is True
        and paper_contract.get('historical_tokenizer_runtime_audit', {}).get(
            'sha256'
        ) == tokenizer_runtime_artifact['sha256']
        and Path(
            paper_contract.get('historical_tokenizer_runtime_audit', {}).get(
                'path', ''
            )
        ).resolve() == Path(tokenizer_runtime_artifact['path']).resolve(),
        'hhc_paper_training_contract_audit/'
        'historical_mgt_tokenizer_runtime_audit',
        'paper-contract audit does not bind the current historical tokenizer '
        'runtime evidence',
    )
    require_artifact(selection.get('run_complete') is True,
                     'hhc_mgt_final_selection', 'run is not complete')
    require_artifact(selection.get('expected_final_step') == 300000,
                     'hhc_mgt_final_selection', 'expected final step 300000')
    expected_selection_contract = {
        'contact_timing_source': 'validation_dataset_annotation_oracle',
        'contact_timing_is_test_data': False,
        'contact_as_transformer_training_input': False,
        'sampler': 'sample_intra_cas', 'sampling_steps': 50,
        'temperature': 0.5, 'cfg_scale': 3.5,
        'contact_guidance_scale': 1.0, 'contact': True,
        'contact_conditioning': False, 'eval_iter': 10000,
    }
    require_artifact(
        selection.get('test_metrics_used_for_selection') is False
        and selection.get('selection_metric')
        == 'minimum fixed-protocol validation FID'
        and selection.get('validation_inference_contract')
        == expected_selection_contract,
        'hhc_mgt_final_selection',
        'checkpoint selection condition is absent or changed',
    )
    validate_bound_file('hhc_mgt_final_selection', selection,
                        'selected_checkpoint', 'selected_checkpoint_sha256')
    validate_bound_file('hhc_mgt_final_selection', selection,
                        'training_manifest', 'training_manifest_sha256')
    validate_bound_file(
        'hhc_mgt_final_selection', selection,
        'final_latest_checkpoint', 'final_latest_checkpoint_sha256',
    )
    final_state = selection.get('final_state_audit', {})
    require_artifact(
        final_state.get('status') == 'pass'
        and final_state.get('global_step') == 300000
        and final_state.get('model_state', {}).get(
            'all_floating_tensors_finite'
        ) is True
        and final_state.get('optimizer_state', {}).get(
            'all_floating_tensors_finite'
        ) is True
        and final_state.get('has_loop_state') is True,
        'hhc_mgt_final_selection',
        'final 300k checkpoint state is incomplete or non-finite',
    )

    cosh_selection = records['coshmdm_validation_selection']['payload']
    cosh_candidates = cosh_selection.get('candidates', [])
    cosh_candidate_steps = cosh_selection.get('candidate_steps', [])
    candidate_contract_valid = (
        isinstance(cosh_candidates, list)
        and bool(cosh_candidates)
        and cosh_candidate_steps
        == [candidate.get('step') for candidate in cosh_candidates]
        and cosh_candidate_steps == sorted(set(cosh_candidate_steps))
        and all(
            isinstance(candidate.get('step'), int)
            and 0 < candidate['step'] <= 200000
            and candidate['step'] % 20000 == 0
            and isinstance(candidate.get('validation_fid'), (int, float))
            and math.isfinite(float(candidate['validation_fid']))
            and candidate.get('checkpoint_source_stable_during_export') is True
            and len(candidate.get('checkpoint', {}).get('sha256', '')) == 64
            and candidate.get('checkpoint', {}).get('sha256', '')[:16]
            in Path(candidate.get('generations', {}).get('path', '')).name
            for candidate in cosh_candidates
        )
    )
    selected_cosh_candidate = (
        min(
            cosh_candidates,
            key=lambda item: (float(item['validation_fid']), item['step']),
        )
        if candidate_contract_valid else {}
    )
    require_artifact(
        cosh_selection.get('schema')
        == 'hhc_coshmdm_capped_validation_selection_v1'
        and cosh_selection.get('status') == 'complete'
        and cosh_selection.get('test_metrics_used_for_selection') is False
        and cosh_selection.get('formal_hhc_validation_record_set_exact') is True
        and cosh_selection.get('training_cap_status') in {
            'stopped_gracefully', 'sigterm_sent_after_180s_grace',
            'trainer_exited_before_monitor_signal',
        }
        and len(cosh_selection.get(
            'formal_hhc_validation_selection_sha256', ''
        )) == 64
        and 0 < int(cosh_selection.get('selected_step', 0)) <= 200000,
        'coshmdm_validation_selection',
        'CoShMDM capped validation selection is incomplete or used test data',
    )
    require_artifact(
        candidate_contract_valid
        and selected_cosh_candidate.get('step')
        == cosh_selection.get('selected_step')
        and float(selected_cosh_candidate.get('validation_fid'))
        == float(cosh_selection.get('selected_validation_fid'))
        and selected_cosh_candidate.get('checkpoint', {}).get('path')
        == cosh_selection.get('selected_checkpoint')
        and selected_cosh_candidate.get('checkpoint', {}).get('sha256')
        == cosh_selection.get('selected_checkpoint_sha256'),
        'coshmdm_validation_selection',
        'CoShMDM selected row is not the reproducible minimum-FID candidate',
    )
    cosh_selected_step = int(cosh_selection['selected_step'])

    timotion_selection = records['timotion_validation_selection']['payload']
    timotion_candidates = timotion_selection.get('candidates', [])
    timotion_expected_epochs = list(range(100, 1501, 100))
    timotion_expected_steps = [epoch * 71 for epoch in timotion_expected_epochs]
    timotion_candidates_valid = (
        timotion_selection.get('schema')
        == 'hhc_timotion_interx_validation_selection_v1'
        and timotion_selection.get('status') == 'complete'
        and timotion_selection.get('test_metrics_used_for_selection') is False
        and timotion_selection.get('validation_samples') == 544
        and timotion_selection.get('validation_seed') == 500123
        and timotion_selection.get('candidate_epochs')
        == timotion_expected_epochs
        and timotion_selection.get('candidate_steps')
        == timotion_expected_steps
        and len(timotion_candidates) == len(timotion_expected_epochs)
        and all(
            candidate.get('epoch') == epoch
            and candidate.get('step') == step
            and candidate.get('samples') == 544
            and candidate.get('test_set_used') is False
            and set(candidate.get('values', {})) == {
                'val/FID', 'val/R_precision_top1',
                'val/R_precision_top2', 'val/R_precision_top3',
                'val/Matching', 'val/Diversity',
            }
            and all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in candidate.get('values', {}).values()
            )
            for candidate, epoch, step in zip(
                timotion_candidates,
                timotion_expected_epochs,
                timotion_expected_steps,
            )
        )
    )
    require_artifact(
        timotion_candidates_valid,
        'timotion_validation_selection',
        'TiMotion held-out validation candidate grid is incomplete or changed',
    )
    selected_timotion_candidate = min(
        timotion_candidates,
        key=lambda item: (float(item['values']['val/FID']), item['step']),
    )
    require_artifact(
        timotion_selection.get('selection_metric')
        == 'minimum official validation FID'
        and selected_timotion_candidate.get('epoch')
        == timotion_selection.get('selected_epoch')
        and selected_timotion_candidate.get('step')
        == timotion_selection.get('selected_step')
        and float(selected_timotion_candidate['values']['val/FID'])
        == float(timotion_selection.get('selected_validation_fid'))
        and selected_timotion_candidate.get('checkpoint', {}).get('path')
        == timotion_selection.get('selected_checkpoint')
        and selected_timotion_candidate.get('checkpoint', {}).get('sha256')
        == timotion_selection.get('selected_checkpoint_sha256'),
        'timotion_validation_selection',
        'TiMotion selected row is not the reproducible minimum-FID candidate',
    )
    for candidate in timotion_candidates:
        for artifact_name in ('checkpoint', 'generations', 'metrics'):
            validate_artifact_record(
                'timotion_validation_selection/'
                f"epoch_{candidate['epoch']}/{artifact_name}",
                candidate[artifact_name],
            )
    for artifact_name in (
        'training_manifest', 'validation_history',
        'final_full_state_checkpoint', 'implementation',
    ):
        validate_artifact_record(
            f'timotion_validation_selection/{artifact_name}',
            timotion_selection.get(artifact_name, {}),
        )

    # Re-open the finalized validation ledger instead of trusting only the
    # selector's copied candidate table.  The live callback deliberately keeps
    # this ledger in ``running`` state; the post-training finalizer may mark it
    # complete only after the full 106,500-update manifest exists.  Binding the
    # ledger here prevents an interrupted training run or a hand-assembled
    # subset of validation checkpoints from entering the final comparison.
    timotion_history_path = validate_artifact_record(
        'timotion_validation_selection/validation_history',
        timotion_selection.get('validation_history', {}),
    )
    timotion_history = json.loads(
        timotion_history_path.read_text(encoding='utf-8')
    )
    timotion_history_manifest_path = validate_artifact_record(
        'timotion_validation_history/training_manifest',
        timotion_history.get('training_manifest', {}),
    )
    timotion_history_finalizer_path = validate_artifact_record(
        'timotion_validation_history/implementation',
        timotion_history.get('implementation', {}),
    )
    timotion_history_records = timotion_history.get('records', [])
    timotion_history_by_epoch = {
        record.get('epoch'): record for record in timotion_history_records
    }
    require_artifact(
        timotion_history.get('schema')
        == 'timotion_official_interx_validation_history_v1'
        and timotion_history.get('status') == 'complete'
        and timotion_history.get('final_global_step') == 106500
        and timotion_history.get('completed_validation_epochs')
        == timotion_expected_epochs
        and len(timotion_history_records) == len(timotion_expected_epochs)
        and len(timotion_history_by_epoch) == len(timotion_expected_epochs)
        and sorted(timotion_history_by_epoch) == timotion_expected_epochs,
        'timotion_validation_history',
        'TiMotion validation ledger was not finalized after all 106,500 updates',
    )
    require_artifact(
        same_resolved_path(
            timotion_history_manifest_path,
            timotion_selection.get('training_manifest', {}).get('path'),
        )
        and timotion_history.get('training_manifest', {}).get('sha256')
        == timotion_selection.get('training_manifest', {}).get('sha256')
        and same_resolved_path(
            timotion_history_finalizer_path,
            'tools/finalize_timotion_validation_history.py',
        )
        and timotion_history.get('implementation', {}).get('sha256')
        == sha256_file('tools/finalize_timotion_validation_history.py'),
        'timotion_validation_history',
        'TiMotion validation ledger is not bound to the completed manifest '
        'and current finalizer',
    )
    for candidate in timotion_candidates:
        history_record = timotion_history_by_epoch[candidate['epoch']]
        normalized_history_record = {
            key: history_record.get(key)
            for key in (
                'epoch', 'step', 'samples', 'test_set_used', 'checkpoint',
                'generations', 'metrics', 'values',
            )
        }
        require_artifact(
            normalized_history_record == candidate,
            'timotion_validation_history',
            f"epoch {candidate['epoch']} differs from the selected candidate",
        )

    for label, expected_step in (
        ('coshmdm_final_state_audit', cosh_selected_step),
        ('timotion_final_state_audit', 106500),
    ):
        audit = records[label]['payload']
        require_artifact(audit.get('status') == 'pass', label, 'status is not pass')
        require_artifact(audit.get('global_step') == expected_step, label,
                         f'expected global step {expected_step}')
        validate_bound_file(label, audit)
    timotion_final_audit = records['timotion_final_state_audit']['payload']
    require_artifact(
        same_resolved_path(
            timotion_selection['final_full_state_checkpoint']['path'],
            timotion_final_audit.get('checkpoint'),
        )
        and timotion_selection['final_full_state_checkpoint']['sha256']
        == timotion_final_audit.get('checkpoint_sha256'),
        'timotion_validation_selection',
        'final training state is not bound separately from selected weights',
    )

    expected_cosh_runtime = {
        'dataset': 'interx',
        'stats_prefix': 'interx_global',
        'normalization_mean_sha256': (
            'aa281b878b8a71c6d10c001059b29374921c18545c082102bf69a404982f568d'
        ),
        'normalization_std_sha256': (
            '30a83e92965a465301a3642eb55b2ba705d3dceaba05e45846aa433d3900b104'
        ),
    }
    coshmdm_contract = records['coshmdm_training_contract_audit']['payload']
    coshmdm_training = coshmdm_contract.get('training', {})
    coshmdm_dataset = coshmdm_contract.get('dataset', {})
    coshmdm_release = coshmdm_contract.get(
        'released_checkpoint_compatibility', {}
    )
    require_artifact(
        coshmdm_contract.get('schema')
        == 'hhc_coshmdm_interx_training_contract_v1'
        and coshmdm_contract.get('status') == 'pass'
        and coshmdm_contract.get('frozen_files_independently_rehashed') is True
        and len(coshmdm_contract.get('frozen_files', {})) == 13
        and coshmdm_contract.get('repository', {}).get(
            'matches_training_launch') is True
        and coshmdm_training.get('epochs') == 2000
        and coshmdm_training.get('expected_optimizer_updates') == 568000
        and coshmdm_training.get('effective_batch_size') == 64
        and coshmdm_release.get('documented_pretrained_model_links') == 1
        and coshmdm_release.get(
            'documented_interx_checkpoint_available'
        ) is False
        and coshmdm_release.get('tracked_checkpoint_paths') == []
        and coshmdm_release.get('upstream_checkpoint_download_label')
        == 'InterHuman'
        and coshmdm_release.get('upstream_pretrain_download_sha256')
        == '1fa25981b86a93f3957a7a91854be0ea9a28d89066ee54626729a80d7a38eaac'
        and coshmdm_dataset.get('independent_split_counts')
        == {'train': 9110, 'val': 570, 'test': 1708}
        and not any(coshmdm_dataset.get(
            'independent_split_overlap', {}).values()),
        'coshmdm_training_contract_audit',
        'CoShMDM source/data/budget contract is incomplete or changed',
    )
    validate_artifact_record(
        'coshmdm_training_contract_audit/training_manifest',
        coshmdm_contract.get('training_manifest', {}),
    )
    validate_artifact_record(
        'coshmdm_training_contract_audit/statistics_manifest',
        coshmdm_contract.get('statistics', {}).get('manifest', {}),
    )
    cosh_resume = records['coshmdm_resume_provenance']['payload']
    cosh_startup = records['coshmdm_statsfix_startup_audit']['payload']
    cosh_first_epoch = cosh_startup.get('first_epoch', {})
    require_artifact(
        cosh_startup.get('schema') == 'hhc_coshmdm_statsfix_startup_audit_v2'
        and cosh_startup.get('status') == 'pass'
        and cosh_startup.get('runtime_contract') == expected_cosh_runtime
        and cosh_first_epoch.get('optimizer_updates') == 284
        and cosh_first_epoch.get(
            'dropped_incomplete_accumulation_micro_batches') == 1
        and cosh_first_epoch.get('last_logged_decade_update') == 280
        and cosh_first_epoch.get('first_epoch_1_logged_update') == 290,
        'coshmdm_statsfix_startup_audit',
        'CoShMDM stats-fixed runtime did not prove the exact epoch-0 batch budget',
    )
    cosh_epoch_series = cosh_startup.get('epoch_boundary_series', {})
    cosh_maximum_logged_update = int(
        cosh_resume.get('maximum_logged_update', 0)
    )
    expected_completed_cosh_epochs = cosh_maximum_logged_update // 284
    require_artifact(
        cosh_epoch_series.get('recorded_completed_epochs')
        == expected_completed_cosh_epochs
        and cosh_epoch_series.get('first_epoch') == 0
        and cosh_epoch_series.get('last_epoch')
        == expected_completed_cosh_epochs - 1
        and cosh_epoch_series.get('steps_contiguous_from_zero') is True
        and cosh_epoch_series.get('all_discard_values_equal_one') is True,
        'coshmdm_statsfix_startup_audit',
        'CoShMDM did not prove one discarded half-batch at every completed '
        'epoch inside the disclosed two-day window',
    )
    for name in ('runtime_manifest', 'training_log', 'implementation'):
        validate_artifact_record(
            f'coshmdm_statsfix_startup_audit/{name}',
            cosh_startup.get(name, {}),
        )
    event_files = cosh_startup.get('event_files', [])
    require_artifact(
        bool(event_files), 'coshmdm_statsfix_startup_audit',
        'no TensorBoard event files bind the epoch-boundary proof',
    )
    for index, record in enumerate(event_files):
        validate_artifact_record(
            f'coshmdm_statsfix_startup_audit/event_file_{index}', record
        )
    validate_coshmdm_representation_adapter(
        records['coshmdm_representation_adapter_audit']['payload']
    )
    cosh_trajectory = records[
        'coshmdm_training_trajectory_diagnosis'
    ]['payload']
    trajectory_milestones = cosh_trajectory.get('milestones', [])
    trajectory_checks = cosh_trajectory.get('checks', {})
    require_artifact(
        cosh_trajectory.get('schema')
        == 'hhc_coshmdm_training_trajectory_diagnosis_v1'
        and cosh_trajectory.get('status') == 'complete'
        and [item.get('step') for item in trajectory_milestones]
        == [20000, 40000, 60000, 80000, 100000]
        and trajectory_checks.get('all_logged_losses_finite') is True
        and trajectory_checks.get('all_milestone_weights_finite') is True
        and trajectory_checks.get(
            'training_total_loss_decreases_at_every_milestone'
        ) is True
        and trajectory_checks.get(
            'validation_fid_worsens_after_selected_checkpoint'
        ) is True
        and trajectory_checks.get('historical_gradient_norms_available') is False
        and cosh_trajectory.get('selection_manifest', {}).get('sha256')
        == records['coshmdm_validation_selection']['artifact']['sha256'],
        'coshmdm_training_trajectory_diagnosis',
        'CoShMDM loss/weight/FID trajectory diagnosis is incomplete or stale',
    )
    for item in trajectory_milestones:
        require_artifact(
            item.get('checkpoint', {}).get('nonfinite_weight_values') == 0
            and math.isfinite(float(
                item.get('checkpoint', {}).get('weight_l2_norm', float('nan'))
            ))
            and math.isfinite(float(
                item.get('training_scalars', {}).get('total', {}).get(
                    'trailing_window', {}
                ).get('median', float('nan'))
            )),
            'coshmdm_training_trajectory_diagnosis',
            f"invalid milestone evidence at step {item.get('step')}",
        )

    representation_oracle = records[
        'coshmdm_representation_oracle_metrics'
    ]['payload']
    embedding_gap = records['coshmdm_embedding_gap_diagnosis']['payload']
    gaps = embedding_gap.get('gaps', {})
    require_artifact(
        representation_oracle.get('schema')
        == 'fair_interx_main_only_metrics_v1'
        and representation_oracle.get('status') == 'complete'
        and representation_oracle.get('samples') == 1696
        and representation_oracle.get('main_generation_metadata', {}).get(
            'diagnostic_only'
        ) is True
        and representation_oracle.get('main_generation_metadata', {}).get(
            'not_a_body_controlled_fid'
        ) is True
        and embedding_gap.get('schema')
        == 'hhc_coshmdm_embedding_gap_diagnosis_v1'
        and embedding_gap.get('status') == 'complete'
        and embedding_gap.get('not_a_body_controlled_fid') is True
        and embedding_gap.get('not_a_reported_comparison_row') is True
        and math.isclose(
            float(gaps.get(
                'official_ground_truth_to_representation_oracle', {}
            ).get('fid', float('nan'))),
            float(representation_oracle.get('fid', float('nan'))),
            rel_tol=0.0, abs_tol=1e-4,
        )
        and embedding_gap.get('representation_oracle', {}).get('sha256')
        == representation_oracle.get('main_generations_sha256')
        and embedding_gap.get('evaluator_provenance', {}).get(
            'checkpoint', {}
        ).get('sha256')
        == 'b1699f13276c044d8b7ae5f6063eb241b7ef96d7e92c878857abd5081e40701e',
        'coshmdm_embedding_gap_diagnosis',
        'diagnostic representation-oracle embedding decomposition is invalid',
    )
    for label in ('requests', 'model_generations', 'representation_oracle'):
        validate_artifact_record(
            f'coshmdm_embedding_gap_diagnosis/{label}',
            embedding_gap.get(label, {}),
        )
    validate_artifact_record(
        'coshmdm_embedding_gap_diagnosis/implementation',
        embedding_gap.get('implementation', {}),
    )
    resume_attempts = cosh_resume.get('training_attempts', [])
    require_artifact(
        cosh_resume.get('schema') == 'hhc_coshmdm_training_provenance_v2'
        and cosh_resume.get('status') == 'pass_user_capped'
        and cosh_resume.get('initial_global_step') == 0
        and cosh_resume.get('expected_final_global_step') == 568000
        and cosh_resume.get('original_author_budget_completed') is False
        and cosh_resume.get('user_capped_run_complete') is True
        and cosh_resume.get('selected_global_step') == cosh_selected_step
        and cosh_resume.get('maximum_logged_update') >= cosh_selected_step
        and cosh_resume.get('runtime_contract') == expected_cosh_runtime
        and cosh_resume.get('training_classification')
        == ('fresh Inter-X retraining with checkpoint-bound interx_global '
            'normalization; user-capped to two additional days; validation-'
            'selected milestone; mixed-stat predecessor excluded')
        and len(resume_attempts) >= 1
        and not any(item.get('segmentation_fault') for item in resume_attempts)
        and max(item.get('maximum_logged_update') or 0
                for item in resume_attempts)
        == cosh_resume.get('maximum_logged_update')
        and 'checkpoint-embedded runtime contract exactly equals'
        in cosh_resume.get('resume_policy', '')
        and 'distributionally but not bitwise equivalent'
        in cosh_resume.get('reproducibility_limitation', '')
        ,
        'coshmdm_resume_provenance',
        'CoShMDM corrected training provenance is incomplete or overstated',
    )
    for artifact_name in (
        'budget_cap_manifest', 'validation_selection_manifest'
    ):
        validate_artifact_record(
            f'coshmdm_resume_provenance/{artifact_name}',
            cosh_resume.get(artifact_name, {}),
        )
    validate_artifact_record(
        'coshmdm_resume_provenance/training_manifest',
        cosh_resume.get('training_manifest', {}),
    )
    validate_artifact_record(
        'coshmdm_resume_provenance/runtime_manifest',
        cosh_resume.get('runtime_manifest', {}),
    )
    require_artifact(
        cosh_startup.get('runtime_manifest', {}).get('sha256')
        == cosh_resume.get('runtime_manifest', {}).get('sha256')
        and same_resolved_path(
            cosh_startup.get('runtime_manifest', {}).get('path'),
            cosh_resume.get('runtime_manifest', {}).get('path'),
        )
        and same_resolved_path(
            cosh_startup.get('training_log', {}).get('path'),
            resume_attempts[0].get('artifact', {}).get('path'),
        ),
        'coshmdm_statsfix_startup_audit',
        'epoch-0 proof is not bound to the final CoShMDM runtime/log',
    )
    validate_artifact_record(
        'coshmdm_resume_provenance/excluded_mixed_stats_run',
        cosh_resume.get('excluded_mixed_stats_run', {}),
    )
    validate_artifact_record(
        'coshmdm_resume_provenance/milestone_20k_checkpoint_audit',
        cosh_resume.get('milestone_20k_checkpoint_audit', {}),
    )
    cosh_milestone_20k = cosh_resume.get('milestone_20k_checkpoint', {})
    validate_artifact_record(
        'coshmdm_resume_provenance/milestone_20k_checkpoint',
        cosh_milestone_20k,
    )
    require_artifact(
        cosh_milestone_20k.get('sha256')
        == '140a2496e941615c3377cb186f187c2c283455e2fb033632ba5694d5387a338d'
        and Path(cosh_milestone_20k.get('path', '')).name
        == 'milestone-000020000.ckpt',
        'coshmdm_resume_provenance/milestone_20k_checkpoint',
        'preserved stats-fixed 20k checkpoint identity changed',
    )
    validate_artifact_record(
        'coshmdm_resume_provenance/queue_source',
        cosh_resume.get('queue_source', {}),
    )
    validate_artifact_record(
        'coshmdm_resume_provenance/implementation',
        cosh_resume.get('implementation', {}),
    )
    for index, attempt in enumerate(resume_attempts):
        validate_artifact_record(
            f'coshmdm_resume_provenance/training_attempt_{index}',
            attempt.get('artifact', {}),
        )
    resume_final_audit = cosh_resume.get('final_checkpoint_audit', {})
    validate_artifact_record(
        'coshmdm_resume_provenance/final_checkpoint_audit',
        resume_final_audit,
    )
    require_artifact(
        resume_final_audit.get('sha256')
        == records['coshmdm_final_state_audit']['artifact']['sha256']
        and same_resolved_path(
            resume_final_audit.get('path'),
            records['coshmdm_final_state_audit']['artifact']['path'],
        ),
        'coshmdm_resume_provenance/coshmdm_final_state_audit',
        'resume provenance binds a different final checkpoint audit',
    )

    timotion_contract = records['timotion_training_contract_audit']['payload']
    timotion_adapter_equivalence = records[
        'timotion_cosh_adapter_equivalence_audit'
    ]['payload']
    require_artifact(
        timotion_adapter_equivalence.get('schema')
        == 'timotion_cosh_adapter_equivalence_audit_v1'
        and timotion_adapter_equivalence.get('status') == 'pass'
        and all(timotion_adapter_equivalence.get('checks', {}).values())
        and timotion_adapter_equivalence.get('contract') == {
            'input': 'released two-person 262-D representation',
            'output': 'official raw Inter-X 672-D representation',
            'formal_path': (
                'shape-specific all-joint root fit; identity hands'
            ),
            'ground_truth_motion_consumed': False,
        }
        and all(
            case.get('bitwise_equal') is True
            and case.get('max_abs') == 0.0
            and case.get('finite') is True
            for case in timotion_adapter_equivalence.get('cases', {}).values()
        )
        and len(timotion_adapter_equivalence.get('cases', {})) == 8,
        'timotion_cosh_adapter_equivalence_audit',
        'TiMotion and CoShMDM do not share a proven identical 262-D to '
        'official-672-D evaluation adapter',
    )
    for source_label, source_record in (
        timotion_adapter_equivalence.get('sources', {})
    ).items():
        validate_artifact_record(
            f'timotion_cosh_adapter_equivalence_audit/{source_label}',
            source_record,
        )
    validate_artifact_record(
        'timotion_cosh_adapter_equivalence_audit/implementation',
        timotion_adapter_equivalence.get('implementation', {}),
    )
    timotion_dataset = timotion_contract.get('dataset', {})
    timotion_budget = timotion_contract.get('paper_budget', {})
    timotion_release = timotion_contract.get(
        'released_checkpoint_compatibility', {}
    )
    timotion_gradient = timotion_contract.get('gradient_update_contract', {})
    require_artifact(
        timotion_contract.get('schema')
        == 'hhc_timotion_interx_training_contract_v1'
        and timotion_contract.get('status') == 'pass'
        and timotion_contract.get('tree_contents_independently_rehashed') is True
        and timotion_contract.get('public_repository_revision')
        == '172580fd7f2551a7cc10fde1f15ff73315742e65'
        and timotion_budget.get('epochs') == 1500
        and timotion_budget.get('global_batch') == 256
        and timotion_budget.get('optimizer_updates') == 106500
        and timotion_budget.get('sample_exposures') == 27264000
        and timotion_budget.get(
            'public_readme_default_1900_overridden_by_paper'
        ) is True
        and timotion_release.get('documented_pretrained_model_links') == 2
        and timotion_release.get(
            'documented_interx_checkpoint_available'
        ) is False
        and timotion_release.get('tracked_interx_paths') == []
        and timotion_release.get('tracked_checkpoint_paths') == []
        and timotion_release.get('upstream_readme_sha256')
        == '48377e5b2031acd18ba8bd30b97a6cbcc692bc3514b960ad18e2bee2421a22eb'
        and timotion_release.get('upstream_dataset_config_sha256')
        == 'ff441a5d6c577196f1a5c0975c9dd32e1acd00fb2cac1be8fc575c21db56a4e7'
        and timotion_release.get('upstream_evaluator_download_sha256')
        == 'fe36d172784a55f37a418ea6896515d12694a51339439ef360f92367201a1df0'
        and timotion_dataset.get('union_count') == 11388
        and timotion_dataset.get('converted_pair_count') == 11388
        and timotion_dataset.get('all_h5_exact_id_match') is True
        and timotion_dataset.get('all_train_ids_pass_loader_minimum') is True
        and timotion_dataset.get('normalization_train_only_recompute') is True
        and timotion_dataset.get('below_15_frames_by_split', {}).get(
            'train', {}
        ).get('count') == 0,
        'timotion_training_contract_audit',
        'TiMotion paper/data/source/restart contract is incomplete or changed',
    )
    require_artifact(
        timotion_gradient == {
            'accumulated_loss_divisor': 8,
            'telemetry_is_pre_clip': True,
            'gradient_clip_global_norm': 0.5,
            'clip_precedes_optimizer_step': True,
            'matches_released_tools_train_py': True,
        },
        'timotion_training_contract_audit',
        'TiMotion accumulated-gradient clipping no longer matches the '
        'released eight-GPU update ordering',
    )
    timotion_manifest_path = validate_artifact_record(
        'timotion_training_contract_audit/manifest',
        timotion_contract.get('manifest', {}),
    )
    timotion_training_manifest = json.loads(
        timotion_manifest_path.read_text(encoding='utf-8')
    )
    require_artifact(
        same_resolved_path(
            timotion_selection['training_manifest']['path'],
            timotion_contract.get('manifest', {}).get('path'),
        )
        and timotion_selection['training_manifest']['sha256']
        == timotion_contract.get('manifest', {}).get('sha256'),
        'timotion_validation_selection',
        'checkpoint selection is not bound to the audited training manifest',
    )

    timotion_padding = records[
        'timotion_padding_ro_stabilization_audit'
    ]['payload']
    timotion_padding_checks = timotion_padding.get('checks', {})
    require_artifact(
        timotion_padding.get('schema')
        == 'timotion_padding_ro_stabilization_audit_v1'
        and timotion_padding.get('status') == 'pass'
        and timotion_padding_checks
        and all(timotion_padding_checks.values())
        and timotion_padding.get('failed_smoke_nonfinite_motion_values')
        == 18324708
        and timotion_padding.get('synthetic_runtime', {}).get(
            'old_masked_loss_finite'
        ) is False
        and timotion_padding.get('synthetic_runtime', {}).get(
            'new_nonfinite_target_quaternions'
        ) == 0
        and timotion_padding.get('source', {}).get('sha256')
        == timotion_contract.get('artifacts', {}).get('losses', {}).get(
            'sha256'
        ),
        'timotion_padding_ro_stabilization_audit',
        'TiMotion padding-only RO stabilization is incomplete or is not '
        'bound to the formally trained loss source',
    )
    remote_wandb_path = validate_artifact_record(
        'timotion_padding_ro_stabilization_audit/remote_wandb_audit',
        timotion_padding.get('remote_wandb_audit', {}),
    )
    remote_wandb = json.loads(remote_wandb_path.read_text(encoding='utf-8'))
    require_artifact(
        remote_wandb.get('schema')
        == 'hhc_wandb_online_metric_history_audit_v1'
        and remote_wandb.get('status') == 'pass'
        and remote_wandb.get('run_id') == 'oi4wa5qn'
        and remote_wandb.get('expected_final_step') == 50
        and all(remote_wandb.get('checks', {}).values())
        and remote_wandb.get('metrics', {}).get(
            'train/nonfinite_gradients', {}
        ).get('maximum') == 0.0
        and remote_wandb.get('metrics', {}).get(
            'train/nonfinite_weights', {}
        ).get('maximum') == 0.0,
        'timotion_padding_ro_stabilization_audit',
        'the finite replacement smoke is not backed by a completed remote '
        'online-W&B metric history',
    )
    timotion_exclusion = records[
        'timotion_interrupted_run_exclusion'
    ]['payload']
    require_artifact(
        timotion_exclusion.get('schema')
        == 'timotion_training_run_exclusion_v1'
        and timotion_exclusion.get('status') == 'excluded'
        and timotion_exclusion.get('run_id') == 'jqh616am'
        and timotion_exclusion.get('remote_state') == 'failed'
        and timotion_exclusion.get('result_eligible') is False
        and timotion_exclusion.get('checkpoint_retained') is False
        and 'already executed the released global gradient clip norm 0.5'
        in timotion_exclusion.get('important_correction', '')
        and 'optimizer step 0'
        in timotion_exclusion.get('replacement_policy', ''),
        'timotion_interrupted_run_exclusion',
        'the interrupted no-checkpoint TiMotion run is not explicitly '
        'excluded or its gradient-clip correction is missing',
    )
    timotion_accumulation = records[
        'timotion_gradient_accumulation_audit'
    ]['payload']
    accumulation_runtime = timotion_accumulation.get(
        'runtime_equivalence', {}
    )
    require_artifact(
        timotion_accumulation.get('schema')
        == 'timotion_gradient_accumulation_audit_v1'
        and timotion_accumulation.get('status') == 'pass'
        and timotion_accumulation.get('contract', {}).get(
            'paper_global_batch'
        ) == 256
        and timotion_accumulation.get('contract', {}).get(
            'accumulation_steps'
        ) == 8
        and timotion_accumulation.get('contract', {}).get(
            'gradient_clip_global_norm'
        ) == 0.5
        and accumulation_runtime.get(
            'pre_clip_gradient_max_abs_difference', math.inf
        ) <= 1e-14
        and accumulation_runtime.get(
            'post_clip_gradient_max_abs_difference', math.inf
        ) <= 1e-14
        and accumulation_runtime.get(
            'post_adamw_weight_max_abs_difference', math.inf
        ) <= 1e-14
        and timotion_accumulation.get('sources', {}).get(
            'fair_runner', {}
        ).get('sha256')
        == timotion_contract.get('artifacts', {}).get('runner', {}).get(
            'sha256'
        ),
        'timotion_gradient_accumulation_audit',
        'the one-GPU accumulated update is not numerically equivalent to '
        'the released global-batch gradient/clip/AdamW update',
    )
    timotion_warmup = records[
        'timotion_warmup_stability_audit'
    ]['payload']
    warmup_metrics = timotion_warmup.get('metrics', {})
    require_artifact(
        timotion_warmup.get('schema')
        == 'hhc_wandb_warmup_stability_audit_v1'
        and timotion_warmup.get('status') == 'pass'
        and timotion_warmup.get('warmup_epochs') == 10
        and timotion_warmup.get('completed_epochs', 0) >= 10
        and timotion_warmup.get('optimizer_updates_per_epoch') == 71
        and timotion_warmup.get('telemetry_interval_optimizer_steps') == 50
        and timotion_warmup.get('minimum_expected_telemetry_events') == 14
        and all(timotion_warmup.get('checks', {}).values())
        and timotion_warmup.get('run_path', '').endswith(
            '/' + timotion_training_manifest.get('observability', {}).get(
                'wandb_run_id', '__missing__'
            )
        )
        and warmup_metrics.get('train/nonfinite_gradients', {}).get(
            'maximum'
        ) == 0.0
        and warmup_metrics.get('train/nonfinite_weights', {}).get(
            'maximum'
        ) == 0.0
        and warmup_metrics.get('train/weight_l2_norm', {}).get(
            'relative_span_from_first', math.inf
        ) <= timotion_warmup.get('max_weight_relative_span', 0.0)
        and warmup_metrics.get('train/weight_max_abs', {}).get(
            'relative_span_from_first', math.inf
        ) <= timotion_warmup.get('max_weight_relative_span', 0.0),
        'timotion_warmup_stability_audit',
        'the formal TiMotion replacement lacks a finite, weight-stable '
        'online-W&B audit through all ten warm-up epochs',
    )
    timotion_spike = records[
        'timotion_transient_velocity_spike_audit'
    ]['payload']
    require_artifact(
        timotion_spike.get('schema')
        == 'timotion_transient_velocity_spike_audit_v1'
        and timotion_spike.get('status') == 'pass'
        and all(timotion_spike.get('checks', {}).values())
        and timotion_spike.get('interpretation_policy', '').startswith(
            'post-hoc operational diagnosis only'
        )
        and timotion_spike.get('run_path', '').endswith(
            '/' + timotion_training_manifest.get('observability', {}).get(
                'wandb_run_id', '__missing__'
            )
        )
        and timotion_spike.get('telemetry_latest', {}).get(
            'train/nonfinite_gradients'
        ) == 0.0
        and timotion_spike.get('telemetry_latest', {}).get(
            'train/nonfinite_weights'
        ) == 0.0,
        'timotion_transient_velocity_spike_audit',
        'the TiMotion velocity-loss excursion lacks a finite, source-bound '
        'post-hoc W&B diagnosis',
    )
    timotion_scheduler = records[
        'timotion_scheduler_boundary_audit'
    ]['payload']
    scheduler_contract = timotion_scheduler.get('contract', {})
    scheduler_transitions = {
        row.get('epoch_zero_based'): row
        for row in timotion_scheduler.get('transitions', [])
    }
    require_artifact(
        timotion_scheduler.get('schema')
        == 'timotion_released_scheduler_boundary_audit_v1'
        and timotion_scheduler.get('status') == 'pass'
        and all(timotion_scheduler.get('checks', {}).values())
        and scheduler_contract.get('base_learning_rate') == 1e-4
        and scheduler_contract.get('warmup_parameter') == 10
        and scheduler_contract.get('cosine_horizon_epochs') == 2000
        and scheduler_contract.get('optimizer_updates_per_epoch') == 71
        and scheduler_contract.get('port_policy')
        == 'preserve released scheduler behavior exactly'
        and scheduler_transitions.get(10, {}).get(
            'observed_learning_rate', 0.0
        ) > 1e-4
        and scheduler_transitions.get(11, {}).get(
            'observed_learning_rate', math.inf
        ) < 1e-4
        and timotion_scheduler.get('source', {}).get('sha256')
        == timotion_contract.get('artifacts', {}).get(
            'scheduler_and_model_utils', {}
        ).get('sha256'),
        'timotion_scheduler_boundary_audit',
        'the formal TiMotion learning-rate trace no longer reproduces the '
        'released inclusive warm-up boundary and cosine transition',
    )
    timotion_formal_wandb = records[
        'timotion_formal_wandb_history_audit'
    ]['payload']
    require_artifact(
        timotion_formal_wandb.get('schema')
        == 'hhc_wandb_online_metric_history_audit_v1'
        and timotion_formal_wandb.get('status') == 'pass'
        and timotion_formal_wandb.get('expected_final_step') == 106500
        and timotion_formal_wandb.get('metric_profile')
        == 'diffusion_regression_20260831_launch'
        and timotion_formal_wandb.get('remote_state') == 'finished'
        and all(timotion_formal_wandb.get('checks', {}).values())
        and timotion_formal_wandb.get('checks', {}).get(
            'all_training_metrics_reach_expected_step'
        ) is True
        and timotion_formal_wandb.get('checks', {}).get(
            'all_validation_metrics_follow_final_training_event'
        ) is True
        and timotion_formal_wandb.get('checks', {}).get(
            'nonfinite_counts_zero'
        ) is True
        and timotion_formal_wandb.get('optimizer_step_axis')
        == 'trainer/global_step'
        and timotion_formal_wandb.get('implementation', {}).get('sha256')
        == sha256_file('tools/audit_wandb_online_metric_history.py')
        and timotion_formal_wandb.get('run_id')
        == timotion_training_manifest.get('observability', {}).get(
            'wandb_run_id'
        )
        and complete_wandb_norm_telemetry(
            timotion_formal_wandb, 106500
        )
        and timotion_formal_wandb.get('metrics', {}).get(
            'train/nonfinite_gradients', {}
        ).get('maximum') == 0.0
        and timotion_formal_wandb.get('metrics', {}).get(
            'train/nonfinite_weights', {}
        ).get('maximum') == 0.0,
        'timotion_formal_wandb_history_audit',
        'the completed TiMotion result is not backed by a full online-W&B '
        'history through optimizer step 106,500',
    )

    optimization = records['optimization_ablation_readiness']['payload']
    require_artifact(
        optimization.get('schema') == 'hhc_optimization_ablation_readiness_v1'
        and optimization.get('status') in {'ready', 'pending'}
        and optimization.get('expected_samples') == 1696
        and optimization.get('expected_converted_npz_files') == 3392
        and optimization.get('authentication_secret_recorded') is False
        and isinstance(
            optimization.get('codex_chatgpt_login_available'), bool
        )
        and optimization.get('ready') == (optimization.get('status') == 'ready'),
        'optimization_ablation_readiness',
        'strict optimizer readiness contract is incomplete or changed',
    )
    validate_artifact_record(
        'optimization_ablation_readiness/implementation',
        optimization.get('implementation', {}),
    )
    readiness_artifacts = optimization.get('artifacts', {})
    for artifact_name in (
        'requests', 'expected_split', 'contact_extraction_manifest',
        'contact_cache_audit',
    ):
        record = readiness_artifacts.get(artifact_name, {})
        require_artifact(
            record.get('exists') is True,
            'optimization_ablation_readiness',
            f'required readiness artifact is missing: {artifact_name}',
        )
        validate_artifact_record(
            f'optimization_ablation_readiness/{artifact_name}', record
        )
    for artifact_name in ('generation_records', 'conversion_manifest'):
        record = readiness_artifacts.get(artifact_name, {})
        if record.get('exists') is True:
            validate_artifact_record(
                f'optimization_ablation_readiness/{artifact_name}', record
            )
    if optimization.get('status') == 'pending':
        require_artifact(
            bool(optimization.get('blockers')),
            'optimization_ablation_readiness',
            'pending status has no explicit blocker',
        )
    else:
        generation_integrity = optimization.get(
            'generation_record_integrity', {}
        )
        require_artifact(
            optimization.get('converted_npz_files') == 3392
            and not optimization.get('conversion_manifest_mismatches')
            and generation_integrity.get('records') == 1696
            and generation_integrity.get('unique_sample_keys') == 1696
            and generation_integrity.get('request_indices_complete') is True
            and generation_integrity.get(
                'all_npz_paths_and_hashes_verified'
            ) is True
            and not generation_integrity.get('errors'),
            'optimization_ablation_readiness',
            'ready optimizer inputs are not fully bound per sample',
        )

    external = records['review_external_prerequisites']['payload']
    require_artifact(
        external.get('schema') == 'hhc_review_external_prerequisites_v1'
        and external.get('status') == 'pass'
        and isinstance(external.get('unresolved_external_prerequisites'), list)
        and external.get('all_external_prerequisites_ready')
        == (not external.get('unresolved_external_prerequisites')),
        'review_external_prerequisites',
        'external review prerequisite audit is incomplete or inconsistent',
    )
    interhuman_external = external.get('interhuman', {})
    require_artifact(
        interhuman_external.get('local_retraining_authorized') is False
        and interhuman_external.get('execution_scope') == (
            'after Inter-X completion, ingest and protocol-audit the '
            'results produced on the user-designated external server'
        )
        and interhuman_external.get('baseline_checkpoint_policy') == (
            'use compatible author checkpoints first; retrain a baseline '
            'only when its released checkpoint is unavailable or '
            'protocol/data incompatible'
        )
        and 'do not rerun the InterHuman baseline on this machine'
        in interhuman_external.get('required_experiment', ''),
        'review_external_prerequisites',
        'InterHuman scope would rerun an externally assigned baseline or '
        'ignore the author-checkpoint-first policy',
    )
    strict_llm_cache = external.get('strict_llm_contact_cache', {})
    require_artifact(
        strict_llm_cache.get('ready') is True
        and strict_llm_cache.get('samples') == 1696
        and strict_llm_cache.get('metadata_records') == 1696
        and not strict_llm_cache.get('blockers'),
        'review_external_prerequisites',
        'strict 1696-sample Luna contact cache is not complete',
    )
    optimization_inputs = external.get('optimization_inputs', {})
    require_artifact(
        optimization_inputs.get('classification')
        == 'internal_downstream_dependency'
        and optimization_inputs.get('ready') == optimization.get('ready')
        and optimization_inputs.get('blockers')
        == optimization.get('blockers'),
        'review_external_prerequisites',
        'optimizer generation inputs are conflated with external prerequisites',
    )
    prosepose_cache = external.get('chi3d', {}).get(
        'prosepose_test_named_cache_audit', {}
    )
    require_artifact(
        prosepose_cache.get('usable_as_official_chi3d_test') is False,
        'review_external_prerequisites',
        'ProsePose test-named cache is incorrectly usable as official CHI3D test',
    )
    if prosepose_cache.get('cache_detected'):
        require_artifact(
            prosepose_cache.get('contact_subjects') == ['s03']
            and prosepose_cache.get('contact_record_count') == 504
            and prosepose_cache.get('classification') == (
                'internal_experiment_split_of_official_training_subject_s03; '
                'filename_test_does_not_mean_official_chi3d_test_package'
            ),
            'review_external_prerequisites',
            'detected ProsePose CHI3D cache exclusion is absent or changed',
        )
        validate_artifact_record(
            'review_external_prerequisites/prosepose_contact_records',
            prosepose_cache.get('contact_records', {}),
        )
        validate_artifact_record(
            'review_external_prerequisites/prosepose_optimization_cache',
            prosepose_cache.get('processed_optimization_cache', {}),
        )
    else:
        require_artifact(
            prosepose_cache.get('classification')
            == 'cache_absent_not_official_chi3d_test_package',
            'review_external_prerequisites',
            'absent ProsePose cache classification is inconsistent',
        )

    wallclock = records['training_wallclock_audit']['payload']
    require_artifact(
        wallclock.get('schema') == 'hhc_fair_training_wallclock_v1'
        and wallclock.get('status') == 'pass'
        and set(wallclock.get('records', {}))
        == {'hhc_mgt', 'coshmdm', 'timotion'},
        'training_wallclock_audit',
        'main training wall-clock records are incomplete',
    )
    hardware = wallclock.get('hardware_observation', {})
    observed_gpus = {
        gpu.get('index'): gpu for gpu in hardware.get('gpus', [])
        if isinstance(gpu, dict)
    }
    require_artifact(
        hardware.get('source') == 'nvidia-smi --query-gpu=index,name,uuid'
        and {0, 1}.issubset(observed_gpus)
        and all(observed_gpus[index].get('name') for index in (0, 1))
        and all(observed_gpus[index].get('uuid') for index in (0, 1)),
        'training_wallclock_audit',
        'GPU hardware observation is incomplete',
    )
    expected_training = {
        'hhc_mgt': (300000, 96, 1),
        'coshmdm': (cosh_selected_step, 64, 0),
        'timotion': (106500, 256, 0),
    }
    for name, (updates, batch_size, gpu_index) in expected_training.items():
        record = wallclock['records'][name]
        require_artifact(
            record.get('optimizer_updates') == updates
            and record.get('effective_batch_size') == batch_size
            and record.get('sample_exposures') == updates * batch_size
            and record.get('assigned_gpu_index') == gpu_index
            and record.get('gpu_name_observed_at_audit')
            == observed_gpus[gpu_index].get('name')
            and record.get('gpu_uuid_observed_at_audit')
            == observed_gpus[gpu_index].get('uuid')
            and math.isfinite(float(record.get('elapsed_seconds', float('nan'))))
            and float(record.get('elapsed_seconds', 0)) > 0,
            'training_wallclock_audit',
            f'{name} training budget/time is invalid',
        )
        for artifact_name in ('training_manifest', 'final_checkpoint'):
            validate_artifact_record(
                f'training_wallclock_audit/{name}/{artifact_name}',
                record.get(artifact_name, {}),
            )
        if 'final_state_audit' in record:
            validate_artifact_record(
                f'training_wallclock_audit/{name}/final_state_audit',
                record['final_state_audit'],
            )
    require_artifact(
        wallclock['records']['coshmdm'].get(
            'original_author_target_optimizer_updates') == 568000
        and 'two-day wall-clock window' in wallclock['records']['coshmdm'].get(
            'training_budget_contract', ''
        ),
        'training_wallclock_audit',
        'CoShMDM wall-clock record does not disclose the truncated author budget',
    )


def sampler_identity_value_matches(key, actual, expected):
    """Compare sampler identity values without trusting symlink spellings.

    The skeleton bank is content-bound separately.  Its recorded path must
    still exist and resolve to the exact expected directory, but a symlinked
    workspace path and its canonical target are the same asset rather than a
    sampler difference.
    """
    if key == 'skeleton_root':
        try:
            actual_path = Path(actual)
            expected_path = Path(expected)
            return (
                actual_path.is_dir()
                and expected_path.is_dir()
                and actual_path.samefile(expected_path)
            )
        except (OSError, TypeError):
            return False
    return actual == expected


def validate_method_artifacts(label, summary, multimodality, physical, timing,
                              expected_samples=1696):
    require_artifact(summary.get('replications') == 20, label,
                     'semantic summary does not contain 20 replications')
    require_artifact(summary.get('samples_per_replication') == expected_samples,
                     label, f'semantic summary does not bind {expected_samples} samples')
    require_artifact(len(summary.get('inputs', [])) == 20, label,
                     'semantic summary input count is not 20')
    for metric_name in (
        'top1', 'top2', 'top3', 'fid',
        'matching', 'diversity', 'multimodality',
    ):
        record = summary.get('metrics', {}).get(metric_name, {})
        values = record.get('values', [])
        require_artifact(len(values) == 20, label,
                         f'{metric_name} does not contain 20 values')
        require_artifact(all(math.isfinite(float(value)) for value in values),
                         label, f'{metric_name} contains a non-finite value')
        require_artifact(
            all(math.isfinite(float(record.get(key, float('nan'))))
                for key in ('mean', 'ci95')),
            label, f'{metric_name} mean/CI is not finite',
        )

    # Do not trust a directory label or run manifest alone: the aggregate must
    # carry the sampler identity extracted from every main-generation HDF5.
    # This specifically prevents the completed native InterMask 20-step audit
    # from being copied/renamed into the requested common-50 table.
    expected_sampler_identities = {
        HHC_NO_TIMING_LABEL: {
            'sampler': 'sample_intra_cas',
            'sampler_semantics': 'paper_high_confidence_tokens_fixed',
            'cascade_order': 'root,body,hands',
            'sampling_steps': 50,
            'sampling_steps_unit': 'outer_maskgit_iterations',
            'maximum_active_channel_stages_per_outer_iteration': 3,
            'maximum_model_forward_calls_per_outer_iteration_with_cfg': 6,
            'resolved_channels_skip_forward_but_not_outer_iterations': True,
            'temperature': 0.5,
            'cfg_scale': 3.5,
            'contact_guidance': False,
            'contact_guidance_scale': 1.0,
            'contact_map_contract': 'unused',
            'generation_length_policy': 'exact_request_length',
        },
        HHC_ORACLE_TIMING_LABEL: {
            'sampler': 'sample_intra_cas',
            'sampler_semantics': 'paper_high_confidence_tokens_fixed',
            'cascade_order': 'root,body,hands',
            'sampling_steps': 50,
            'sampling_steps_unit': 'outer_maskgit_iterations',
            'maximum_active_channel_stages_per_outer_iteration': 3,
            'maximum_model_forward_calls_per_outer_iteration_with_cfg': 6,
            'resolved_channels_skip_forward_but_not_outer_iterations': True,
            'temperature': 0.5,
            'cfg_scale': 3.5,
            'contact_guidance': True,
            'contact_guidance_scale': 1.0,
            'contact_map_contract': 'paper_formula_train_only',
            'generation_length_policy': 'exact_request_length',
        },
        'intermask_official_exact_common50': {
            'time_steps': 50,
            'cond_scale': 2.0,
            'temperature': 1.0,
            'topk_filter_thres': 0.9,
            'length_policy': 'exact',
        },
        'interact2ar_official_fair_no_gt_first_frame': {
            'sampling_steps': 10,
            'sampling_process': 'author_autoregressive_ancestral_10_steps',
            'strategy': 'author_autoregressive_ancestral',
            'cfg_weight': 2.5,
            'first_frame_conditioning': 'none',
            'generation_length_policy': 'exact_request_length_grouped',
        },
        COSHMDM_LABEL: {
            'strategy': 'ddim50',
            'cfg_weight': 3.5,
            'sampling_steps': 50,
            'sampling_process': 'DDIM50_eta0_CFG3.5',
            'ddim_eta': 0.0,
            'training_diffusion_steps': 1000,
            'normalization_prefix': 'interx_global',
            'request_limit': -1,
            'generation_length_policy': 'exact_request_length_grouped',
        },
        'timotion_transformer_lpa_interx_retrained_fair': {
            'sampling_steps': 50,
            'sampling_process': 'DDIM50_eta0_CFG3.5',
            'cfg_weight': 3.5,
            'ddim_eta': 0.0,
            'training_diffusion_steps': 1000,
            'normalization_prefix': 'interx_global',
            'request_limit': -1,
            'first_frame_conditioning': 'none',
            'generation_length_policy': 'exact_request_length_grouped',
        },
    }
    coshmdm_family_labels = {
        COSHMDM_LABEL,
        'timotion_transformer_lpa_interx_retrained_fair',
    }
    if label in coshmdm_family_labels:
        skeleton_root = (
            Path(__file__).resolve().parents[1] / 'dataset/interx/skeleton'
        ).resolve()
        expected_sampler_identities[label].update({
            'adapter': (
                '22-body rotations + shape-specific all-joint rigid root fit '
                'v2; identity hands'
            ),
            'skeleton_root': str(skeleton_root),
            'skeleton_files_sha256': interx_skeleton_bank_sha256(
                skeleton_root
            ),
        })
    if label == 'timotion_transformer_lpa_interx_retrained_fair':
        expected_sampler_identities[label].update({
            'method': 'public_TIMotion_Transformer_LPA',
            'architecture': 'layers5_heads16_latent512_LPA_AdaLN_conv3x1',
        })
    if label.startswith('hhc_generalization_'):
        expected_sampler_identities[label] = {
            **expected_sampler_identities[HHC_ORACLE_TIMING_LABEL],
        }
    if label in expected_sampler_identities:
        provenance = summary.get('provenance', {})
        require_artifact(
            provenance.get('legacy_provenance_allowed') is False
            and not provenance.get('legacy_provenance_reasons')
            and provenance.get('generation_files_sha256_verified') is True
            and provenance.get('unique_requests_verified') is True,
            label, 'semantic aggregate provenance is incomplete or legacy',
        )
        identity = provenance.get('sampler_identity')
        require_artifact(isinstance(identity, dict), label,
                         'semantic aggregate has no HDF5 sampler identity')
        for key, expected in expected_sampler_identities[label].items():
            require_artifact(
                sampler_identity_value_matches(
                    key, identity.get(key), expected
                ), label,
                f'HDF5 sampler identity {key}={identity.get(key)!r}, '
                f'expected {expected!r}',
            )

    require_artifact(multimodality.get('replications') == 20, label,
                     '435-pair MM summary does not contain 20 replications')
    require_artifact(multimodality.get('legacy_provenance_allowed') is False,
                     label, '435-pair MM summary permits legacy provenance')
    require_artifact(
        [record.get('replication')
         for record in multimodality.get('per_replication', [])]
        == list(range(20)),
        label, '435-pair MM replication IDs are incomplete or reordered',
    )
    exhaustive = multimodality.get('exhaustive_all_pairs', {})
    require_artifact(
        all(math.isfinite(float(exhaustive.get(key, float('nan'))))
            for key in ('mean', 'ci95')),
        label, '435-pair MM mean/CI is not finite',
    )
    require_artifact(
        len(exhaustive.get('values', [])) == 20
        and multimodality.get('official_10_pairs', {}).get('values')
        == summary.get('metrics', {}).get('multimodality', {}).get('values'),
        label, '435-pair MM summary is not tied to official per-repeat values',
    )

    require_artifact(physical.get('samples') == expected_samples, label,
                     f'physical score expected {expected_samples} samples')
    require_artifact(physical.get('repeat') == 0, label,
                     'physical score is not replication 0')
    require_artifact(
        physical.get('protocol') == 'interx_official_v1_physical_joint_v1'
        and physical.get('definition') == (
            'LLM-free joint metrics using the frozen binary contact schedule '
            'and subject-specific Inter-X skeletons; penetration remains a '
            'separate mesh metric.'
        )
        and physical.get('contact_threshold_m') == 0.1
        and physical.get('ground_height_threshold_m') == 0.05,
        label,
        'physical score does not use the frozen LLM-free Inter-X contract',
    )
    require_artifact(physical.get('generation_request_hash_verified') is True,
                     label, 'physical generation request hash was not verified')
    validate_bound_file(label, physical, 'requests', 'requests_sha256')
    validate_bound_file(label, physical, 'generations', 'generations_sha256')
    require_artifact(
        all(math.isfinite(float(physical.get(key, float('nan')))) for key in (
            'contact_f1', 'contact_temporal_iou', 'false_contact_rate',
            'minimum_interperson_joint_distance_m', 'foot_sliding_mps',
        )),
        label, 'physical report contains a missing/non-finite primary metric',
    )

    require_artifact(
        timing.get('schema') == 'fair_interx_generation_wall_timing_summary_v1',
        label, 'unexpected timing schema',
    )
    require_artifact(timing.get('replications') == 20, label,
                     'timing summary does not contain 20 replications')
    require_artifact(timing.get('label') == label, label,
                     'timing summary label does not match method')
    timing_inputs = timing.get('inputs', [])
    require_artifact(len(timing_inputs) == 20, label,
                     'timing summary does not bind 20 timing reports')
    for record in timing_inputs:
        validate_artifact_record(label, record)
    for timing_name in ('main_elapsed_seconds', 'main_seconds_per_sample_amortized'):
        timing_record = timing.get(timing_name, {})
        require_artifact(
            all(math.isfinite(float(timing_record.get(key, float('nan'))))
                for key in ('mean', 'ci95_half_width', 'median', 'p95')),
            label, f'{timing_name} contains non-finite statistics',
        )


def validate_final_mesh_physical_main_table(payload, fair_root,
                                            joint_physical_by_label):
    label = 'final_mesh_physical_main_table'
    expected_order = [method_label for _, method_label in METHODS]
    require_artifact(
        payload.get('schema')
        == 'hhc_final_interx_mesh_physical_main_table_v1'
        and payload.get('status') == 'complete'
        and payload.get('replication_id') == 0
        and payload.get('samples_per_method') == 1696
        and payload.get('method_order') == expected_order
        and payload.get('metric_keys') == list(MESH_PHYSICAL_METRIC_KEYS)
        and payload.get('base_seed') == 1234
        and payload.get('query_chunk') == 512
        and payload.get('device') == 'cuda:0'
        and payload.get('statistical_scope') == (
            'deterministic replication-0 descriptive main-table geometry; '
            'no 20-replication confidence interval or significance claim'
        )
        and set(payload.get('methods', {})) == set(expected_order),
        label,
        'six-method full-test replication-0 mesh protocol is incomplete or changed',
    )
    requests_path = validate_artifact_record(
        f'{label}/requests', payload.get('requests', {})
    )
    require_artifact(
        same_resolved_path(
            requests_path,
            'output/fair_interx_requests/replication_00.npz',
        ),
        label, 'mesh table does not use frozen replication-00 requests',
    )
    repository = Path(__file__).resolve().parents[1]
    expected_implementation = {
        'orchestrator': repository / 'tools/run_final_interx_mesh_physical.py',
        'converter': repository / 'tools/export_fair_h5_to_optimizer_npz.py',
        'scorer': repository / 'tools/score_volumetric_smpl_npz.py',
        'volumetric_backend': (
            repository / 'optimization/volumetric_smpl_backend.py'
        ),
        'npz_parameter_loader': repository / 'eval/calc_pene_from_npz.py',
        'rotation_conversions': repository / 'utils/rotation_conversions.py',
    }
    implementation = payload.get('implementation', {})
    require_artifact(
        set(implementation) == set(expected_implementation), label,
        'mesh implementation set changed',
    )
    for name, expected_path in expected_implementation.items():
        path = validate_artifact_record(
            f'{label}/implementation/{name}', implementation[name]
        )
        require_artifact(
            path.resolve() == expected_path.resolve(), label,
            f'mesh implementation path changed for {name}',
        )

    rows = {}
    for method_label in expected_order:
        method = payload['methods'][method_label]
        require_artifact(
            method.get('label') == method_label
            and method.get('samples') == 1696
            and method.get('repeat_id') == 0
            and set(method.get('aggregate', {}))
            == set(MESH_PHYSICAL_METRIC_KEYS)
            and method.get('bindings_verified') == {
                'same_frozen_requests': True,
                'exact_generation_hdf5': True,
                'request_betas_and_gender': True,
                'complete_generation_record_ledger': True,
                'complete_per_sample_score_ledger': True,
                'released_volumetric_smpl_weights': True,
                'scoring_contract': True,
            },
            f'{label}/{method_label}',
            'method mesh record is incomplete or changed',
        )
        aggregate = method['aggregate']
        for metric_name, statistics in aggregate.items():
            require_artifact(
                set(statistics) == {'mean', 'median'}
                and all(math.isfinite(float(value))
                        for value in statistics.values()),
                f'{label}/{method_label}',
                f'{metric_name} statistics are incomplete or non-finite',
            )
        method_artifacts = method.get('artifacts', {})
        require_artifact(
            set(method_artifacts) == {
                'requests', 'generation', 'method_run_manifest',
                'conversion_manifest', 'conversion_records', 'metrics',
                'method_summary',
            },
            f'{label}/{method_label}', 'method artifact set changed',
        )
        resolved = {
            name: validate_artifact_record(
                f'{label}/{method_label}/{name}', record
            )
            for name, record in method_artifacts.items()
        }
        expected_generation = (
            Path(fair_root).resolve() / method_label /
            'replication_00_main.h5'
        )
        require_artifact(
            resolved['requests'] == requests_path
            and resolved['generation'] == expected_generation.resolve()
            and resolved['method_run_manifest'] == (
                Path(fair_root).resolve() / method_label / 'run_manifest.json'
            ).resolve(),
            f'{label}/{method_label}',
            'method request/generation/run-manifest path differs',
        )
        joint_physical = joint_physical_by_label[method_label]
        require_artifact(
            method_artifacts['requests']['sha256']
            == joint_physical.get('requests_sha256')
            and method_artifacts['generation']['sha256']
            == joint_physical.get('generations_sha256'),
            f'{label}/{method_label}',
            'mesh and joint/contact metrics do not score the same request/HDF5',
        )
        conversion = json.loads(
            resolved['conversion_manifest'].read_text(encoding='utf-8')
        )
        require_artifact(
            conversion.get('schema')
            == 'hhc_fair_h5_to_optimizer_npz_v1'
            and conversion.get('status') == 'complete'
            and conversion.get('samples') == 1696
            and conversion.get('repeat_id') == 0
            and conversion.get('resume_contract')
            == CONVERSION_RESUME_CONTRACT
            and conversion.get('requests_sha256')
            == method_artifacts['requests']['sha256']
            and conversion.get('generations_sha256')
            == method_artifacts['generation']['sha256']
            and conversion.get('generation_records_sha256')
            == method_artifacts['conversion_records']['sha256']
            and conversion.get('implementation', {}).get('converter')
            == implementation['converter']
            and conversion.get('implementation', {}).get(
                'rotation_conversions'
            ) == implementation['rotation_conversions'],
            f'{label}/{method_label}',
            'conversion manifest is not bound to the final method inputs/code',
        )
        metrics = json.loads(resolved['metrics'].read_text(encoding='utf-8'))
        checkpoint_records = metrics.get(
            'volumetric_smpl_checkpoints', {}
        )
        require_artifact(
            metrics.get('schema')
            == 'hhc_volumetric_smpl_physical_metrics_v1'
            and metrics.get('status') == 'complete'
            and metrics.get('sample_count') == 1696
            and metrics.get('base_seed') == 1234
            and metrics.get('max_frames') == 150
            and metrics.get('query_chunk') == 512
            and metrics.get('device') == 'cuda:0'
            and metrics.get('aggregate') == aggregate
            and metrics.get('scoring_contract_sha256')
            == canonical_sha256(metrics.get('scoring_contract', {}))
            and metrics.get('scoring_contract', {}).get('device') == 'cuda:0'
            and {
                gender: record.get('sha256')
                for gender, record in checkpoint_records.items()
            } == VOLUMETRIC_CHECKPOINT_SHA256
            and metrics.get('implementation', {}).get('scorer')
            == implementation['scorer']
            and metrics.get('implementation', {}).get('volumetric_backend')
            == implementation['volumetric_backend']
            and metrics.get('implementation', {}).get('npz_parameter_loader')
            == implementation['npz_parameter_loader'],
            f'{label}/{method_label}',
            'mesh metric manifest is not bound to the common released backend',
        )
        for gender, checkpoint in checkpoint_records.items():
            validate_artifact_record(
                f'{label}/{method_label}/checkpoint/{gender}', checkpoint
            )
        method_summary = json.loads(
            resolved['method_summary'].read_text(encoding='utf-8')
        )
        require_artifact(
            method_summary.get('label') == method_label
            and method_summary.get('aggregate') == aggregate
            and method_summary.get('bindings_verified')
            == method.get('bindings_verified'),
            f'{label}/{method_label}', 'method completion summary differs',
        )
        rows[method_label] = {
            metric_name: {
                'mean': statistics['mean'],
                'median': statistics['median'],
            }
            for metric_name, statistics in aggregate.items()
        }
    return rows


def validate_shape_augmentation_results(shape):
    label = 'shape_augmentation_results'
    preflight = shape.get('input_preflight', {})
    require_artifact(
        shape.get('schema') == 'hhc_interx_shape_augmentation_results_v1'
        and shape.get('status') == 'complete'
        and shape.get('official_split_unchanged') is True
        and shape.get('pilot_samples') == 25
        and shape.get('augmented_samples_per_arm') == 6238
        and set(shape.get('results', {})) == {'random', 'optimized'},
        label, 'matched random/optimized augmentation result is incomplete',
    )
    require_artifact(
        preflight.get('selected_samples') == 6238
        and preflight.get('positive_timing_samples') == 5868
        and preflight.get('semantic_contact_active_samples') == 5666
        and preflight.get('physical_timing_only_samples') == 202
        and preflight.get('regions_inactive_because_timing_is_zero') == 137
        and preflight.get('structured_contact_rows') == 7434
        and preflight.get('boundary_whitespace_caption_matches') == 1,
        label, 'shape augmentation input-preflight counts changed',
    )
    require_artifact(
        shape['results']['random'].get('backend_lifetime')
        == 'not_instantiated_for_random_shape_only_control'
        and shape['results']['optimized'].get('backend_lifetime')
        == (
            'one frozen male/female backend per serial process; fresh '
            'loss state and optimizer per sample'
        ),
        label, 'shape augmentation backend lifecycle contract changed',
    )
    require_artifact(
        {
            'augmentation_input_preflight', 'interx_protocol_audit',
            'shared_backend_equivalence', 'root_tokenizer_wiring',
        } <= set(shape.get('artifacts', {})),
        label, 'shape augmentation result lacks input/protocol provenance',
    )
    for name, record in shape['artifacts'].items():
        validate_artifact_record(f'{label}/{name}', record)
    wiring_path = Path(shape['artifacts']['root_tokenizer_wiring']['path'])
    wiring = json.loads(wiring_path.read_text(encoding='utf-8'))
    require_artifact(
        wiring.get('schema') == 'hhc_shape_root_tokenizer_wiring_audit_v1'
        and wiring.get('status') == 'pass'
        and wiring.get('contract') == 'deterministic_historical_root_v1'
        and wiring.get('missing_checks') == []
        and all(wiring.get('checks', {}).values()),
        label, 'shape root-tokenizer entrypoint wiring is incomplete',
    )
    for name, record in wiring.get('artifacts', {}).items():
        validate_artifact_record(f'{label}/root_tokenizer_wiring/{name}', record)
    equivalence_path = Path(
        shape['artifacts']['shared_backend_equivalence']['path']
    )
    equivalence = json.loads(equivalence_path.read_text(encoding='utf-8'))
    comparisons = (
        equivalence.get('fresh_vs_shared', {}),
        equivalence.get('shared_first_vs_second', {}),
    )
    require_artifact(
        equivalence.get('schema')
        == 'hhc_shared_volumetric_backend_equivalence_v1'
        and equivalence.get('status') == 'pass'
        and equivalence.get('contract') == {
            'device': 'cpu', 'frames': 3, 'epochs': 1,
            'query_chunk': 512, 'seed': 1234, 'threads': 1,
            'deterministic_algorithms': True,
        }
        and equivalence.get('checkpoint_hashes_valid') is True
        and equivalence.get('finite_loss_histories') == {
            'fresh': True, 'shared_first': True, 'shared_second': True,
        }
        and equivalence.get('diagnostic_flags') == {
            'fresh_backend_reused': False,
            'shared_first_backend_reused': True,
            'shared_second_backend_reused': True,
        }
        and all(
            comparison.get('bitwise_equal') is True
            and comparison.get('finite') is True
            and comparison.get('max_abs_difference') == 0.0
            for comparison in comparisons
        ),
        label, 'shared VolumetricSMPL reuse equivalence is unproven',
    )
    expected_volume_hashes = {
        'male': (
            '96b22f6409e67dc950dafdd83bb8dc4779381ecc2b2c752e1c6fa3a205b2e18c'
        ),
        'female': (
            '2b582f009c2daeec50c90b5f39689a248bac96a4cc9983548ad4be677b72ccdb'
        ),
    }
    volume_records = equivalence.get('volumetric_smpl_checkpoints', {})
    implementations = equivalence.get('implementation', {})
    require_artifact(
        set(volume_records) == set(expected_volume_hashes)
        and all(
            volume_records[gender].get('sha256') == digest
            and sha256_file(Path(volume_records[gender]['path'])) == digest
            for gender, digest in expected_volume_hashes.items()
        )
        and len(implementations) == 5
        and all(
            Path(path).is_file() and sha256_file(Path(path)) == digest
            for path, digest in implementations.items()
        ),
        label, 'shared backend audit weights/implementation are stale',
    )


def validate_llm_robustness_results(payload):
    label = 'llm_robustness'
    expected_strata = {
        'timing_negative', 'laterality_positive',
        'multiple_contact_cues_positive', 'role_reference_positive',
    }
    require_artifact(
        payload.get('schema') == 'hhc_interx_llm_robustness_evaluation_v1'
        and payload.get('status') == 'complete'
        and payload.get('sample_count') == 128,
        label, 'expected the completed preregistered 128-sample study',
    )
    require_artifact(
        set(payload.get('stratum_counts', {})) == expected_strata
        and all(value == 32 for value in payload['stratum_counts'].values()),
        label, 'expected four disjoint 32-sample robustness strata',
    )
    require_artifact(
        set(payload.get('results', {})) == {
            'luna', 'terra_reference', 'rule',
        },
        label, 'robustness source set changed',
    )
    require_artifact(
        'not human region ground truth' in payload.get(
            'reference_interpretation', ''
        )
        and 'no human 75-region labels' in payload.get('oracle_scope', ''),
        label, 'reference/oracle scope is overstated',
    )
    for source_name, result in payload['results'].items():
        presence = result.get('contact_presence_vs_manual_timing', {})
        require_artifact(
            presence.get('samples') == 128
            and all(math.isfinite(float(presence.get(key, float('nan'))))
                    for key in ('precision', 'recall', 'f1', 'accuracy')),
            label, f'{source_name} presence metrics are incomplete',
        )
        require_artifact(
            set(result.get('by_stratum', {})) == expected_strata
            and all(record.get('samples') == 32
                    for record in result['by_stratum'].values()),
            label, f'{source_name} stratum metrics are incomplete',
        )
        usage = result.get('resource_usage', {})
        wall = usage.get('wall_seconds') or {}
        require_artifact(
            wall.get('samples') == 128
            and all(math.isfinite(float(wall.get(key, float('nan'))))
                    for key in (
                        'total', 'mean_per_sample', 'median_per_sample',
                        'p95_per_sample',
                    )),
            label, f'{source_name} wall-time evidence is incomplete',
        )
        if source_name != 'rule':
            tokens = usage.get('tokens') or {}
            require_artifact(
                usage.get('wall_untimed_attempts') == 0
                and tokens.get('samples') == 128
                and all(math.isfinite(float(tokens.get(key, float('nan'))))
                        for key in (
                            'total', 'mean_per_sample', 'median_per_sample',
                            'p95_per_sample',
                        )),
                label, f'{source_name} CLI timing/token evidence is incomplete',
            )
        if source_name != 'terra_reference':
            agreement = result.get(
                'part_pair_agreement_vs_terra_reference', {}
            )
            require_artifact(
                set(agreement) == {'ordered', 'unordered'}
                and all(
                    record.get('samples') == 128
                    and all(math.isfinite(float(record.get(key, float('nan'))))
                            for key in ('precision', 'recall', 'f1',
                                        'sample_exact_match_rate'))
                    for record in agreement.values()
                ),
                label, f'{source_name} part-pair agreement is incomplete',
            )
    require_artifact(bool(payload.get('artifacts')), label,
                     'robustness result has no bound artifacts')
    for name, record in payload['artifacts'].items():
        validate_artifact_record(f'{label}/{name}', record)


def validate_hhc_validation_sampler_calibration(payload):
    label = 'hhc_validation_sampler_calibration'
    expected_grid = (
        ('temp0p5_cfg3p5', {'sampling_steps': 50, 'temperature': 0.5,
                            'cfg_scale': 3.5}),
        ('temp0p5_cfg1p0', {'sampling_steps': 50, 'temperature': 0.5,
                            'cfg_scale': 1.0}),
        ('temp0p5_cfg2p5', {'sampling_steps': 50, 'temperature': 0.5,
                            'cfg_scale': 2.5}),
        ('temp0p5_cfg4p5', {'sampling_steps': 50, 'temperature': 0.5,
                            'cfg_scale': 4.5}),
        ('temp0p3_cfg3p5', {'sampling_steps': 50, 'temperature': 0.3,
                            'cfg_scale': 3.5}),
        ('temp0p8_cfg3p5', {'sampling_steps': 50, 'temperature': 0.8,
                            'cfg_scale': 3.5}),
    )
    expected_grid_values = [
        {'temperature': config['temperature'], 'cfg_scale': config['cfg_scale']}
        for _, config in expected_grid
    ]
    bridge = payload.get('formal_row_bridge', {})
    require_artifact(
        payload.get('schema') == 'hhc_validation_sampler_calibration_v1'
        and payload.get('status') == 'complete'
        and payload.get('test_split_used') is False
        and payload.get('sampling_order_contract')
        == 'exact_hhc_training_validation_loader_batch_and_within_batch_order'
        and payload.get('formal_row') == {
            'sampling_steps': 50, 'temperature': 0.5, 'cfg_scale': 3.5,
        }
        and payload.get('grid') == expected_grid_values
        and bridge.get('pass') is True
        and bridge.get('maximum_absolute_difference') == 1e-4
        and bridge.get('bitwise_equivalence_claimed') is False
        and bridge.get('tolerance_basis')
        == ('checkpoint selection records FID to four decimals; standalone '
            'HDF5 re-embedding must reproduce within one 0.0001 unit')
        and math.isfinite(float(bridge.get(
            'training_validation_fid', float('nan')
        )))
        and math.isfinite(float(bridge.get(
            'offline_replay_fid', float('nan')
        )))
        and 0 <= float(bridge.get('absolute_difference', float('inf'))) <= 1e-4,
        label,
        'validation-only common-50 calibration or formal-row bridge failed',
    )
    results = payload.get('results', {})
    require_artifact(
        set(results) == {name for name, _ in expected_grid}
        and payload.get('lowest_validation_fid_row') in results,
        label, 'expected the six preregistered validation rows',
    )
    artifacts = payload.get('artifacts', {})
    expected_artifacts = {
        'selection_manifest', 'checkpoint', 'request_manifest', 'requests',
        'formal_hhc_validation_selection', 'vq_checkpoint', 'contact_map',
        'exporter', 'scorer', 'implementation',
    }
    require_artifact(
        set(artifacts) == expected_artifacts,
        label, 'calibration artifact set changed',
    )
    resolved_artifacts = {
        name: validate_artifact_record(f'{label}/{name}', record)
        for name, record in artifacts.items()
    }
    request_manifest = json.loads(
        resolved_artifacts['request_manifest'].read_text(encoding='utf-8')
    )
    require_artifact(
        request_manifest.get('schema')
        == 'hhc_mgt_validation_bridge_request_v1'
        and request_manifest.get('status') == 'complete'
        and request_manifest.get('samples') == 544
        and request_manifest.get('seed') == 500123
        and request_manifest.get('test_split_used') is False
        and request_manifest.get('sampling_order_contract')
        == 'exact_hhc_training_validation_loader_batch_and_within_batch_order'
        and request_manifest.get(
            'ordered_sampling_records_match_training_selection') is True
        and request_manifest.get(
            'same_record_set_as_common_validation_request') is True
        and request_manifest.get('request', {}).get('sha256')
        == artifacts['requests']['sha256']
        and request_manifest.get(
            'formal_hhc_validation_selection', {}).get('sha256')
        == artifacts['formal_hhc_validation_selection']['sha256'],
        label, 'calibration request is not the exact formal validation set',
    )
    for name in (
        'source_common_request', 'source_common_request_manifest',
        'formal_hhc_validation_selection', 'request', 'implementation',
    ):
        validate_artifact_record(
            f'{label}/request_manifest/{name}', request_manifest.get(name, {})
        )
    for name, expected_config in expected_grid:
        result = results[name]
        require_artifact(
            result.get('configuration') == expected_config
            and all(math.isfinite(float(result.get(metric, float('nan'))))
                    for metric in (
                        'fid', 'diversity', 'top1', 'top2', 'top3', 'matching'
                    )),
            label, f'{name} configuration or metrics are incomplete',
        )
        generation_path = validate_artifact_record(
            f'{label}/{name}/generation', result.get('generation', {})
        )
        metrics_path = validate_artifact_record(
            f'{label}/{name}/metrics', result.get('metrics', {})
        )
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        require_artifact(
            metrics.get('schema') == 'fair_interx_main_only_metrics_v1'
            and metrics.get('status') == 'complete'
            and metrics.get('samples') == 544
            and metrics.get('multimodality') is None
            and metrics.get('requests_sha256') == artifacts['requests']['sha256']
            and metrics.get('main_generations_sha256')
            == result['generation']['sha256']
            and same_resolved_path(
                metrics.get('main_generations'), generation_path
            )
            and all(float(metrics[metric]) == float(result[metric])
                    for metric in (
                        'fid', 'diversity', 'top1', 'top2', 'top3', 'matching'
                    )),
            label, f'{name} summary is not bound to its main-only score',
        )
        validate_main_only_evaluator_provenance(f'{label}/{name}', metrics)
    lowest = min(results, key=lambda name: float(results[name]['fid']))
    require_artifact(
        payload['lowest_validation_fid_row'] == lowest,
        label, 'declared minimum-validation-FID row is wrong',
    )


def validate_review_evidence(evidence):
    paired = evidence['paired_fair_method_comparison']
    require_artifact(
        paired.get('schema') == 'fair_interx_paired_method_comparison_v1',
        'paired_fair_method_comparison', 'unexpected schema',
    )
    require_artifact(paired.get('replications') == 20,
                     'paired_fair_method_comparison',
                     'expected 20 paired replications')
    require_artifact(
        paired.get('reference', {}).get('label') == 'hhc_no_timing'
        and same_resolved_path(
            paired.get('reference', {}).get('directory'),
            Path('output/fair_eval') / HHC_NO_TIMING_LABEL,
        ),
        'paired_fair_method_comparison', 'fair reference is not no-timing HHC',
    )
    require_artifact(
        set(paired.get('comparators', {})) == {
            'intermask', 'interact2ar', 'coshmdm', 'timotion',
        },
        'paired_fair_method_comparison', 'comparator set changed',
    )
    expected_comparator_directories = {
        'intermask': 'intermask_official_exact_common50',
        'interact2ar': 'interact2ar_official_fair_no_gt_first_frame',
        'coshmdm': COSHMDM_LABEL,
        'timotion': 'timotion_transformer_lpa_interx_retrained_fair',
    }
    require_artifact(
        all(same_resolved_path(
            paired['comparators'][name].get('directory'),
            Path('output/fair_eval') / directory,
        ) for name, directory in expected_comparator_directories.items()),
        'paired_fair_method_comparison',
        'paired comparator directory differs from a final method row',
    )
    oracle = evidence['paired_oracle_timing_effect']
    require_artifact(oracle.get('replications') == 20,
                     'paired_oracle_timing_effect',
                     'expected 20 paired replications')
    require_artifact(
        oracle.get('reference', {}).get('label') == 'hhc_oracle'
        and same_resolved_path(
            oracle.get('reference', {}).get('directory'),
            Path('output/fair_eval') / HHC_ORACLE_TIMING_LABEL,
        )
        and set(oracle.get('comparators', {})) == {'hhc_no_timing'}
        and same_resolved_path(
            oracle['comparators']['hhc_no_timing'].get('directory'),
            Path('output/fair_eval') / HHC_NO_TIMING_LABEL,
        ),
        'paired_oracle_timing_effect', 'oracle/no-timing contrast changed',
    )
    for evidence_label, comparison in (
        ('paired_fair_method_comparison', paired),
        ('paired_oracle_timing_effect', oracle),
    ):
        for comparator in comparison.get('comparators', {}).values():
            for metric_record in comparator.get('metrics', {}).values():
                require_artifact(
                    math.isfinite(float(metric_record.get(
                        'holm_familywise_p', float('nan')
                    ))), evidence_label, 'Holm-adjusted p-value is not finite',
                )

    map_sensitivity = evidence[
        'paired_contact_map_implementation_sensitivity_rep0'
    ]
    require_artifact(
        map_sensitivity.get('schema')
        == 'fair_interx_paired_method_comparison_v1'
        and map_sensitivity.get('replications') == 1
        and map_sensitivity.get('replication_ids') == [0]
        and map_sensitivity.get('reference', {}).get('label') == 'fair_map'
        and same_resolved_path(
            map_sensitivity.get('reference', {}).get('directory'),
            Path('output/fair_eval') / HHC_ORACLE_TIMING_LABEL,
        )
        and set(map_sensitivity.get('comparators', {}))
        == {'legacy_bug_map'}
        and same_resolved_path(
            map_sensitivity['comparators']['legacy_bug_map'].get('directory'),
            Path('output/fair_eval') /
            'hhc_formal_raw12288_repaired_mgt_common50_contact_legacy_full_axis_bug_rep00',
        ),
        'paired_contact_map_implementation_sensitivity_rep0',
        'expected the disclosed replication-0 fair/legacy-map sensitivity',
    )
    for metric_record in map_sensitivity[
            'comparators']['legacy_bug_map'].get('metrics', {}).values():
        require_artifact(
            math.isfinite(float(metric_record.get(
                'paired_delta_reference_minus_comparator_mean', float('nan')
            )))
            and math.isfinite(float(metric_record.get(
                'reference_advantage_mean', float('nan')
            ))),
            'paired_contact_map_implementation_sensitivity_rep0',
            'replication-0 effect-size record is incomplete',
        )

    mgt_sensitivity = evidence['paired_hhc_mgt_training_state_sensitivity_rep0']
    require_artifact(
        mgt_sensitivity.get('schema')
        == 'fair_interx_paired_method_comparison_v1'
        and mgt_sensitivity.get('replications') == 1
        and mgt_sensitivity.get('replication_ids') == [0]
        and mgt_sensitivity.get('reference', {}).get('label')
        == 'corrected_paper'
        and same_resolved_path(
            mgt_sensitivity.get('reference', {}).get('directory'),
            Path('output/fair_eval') / HHC_ORACLE_TIMING_LABEL,
        )
        and set(mgt_sensitivity.get('comparators', {})) == {
            'precleanup_latest_complete_50k',
            'precleanup_git_default_280k',
        }
        and same_resolved_path(
            mgt_sensitivity['comparators'][
                'precleanup_latest_complete_50k'
            ].get('directory'),
            'output/fair_eval/hhc_precleanup_latest_complete_mgt_common50_contact_rep00',
        )
        and same_resolved_path(
            mgt_sensitivity['comparators'][
                'precleanup_git_default_280k'
            ].get('directory'),
            'output/fair_eval/hhc_precleanup_git_default_mgt_common50_contact_rep00',
        ),
        'paired_hhc_mgt_training_state_sensitivity_rep0',
        'expected the disclosed corrected/pre-cleanup MGT replication-0 contrast',
    )
    for comparator in mgt_sensitivity['comparators'].values():
        for metric_record in comparator.get('metrics', {}).values():
            require_artifact(
                math.isfinite(float(metric_record.get(
                    'paired_delta_reference_minus_comparator_mean',
                    float('nan'),
                )))
                and math.isfinite(float(metric_record.get(
                    'reference_advantage_mean', float('nan')
                ))),
                'paired_hhc_mgt_training_state_sensitivity_rep0',
                'replication-0 MGT effect-size record is incomplete',
            )

    timing = evidence['contact_timing_ablation']
    require_artifact(
        timing.get('schema') == 'hhc_contact_timing_ablation_summary_v1',
        'contact_timing_ablation', 'unexpected schema',
    )
    require_artifact(
        set(timing.get('rows', {})) == {
            'request_oracle', 'none', 'text_predicted', 'full_valid_weak',
        },
        'contact_timing_ablation', 'timing ablation rows changed',
    )
    sampling = evidence['sampling_pareto']
    require_artifact(
        sampling.get('schema') == 'hhc_sampling_pareto_summary_v1'
        and sampling.get('replication') == 0
        and len(sampling.get('rows', [])) == 8,
        'sampling_pareto', 'expected the preregistered eight-row replication-0 grid',
    )
    validate_hhc_validation_sampler_calibration(
        evidence['hhc_validation_sampler_calibration']
    )

    qualitative = evidence['qualitative_comparison']
    expected_qualitative_methods = [label for _, label in METHODS]
    require_artifact(
        qualitative.get('schema') == 'hhc_fair_interx_qualitative_render_v1'
        and qualitative.get('status') == 'complete'
        and qualitative.get('methods') == expected_qualitative_methods
        and qualitative.get('same_request_repeat_camera_contract_verified') is True
        and qualitative.get('strict_request_hashes_verified') is True
        and len(qualitative.get('figures', [])) == 9,
        'qualitative_comparison',
        'expected nine strict same-request/same-camera comparison figures',
    )
    selection_path = validate_artifact_record(
        'qualitative_comparison/selection', qualitative.get('selection', {})
    )
    for name in ('requests', 'implementation'):
        validate_artifact_record(
            f'qualitative_comparison/{name}', qualitative.get(name, {})
        )
    require_artifact(
        set(qualitative.get('source_generations', {}))
        == set(expected_qualitative_methods),
        'qualitative_comparison', 'qualitative source method set changed',
    )
    for method, record in qualitative['source_generations'].items():
        validate_artifact_record(
            f'qualitative_comparison/source/{method}', record
        )
    expected_categories = {
        'action_hug', 'action_high_five', 'action_handshake', 'action_kick',
        'gender_male_male', 'gender_female_female', 'gender_mixed',
        'shape_max_pair_beta_distance', 'shape_max_subject_beta_norm',
    }
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    require_artifact(
        selection.get('schema')
        == 'hhc_qualitative_comparison_selection_v1'
        and selection.get('status') == 'pass'
        and selection.get('selection_seed') == 20260826
        and selection.get('selection_is_metric_independent') is True
        and selection.get('methods') == expected_qualitative_methods
        and selection.get('view_contract', {}).get(
            'same_request_and_repeat_across_methods'
        ) is True
        and selection.get('view_contract', {}).get(
            'same_camera_and_axis_limits_within_case'
        ) is True
        and selection.get('view_contract', {}).get('frame_positions')
        == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        and {record.get('category') for record in selection.get('records', [])}
        == expected_categories,
        'qualitative_comparison',
        'qualitative selection is not the frozen metric-independent protocol',
    )
    require_artifact(
        {record.get('category') for record in qualitative['figures']}
        == expected_categories,
        'qualitative_comparison', 'qualitative category set changed',
    )
    for figure in qualitative['figures']:
        validate_artifact_record(
            f'qualitative_comparison/figure/{figure.get("category")}',
            figure.get('figure', {}),
        )

    predictor = evidence['contact_timing_predictor_score']
    require_artifact(predictor.get('samples') == 1696,
                     'contact_timing_predictor_score', 'expected 1696 samples')
    validate_bound_file('contact_timing_predictor_score', predictor,
                        'requests', 'requests_sha256')
    validate_bound_file('contact_timing_predictor_score', predictor,
                        'predictions', 'predictions_sha256')

    validate_shape_augmentation_results(evidence['shape_augmentation_results'])
    validate_llm_robustness_results(evidence['llm_robustness'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fair-root', type=Path, default=Path('output/fair_eval'))
    parser.add_argument('--generalization-root', type=Path,
                        default=Path('output/fair_eval_generalization'))
    parser.add_argument('--output-json', type=Path,
                        default=Path('output/review_results_report.json'))
    parser.add_argument('--output-markdown', type=Path,
                        default=Path('docs/review_results_report.md'))
    parser.add_argument(
        '--mesh-physical-manifest', type=Path,
        default=Path('output/fair_eval_mesh_physical/manifest.json'),
    )
    args = parser.parse_args()
    rows, artifacts, method_manifests = [], {}, {}
    joint_physical_by_label = {}
    for display_name, label in METHODS:
        directory = args.fair_root / label
        run_manifest, run_manifest_artifact = load_artifact(
            directory / 'run_manifest.json'
        )
        validate_method_manifest(label, run_manifest)
        method_manifests[label] = run_manifest
        summary, summary_artifact = load_artifact(directory / 'summary.json')
        mm, mm_artifact = load_artifact(
            directory / 'multimodality_audit_summary.json'
        )
        physical, physical_artifact = load_artifact(
            directory / 'replication_00_physical.json'
        )
        timing, timing_artifact = load_artifact(
            directory / 'generation_timing_summary.json'
        )
        validate_method_artifacts(label, summary, mm, physical, timing)
        joint_physical_by_label[label] = physical
        row = {
            'method': display_name, 'label': label,
            **{
                key: metric(summary, key) for key in (
                    'top1', 'top2', 'top3', 'fid',
                    'matching', 'diversity', 'multimodality',
                )
            },
            'multimodality_all_435_pairs': {
                key: mm['exhaustive_all_pairs'][key]
                for key in ('mean', 'ci95')
            },
            'physical_replication_0': {
                key: physical[key] for key in (
                    'contact_f1', 'contact_temporal_iou', 'false_contact_rate',
                    'minimum_interperson_joint_distance_m', 'foot_sliding_mps',
                )
            },
            'generation_timing': {
                'main_elapsed_seconds': timing['main_elapsed_seconds'],
                'main_seconds_per_sample_amortized': (
                    timing['main_seconds_per_sample_amortized']
                ),
            },
        }
        rows.append(row)
        artifacts[label] = {
            'run_manifest': run_manifest_artifact,
            'semantic_summary': summary_artifact,
            'multimodality_summary': mm_artifact,
            'physical_replication_0': physical_artifact,
            'generation_timing_summary': timing_artifact,
        }

    mesh_physical, mesh_physical_artifact = load_artifact(
        args.mesh_physical_manifest
    )
    mesh_physical_rows = validate_final_mesh_physical_main_table(
        mesh_physical, args.fair_root, joint_physical_by_label
    )
    for row in rows:
        row['mesh_physical_replication_0'] = mesh_physical_rows[row['label']]
    artifacts['final_mesh_physical_main_table'] = mesh_physical_artifact

    ordinary_only, ordinary_only_artifact = load_artifact(
        args.fair_root / 'final_ordinary_only_metric_audit.json'
    )
    validate_final_ordinary_only_metric_audit(ordinary_only)
    artifacts['final_ordinary_only_metric_audit'] = ordinary_only_artifact

    training_and_protocol = {}
    for label, path in TRAINING_AND_PROTOCOL_ARTIFACTS:
        payload, artifact = load_artifact(path)
        training_and_protocol[label] = {
            'payload': payload,
            'artifact': artifact,
        }
    if training_and_protocol[
        'hhc_post_50k_training_dispatch'
    ]['payload'].get('action') == 'continue_global_control_to_300k_then_shape':
        promoted_payload, promoted_artifact = load_artifact(Path(
            'output/hhc_global_masked_loss_control_300k_results.json'
        ))
        training_and_protocol[
            'hhc_global_masked_loss_control_300k_results'
        ] = {
            'payload': promoted_payload,
            'artifact': promoted_artifact,
        }
    validate_training_and_protocol(training_and_protocol)
    validate_method_checkpoint_bindings(method_manifests, training_and_protocol)
    required_review_paths = {
        'paired_fair_method_comparison': (
            args.fair_root / 'paired_fair_method_comparison.json'
        ),
        'paired_oracle_timing_effect': (
            args.fair_root / 'paired_oracle_timing_effect.json'
        ),
        'paired_contact_map_implementation_sensitivity_rep0': (
            args.fair_root /
            'paired_contact_map_implementation_sensitivity_rep0.json'
        ),
        'paired_hhc_mgt_training_state_sensitivity_rep0': (
            args.fair_root /
            'paired_hhc_mgt_training_state_sensitivity_rep0.json'
        ),
        'contact_timing_ablation': (
            args.fair_root / 'hhc_contact_timing_ablation_rep0/summary.json'
        ),
        'sampling_pareto': (
            args.fair_root / 'hhc_sampling_pareto_rep0/summary.json'
        ),
        'hhc_validation_sampler_calibration': Path(
            'output/hhc_validation_sampler_calibration/manifest.json'
        ),
        'contact_timing_predictor_score': Path(
            'output/contact_timing/text_knn_replication_00_score.json'
        ),
        'qualitative_comparison': Path(
            'output/qualitative_comparison/manifest.json'
        ),
        'shape_augmentation_results': Path(
            'output/interx_shape_augmentation_results.json'
        ),
        'llm_robustness': Path(
            'output/interx_llm_robustness/evaluation.json'
        ),
    }
    review_evidence = {}
    for label, path in required_review_paths.items():
        review_evidence[label], evidence_artifact = load_artifact(path)
        review_evidence[label + '_artifact'] = evidence_artifact
    validate_review_evidence(review_evidence)
    calibration = review_evidence['hhc_validation_sampler_calibration']
    calibration_artifacts = calibration['artifacts']
    hhc_selection_record = training_and_protocol['hhc_mgt_final_selection']
    hhc_selection = hhc_selection_record['payload']
    validation_selection_record = training_and_protocol[
        'mgt_validation_selection'
    ]['artifact']
    require_artifact(
        calibration_artifacts['selection_manifest']['sha256']
        == hhc_selection_record['artifact']['sha256']
        and same_resolved_path(
            calibration_artifacts['selection_manifest']['path'],
            hhc_selection_record['artifact']['path'],
        )
        and calibration_artifacts['checkpoint']['sha256']
        == hhc_selection['selected_checkpoint_sha256']
        and same_resolved_path(
            calibration_artifacts['checkpoint']['path'],
            hhc_selection['selected_checkpoint'],
        )
        and calibration_artifacts[
            'formal_hhc_validation_selection']['sha256']
        == validation_selection_record['sha256']
        and same_resolved_path(
            calibration_artifacts['formal_hhc_validation_selection']['path'],
            validation_selection_record['path'],
        )
        and float(calibration['formal_row_bridge'][
            'training_validation_fid'])
        == float(hhc_selection['selected_validation_fid']),
        'hhc_validation_sampler_calibration',
        'calibration is not bound to the final HHC validation selection',
    )
    optimization_readiness = training_and_protocol[
        'optimization_ablation_readiness'
    ]['payload']
    optimization_ablation = None
    optimization_formal = None
    if optimization_readiness['ready']:
        optimization_payload, optimization_artifact = load_artifact(
            Path('output/optimization_ablation_pilot/ablation_manifest.json')
        )
        optimization_summaries = validate_optimization_ablation(
            optimization_payload
        )
        optimization_ablation = {
            'payload': optimization_payload,
            'artifact': optimization_artifact,
            'summaries': optimization_summaries,
        }
        formal_payload, formal_artifact = load_artifact(
            Path('output/optimization_formal_rep0/formal_manifest.json')
        )
        validate_formal_optimization(formal_payload)
        optimization_formal = {
            'payload': formal_payload,
            'artifact': formal_artifact,
        }
    external_prerequisites = training_and_protocol[
        'review_external_prerequisites'
    ]['payload']
    pending_external = list(
        external_prerequisites['unresolved_external_prerequisites']
    )
    review_coverage = training_and_protocol[
        'review_requirement_coverage'
    ]['payload']
    current_interx_pending = list(
        review_coverage.get('current_interx_pending', [])
    )
    review_completion_status = resolve_review_completion_status(
        optimization_complete=(
            optimization_ablation is not None
            and optimization_formal is not None
        ),
        current_interx_pending=current_interx_pending,
        external_prerequisites_pending=pending_external,
    )
    report = {
        'schema': 'hhc_review_results_report_v2',
        'review_completion_status': review_completion_status,
        'current_interx_pending': current_interx_pending,
        'unresolved_external_prerequisites': pending_external,
        'protocol': (
            '20 frozen request replications, official Inter-X evaluator, common '
            'generation counts and main/MM seeds; full 1696-sample '
            'replication-0 joint-contact and released-VolumetricSMPL mesh '
            'geometry table'
        ),
        'main_table': rows,
        'mesh_physical_main_table_protocol': mesh_physical,
        'training_and_protocol_provenance': training_and_protocol,
        'required_review_evidence': review_evidence,
        'optimization_ablation': optimization_ablation,
        'optimization_formal': optimization_formal,
        'external_prerequisites': external_prerequisites,
        'artifacts': artifacts,
        'implementation_sha256': sha256_file(__file__),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, args.output_json)

    lines = [
        '# Strict Inter-X review results', '',
        f"Completion status: `{review_completion_status}`. Current Inter-X or "
        f"manuscript requirements still pending: `{current_interx_pending}`. "
        f"External prerequisites still pending: `{pending_external}`.", '',
        'Semantic main rows use 20 frozen request replications and the same '
        'official evaluator. Geometry/contact rows use the same frozen '
        'replication 0 generation and request beta/gender for every method.',
        '',
        '| Method | Top-1 ↑ | FID ↓ | Matching ↓ | Diversity → | MM (10) ↑ | MM (435) ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {format_metric(row['top1'])} | "
            f"{format_metric(row['fid'])} | "
            f"{format_metric(row['matching'])} | {format_metric(row['diversity'])} | "
            f"{format_metric(row['multimodality'])} | "
            f"{format_metric(row['multimodality_all_435_pairs'])} |"
        )
    lines += [
        '', '## Main contact and mesh-penetration results', '',
        'These are deterministic descriptive results on all 1,696 valid '
        'replication-0 requests. Contact metrics use the frozen manual binary '
        'timing and subject-specific Inter-X skeletons. Mesh metrics use '
        'request beta/gender and the same released male/female VolumetricSMPL '
        'weights. They do not carry a 20-replication confidence interval or '
        'significance claim.', '',
        '| Method | Contact F1 ↑ | Temporal IoU ↑ | False contact ↓ | Foot sliding (m/s) ↓ | Penetrating vertices (%) ↓ | Collision frames (%) ↓ | Eq. 6 (m²/frame) ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        joint = row['physical_replication_0']
        mesh = row['mesh_physical_replication_0']
        lines.append(
            f"| {row['method']} | {joint['contact_f1']:.4f} | "
            f"{joint['contact_temporal_iou']:.4f} | "
            f"{joint['false_contact_rate']:.4f} | "
            f"{joint['foot_sliding_mps']:.4f} | "
            f"{100.0 * mesh['penetrating_vertex_fraction']['mean']:.4f} | "
            f"{100.0 * mesh['collision_frame_rate']['mean']:.4f} | "
            f"{mesh['paper_eq6_squared_sum_m2_per_frame']['mean']:.4e} |"
        )
    lines += ['', '## Training and protocol provenance', '']
    for label, record in training_and_protocol.items():
        artifact = record['artifact']
        lines.append(
            f"- `{label}`: `{artifact['path']}` "
            f"(SHA-256 `{artifact['sha256']}`)"
        )
    lines += [
        '',
        'Every final method manifest binds content-addressed snapshots of all '
        'declared implementation files. Repository revision/diff records '
        'describe the launch state and remain the training-generation equality '
        'key; unrelated edits to the live worktree cannot relabel or invalidate '
        'the frozen executable bytes.',
    ]
    cosh_resume_report = training_and_protocol[
        'coshmdm_resume_provenance'
    ]['payload']
    lines += [
        '', '### CoShMDM normalization and restart disclosure', '',
        'The reported CoShMDM row is a fresh Inter-X retraining from update 0. '
        'Every launch explicitly selects `interx_global`, and its train-split '
        'mean/std hashes are embedded in every checkpoint and checked on '
        'resume. An earlier continuation that silently fell back to the '
        'released InterHuman `global_mean/std` is preserved only as diagnostic '
        'evidence and contributes no state to the reported model. If an OS '
        'restart occurs, Lightning restores model, AdamW, scheduler and loop '
        'progress, but the released two-worker caption/crop/person-order RNG '
        'streams are not serialized; such a continuation is distributionally, '
        'not bitwise, equivalent to an uninterrupted run.',
    ]
    protocol_audit = training_and_protocol['interx_protocol_audit']['payload']
    protocol_assets = protocol_audit['asset_protocol']
    lines += [
        '', '### Official evaluator and GT identity', '',
        f"The official test split has {protocol_audit['split_entries']:,} IDs; "
        f"{protocol_audit['valid_entries']:,} pass the shared length filter and "
        f"batch-32 `drop_last=True` evaluates exactly "
        f"{protocol_audit['evaluated_entries']:,} GT motions per replication. "
        'The current loader, official Inter-X evaluator and InterMask source '
        'match this same contract. The evaluator weight SHA-256 is '
        f"`{protocol_audit['official_evaluator_weights_sha256']}`.",
        '',
        'All six main-table methods are additionally covered by the frozen '
        'ordinary-only audit: 20 replication records per method, 120 total. '
        'InterMask and Interact2AR reuse their unchanged generated HDF5 files '
        'and are re-scored with the current evaluator; no model is retrained '
        'or re-sampled. Body-controlled FID/diversity fields are rejected.',
        '',
        'MMM-linked train/val/test HDF5 assets, raw 672-D Interx mean/std, '
        'evaluator options, male/female SMPL-X models and male/female '
        'VolumetricSMPL checkpoints are byte-identity gated. Raw mean/std '
        f"hashes are `{protocol_assets['statistics']['Interx_Mean.npy']['sha256']}` "
        f"and `{protocol_assets['statistics']['Interx_Std.npy']['sha256']}`.",
    ]
    selection = training_and_protocol['hhc_mgt_final_selection']['payload']
    selection_contract = selection['validation_inference_contract']
    validation_pair = training_and_protocol[
        'hhc_validation_observational_pair_audit'
    ]['payload']
    lines += [
        '', '### HHC checkpoint selection condition', '',
        'The HHC checkpoint is selected only by fixed validation FID; no test '
        'metric is used. Validation sampling uses the validation split oracle '
        f"contact timing with `{selection_contract['sampler']}`, "
        f"{selection_contract['sampling_steps']} outer steps, temperature "
        f"{selection_contract['temperature']}, CFG "
        f"{selection_contract['cfg_scale']} and contact guidance "
        f"{selection_contract['contact_guidance_scale']}. Transformer training "
        'does not consume contact timing as an input. The no-timing and oracle '
        'test rows share this same selected checkpoint.',
        '',
        f"At update {validation_pair['expected_step']:,}, the permanently "
        'retained checkpoints saved immediately before and after validation '
        f"have all {validation_pair['model_tensor_count']:,} model "
        'tensors, optimizer state, scheduler state, and hyperparameters '
        'bitwise identical. Only callback/loop validation metadata differs, '
        'providing runtime evidence that checkpoint-selection validation is '
        'observational and does not mutate training state.',
    ]
    conditional_support = training_and_protocol[
        'hhc_mgt_train_val_conditional_support_audit'
    ]['payload']
    conditional_marginals = conditional_support['marginal_code_support']
    conditional_tuples = conditional_support[
        'same_frame_conditional_tuple_support'
    ]
    conditional_transitions = conditional_support[
        'within_part_transition_support'
    ]
    lines += [
        '', '### MGT exact-token generalization diagnosis', '',
        'Validation marginal code occurrence coverage in training is '
        f"`{100 * conditional_marginals['body']['validation_occurrence_coverage']:.3f}%/"
        f"{100 * conditional_marginals['hand']['validation_occurrence_coverage']:.3f}%/"
        f"{100 * conditional_marginals['root']['validation_occurrence_coverage']:.3f}%` "
        'for Body/Hand/Root, ruling out a wholesale code-ID support mismatch. '
        'However, same-frame Root-to-Body and Root+Body-to-Hand tuple occurrence '
        f"coverage is only `{100 * conditional_tuples['root_to_body_pair']['validation_occurrence_coverage']:.3f}%` "
        f"and `{100 * conditional_tuples['root_body_to_hand_triplet']['validation_occurrence_coverage']:.3f}%`; "
        'Body/Hand/Root transition occurrence coverage is '
        f"`{100 * conditional_transitions['body']['validation_occurrence_coverage']:.2f}%/"
        f"{100 * conditional_transitions['hand']['validation_occurrence_coverage']:.2f}%/"
        f"{100 * conditional_transitions['root']['validation_occurrence_coverage']:.2f}%`. "
        'This contextualizes low held-out exact-token accuracy as a high-cardinality '
        'conditional-composition problem. It is not treated as the cause of the '
        'current-versus-historical FID gap because the historical low-FID MGT '
        'exhibits the same exact-token gap; decoded validation FID remains the '
        'model-promotion criterion.',
    ]
    shape_pair = training_and_protocol[
        'hhc_mgt_shape_dependence_pair'
    ]['payload']
    permuted_shape = shape_pair['effects']['permuted_beta_and_gender']
    lines += [
        '', '### MGT shape-condition shortcut diagnosis', '',
        'With text, target tokens and teacher-forced Root→Body→Hands cascade '
        'contexts fixed, replacing each requested shape by a deterministic '
        'cross-sample derangement changes aggregate exact-token accuracy by '
        f"`{-permuted_shape['training']['requested_accuracy_advantage_percent_points']:.3f}` "
        'points on train and '
        f"`{-permuted_shape['validation']['requested_accuracy_advantage_percent_points']:.3f}` "
        'points on validation (permuted minus requested). The pre-registered '
        f"decision is `{shape_pair['decision']}`. This is mechanism evidence "
        'for the matched random-shape augmentation study only; decoded-motion '
        'validation/test FID remains the result and promotion criterion.',
    ]
    promoted_key = 'hhc_global_masked_loss_control_300k_results'
    if promoted_key in training_and_protocol:
        promoted = training_and_protocol[promoted_key]['payload']
        lines += [
            '', '### Promoted global masked-token loss control', '',
            'The preregistered 50k gate promoted the same online W&B run to '
            '300k. The table below is a loss-reduction diagnostic on the '
            'unchanged official Inter-X requests and common-50 sampler; it is '
            'reported separately from the paper-objective main HHC row.', '',
            '| Timing | FID ↓ | Top-1 ↑ | Matching ↓ | Diversity | MModality |',
            '|---|---:|---:|---:|---:|---:|',
        ]
        for guidance, display in (
            ('no_contact', 'None'), ('contact', 'Oracle'),
        ):
            metrics = promoted['methods'][guidance]['metrics']
            lines.append(
                f"| {display} | {format_metric(metrics['fid'])} | "
                f"{format_metric(metrics['top1'])} | "
                f"{format_metric(metrics['matching'])} | "
                f"{format_metric(metrics['diversity'])} | "
                f"{format_metric(metrics['multimodality'])} |"
            )
    map_audit = training_and_protocol[
        'hhc_contact_guidance_map_audit'
    ]['payload']
    map_comparison = review_evidence[
        'paired_contact_map_implementation_sensitivity_rep0'
    ]['comparators']['legacy_bug_map']['metrics']
    lines += [
        '', '### Contact-map implementation sensitivity', '',
        'The fair main rows use the correctly indexed train-only map. The '
        'preserved MMM map consumes 1,706 official-test samples and treats '
        'temporal rows 0/1/2 of cached `[1,T,3]` indices as the three '
        'kinematic parts; exact bug replay is byte-identical to the saved '
        'author-development map. It is therefore excluded from the main '
        'table. The following replication-0 values are combined '
        'implementation-sensitivity effect sizes only; they do not isolate '
        'axis, split-scope, dead-code, and historical-cache-crop effects and '
        'carry no inferential claim.', '',
        '| Metric | Fair train-only map | MMM legacy map | Fair - legacy |',
        '|---|---:|---:|---:|',
    ]
    for metric_name in ('fid', 'top1', 'matching'):
        record = map_comparison[metric_name]
        lines.append(
            f"| {metric_name} | {record['reference_mean']:.6f} | "
            f"{record['comparator_mean']:.6f} | "
            f"{record['paired_delta_reference_minus_comparator_mean']:.6f} |"
        )
    lines.append(
        'Audit map hashes: fair `'
        + map_audit['paper_formula_fair_map']['sha256']
        + '`, legacy `'
        + map_audit['mmm_legacy_full_axis_bug_map']['sha256'] + '`.'
    )
    adapter_audit = training_and_protocol[
        'coshmdm_representation_adapter_audit'
    ]['payload']
    timotion_adapter_equivalence = training_and_protocol[
        'timotion_cosh_adapter_equivalence_audit'
    ]['payload']
    projection_repeat = adapter_audit['projection_once_vs_twice']['all_rmse']
    information_loss = adapter_audit['raw_to_projection_information_loss']
    lines += [
        '', '### CoShMDM/TiMotion adapter geometry audit', '',
        'The shape-specific all-joint root adapter is audited on all 1,696 '
        'official test requests and all 59 skeleton groups. Its once-to-twice '
        f"projection RMSE is `{projection_repeat['mean']:.3e}` on average "
        f"and `{projection_repeat['maximum']:.3e}` at worst. All non-root "
        'body rotations are preserved exactly over every requested frame. '
        'The remaining raw-to-projected differences are the intentionally '
        'unmodelled initial absolute root transform and 33 hand-joint '
        'rotations; they are not sampler errors. Every quantitative method '
        'row uses the frozen official full-space Inter-X evaluator. A '
        'projected-GT representation oracle is computed only as the diagnostic '
        'decomposition below; it is explicitly not a body-controlled FID and '
        'is not a comparison-table row. TiMotion\'s separately implemented '
        'exporter is bitwise identical to the CoShMDM adapter in all eight '
        'audited float32/float64, pair/flat, shape-free/shape-specific cases '
        f"(maximum absolute difference `{max(case['max_abs'] for case in timotion_adapter_equivalence['cases'].values()):.1f}`).", '',
        'Raw-to-projection RMSE audit: root rotation `'
        f"{information_loss['root_rotation6d']['rmse']:.5f}`; root "
        f"translation `{information_loss['root_translation_m']['rmse']:.5f} m`; "
        f"hands `{information_loss['hand_rotation6d']['rmse']:.5f}`; "
        'non-root body rotations `0.00000`.'
    ]
    timotion_padding = training_and_protocol[
        'timotion_padding_ro_stabilization_audit'
    ]['payload']
    timotion_smoke_path = Path(
        timotion_padding['remote_wandb_audit']['path']
    )
    if not timotion_smoke_path.is_file():
        timotion_smoke_path = Path('tools/provenance_snapshots') / (
            timotion_padding['remote_wandb_audit']['sha256']
            + '__' + timotion_smoke_path.name
        )
    timotion_smoke = json.loads(
        timotion_smoke_path.read_text(encoding='utf-8')
    )
    timotion_metrics = timotion_smoke['metrics']
    lines += [
        '', '### TiMotion padding-RO numerical stabilization', '',
        'The released RO loss normalizes hip-to-hip vectors before applying '
        'the temporal mask. Coincident hips in padding therefore reproduce a '
        '`0/0` NaN, even though no valid frame has a zero vector. The fair '
        'port substitutes a finite x-axis only outside the loss support; valid '
        'predictions, targets and RO values remain bitwise unchanged. The '
        'excluded bad smoke contains '
        f"`{timotion_padding['failed_smoke_nonfinite_motion_values']:,}` "
        'non-finite exported motion values. The replacement online-W&B smoke '
        f"(`{timotion_smoke['run_id']}`) completed 50 updates with gradient "
        'L2/max-abs `'
        f"{timotion_metrics['train/gradient_l2_norm']['maximum']:.6f}`/`"
        f"{timotion_metrics['train/gradient_max_abs']['maximum']:.6f}`, "
        'weight L2/max-abs `'
        f"{timotion_metrics['train/weight_l2_norm']['maximum']:.6f}`/`"
        f"{timotion_metrics['train/weight_max_abs']['maximum']:.6f}`, and zero "
        'non-finite gradients or weights. These smoke metrics establish '
        'numerical readiness and are not a comparison result.',
    ]
    timotion_exclusion = training_and_protocol[
        'timotion_interrupted_run_exclusion'
    ]['payload']
    timotion_accumulation = training_and_protocol[
        'timotion_gradient_accumulation_audit'
    ]['payload']
    timotion_warmup = training_and_protocol[
        'timotion_warmup_stability_audit'
    ]['payload']
    timotion_spike = training_and_protocol[
        'timotion_transient_velocity_spike_audit'
    ]['payload']
    timotion_scheduler = training_and_protocol[
        'timotion_scheduler_boundary_audit'
    ]['payload']
    timotion_selection = training_and_protocol[
        'timotion_validation_selection'
    ]['payload']
    accumulation_runtime = timotion_accumulation['runtime_equivalence']
    warmup_metrics = timotion_warmup['metrics']
    lines += [
        '', '### TiMotion accumulated-update and run-exclusion disclosure', '',
        'The one-GPU port divides each of eight mean-reduced 32-example '
        'losses by eight before accumulation. Its runtime arithmetic audit '
        'matches a 256-example global batch through the released global-norm '
        'clip `0.5` and AdamW step: maximum pre/post-clip gradient differences '
        f"are `{accumulation_runtime['pre_clip_gradient_max_abs_difference']:.3e}`/`"
        f"{accumulation_runtime['post_clip_gradient_max_abs_difference']:.3e}`, "
        'and the post-update weight difference is `'
        f"{accumulation_runtime['post_adamw_weight_max_abs_difference']:.3e}`. "
        'Gradient telemetry is deliberately measured before clipping.', '',
        f"Formal run `{timotion_exclusion['run_id']}` was interrupted before "
        'any checkpoint was written and is explicitly marked failed and '
        '`result_eligible=false`. Its replacement starts from optimizer step '
        '0 under the same audited contract; no weights from the interrupted '
        'run enter the reported result.',
        '',
        f"The replacement run `{timotion_warmup['run_path'].rsplit('/', 1)[-1]}` "
        'also passes its frozen ten-epoch online-W&B warm-up audit: all '
        f"{timotion_warmup['minimum_expected_telemetry_events']} expected "
        'telemetry events are present, gradient and weight non-finite counts '
        'remain zero, and the global weight L2/max-abs relative spans are '
        f"`{warmup_metrics['train/weight_l2_norm']['relative_span_from_first']:.3%}`/`"
        f"{warmup_metrics['train/weight_max_abs']['relative_span_from_first']:.3%}`. "
        'The pre-clip gradient range is retained as diagnostic evidence; the '
        'actual optimizer update remains clipped to global norm `0.5`.',
        '',
        'A separate post-hoc operational audit discloses the largest '
        f"post-convergence velocity excursion at completed epoch "
        f"{timotion_spike['peak']['completed_epoch']}: mean train/validation "
        f"loss `{timotion_spike['peak']['train_loss_mean']:.6f}`/`"
        f"{timotion_spike['peak']['validation_loss']:.6f}`, with the two "
        'velocity terms explaining `'
        f"{timotion_spike['decomposition']['velocity_excess_fraction_of_total_excess']:.2%}` "
        'of the train-loss excess while the simple diffusion loss remains `'
        f"{timotion_spike['peak']['train_simple_mean']:.6f}`. At its latest "
        f"complete audited epoch {timotion_spike['latest_complete_epoch']['completed_epoch']}, "
        'train/validation loss is `'
        f"{timotion_spike['latest_complete_epoch']['train_loss_mean']:.6f}`/`"
        f"{timotion_spike['latest_complete_epoch']['validation_loss']:.6f}`; "
        'gradient and weight non-finite counts remain zero. This disclosure '
        'is not a checkpoint-selection rule and does not alter the released '
        'loss.',
        '',
        'The released scheduler uses the inclusive condition '
        '`epoch <= warmup`. Accordingly, the formal W&B trace exactly matches '
        'its one-epoch boundary overshoot: zero-based epoch 10 uses `'
        f"{timotion_scheduler['transitions'][10]['observed_learning_rate']:.10f}`, "
        'then epoch 11 returns to the cosine branch at `'
        f"{timotion_scheduler['transitions'][11]['observed_learning_rate']:.10f}`. "
        'This source-bound behavior is preserved for reproducibility rather '
        'than silently repaired in the comparison port.',
        '',
        'The reported TiMotion generations use the minimum-FID checkpoint '
        f"from all 15 held-out validations: epoch {timotion_selection['selected_epoch']} "
        f"(update {timotion_selection['selected_step']:,}, validation FID "
        f"`{timotion_selection['selected_validation_fid']:.6f}`). The final "
        '106,500-update full-state checkpoint is retained and audited '
        'separately; no test-set score enters checkpoint selection.',
    ]
    embedding_gap = training_and_protocol[
        'coshmdm_embedding_gap_diagnosis'
    ]['payload']
    trajectory = training_and_protocol[
        'coshmdm_training_trajectory_diagnosis'
    ]['payload']
    gaps = embedding_gap['gaps']
    lines += [
        '', '### CoShMDM high-FID diagnosis', '',
        'On frozen replication 0, the official GT-to-CoShMDM FID is '
        f"`{gaps['official_ground_truth_to_coshmdm_model']['fid']:.6f}`. "
        'Projecting GT through CoShMDM\'s observable 262-D representation and '
        'back to the same official 672-D evaluator gives diagnostic FID '
        f"`{gaps['official_ground_truth_to_representation_oracle']['fid']:.6f}`; "
        'the representation-oracle-to-model FID is '
        f"`{gaps['representation_oracle_to_coshmdm_model']['fid']:.6f}`. "
        'FID is not additive, so these values diagnose two simultaneous gaps '
        'rather than partitioning the main-row score arithmetically. The '
        'GT-to-model gap is dominated by the evaluator-embedding centroid term '
        f"({gaps['official_ground_truth_to_coshmdm_model']['mean_term_fraction'] * 100:.1f}%).",
        '',
        '| Step | Train total loss, trailing-1k median | Weight L2 | Val FID |',
        '|---:|---:|---:|---:|',
    ]
    for item in trajectory['milestones']:
        lines.append(
            f"| {item['step']:,} | "
            f"{item['training_scalars']['total']['trailing_window']['median']:.6f} | "
            f"{item['checkpoint']['weight_l2_norm']:.3f} | "
            f"{item['validation']['fid']:.6f} |"
        )
    lines += [
        '',
        'All logged objectives and all 3.054×10^8 milestone weight values are '
        'finite. Training loss decreases at every milestone while validation '
        'FID worsens after the selected 40k checkpoint; this contradicts a '
        'NaN/exploding-loss explanation and supports objective/generalization/'
        'representation mismatch. The historical TensorBoard run did not log '
        'gradients, so gradient norms cannot be reconstructed retrospectively.',
    ]
    wallclock = training_and_protocol['training_wallclock_audit']['payload']
    lines += [
        '', '## Main training compute', '',
        'The raw12288 VQ is a byte-identical imported pre-cleanup checkpoint; '
        'only the corrected HHC MGT is retrained in the current study.', '',
        '| Stage | Optimizer updates | Effective batch | Sample exposures | Wall hours |',
        '|---|---:|---:|---:|---:|',
    ]
    for name in ('hhc_mgt', 'coshmdm', 'timotion'):
        record = wallclock['records'][name]
        lines.append(
            f"| {name} | {record['optimizer_updates']:,} | "
            f"{record['effective_batch_size']:,} | "
            f"{record['sample_exposures']:,} | "
            f"{record['elapsed_hours']:.2f} |"
        )
    optimization_readiness = training_and_protocol[
        'optimization_ablation_readiness'
    ]['payload']
    lines += ['', '## Contact optimization ablation status', '']
    if optimization_ablation is None:
        lines.append(
            'The strict paired optimization matrix is pending and is not '
            'included in claimed completed results. Blockers: `'
            + '`, `'.join(optimization_readiness['blockers']) + '`.'
        )
    else:
        lines.append(
            'All 13 preregistered paired optimizer pilot ablations completed. '
            f"Manifest: `{optimization_ablation['artifact']['path']}` "
            f"(SHA-256 `{optimization_ablation['artifact']['sha256']}`)."
        )
        lines += [
            '',
            '| Optimizer condition | Penetrating-vertex fraction Δ [95% CI], p | Eq. 6 Δ [95% CI], p | Foot sliding Δ [95% CI], p | Mean s/sample |',
            '|---|---:|---:|---:|---:|',
        ]
        ablation_metrics = (
            ('penetrating_vertex_fraction', 'penetration'),
            ('paper_eq6_squared_sum_m2_per_frame', 'eq6'),
            ('mean_foot_sliding_m_per_frame', 'foot'),
        )
        for name, summary in optimization_ablation['summaries'].items():
            rendered = {}
            for metric_name, short_name in ablation_metrics:
                metric_record = summary['comparisons'][metric_name]
                interval = metric_record['paired_delta_bootstrap_95ci']
                rendered[short_name] = (
                    f"{metric_record['paired_delta_after_minus_before_mean']:.3e} "
                    f"[{interval[0]:.3e}, {interval[1]:.3e}], "
                    f"{metric_record['wilcoxon_two_sided_p']:.3g}"
                )
            runtime = summary['runtime'][
                'optimizer_wall_seconds_mean_per_sample'
            ]
            lines.append(
                f"| {name} | {rendered['penetration']} | "
                f"{rendered['eq6']} | {rendered['foot']} | {runtime:.3f} |"
            )
        for condition in ('hhc', 'gt'):
            formal_record = optimization_formal['payload'][
                'output_artifacts'][condition]
            metrics = json.loads(Path(
                formal_record['official_main_metrics']['path']
            ).read_text(encoding='utf-8'))
            completion = json.loads(Path(
                formal_record['optimization_completion_manifest']['path']
            ).read_text(encoding='utf-8'))
            runtime = completion['runtime']
            lines.append(
                f"Full `{condition}` + optimization (1,696 samples): "
                f"FID `{metrics['fid']:.5f}`, Top-1/2/3 "
                f"`{metrics['top1']:.5f}/{metrics['top2']:.5f}/"
                f"{metrics['top3']:.5f}`, Matching "
                f"`{metrics['matching']:.5f}`. MModality is N/A because "
                'the formal optimizer operates on one repeat. CUDA-synchronized '
                f"optimizer wall time: mean `{runtime['optimizer_wall_seconds_mean_per_sample']:.3f}` s/sample, "
                f"P95 `{runtime['optimizer_wall_seconds_p95_per_sample']:.3f}` s/sample."
            )
    lines += ['', '## External review prerequisites', '']
    if pending_external:
        lines.append(
            'The following externally supplied assets are still unavailable; '
            'the report therefore does not claim those review experiments as '
            'complete: `' + '`, `'.join(pending_external) + '`.'
        )
    else:
        lines.append('All externally supplied review prerequisites are present.')
    shape = review_evidence['shape_augmentation_results']
    shape_preflight = shape['input_preflight']
    shape_paired = json.loads(Path(
        shape['artifacts']['paired_method_comparison']['path']
    ).read_text(encoding='utf-8'))
    shape_paired_contact = json.loads(Path(
        shape['artifacts']['paired_method_comparison_contact_guidance']['path']
    ).read_text(encoding='utf-8'))
    lines += [
        '', '## Training-time random-shape augmentation', '',
        'Both arms append the same 6,238 train-only clips with identical '
        'random donor shapes, captions, manual contact timing, fixed raw12288 '
        'VQ and 300k MGT budget. The optimized arm differs only by the paper '
        'contact/physical refinement. Official train/val/test splits and raw '
        'Mean/Std remain unchanged. The full input preflight found '
        f"{shape_preflight['semantic_contact_active_samples']:,} timed-semantic "
        f"clips, {shape_preflight['physical_timing_only_samples']:,} "
        'timing-positive physical-only clips and '
        f"{shape_preflight['regions_inactive_because_timing_is_zero']:,} "
        'zero-timing clips whose text regions remain inactive. Exactly '
        f"{shape_preflight['boundary_whitespace_caption_matches']} caption "
        'required boundary-whitespace normalization; both raw and plan hashes '
        'are retained.', '',
        '**Without sampling-time contact guidance**', '',
        '| Arm | Metric | Baseline advantage | 95% bootstrap CI | Holm p |',
        '|---|---|---:|---:|---:|',
    ]
    for arm, comparison in shape_paired['comparators'].items():
        for metric_name, record in comparison['metrics'].items():
            interval = record['reference_advantage_bootstrap_95ci']
            lines.append(
                f"| {arm} | {metric_name} | "
                f"{record['reference_advantage_mean']:.4f} | "
                f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
                f"{record['holm_familywise_p']:.4g} |"
            )
    lines += [
        '', '**With sampling-time contact guidance**', '',
        '| Arm | Metric | Baseline advantage | 95% bootstrap CI | Holm p |',
        '|---|---|---:|---:|---:|',
    ]
    for arm, comparison in shape_paired_contact['comparators'].items():
        for metric_name, record in comparison['metrics'].items():
            interval = record['reference_advantage_bootstrap_95ci']
            lines.append(
                f"| {arm} | {metric_name} | "
                f"{record['reference_advantage_mean']:.4f} | "
                f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
                f"{record['holm_familywise_p']:.4g} |"
            )
    lines += [
        '', '## Full-split LLM contact-region coverage', '',
        'Luna first-caption caches cover every official Inter-X sequence. '
        'Manual timing coverage is reported separately and missing timing is '
        'never imputed as a negative label.', '',
        '| Split | Samples | Structured contact rows | Timing covered | Timing positive | Positive timing / empty regions | Zero timing / nonempty regions |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for display_name, split_label in (
        ('train', 'train'), ('validation', 'val_full'),
        ('test auxiliary first-caption', 'test_full'),
    ):
        cache_counts = training_and_protocol[
            f'interx_luna_{split_label}_cache_audit'
        ]['payload']['counts']
        timing_counts = training_and_protocol[
            f'interx_luna_{split_label}_timing_audit'
        ]['payload']['counts']
        lines.append(
            f"| {display_name} | {cache_counts['completed_samples']:,} | "
            f"{cache_counts['structured_contact_rows']:,} | "
            f"{timing_counts['timing_covered_samples']:,} | "
            f"{timing_counts['timing_positive_samples']:,} | "
            f"{timing_counts['positive_timing_empty_regions']:,} | "
            f"{timing_counts['zero_timing_nonempty_regions']:,} |"
        )
    frozen_counts = training_and_protocol[
        'interx_luna_test_frozen_cache_audit'
    ]['payload']['counts']
    terra_counts = training_and_protocol[
        'interx_terra_robustness_cache_audit'
    ]['payload']['counts']
    full_train_usage = training_and_protocol[
        'interx_luna_train_cache_audit'
    ]['payload']['resource_usage']
    full_train_wall = full_train_usage['end_to_end_wall_clock']
    lines += [
        '',
        f"Full-train Luna annotation took `{full_train_wall['total_hours']:.3f}` "
        'hours end to end. This span includes retries, resume/pause and '
        'orchestration overhead; the legacy sidecars contain no per-attempt '
        'timers, so per-sample model latency is N/A rather than inferred.',
        '',
        'Fusion never invents contact to force agreement: timing-positive '
        'samples with empty LLM regions use physical losses only, while LLM '
        'regions on timing-zero samples are semantically inactive.',
        '',
        f"The formal request-bound test cache contains "
        f"{frozen_counts['completed_samples']:,} captions; it is distinct from "
        'the auxiliary first-caption cache. The Terra robustness reference '
        f"contains {terra_counts['completed_samples']:,} frozen captions. All "
        'eight cache/timing artifacts pass count, schema, implementation and '
        'content-hash gates.',
    ]
    llm_robustness = review_evidence['llm_robustness']
    lines += [
        '', '## Contact-region parser robustness', '',
        'The 128 frozen captions comprise four disjoint 32-sample strata. '
        'Manual Inter-X contact timing supplies binary contact-presence labels '
        'only. GPT-5.6 Terra is a stronger-model consistency reference, not '
        'human 75-region ground truth; ordered and person-swap-tolerant part '
        'scores below are agreement with Terra, not oracle accuracy. Luna and '
        'Terra were invoked through Codex CLI with ChatGPT login and no API key; '
        'API-billed cost is therefore N/A rather than an estimate from public '
        'API list prices.',
        '',
        '| Parser | Presence P/R/F1/Acc. | Empty | Failed attempts | Mean/P95 s per sample | Tokens per sample | Ordered/Swap-tolerant part F1 vs Terra |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    llm_names = {
        'luna': 'GPT-5.6 Luna (primary)',
        'terra_reference': 'GPT-5.6 Terra (reference)',
        'rule': 'Deterministic lexical rule',
    }
    for source_name in ('luna', 'terra_reference', 'rule'):
        result = llm_robustness['results'][source_name]
        presence = result['contact_presence_vs_manual_timing']
        usage = result['resource_usage']
        wall = usage['wall_seconds']
        tokens = usage.get('tokens')
        token_text = (
            f"{tokens['mean_per_sample']:.1f}" if tokens is not None else 'N/A'
        )
        agreement = result.get('part_pair_agreement_vs_terra_reference')
        agreement_text = (
            f"{agreement['ordered']['f1']:.3f}/"
            f"{agreement['unordered']['f1']:.3f}"
            if agreement is not None else 'reference'
        )
        lines.append(
            f"| {llm_names[source_name]} | {presence['precision']:.3f}/"
            f"{presence['recall']:.3f}/{presence['f1']:.3f}/"
            f"{presence['accuracy']:.3f} | {result['empty_samples']} | "
            f"{result['failed_attempts']} | {wall['mean_per_sample']:.4f}/"
            f"{wall['p95_per_sample']:.4f} | {token_text} | "
            f"{agreement_text} |"
        )
    historical = training_and_protocol[
        'historical_mmm_wandb_comparison'
    ]['payload']
    old_metrics = historical['historical_run']['final_metrics']
    historical_validation = historical['historical_run']['validation_history']
    current_validation = historical['current_run']['validation_history']
    legacy_differential = training_and_protocol[
        'hhc_paper_training_contract_audit'
    ]['payload']['legacy_vs_active_differential']
    legacy_betas = legacy_differential['different_settings'][
        'optimizer_betas'
    ]['legacy']
    active_betas = legacy_differential['different_settings'][
        'optimizer_betas'
    ]['active']
    lines += [
        '', '## W&B trajectory reference (not a fair comparison)', '',
        'The historical MMM run and current run use the same byte-identical '
        'raw12288 VQ, but different transformer architecture and training '
        'configuration. Its values are shown only as a labelled trajectory '
        'reference. The '
        'legacy validation hook logs FID for the current milestone but logs '
        'Top-1 and Matching as running-best state; those columns are therefore '
        'not per-checkpoint measurements. The relevant pre-cleanup two-person '
        'entrypoint and the paper-config run both use BF16 mixed precision, so '
        'precision is not an observed difference. Unmatched differences '
        f'include AdamW betas {legacy_betas} versus {active_betas}, MultiStep '
        'versus warm-up/cosine scheduling, flattened corruption/all-token loss '
        'versus sorted part masks/masked-token loss, model width/head/block '
        'layout, batch size, and seed; this diagnostic does not attribute the '
        'trajectory gap to any one of them.', '',
        '| Run | Step | Validation FID (current) | Top-1 (running best) | Matching (running best) |',
        '|---|---:|---:|---:|---:|',
        f"| Historical MMM | {int(old_metrics['trainer/global_step']) + 1} | "
        f"{old_metrics['val/FID']:.4f} | {old_metrics['val/TOP1']:.4f} | "
        f"{old_metrics['val/Matching']:.4f} |",
    ]
    current_by_step = {
        int(record['completed_updates']): record
        for record in current_validation
    }
    historical_by_step = {
        int(record['trainer/global_step']) + 1: record
        for record in historical_validation
    }
    lines += [
        '', '| Step | Historical FID | Paper-config FID | Historical Top-1 (running best) | Paper-config Top-1 (running best) |',
        '|---:|---:|---:|---:|---:|',
    ]
    for step in sorted(set(historical_by_step) & set(current_by_step)):
        old_record = historical_by_step[step]
        record = current_by_step[step]
        lines.append(
            f"| {step} | {old_record['val/FID']:.4f} | "
            f"{record['val/FID']:.4f} | {old_record['val/TOP1']:.4f} | "
            f"{record['val/TOP1']:.4f} |"
        )
    calibration = review_evidence['hhc_validation_sampler_calibration']
    bridge = calibration['formal_row_bridge']
    lines += [
        '', '### Validation-only 50-step sampler diagnostic', '',
        'All six rows use the exact 544 records used for checkpoint selection; '
        'the test split is not read and every row retains 50 outer MaskGIT '
        'iterations. This is a diagnostic of the selected checkpoint, not a '
        'replacement for the preregistered formal `temperature=0.5, CFG=3.5` '
        'test row. The formal offline row reproduces its training-time '
        f"validation FID within `{bridge['absolute_difference']:.3g}` absolute "
        'error.', '',
        '| Temperature | CFG | Validation FID | Diversity | Top-1 | Matching |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for name in (
        'temp0p5_cfg3p5', 'temp0p5_cfg1p0', 'temp0p5_cfg2p5',
        'temp0p5_cfg4p5', 'temp0p3_cfg3p5', 'temp0p8_cfg3p5',
    ):
        result = calibration['results'][name]
        config = result['configuration']
        lines.append(
            f"| {config['temperature']:.1f} | {config['cfg_scale']:.1f} | "
            f"{result['fid']:.6f} | {result['diversity']:.6f} | "
            f"{result['top1']:.6f} | {result['matching']:.6f} |"
        )
    mgt_sensitivity = review_evidence[
        'paired_hhc_mgt_training_state_sensitivity_rep0'
    ]
    lines += [
        '', '### Same-protocol pre-cleanup MGT diagnostic', '',
        'These rows use the same frozen replication-0 request file, seed, '
        'raw12288/175k VQ, official evaluator, and '
        '`sample_intra_cas/50/0.5/3.5/1.0` inference contract. They isolate '
        'the observed output difference to the MGT training state/configuration '
        'as a whole, but do not attribute it to any single unmatched training '
        'factor. One replication supplies effect sizes only and carries no '
        'significance claim.', '',
        '| MGT | FID | Top-1 | Matching | MModality |',
        '|---|---:|---:|---:|---:|',
    ]
    corrected_metrics = next(iter(
        mgt_sensitivity['comparators'].values()
    ))['metrics']
    lines.append(
        '| Corrected paper configuration | '
        f"{corrected_metrics['fid']['reference_mean']:.6f} | "
        f"{corrected_metrics['top1']['reference_mean']:.6f} | "
        f"{corrected_metrics['matching']['reference_mean']:.6f} | "
        f"{corrected_metrics['multimodality']['reference_mean']:.6f} |"
    )
    for comparator_name in (
        'precleanup_latest_complete_50k', 'precleanup_git_default_280k',
    ):
        metrics = mgt_sensitivity['comparators'][comparator_name]['metrics']
        lines.append(
            f'| {comparator_name} | '
            f"{metrics['fid']['comparator_mean']:.6f} | "
            f"{metrics['top1']['comparator_mean']:.6f} | "
            f"{metrics['matching']['comparator_mean']:.6f} | "
            f"{metrics['multimodality']['comparator_mean']:.6f} |"
        )
    paired = review_evidence.get('paired_fair_method_comparison')
    if paired is not None:
        lines += [
            '', '## Paired significance against HHC-M (no GT timing)', '',
            'All tests pair the same 20 frozen request files. Positive advantage '
            'means no-timing HHC-M is better; p-values use Holm family-wise '
            'correction.', '',
            '| Comparator | Metric | Advantage | 95% bootstrap CI | Holm p |',
            '|---|---|---:|---:|---:|',
        ]
        for comparator, comparison in paired['comparators'].items():
            for metric_name, record in comparison['metrics'].items():
                interval = record['reference_advantage_bootstrap_95ci']
                lines.append(
                    f"| {comparator} | {metric_name} | "
                    f"{record['reference_advantage_mean']:.4f} | "
                    f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
                    f"{record['holm_familywise_p']:.4g} |"
                )
    paired_timing = review_evidence.get('paired_oracle_timing_effect')
    if paired_timing is not None:
        lines += [
            '', '## Oracle contact-timing effect', '',
            'This is an input-information ablation, not a fair baseline '
            'comparison. Positive advantage means oracle timing is better.', '',
            '| Metric | Oracle advantage | 95% bootstrap CI | Holm p |',
            '|---|---:|---:|---:|',
        ]
        timing_metrics = paired_timing['comparators']['hhc_no_timing']['metrics']
        for metric_name, record in timing_metrics.items():
            interval = record['reference_advantage_bootstrap_95ci']
            lines.append(
                f"| {metric_name} | {record['reference_advantage_mean']:.4f} | "
                f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
                f"{record['holm_familywise_p']:.4g} |"
            )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    temporary_md = args.output_markdown.with_suffix('.md.tmp')
    temporary_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.replace(temporary_md, args.output_markdown)
    print(json.dumps({'json': str(args.output_json),
                      'markdown': str(args.output_markdown), 'status': 'complete'}))


if __name__ == '__main__':
    main()
