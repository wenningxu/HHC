#!/usr/bin/env python
"""Validate and freeze the matched Inter-X shape-augmentation experiment."""

import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_AUDIT = Path('output/interx_shape_augmentation_plan_v2_audit.json')
LUNA_AUDIT = Path('output/interx_luna_train_timing_consistency_audit.json')
LUNA_CACHE_AUDIT = Path('output/interx_luna_train_cache_audit.json')
PREFLIGHT = Path(
    'output/interx_shape_aug_preflight_seed123_v2/_preflight_manifest.json'
)
PROTOCOL_AUDIT = Path('output/interx_protocol_audit.json')
BACKEND_EQUIVALENCE_AUDIT = Path(
    'output/optimization_backend_audit/'
    'shared_backend_equivalence_cpu_f3_e1.json'
)
VOLUMETRIC_CONTRACT_GATE = Path(
    'output/interx_volumetric_smpl_contract_gate.json'
)
PILOT_COMPARISON = Path(
    'output/interx_shape_aug_pilot_flathand_query10475_paired_physical_seed20260828.json'
)
PILOT_ASSESSMENT = Path(
    'output/interx_shape_aug_pilot_flathand_query10475_assessment_seed20260828.json'
)
PAIRED_METHODS_NO_CONTACT = Path(
    'output/fair_eval/paired_shape_augmentation_comparison.json'
)
PAIRED_METHODS_CONTACT = Path(
    'output/fair_eval/paired_shape_augmentation_contact_guidance_comparison.json'
)
LAUNCH_CONTRACT_AUDIT = Path(
    'output/interx_shape_augmentation_launch_contract_audit.json'
)
ROOT_TOKENIZER_WIRING_AUDIT = Path(
    'output/shape_root_tokenizer_wiring_audit.json'
)
DISK_CAPACITY_AUDIT = Path(
    'output/interx_shape_augmentation_disk_capacity_audit.json'
)
EXPECTED_AUGMENTED = 6238
EXPECTED_FORMAL_SAMPLES = 1696

ARMS = {
    'random': {
        'mode': 'random_shape_only',
        'query_chunk': 512,
        'completion': Path('output/interx_shape_aug_random_seed123_v2/_completion_manifest.json'),
        'materialization': Path(
            'dataset/interx_shape_aug_random_seed123_v2/materialization_manifest.json'
        ),
        'selection': Path(
            'output/hhc_formal_raw12288_repaired_mgt_shapeaug_random_selection.json'
        ),
        'training_state': Path(
            'output/hhc_formal_raw12288_repaired_mgt_shapeaug_random_training_state.json'
        ),
        'training_audit': Path(
            'output/hhc_formal_raw12288_repaired_mgt_shapeaug_random_training_contract_audit.json'
        ),
        'validation_pair_audit': Path(
            'output/hhc_formal_raw12288_repaired_mgt_shapeaug_random_20k_pre_post_validation_pair_audit.json'
        ),
        'wandb_audit': Path(
            'output/hhc_formal_raw12288_repaired_mgt_shapeaug_random_wandb_history_audit.json'
        ),
        'cache_manifest': Path(
            'checkpoints/vq_precleanup_raw12288_shapeaug_random_cache/'
            'codebook/_cache_manifest.json'
        ),
        'contact_map': Path(
            'checkpoints/vq_precleanup_raw12288_shapeaug_random_cache/'
            'code_contact_probs.npy'
        ),
        'labels': (
            'hhc_formal_raw12288_repaired_mgt_shapeaug_random_common50_no_contact',
            'hhc_formal_raw12288_repaired_mgt_shapeaug_random_common50_contact',
        ),
    },
    'optimized': {
        'mode': 'optimized',
        'query_chunk': 10475,
        'completion': Path(
            'output/interx_shape_aug_full_batch8_query10475_seed20260828_aggregate/'
            '_completion_manifest.json'
        ),
        'materialization': Path(
            'dataset/interx_shape_aug_optimized_flathand_seed123_v2/'
            'materialization_manifest.json'
        ),
        'selection': Path(
            'output/hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_selection.json'
        ),
        'training_state': Path(
            'output/hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_training_state.json'
        ),
        'training_audit': Path(
            'output/hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_training_contract_audit.json'
        ),
        'validation_pair_audit': Path(
            'output/hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_20k_pre_post_validation_pair_audit.json'
        ),
        'wandb_audit': Path(
            'output/hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_wandb_history_audit.json'
        ),
        'cache_manifest': Path(
            'checkpoints/vq_precleanup_raw12288_shapeaug_optimized_cache/'
            'codebook/_cache_manifest.json'
        ),
        'contact_map': Path(
            'checkpoints/vq_precleanup_raw12288_shapeaug_optimized_cache/'
            'code_contact_probs.npy'
        ),
        'labels': (
            'hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_common50_no_contact',
            'hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_common50_contact',
        ),
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        'path': str(path), 'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def load(path):
    record = artifact(path)
    return json.loads(Path(record['path']).read_text(encoding='utf-8')), record


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_path(left, right):
    return bool(left and right) and Path(left).resolve() == Path(right).resolve()


def complete_wandb_norm_telemetry(payload, expected_final_step):
    metrics = payload.get('metrics', {})
    required = (
        'train/gradient_l2_norm', 'train/gradient_max_abs',
        'train/weight_l2_norm', 'train/weight_max_abs',
        'train/nonfinite_gradients', 'train/nonfinite_weights',
    )
    for name in required:
        summary = metrics.get(name, {})
        if (
            not isinstance(summary.get('count'), int)
            or summary['count'] <= 0
            or not isinstance(summary.get('last_step'), int)
            or summary['last_step'] < expected_final_step
            or not all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in (
                    summary.get('minimum'), summary.get('maximum')
                )
            )
        ):
            return False
    return (
        metrics['train/nonfinite_gradients']['minimum'] == 0
        and metrics['train/nonfinite_gradients']['maximum'] == 0
        and metrics['train/nonfinite_weights']['minimum'] == 0
        and metrics['train/nonfinite_weights']['maximum'] == 0
    )


def materialization_binds_completion(materialization, completion,
                                     completion_artifact):
    """Accept a released single run or an exact multi-shard aggregate."""
    if completion.get('schema') == 'hhc_interx_shape_augmentation_run_v1':
        return (
            materialization.get('optimization_completion_sha256')
            == completion_artifact['sha256']
        )
    if completion.get('schema') != (
        'hhc_interx_shape_augmentation_shard_aggregate_v1'
    ):
        return False
    recorded = materialization.get('optimization_completions')
    shards = completion.get('shards')
    if not isinstance(recorded, list) or not isinstance(shards, list):
        return False
    expected = [shard.get('completion_manifest', {}) for shard in shards]
    if len(recorded) != len(expected):
        return False
    return all(
        same_path(actual.get('path'), wanted.get('path'))
        and actual.get('sha256') == wanted.get('sha256')
        for actual, wanted in zip(recorded, expected)
    )


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def volumetric_checkpoint_contract_valid(mode, completion, protocol):
    expected = protocol.get('asset_protocol', {}).get(
        'volumetric_smpl_checkpoint_sha256', {}
    )
    recorded = completion.get('optimizer', {}).get(
        'volumetric_smpl_checkpoints', {}
    )
    if mode == 'random_shape_only':
        return recorded == {}
    if mode != 'optimized':
        return False
    return (
        set(recorded) == {'male', 'female'}
        and set(expected) == {'male', 'female'}
        and all(
            recorded[gender].get('sha256') == expected[gender]
            and artifact(Path(recorded[gender]['path']))['sha256']
            == expected[gender]
            for gender in ('male', 'female')
        )
    )


def validate_backend_equivalence(audit, protocol):
    expected_contract = {
        'device': 'cpu',
        'frames': 3,
        'epochs': 1,
        'query_chunk': 512,
        'seed': 1234,
        'threads': 1,
        'deterministic_algorithms': True,
    }
    expected_flags = {
        'fresh_backend_reused': False,
        'shared_first_backend_reused': True,
        'shared_second_backend_reused': True,
    }
    expected_checkpoints = protocol.get('asset_protocol', {}).get(
        'volumetric_smpl_checkpoint_sha256', {}
    )
    recorded_checkpoints = audit.get('volumetric_smpl_checkpoints', {})
    comparisons = (
        audit.get('fresh_vs_shared', {}),
        audit.get('shared_first_vs_second', {}),
    )
    expected_implementations = {
        str(path.resolve()) for path in (
            ROOT / 'tools/audit_shared_volumetric_backend_equivalence.py',
            ROOT / 'tools/run_interx_shape_augmentation.py',
            ROOT / 'optimize/optimize_full_body.py',
            ROOT / 'optimization/losses.py',
            ROOT / 'optimization/volumetric_smpl_backend.py',
        )
    }
    implementations = audit.get('implementation', {})
    return (
        audit.get('schema')
        == 'hhc_shared_volumetric_backend_equivalence_v1'
        and audit.get('status') == 'pass'
        and audit.get('contract') == expected_contract
        and audit.get('diagnostic_flags') == expected_flags
        and audit.get('finite_loss_histories') == {
            'fresh': True,
            'shared_first': True,
            'shared_second': True,
        }
        and audit.get('checkpoint_hashes_valid') is True
        and all(
            comparison.get('bitwise_equal') is True
            and comparison.get('finite') is True
            and comparison.get('max_abs_difference') == 0.0
            for comparison in comparisons
        )
        and set(recorded_checkpoints) == {'male', 'female'}
        and set(expected_checkpoints) == {'male', 'female'}
        and all(
            recorded_checkpoints[gender].get('sha256')
            == expected_checkpoints[gender]
            and artifact(Path(recorded_checkpoints[gender]['path']))['sha256']
            == expected_checkpoints[gender]
            for gender in ('male', 'female')
        )
        and set(implementations) == expected_implementations
        and all(
            sha256_file(Path(path)) == digest
            for path, digest in implementations.items()
        )
    )
def validate_method(
        label, selection, contact_map, materialization, cache_manifest,
        expected_contact_guidance):
    directory = ROOT / 'output/fair_eval' / label
    run, run_artifact = load(directory / 'run_manifest.json')
    summary, summary_artifact = load(directory / 'summary.json')
    mm, mm_artifact = load(directory / 'multimodality_audit_summary.json')
    physical, physical_artifact = load(directory / 'replication_00_physical.json')
    timing, timing_artifact = load(directory / 'generation_timing_summary.json')
    require(
        run.get('method') == 'hhc'
        and run.get('label') == label
        and run.get('replication_range') == [0, 20]
        and run.get('samples_per_replication') == EXPECTED_FORMAL_SAMPLES
        and run.get('sampling_steps') == 50
        and run.get('temperature') == 0.5
        and run.get('cfg_scale') == 3.5
        and run.get('contact_guidance_scale') == 1.0
        and run.get('hhc_vq_contract') == 'precleanup_raw12288'
        and run.get('hhc_mgt_contract') == 'current'
        and run.get('hhc_contact_map_contract')
        == 'paper_formula_train_only'
        and run.get('hhc_stats_prefix') == 'Interx'
        and same_path(
            run.get('hhc_stats_dir'), materialization['output_root']
        )
        and same_path(run.get('selection_manifest'), selection['path'])
        and same_path(run.get('hhc_contact_map'), contact_map['path']),
        f'{label}: generation contract differs from its augmented arm',
    )
    if expected_contact_guidance:
        require(
            run.get('contact_guidance') is True
            and run.get('contact_timing_source') == 'request_oracle'
            and run.get('contact_timing_source_requested') == 'request_oracle',
            f'{label}: contact-guided arm is not bound to frozen request timing',
        )
    else:
        require(
            run.get('contact_guidance') is False
            and run.get('contact_timing_source') == 'none'
            and run.get('contact_timing_source_requested') == 'none'
            and run.get('contact_timing_file') is None,
            f'{label}: no-timing arm consumed or requested contact timing',
        )
    provenance_artifacts = run.get('provenance', {}).get('artifacts', {})
    normalization = materialization['normalization']
    require(
        provenance_artifacts.get('normalization_mean', {}).get('sha256')
        == normalization['mean_sha256']
        and provenance_artifacts.get('normalization_std', {}).get('sha256')
        == normalization['std_sha256']
        and provenance_artifacts.get('mgt_selection_manifest', {}).get('sha256')
        == selection['sha256']
        and (
            not expected_contact_guidance
            or provenance_artifacts.get('contact_map', {}).get('sha256')
            == contact_map['sha256']
        )
        and (
            not expected_contact_guidance
            or provenance_artifacts.get('contact_map_manifest', {}).get('sha256')
            == cache_manifest['sha256']
        ),
        f'{label}: generation artifacts are not hash-bound to the augmented arm',
    )
    require(
        summary.get('replications') == 20
        and summary.get('samples_per_replication') == EXPECTED_FORMAL_SAMPLES
        and summary.get('provenance', {}).get('legacy_provenance_allowed')
        is False
        and summary.get('provenance', {}).get(
            'generation_files_sha256_verified') is True
        and summary.get('provenance', {}).get('unique_requests_verified')
        is True,
        f'{label}: semantic summary is not the strict 20x1696 protocol',
    )
    sampler_identity = summary.get('provenance', {}).get(
        'sampler_identity', {}
    )
    expected_sampler = {
        'sampler': 'sample_intra_cas',
        'sampler_semantics': 'paper_high_confidence_tokens_fixed',
        'cascade_order': 'root,body,hands',
        'sampling_steps': 50,
        'sampling_steps_unit': 'outer_maskgit_iterations',
        'channel_updates_per_outer_step': 3,
        'model_forward_calls_per_outer_step_with_cfg': 6,
        'temperature': 0.5,
        'cfg_scale': 3.5,
        'contact_guidance': expected_contact_guidance,
        'contact_guidance_scale': 1.0,
        'contact_map_contract': (
            'paper_formula_train_only'
            if expected_contact_guidance else 'unused'
        ),
        'contact_timing_source': (
            'request_oracle' if expected_contact_guidance else 'none'
        ),
        'generation_length_policy': 'exact_request_length',
    }
    require(
        all(sampler_identity.get(key) == value
            for key, value in expected_sampler.items()),
        f'{label}: HDF5 sampler identity differs from common50 paper sampling',
    )
    require(
        mm.get('replications') == 20
        and physical.get('samples') == EXPECTED_FORMAL_SAMPLES
        and timing.get('replications') == 20,
        f'{label}: MM/physical/timing evidence is incomplete',
    )
    return {
        'run_manifest': run_artifact,
        'semantic_summary': summary_artifact,
        'multimodality_summary': mm_artifact,
        'physical_replication_0': physical_artifact,
        'generation_timing_summary': timing_artifact,
    }


def main():
    output = ROOT / 'output/interx_shape_augmentation_results.json'
    plan, plan_artifact = load(PLAN_AUDIT)
    luna, luna_artifact = load(LUNA_AUDIT)
    luna_cache, luna_cache_artifact = load(LUNA_CACHE_AUDIT)
    preflight, preflight_artifact = load(PREFLIGHT)
    protocol, protocol_artifact = load(PROTOCOL_AUDIT)
    backend_equivalence, backend_equivalence_artifact = load(
        BACKEND_EQUIVALENCE_AUDIT
    )
    volumetric_gate, volumetric_gate_artifact = load(
        VOLUMETRIC_CONTRACT_GATE
    )
    pilot, pilot_artifact = load(PILOT_COMPARISON)
    pilot_assessment, pilot_assessment_artifact = load(PILOT_ASSESSMENT)
    paired, paired_artifact = load(PAIRED_METHODS_NO_CONTACT)
    paired_contact, paired_contact_artifact = load(PAIRED_METHODS_CONTACT)
    launch_contract, launch_contract_artifact = load(LAUNCH_CONTRACT_AUDIT)
    root_wiring, root_wiring_artifact = load(ROOT_TOKENIZER_WIRING_AUDIT)
    disk_capacity, disk_capacity_artifact = load(DISK_CAPACITY_AUDIT)
    require(
        plan.get('status') == 'pass'
        and plan.get('counts', {}).get('eligible_timing_covered_samples')
        == EXPECTED_AUGMENTED,
        'shape augmentation plan audit is not the frozen 6,238-sample v2 plan',
    )
    require(
        luna.get('status') == 'complete'
        and luna.get('fusion_readiness_passed') is True
        and luna.get('counts', {}).get('expected_samples') == 9110
        and luna.get('counts', {}).get('audited_samples') == 9110,
        'full-train Luna cache did not pass the strict fusion audit',
    )
    require(
        luna_cache.get('schema')
        == 'hhc_interx_codex_contact_cache_audit_v1'
        and luna_cache.get('status') == 'pass'
        and luna_cache.get('complete') is True
        and luna_cache.get('integrity_passed') is True
        and luna_cache.get('counts', {}).get('expected_samples') == 9110
        and luna_cache.get('counts', {}).get('completed_samples') == 9110,
        'full-train Luna cache did not pass strict provenance/caption audit',
    )
    require(
        preflight.get('schema')
        == 'hhc_interx_shape_augmentation_preflight_v1'
        and preflight.get('status') == 'pass'
        and preflight.get('selected_samples') == EXPECTED_AUGMENTED
        and preflight.get('positive_timing_samples') == 5868
        and preflight.get('semantic_contact_active_samples') == 5666
        and preflight.get('physical_timing_only_samples') == 202
        and preflight.get('regions_inactive_because_timing_is_zero') == 137
        and preflight.get('structured_contact_rows') == 7434
        and preflight.get('boundary_whitespace_caption_matches') == 1
        and preflight.get('inputs', {}).get('plan_sha256')
        == plan.get('plan_sha256')
        and preflight.get('inputs', {}).get('contact_manifest_sha256')
        == luna_cache.get('inputs', {}).get('manifest_sha256')
        and preflight.get('implementation', {}).get('sha256')
        == sha256_file(preflight['implementation']['path']),
        'full shape-augmentation input preflight is missing or stale',
    )
    require(
        protocol.get('status') == 'pass'
        and set(protocol.get('asset_protocol', {}).get(
            'volumetric_smpl_checkpoint_sha256', {}
        )) == {'male', 'female'},
        'official Inter-X protocol does not bind both VolumetricSMPL weights',
    )
    require(
        validate_backend_equivalence(backend_equivalence, protocol),
        'shared VolumetricSMPL backend equivalence audit is missing or stale',
    )
    require(
        volumetric_gate.get('schema')
        == 'hhc_interx_volumetric_smpl_contract_gate_v1'
        and volumetric_gate.get('status') == 'pass'
        and volumetric_gate.get('query_chunk') == 10475
        and volumetric_gate.get('representation_contract', {}).get(
            'smplx_flat_hand_mean') is True
        and volumetric_gate.get('implementation', {}).get('sha256')
        == sha256_file(
            ROOT / 'tools/require_interx_volumetric_smpl_contract.py'
        )
        and volumetric_gate.get('verified_artifacts', {}).get(
            'production_backend', {}).get('sha256')
        == sha256_file(ROOT / 'optimization/volumetric_smpl_backend.py')
        and volumetric_gate.get('verified_artifacts', {}).get(
            'geometry_audit', {}).get('sha256')
        == sha256_file(
            ROOT / 'output/'
            'volumetric_smpl_interx_flat_hand_geometry_audit.json'
        )
        and volumetric_gate.get('verified_artifacts', {}).get(
            'query_assessment', {}).get('sha256')
        == sha256_file(
            ROOT / 'output/'
            'volumetric_smpl_flat_hand_query_chunk_assessment.json'
        ),
        'flat-hand geometry/query-chunk contract gate is missing or stale',
    )
    require(
        pilot.get('schema') == 'hhc_paired_physical_comparison_v1'
        and pilot.get('sample_count') == 25,
        'matched random-vs-optimized physical pilot is incomplete',
    )
    require(
        pilot_assessment.get('schema')
        == 'hhc_shape_augmentation_pilot_assessment_v1'
        and pilot_assessment.get('status') == 'pass'
        and pilot_assessment.get('full_augmentation_authorized') is True
        and all(pilot_assessment.get('checks', {}).values())
        and all(
            record.get('pass') is True
            for record in pilot_assessment.get(
                'metric_decisions', {}).values()
        ),
        'flat-hand matched pilot did not authorize full augmentation',
    )
    require(
        launch_contract.get('schema')
        == 'hhc_interx_shape_augmentation_launch_contract_v1'
        and launch_contract.get('status') == 'pass'
        and launch_contract.get('launcher', {}).get('sha256')
        == sha256_file(ROOT / 'tools/queue_interx_shape_augmentation_gpu1.sh')
        and launch_contract.get('state_resolver', {}).get('sha256')
        == sha256_file(ROOT / 'tools/resolve_mgt_training_state.py')
        and launch_contract.get('root_tokenizer', {}).get('sha256')
        == sha256_file(
            ROOT / 'dataset/dataset_interx_tokenize_historical_root.py'
        )
        and launch_contract.get('root_tokenizer', {}).get('contract')
        == {
            'explicit_contract': True,
            'uses_root_statistics': True,
            'does_not_apply_extra_coordinate_preprocessing': True,
            'reuses_deterministic_crop': True,
        },
        'shape-augmentation launch contract is missing or stale',
    )
    require(
        root_wiring.get('schema')
        == 'hhc_shape_root_tokenizer_wiring_audit_v1'
        and root_wiring.get('status') == 'pass'
        and root_wiring.get('contract')
        == 'deterministic_historical_root_v1'
        and all(root_wiring.get('checks', {}).values())
        and root_wiring.get('missing_checks') == []
        and root_wiring.get('artifacts', {}).get(
            'training_entrypoint', {}).get('sha256')
        == sha256_file(ROOT / 'train/train_t2i_trans.py')
        and root_wiring.get('artifacts', {}).get(
            'root_tokenizer', {}).get('sha256')
        == sha256_file(
            ROOT / 'dataset/dataset_interx_tokenize_historical_root.py'
        ),
        'shape root-tokenizer training wiring is missing or stale',
    )
    require(
        disk_capacity.get('schema')
        == 'hhc_shape_augmentation_disk_capacity_audit_v1'
        and disk_capacity.get('status') == 'pass'
        and all(disk_capacity.get('checks', {}).values())
        and disk_capacity.get('policy', '').startswith(
            'conservative apparent-byte projection'
        )
        and disk_capacity.get('projection', {}).get('arms') == 2
        and disk_capacity.get('projection', {}).get('labels_per_arm') == 2
        and disk_capacity.get('projection', {}).get(
            'free_space_after_queue_gib'
        ) == 100
        and disk_capacity.get('implementation', {}).get('sha256')
        == sha256_file(ROOT / 'tools/audit_shape_augmentation_disk_capacity.py'),
        'shape-augmentation disk-capacity audit is missing or stale',
    )
    require(
        paired.get('schema') == 'fair_interx_paired_method_comparison_v1'
        and paired.get('replications') == 20
        and paired.get('reference', {}).get('label') == 'baseline'
        and same_path(
            paired.get('reference', {}).get('directory'),
            'output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_no_contact',
        )
        and set(paired.get('comparators', {}))
        == {'random_shape_only', 'optimized_shape'}
        and same_path(
            paired['comparators']['random_shape_only'].get('directory'),
            'output/fair_eval/hhc_formal_raw12288_repaired_mgt_shapeaug_random_common50_no_contact',
        )
        and same_path(
            paired['comparators']['optimized_shape'].get('directory'),
            'output/fair_eval/hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_common50_no_contact',
        ),
        'formal paired shape-augmentation comparison is incomplete',
    )
    require(
        paired_contact.get('schema')
        == 'fair_interx_paired_method_comparison_v1'
        and paired_contact.get('replications') == 20
        and paired_contact.get('reference', {}).get('label') == 'baseline'
        and same_path(
            paired_contact.get('reference', {}).get('directory'),
            'output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact',
        )
        and set(paired_contact.get('comparators', {}))
        == {'random_shape_only', 'optimized_shape'}
        and same_path(
            paired_contact['comparators']['random_shape_only'].get('directory'),
            'output/fair_eval/hhc_formal_raw12288_repaired_mgt_shapeaug_random_common50_contact',
        )
        and same_path(
            paired_contact['comparators']['optimized_shape'].get('directory'),
            'output/fair_eval/hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_common50_contact',
        ),
        'contact-guided paired shape-augmentation comparison is incomplete',
    )

    artifacts = {
        'plan_audit': plan_artifact,
        'luna_train_audit': luna_artifact,
        'luna_train_cache_audit': luna_cache_artifact,
        'augmentation_input_preflight': preflight_artifact,
        'interx_protocol_audit': protocol_artifact,
        'shared_backend_equivalence': backend_equivalence_artifact,
        'volumetric_smpl_contract_gate': volumetric_gate_artifact,
        'pilot_paired_physical': pilot_artifact,
        'pilot_assessment': pilot_assessment_artifact,
        'paired_method_comparison': paired_artifact,
        'paired_method_comparison_contact_guidance': paired_contact_artifact,
        'launch_contract': launch_contract_artifact,
        'root_tokenizer_wiring': root_wiring_artifact,
        'disk_capacity_preflight': disk_capacity_artifact,
        'implementation': artifact(Path(__file__)),
    }
    results = {}
    completed_names = {}
    plan_hash = plan['plan_sha256']
    official_mean = artifact(Path('dataset/interx/Interx_Mean.npy'))
    official_std = artifact(Path('dataset/interx/Interx_Std.npy'))
    artifacts['official_raw_mean'] = official_mean
    artifacts['official_raw_std'] = official_std
    for arm, contract in ARMS.items():
        completion, completion_artifact = load(contract['completion'])
        materialization, materialization_artifact = load(
            contract['materialization']
        )
        selection, selection_artifact = load(contract['selection'])
        training_state, training_state_artifact = load(
            contract['training_state']
        )
        training_audit, training_audit_artifact = load(
            contract['training_audit']
        )
        validation_pair, validation_pair_artifact = load(
            contract['validation_pair_audit']
        )
        wandb_audit, wandb_audit_artifact = load(contract['wandb_audit'])
        cache, cache_artifact = load(contract['cache_manifest'])
        contact_map_artifact = artifact(contract['contact_map'])
        expected_backend_lifetime = (
            'one frozen male/female backend per serial process; fresh '
            'loss state and optimizer per sample'
            if contract['mode'] == 'optimized'
            else 'not_instantiated_for_random_shape_only_control'
        )
        volume_contract_valid = volumetric_checkpoint_contract_valid(
            contract['mode'], completion, protocol
        )
        require(
            completion.get('schema') in (
                'hhc_interx_shape_augmentation_run_v1',
                'hhc_interx_shape_augmentation_shard_aggregate_v1',
            )
            and completion.get('complete') is True
            and completion.get('requested_samples') == EXPECTED_AUGMENTED
            and completion.get('completed_samples') == EXPECTED_AUGMENTED
            and completion.get('optimizer', {}).get('mode') == contract['mode']
            and completion.get('optimizer', {}).get('query_chunk')
            == contract['query_chunk']
            and completion.get('optimizer', {}).get(
                'volumetric_smpl_representation', {}).get(
                    'smplx_flat_hand_mean') is True
            and completion.get('optimizer', {}).get(
                'volumetric_smpl_representation', {}).get(
                    'hand_pose_convention') == 'flat_hand_mean'
            and (
                contract['mode'] == 'random_shape_only'
                or completion.get('optimizer', {}).get(
                    'volumetric_smpl_representation')
                == volumetric_gate.get('representation_contract')
            )
            and completion.get('optimizer', {}).get('backend_lifetime')
            == expected_backend_lifetime
            and volume_contract_valid
            and completion.get('plan_sha256') == plan_hash,
            f'{arm}: full augmentation completion is invalid',
        )
        if contract['mode'] == 'optimized':
            require(
                completion.get('schema')
                == 'hhc_interx_shape_augmentation_shard_aggregate_v1'
                and len(completion.get('shards', [])) == 8
                and completion.get('selection', {}).get(
                    'full_selection_positions'
                ) == list(range(EXPECTED_AUGMENTED)),
                'optimized: expected the exact validated eight-shard aggregate',
            )
        completed_names[arm] = completion['completed_names']
        require(
            materialization.get('schema')
            == 'hhc_interx_shape_augmentation_materialization_v1'
            and materialization.get('status') == 'complete'
            and materialization.get('augmentation_mode') == contract['mode']
            and materialization.get('counts', {}).get('official_train') == 9110
            and materialization.get('counts', {}).get('augmented')
            == EXPECTED_AUGMENTED
            and materialization.get('counts', {}).get('combined_train') == 15348
            and materialization.get('normalization', {}).get('policy')
            == 'reuse_original_raw_and_root_interx_stats_no_recompute'
            and materialization.get('normalization', {}).get(
                'generation_and_decode_representation') == 'raw'
            and materialization.get('normalization', {}).get(
                'mgt_tokenizer_contract')
            == 'deterministic_historical_root_v1'
            and materialization.get('normalization', {}).get(
                'copy_is_byte_identical') is True
            and same_path(
                materialization.get('normalization', {}).get('source_mean'),
                official_mean['path'],
            )
            and materialization.get('normalization', {}).get(
                'source_mean_sha256') == official_mean['sha256']
            and materialization.get('normalization', {}).get(
                'mean_sha256') == official_mean['sha256']
            and same_path(
                materialization.get('normalization', {}).get('source_std'),
                official_std['path'],
            )
            and materialization.get('normalization', {}).get(
                'source_std_sha256') == official_std['sha256']
            and materialization.get('normalization', {}).get(
                'std_sha256') == official_std['sha256']
            and materialization.get('normalization', {}).get(
                'source_root_mean_sha256')
            == sha256_file(ROOT / 'dataset/interx/Interx_Mean_root.npy')
            and materialization.get('normalization', {}).get(
                'root_mean_sha256')
            == sha256_file(ROOT / 'dataset/interx/Interx_Mean_root.npy')
            and materialization.get('normalization', {}).get(
                'source_root_std_sha256')
            == sha256_file(ROOT / 'dataset/interx/Interx_Std_root.npy')
            and materialization.get('normalization', {}).get(
                'root_std_sha256')
            == sha256_file(ROOT / 'dataset/interx/Interx_Std_root.npy')
            and materialization_binds_completion(
                materialization, completion, completion_artifact
            ),
            f'{arm}: immutable materialized overlay is invalid',
        )
        require(
            selection.get('run_complete') is True
            and selection.get('expected_final_step') == 300000,
            f'{arm}: selected MGT is not a complete 300k run',
        )
        training_manifest, training_artifact = load(
            Path(selection['training_manifest'])
        )
        resolved = training_manifest.get('resolved_arguments', {})
        require(
            training_state.get('schema')
            == 'hhc_mgt_training_state_resolution_v1'
            and training_state.get('status') == 'complete'
            and training_state.get('training_required') is False
            and same_path(
                training_state.get('active_run', {}).get('manifest'),
                training_artifact['path'],
            )
            and training_state.get('active_run', {}).get('manifest_sha256')
            == training_artifact['sha256'],
            f'{arm}: completed/resumed MGT state is not bound to selection',
        )
        require(
            resolved.get('nb_code') == 12288
            and resolved.get('token_cache_contract')
            == 'deterministic_historical_root_v1'
            and resolved.get('paper_architecture') is True
            and resolved.get('embed_dim_gpt') == 512
            and resolved.get('n_head_gpt') == 8
            and resolved.get('num_layers') == 9
            and resolved.get('num_local_layer') == 9
            and resolved.get('total_iter') == 300000
            and resolved.get('training_mask_contract') == 'paper_kinematic'
            and resolved.get('training_loss_contract') == 'paper_masked_only'
            and resolved.get('optimizer_contract') == 'paper_warmup_cosine'
            and resolved.get('conditioning_contract') == 'paper_text_cfg'
            and resolved.get('training_token_length_contract') == 'paper_50'
            and resolved.get('training_validation_contract')
            == 'paper_isolated_common'
            and resolved.get('sampling_steps') == 50
            and resolved.get('sampler') == 'sample_intra_cas'
            and same_path(
                resolved.get('stats_dir'), materialization['output_root']
            )
            and resolved.get('stats_prefix') == 'Interx'
            and same_path(
                resolved.get('augmentation_manifest'),
                materialization_artifact['path'],
            ),
            f'{arm}: MGT training manifest is not bound to its overlay',
        )
        require(
            training_manifest.get('source_derived_configuration', {}).get(
                'training_module_mode_contract'
            ) == 'formal_transformer_train_frozen_clip_vq_eval'
            and training_audit.get('schema')
            == 'hhc_paper_training_contract_audit_v1'
            and training_audit.get('status')
            == 'pass_with_disclosed_exceptions'
            and training_audit.get('active_training_manifest', {}).get(
                'sha256') == training_artifact['sha256']
            and validation_pair.get('schema')
            == 'hhc_lightning_same_step_pair_audit_v1'
            and validation_pair.get('status') == 'pass'
            and validation_pair.get('expected_step') == 20000
            and validation_pair.get('model_tensor_count') == 947
            and validation_pair.get('differing_model_keys') == []
            and all(validation_pair.get('exact', {}).get(key) is True for key in (
                'model_state', 'optimizer_state', 'lr_scheduler_state',
                'hyper_parameters',
            ))
            and Path(validation_pair.get('before', {}).get('path', '')).name
            == 'prevalidation-000020000.ckpt'
            and Path(validation_pair.get('after', {}).get('path', '')).name
            == 'postvalidation-000020000.ckpt'
            and validation_pair.get('fixed_pair_callback', {}).get(
                'before', {}).get('step') == 20000
            and isinstance(
                validation_pair.get('fixed_pair_callback', {}).get(
                    'before', {}).get('pre_saved'),
                bool,
            )
            and validation_pair.get('fixed_pair_callback', {}).get(
                'before', {}).get('post_saved') is False
            and validation_pair.get('fixed_pair_callback', {}).get('after')
            == {
                'step': 20000,
                'pre_saved': True,
                'post_saved': True,
            },
            f'{arm}: repaired formal MGT training/runtime audit is invalid',
        )
        require(
            training_manifest.get('observability', {}).get('wandb_mode')
            == 'online'
            and wandb_audit.get('schema')
            == 'hhc_wandb_online_metric_history_audit_v1'
            and wandb_audit.get('status') == 'pass'
            and wandb_audit.get('metric_profile')
            == 'masked_token_classification'
            and wandb_audit.get('expected_final_step') == 300000
            and wandb_audit.get('remote_state') == 'finished'
            and all(wandb_audit.get('checks', {}).values())
            and wandb_audit.get('checks', {}).get(
                'all_training_metrics_reach_expected_step'
            ) is True
            and wandb_audit.get('checks', {}).get(
                'all_validation_metrics_follow_final_training_event'
            ) is True
            and wandb_audit.get('checks', {}).get(
                'nonfinite_counts_zero'
            ) is True
            and wandb_audit.get('optimizer_step_axis')
            == 'train/optimizer_step'
            and wandb_audit.get('implementation', {}).get('sha256')
            == sha256_file(
                ROOT / 'tools/audit_wandb_online_metric_history.py'
            )
            and wandb_audit.get('run_id')
            == training_manifest.get('observability', {}).get('wandb_run_id')
            and complete_wandb_norm_telemetry(wandb_audit, 300000)
            and wandb_audit.get('metrics', {}).get(
                'train/nonfinite_gradients', {}
            ).get('maximum') == 0.0
            and wandb_audit.get('metrics', {}).get(
                'train/nonfinite_weights', {}
            ).get('maximum') == 0.0,
            f'{arm}: completed MGT lacks a full finite online-W&B history',
        )
        require(
            cache.get('status') == 'complete'
            and cache.get('contact_map_scope') == 'train_only'
            and cache.get('contact_map_estimator')
            == 'empirical_frequency_unseen_neutral_0.5'
            and cache.get('contact_map_sha256') == contact_map_artifact['sha256']
            and cache.get('token_cache_contract')
            == 'deterministic_historical_root_v1'
            and cache.get('normalization_mean_sha256')
            == sha256_file(ROOT / 'dataset/interx/Interx_Mean_root.npy')
            and cache.get('normalization_std_sha256')
            == sha256_file(ROOT / 'dataset/interx/Interx_Std_root.npy')
            and cache.get('tokenizer_implementation_sha256')
            == sha256_file(
                ROOT / 'dataset/dataset_interx_tokenize_historical_root.py'
            )
            and cache.get('vq_checkpoint_sha256')
            == 'fdf5afaa5d3513f933254fafc831f6046f7c38b5e649db08fb6be7e0ce1006bb',
            f'{arm}: augmented token/contact cache provenance is invalid',
        )
        method_artifacts = {}
        for label, expected_contact_guidance in zip(
                contract['labels'], (False, True)):
            method_artifacts[label] = validate_method(
                label, selection_artifact, contact_map_artifact,
                materialization, cache_artifact, expected_contact_guidance,
            )
        artifacts[f'{arm}_completion'] = completion_artifact
        artifacts[f'{arm}_materialization'] = materialization_artifact
        artifacts[f'{arm}_selection'] = selection_artifact
        artifacts[f'{arm}_training_state'] = training_state_artifact
        artifacts[f'{arm}_training_manifest'] = training_artifact
        artifacts[f'{arm}_training_contract_audit'] = training_audit_artifact
        artifacts[f'{arm}_validation_pair_audit'] = validation_pair_artifact
        artifacts[f'{arm}_wandb_history_audit'] = wandb_audit_artifact
        artifacts[f'{arm}_cache_manifest'] = cache_artifact
        artifacts[f'{arm}_contact_map'] = contact_map_artifact
        results[arm] = {
            'augmentation_mode': contract['mode'],
            'backend_lifetime': expected_backend_lifetime,
            'augmented_samples': EXPECTED_AUGMENTED,
            'combined_train_samples': 15348,
            'selection': selection_artifact,
            'methods': method_artifacts,
        }

    require(
        completed_names['random'] == completed_names['optimized'],
        'random-only and optimized arms do not use identical augmented rows',
    )
    payload = {
        'schema': 'hhc_interx_shape_augmentation_results_v1',
        'status': 'complete',
        'design': (
            'same 6,238 source clips, random donor shapes, captions, manual '
            'timing and 300k MGT budget; optimized arm differs only by the '
            'paper contact/physical refinement'
        ),
        'official_split_unchanged': True,
        'fixed_vq_sha256': (
            'fdf5afaa5d3513f933254fafc831f6046f7c38b5e649db08fb6be7e0ce1006bb'
        ),
        'pilot_samples': 25,
        'augmented_samples_per_arm': EXPECTED_AUGMENTED,
        'input_preflight': {
            key: preflight[key] for key in (
                'selected_samples', 'positive_timing_samples',
                'semantic_contact_active_samples',
                'physical_timing_only_samples',
                'regions_inactive_because_timing_is_zero',
                'structured_contact_rows',
                'boundary_whitespace_caption_matches',
                'caption_binding_policy',
            )
        },
        'results': results,
        'artifacts': artifacts,
    }
    atomic_json(output, payload)
    print(json.dumps({
        'output': str(output), 'status': 'complete',
        'augmented_samples_per_arm': EXPECTED_AUGMENTED,
    }, indent=2))


if __name__ == '__main__':
    main()
