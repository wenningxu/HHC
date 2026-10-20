#!/usr/bin/env python
"""Freeze the matched Inter-X shape-augmentation training/evaluation queue."""

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'tools/queue_interx_shape_augmentation_gpu1.sh'
STATE_RESOLVER = ROOT / 'tools/resolve_mgt_training_state.py'
ROOT_TOKENIZER = ROOT / 'dataset/dataset_interx_tokenize_historical_root.py'
OUTPUT = ROOT / 'output/interx_shape_augmentation_launch_contract_audit.json'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    source = LAUNCHER.read_text(encoding='utf-8')
    required = (
        'tools/require_mgt_repair_gate.py',
        '--gate output/mgt_repair_gate.json',
        'tools/audit_shape_root_tokenizer_wiring.py',
        'tools/audit_online_wandb_training_contract.py',
        'tools/require_interx_volumetric_smpl_contract.py',
        'tools/audit_random_shape_augmentation_completion.py',
        'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
        '--query-chunk 10475',
        'interx_volumetric_smpl_contract_gate.json',
        'tools/audit_shape_augmentation_disk_capacity.py',
        'interx_shape_augmentation_disk_capacity_audit.json',
        'output/hhc_formal_raw12288_repaired_mgt_selection.json',
        'hhc_formal_raw12288_repaired_mgt_common50_no_contact',
        'hhc_formal_raw12288_repaired_mgt_common50_contact',
        'dataset/interx/contact_pairs_codex_luna_strict_v3_train',
        'timing_covered_seed123_v2/plan_manifest.json',
        'interx_shape_aug_random_seed123_v2',
        'interx_shape_aug_full_batch8_query10475_seed20260828_aggregate',
        'tools/run_interx_shape_augmentation_full_single_shard.sh',
        'tools/aggregate_interx_shape_augmentation_shards.py',
        '--expected-samples 6238 --expected-shards 8',
        'optimized_materialize_args',
        'tools/assess_shape_augmentation_pilot.py',
        'pilot_physical_gate_v1.json',
        'interx_shape_aug_pilot_flathand_query10475_assessment_seed20260828.json',
        '--mode random_shape_only',
        '--mode optimized',
        'vq_precleanup_raw12288_shapeaug_${arm}_cache',
        'hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_300k_seed123_v1',
        'hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_selection.json',
        'hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_common50',
        'tools/resolve_mgt_training_state.py',
        '--resume-training-ckpt',
        '--resume-in-place',
        '--token-cache-contract deterministic_historical_root_v1',
        '--nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512',
        '--embed-dim-gpt 512 --n-head-gpt 8 --num-layers 9 --num-local-layer 9',
        '--batch-size 96 --total-iter 300000 --warm-up-iter 1000',
        '--lr 0.0002 --weight-decay 0.000001',
        '--contact --no-contact-conditioning --shape --paper-architecture',
        '--training-mask-contract paper_kinematic',
        '--training-loss-contract paper_masked_only',
        '--optimizer-contract paper_warmup_cosine',
        '--conditioning-contract paper_text_cfg',
        '--training-token-length-contract paper_50',
        '--training-validation-contract paper_isolated_common',
        '--sampler sample_intra_cas --temperature 0.5 --sampling-steps 50',
        '--cfg-scale 3.5 --contact-guidance-scale 1.0',
        '--wandb-mode online --seed 123',
        'tools/audit_lightning_checkpoint_pair.py',
        'prevalidation-000020000.ckpt',
        'postvalidation-000020000.ckpt',
        '--expected-step 20000',
        '--require-fixed-pair-marker',
        'tools/audit_hhc_paper_training_contract.py',
        'tools/audit_wandb_online_metric_history.py',
        '--metric-profile masked_token_classification',
        'shapeaug_${arm}_wandb_history_audit.json',
        '--expected-final-step 300000',
        'run_arm random "${random_overlay}"',
        'run_arm optimized "${optimized_overlay}"',
        'paired_shape_augmentation_comparison.json',
        'paired_shape_augmentation_contact_guidance_comparison.json',
        'tools/build_shape_augmentation_results.py',
    )
    forbidden = (
        'hhc_shapeaug_random_current_mgt_selection.json',
        'hhc_shapeaug_optimized_current_mgt_selection.json',
        'hhc_mgt_shapeaug_random_common50',
        'hhc_mgt_shapeaug_optimized_common50',
        '--nb-code 8096',
        '--token-cache-contract deterministic_raw_v1',
        'body-controlled',
        'body_controlled',
        'pilot_optimized="output/interx_shape_aug_pilot_optimized_seed20260828"',
        'full_optimized="output/interx_shape_aug_optimized_seed123_v2"',
        '--output output/interx_shape_aug_pilot_assessment_seed20260828.json',
    )
    missing = [value for value in required if value not in source]
    present_forbidden = [value for value in forbidden if value in source]
    tokenizer_source = ROOT_TOKENIZER.read_text(encoding='utf-8')
    tokenizer_contract = {
        'explicit_contract': (
            "TOKENIZER_CONTRACT = 'deterministic_historical_root_v1'"
            in tokenizer_source
        ),
        'uses_root_statistics': "motion_rep='vel'" in tokenizer_source,
        'does_not_apply_extra_coordinate_preprocessing': (
            'prepare_interx' not in tokenizer_source
        ),
        'reuses_deterministic_crop': (
            'deterministic_cache_crop' in tokenizer_source
            and 'dataset.dataset_interx_tokenize import' in tokenizer_source
        ),
    }
    ordering = {
        'repair_gate_before_launch_audit': source.index(
            'require_mgt_repair_gate.py'
        ) < source.index('audit_shape_augmentation_launch_contract.py'),
        'launch_audit_before_root_wiring_gate': source.index(
            'audit_shape_augmentation_launch_contract.py'
        ) < source.index('audit_shape_root_tokenizer_wiring.py'),
        'root_wiring_gate_before_training': source.index(
            'audit_shape_root_tokenizer_wiring.py'
        ) < source.rindex('train/train_t2i_trans.py'),
        'state_resolution_before_training': source.index(
            'resolve_mgt_training_state.py'
        ) < source.rindex('train/train_t2i_trans.py'),
        'training_before_selection': source.rindex(
            'train/train_t2i_trans.py'
        ) < source.index('select_best_mgt_checkpoint.py'),
        'selection_before_runtime_audits': source.index(
            'select_best_mgt_checkpoint.py'
        ) < source.index('audit_lightning_checkpoint_pair.py'),
        'runtime_audits_before_fair_evaluation': source.index(
            'audit_hhc_paper_training_contract.py'
        ) < source.index('run_fair_interx_method.py'),
        'online_contract_before_training': source.index(
            'audit_online_wandb_training_contract.py'
        ) < source.rindex('train/train_t2i_trans.py'),
        'volumetric_contract_before_optimized_pilot': source.index(
            'require_interx_volumetric_smpl_contract.py'
        ) < source.index(
            'pilot_optimized="output/interx_shape_aug_pilot_optimized_flathand'
        ),
        'online_history_audit_before_fair_evaluation': source.index(
            'audit_wandb_online_metric_history.py'
        ) < source.index('run_fair_interx_method.py'),
        'pilot_assessment_before_disk_and_full_assets': source.index(
            'assess_shape_augmentation_pilot.py'
        ) < source.index('audit_shape_augmentation_disk_capacity.py', 1000) < source.index(
            'full_random="output/interx_shape_aug_random_seed123_v2"'
        ),
        'all_shards_before_aggregate': source.index(
            'tools/run_interx_shape_augmentation_full_single_shard.sh'
        ) < source.index('tools/aggregate_interx_shape_augmentation_shards.py'),
        'aggregate_before_optimized_materialization': source.index(
            'tools/aggregate_interx_shape_augmentation_shards.py'
        ) < source.index('optimized_materialize_args'),
        'optimized_materialization_before_training': source.index(
            'optimized_materialize_args'
        ) < source.rindex('train/train_t2i_trans.py'),
        'both_arms_before_paired_comparison': source.index(
            'run_arm optimized "${optimized_overlay}"'
        ) < source.index('compare_fair_interx_methods_paired.py'),
        'paired_comparison_before_result_builder': source.rindex(
            'compare_fair_interx_methods_paired.py'
        ) < source.index('build_shape_augmentation_results.py'),
    }
    status = (
        'pass' if not missing and not present_forbidden
        and all(ordering.values()) and all(tokenizer_contract.values())
        else 'fail'
    )
    payload = {
        'schema': 'hhc_interx_shape_augmentation_launch_contract_v1',
        'status': status,
        'launcher': {
            'path': str(LAUNCHER),
            'bytes': LAUNCHER.stat().st_size,
            'sha256': sha256_file(LAUNCHER),
        },
        'state_resolver': {
            'path': str(STATE_RESOLVER),
            'bytes': STATE_RESOLVER.stat().st_size,
            'sha256': sha256_file(STATE_RESOLVER),
        },
        'root_tokenizer': {
            'path': str(ROOT_TOKENIZER),
            'bytes': ROOT_TOKENIZER.stat().st_size,
            'sha256': sha256_file(ROOT_TOKENIZER),
            'contract': tokenizer_contract,
        },
        'required_contract_entries': list(required),
        'missing_contract_entries': missing,
        'forbidden_entries_present': present_forbidden,
        'ordering': ordering,
        'arms': ['random_shape_only', 'optimized'],
        'implementation': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2))
    if status != 'pass':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
