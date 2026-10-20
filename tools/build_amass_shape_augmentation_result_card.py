#!/usr/bin/env python3
"""Build a fail-closed result card for the matched AMASS three-arm study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_interact2ar_joint_augmentation_paired import (
    COMPARISON_SCHEMAS,
    METRICS,
    WRAPPERS,
    replication_set_artifact,
)
from utils.interx_shape_augmentation import (
    validate_materialized_interx_shape_augmentation,
)


LABELS = {
    'baseline': 'hhc_no_mgt_shape_v4_pilot50k_bestval_common50',
    'random': 'hhc_no_mgt_shape_amass_randomaug_pilot50k_bestval_common50',
    'optimized': 'hhc_no_mgt_shape_amass_optimizedaug_pilot50k_bestval_common50',
}
ARM_LABELS = {'random': 'randomaug', 'optimized': 'optimizedaug'}
ALLOWED_TRAINING_DIFFERENCES = {
    'exp_name', 'cache_root', 'codebook_dir', 'contact_map_path',
    'train_asset_root', 'augmentation_manifest', 'augmentation_mode',
    'train_split_file', 'all_split_file', 'train_motion_file',
    'all_motion_file', 'stats_dir', 'out_dir', 'wandb_run_id',
    'resume_in_place', 'resume_training_ckpt',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {
        'path': str(path), 'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def require_artifact(record: dict, path: Path, label: str) -> None:
    path = path.resolve(strict=True)
    require(Path(record.get('path', '')).resolve() == path, f'{label}: path')
    require(record.get('bytes') == path.stat().st_size, f'{label}: size')
    require(record.get('sha256') == sha256_file(path), f'{label}: hash')


def compact_summary(path: Path, label: str) -> dict:
    summary = load(path)
    require(
        summary.get('schema') == 'interact2ar_joint_evaluator_suite_summary_v1'
        and summary.get('status') == 'complete',
        f'{label}: invalid joint summary',
    )
    require(summary.get('replications') == 20, f'{label}: not 20 replications')
    require(
        summary.get('samples_per_replication') == 1696,
        f'{label}: wrong sample count',
    )
    require(set(summary.get('methods', {})) == {label}, f'{label}: method set')
    compact = {}
    for wrapper in WRAPPERS:
        compact[wrapper] = {}
        source = summary['methods'][label][wrapper]
        require(set(source) == set(METRICS), f'{label}/{wrapper}: metrics')
        for metric in METRICS:
            row = source[metric]
            values = row.get('values')
            require(
                isinstance(values, list) and len(values) == 20
                and all(math.isfinite(float(value)) for value in values),
                f'{label}/{wrapper}/{metric}: incomplete/non-finite',
            )
            compact[wrapper][metric] = {
                'mean': float(row['mean']), 'ci95': float(row['ci95'])
            }
    return compact


def validate_acceptance(path: Path, label: str) -> dict:
    value = load(path)
    require(
        value.get('schema') == 'hhc_vs_interact2ar_joint_paper_acceptance_v2'
        and value.get('status') == 'complete'
        and value.get('method') == label,
        f'{label}: invalid Interact2AR acceptance',
    )
    return {
        key: bool(value[key]) for key in (
            'numerical_all_joint_metrics_pass',
            'paper_gt_first_frame_protocol_matched',
            'fair_paper_superiority_claim_pass',
        )
    }


def validate_repeat_collapse(path: Path, label: str, mm_h5: Path) -> dict:
    value = load(path)
    require(
        value.get('schema') == 'hhc_mm_repeat_collapse_diagnostic_v1'
        and value.get('status') == 'complete'
        and set(value.get('methods', {})) == {label},
        f'{label}: invalid MM repeat-collapse diagnostic',
    )
    row = value['methods'][label]
    require(row.get('samples') == 100 and row.get('repeats') == 30,
            f'{label}: MM repeat shape')
    attributes = row.get('generation_attributes', {})
    require(
        attributes.get('sampler') == 'sample_intra_cas'
        and attributes.get('sampling_steps') == 50
        and attributes.get('temperature') == 0.5
        and attributes.get('cfg_scale') == 3.5,
        f'{label}: MM generation contract',
    )
    require_artifact(row.get('input_artifact', {}), mm_h5, f'{label}: MM H5')
    implementation_path = ROOT / 'tools/diagnose_mm_repeat_collapse.py'
    require_artifact(
        value.get('implementation', {}), implementation_path,
        f'{label}: MM diagnostic implementation',
    )
    return {
        'exact_pair_rate': float(row['exact_pair_rate']),
        'unique_repeats_per_prompt': row['unique_repeats_per_prompt'],
        'prompt_exact_pair_rate': row['prompt_exact_pair_rate'],
        'sampled_raw_pair_rms': row['sampled_raw_pair_rms'],
    }


def verify_pair(path: Path, kind: str, baseline: Path, candidate: Path) -> dict:
    value = load(path)
    require(value.get('schema') == COMPARISON_SCHEMAS[kind], f'{kind}: schema')
    require(
        value.get('status') == 'complete'
        and value.get('comparison_kind') == kind
        and value.get('claim_scope') == 'matched_training'
        and value.get('causal_training_ablation_claim_allowed') is True,
        f'{kind}: incomplete or non-causal pair',
    )
    require(Path(value['baseline']).resolve() == baseline.resolve(), f'{kind}: baseline')
    require(Path(value['candidate']).resolve() == candidate.resolve(), f'{kind}: candidate')
    evidence = value.get('evidence', {})
    require(
        evidence.get('baseline_replications') == replication_set_artifact(baseline, 20),
        f'{kind}: stale baseline replications',
    )
    require(
        evidence.get('candidate_replications') == replication_set_artifact(candidate, 20),
        f'{kind}: stale candidate replications',
    )
    implementation = evidence.get('implementation', {})
    compare_path = ROOT / 'tools/compare_interact2ar_joint_augmentation_paired.py'
    require(
        Path(implementation.get('path', '')).resolve() == compare_path.resolve()
        and implementation.get('bytes') == compare_path.stat().st_size
        and implementation.get('sha256') == sha256_file(compare_path),
        f'{kind}: stale comparison implementation',
    )
    return {
        'primary_all_improve': bool(value['primary_all_improve']),
        'primary_all_ci_exclude_zero': bool(value['primary_all_ci_exclude_zero']),
        'metrics': value['metrics'],
    }


def validate_training_arm(arm: str, run: Path) -> tuple[dict, dict]:
    manifest_path = run / 'training_manifest.json'
    manifest = load(manifest_path)
    args = manifest['resolved_arguments']
    require(args.get('shape') is False, f'{arm}: MGT shape is enabled')
    require(args.get('contact') is False, f'{arm}: contact model changed')
    require(args.get('training_mask_contract') ==
            'historical_pipeline_paper_kinematic_ratio', f'{arm}: mask')
    require(args.get('training_loss_contract') == 'historical_all_valid', f'{arm}: loss')
    require(args.get('paper_architecture') is False, f'{arm}: architecture')
    require(args.get('diagnostic_stop_iter') == 50000, f'{arm}: budget')
    require(args.get('sampler') == 'sample_intra_cas', f'{arm}: sampler')
    require(args.get('sampling_steps') == 50, f'{arm}: steps')
    require(args.get('temperature') == 0.5 and args.get('cfg_scale') == 3.5,
            f'{arm}: temperature/CFG')
    require(args.get('wandb_mode') == 'online', f'{arm}: W&B is not online')

    arm_label = ARM_LABELS[arm]
    checkpoint_audit = load(
        ROOT / f'output/hhc_no_shape_amass_{arm_label}_pilot50k_final_checkpoint_audit.json'
    )
    wandb_audit = load(
        ROOT / f'output/hhc_no_shape_amass_{arm_label}_pilot50k_wandb_history_audit.json'
    )
    selection = load(
        ROOT / f'output/hhc_no_shape_amass_{arm_label}_pilot50k_validation_selection.json'
    )
    require(
        checkpoint_audit.get('status') == 'pass'
        and checkpoint_audit.get('global_step') == 50000,
        f'{arm}: checkpoint audit',
    )
    require(
        wandb_audit.get('status') == 'pass'
        and wandb_audit.get('remote_state') == 'finished',
        f'{arm}: W&B audit',
    )
    require(selection.get('status') == 'complete', f'{arm}: validation selection')
    selected = Path(selection['selected_checkpoint']).resolve(strict=True)
    require(
        selection.get('selected_checkpoint_sha256') == sha256_file(selected),
        f'{arm}: selected checkpoint changed',
    )
    overlay = Path(args['train_asset_root']).resolve()
    augmentation = validate_materialized_interx_shape_augmentation(
        overlay, args['augmentation_manifest'], verify_augmented_motion_content=False,
    )
    expected_mode = (
        'random_shape_only' if arm == 'random'
        else 'optimized_positive_plus_random_zero'
    )
    require(augmentation.get('augmentation_mode') == expected_mode, f'{arm}: mode')
    require(augmentation['counts']['augmented'] == 6238, f'{arm}: count')
    return args, {
        'manifest': artifact(manifest_path),
        'checkpoint_audit': artifact(ROOT / f'output/hhc_no_shape_amass_{arm_label}_pilot50k_final_checkpoint_audit.json'),
        'wandb_audit': artifact(ROOT / f'output/hhc_no_shape_amass_{arm_label}_pilot50k_wandb_history_audit.json'),
        'validation_selection': artifact(ROOT / f'output/hhc_no_shape_amass_{arm_label}_pilot50k_validation_selection.json'),
        'augmentation_manifest': artifact(Path(args['augmentation_manifest'])),
        'selected_checkpoint': artifact(selected),
        'selected_step': int(selection['selected_step']),
        'selected_validation_fid': float(selection['selected_validation_fid']),
        'wandb_run_id': wandb_audit['run_id'],
    }


def normalized_training_args(args: dict) -> dict:
    return {key: value for key, value in args.items()
            if key not in ALLOWED_TRAINING_DIFFERENCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimized-run', type=Path, required=True)
    parser.add_argument('--random-run', type=Path, required=True)
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'output/hhc_amass_shape_augmentation_result_card.json',
    )
    args = parser.parse_args()
    runs = {'random': args.random_run.resolve(), 'optimized': args.optimized_run.resolve()}
    training_args = {}
    training_evidence = {}
    for arm in ('random', 'optimized'):
        training_args[arm], training_evidence[arm] = validate_training_arm(arm, runs[arm])
    require(
        normalized_training_args(training_args['random']) ==
        normalized_training_args(training_args['optimized']),
        'random and optimized training differ beyond augmentation assets',
    )

    roots = {
        arm: ROOT / 'output/interact2ar_joint_evaluator_candidates' / label
        for arm, label in LABELS.items()
    }
    score_dirs = {
        arm: roots[arm] / 'scores' / LABELS[arm] for arm in LABELS
    }
    metrics = {
        arm: compact_summary(roots[arm] / 'summary.json', LABELS[arm])
        for arm in LABELS
    }
    acceptance = {
        arm: validate_acceptance(roots[arm] / 'paper_acceptance.json', LABELS[arm])
        for arm in LABELS
    }
    repeat_collapse = {
        arm: validate_repeat_collapse(
            roots[arm] / 'mm_repeat_collapse.json', LABELS[arm],
            ROOT / 'output/fair_eval' / LABELS[arm] / 'replication_00_mm.h5',
        ) for arm in LABELS
    }
    pairs = {
        'baseline_vs_random': verify_pair(
            roots['random'] / 'baseline_vs_amass_augmented_paired.json',
            'random_shape_augmentation', score_dirs['baseline'], score_dirs['random'],
        ),
        'baseline_vs_optimized': verify_pair(
            roots['optimized'] / 'baseline_vs_amass_augmented_paired.json',
            'optimized_shape_augmentation', score_dirs['baseline'], score_dirs['optimized'],
        ),
        'random_vs_optimized': verify_pair(
            ROOT / 'output/interact2ar_joint_evaluator_candidates/'
            'hhc_no_mgt_shape_amass_aug_matrix/random_vs_optimized_paired.json',
            'optimization_effect_with_fixed_amass_donors',
            score_dirs['random'], score_dirs['optimized'],
        ),
    }
    payload = {
        'schema': 'hhc_amass_shape_augmentation_three_arm_result_card_v1',
        'status': 'complete',
        'claim_scope': 'matched_50k_training_three_arm_ablation',
        'arms': {
            'baseline': 'no augmentation, no MGT shape condition',
            'random': '6238 HumanML3D-train AMASS random-shape-only samples',
            'optimized': 'same donors: 5868 optimized-selective positive timing plus 370 random-zero',
        },
        'evaluation': {
            'split': 'official Inter-X test', 'replications': 20,
            'samples_per_replication': 1696,
            'sampler': 'sample_intra_cas common50, temperature 0.5, CFG 3.5',
            'joint_metrics': metrics,
            'mm_repeat_collapse': repeat_collapse,
            'interact2ar_acceptance': acceptance,
        },
        'paired_comparisons': pairs,
        'optimized_comprehensively_beats_interact2ar_numerically': acceptance[
            'optimized'
        ]['numerical_all_joint_metrics_pass'],
        'training_evidence': training_evidence,
        'evaluation_evidence': {
            arm: {
                'summary': artifact(roots[arm] / 'summary.json'),
                'acceptance': artifact(roots[arm] / 'paper_acceptance.json'),
                'mm_repeat_collapse': artifact(
                    roots[arm] / 'mm_repeat_collapse.json'
                ),
            } for arm in LABELS
        },
        'implementation': artifact(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps({
        'status': 'complete', 'output': str(args.output),
        'optimized_comprehensively_beats_interact2ar_numerically': payload[
            'optimized_comprehensively_beats_interact2ar_numerically'
        ],
    }, sort_keys=True))


if __name__ == '__main__':
    main()
