#!/usr/bin/env python
"""Audit the corrected fresh CoShMDM Inter-X training and any safe resumes."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COSH_ROOT = Path('/mnt/8T-HDD/wenning/coshmdm-code')
EXPERIMENT = 'CoShMDM-InterX-fair-statsfix-seed123-bs32x2'
EXPECTED_RUNTIME = {
    'dataset': 'interx',
    'stats_prefix': 'interx_global',
    'normalization_mean_sha256': (
        'aa281b878b8a71c6d10c001059b29374921c18545c082102bf69a404982f568d'
    ),
    'normalization_std_sha256': (
        '30a83e92965a465301a3642eb55b2ba705d3dceaba05e45846aa433d3900b104'
    ),
}
EXPECTED_INITIAL_ARGUMENTS = [
    '--dataset', 'interx', '--gpu', '0', '--seed', '123', '--batch-size', '32',
    '--accumulate-grad-batches', '2', '--no-progress-bar', '--stats-prefix',
    'interx_global', '--exp-name', EXPERIMENT,
]
EXPECTED_MILESTONE_SHA256 = (
    '140a2496e941615c3377cb186f187c2c283455e2fb033632ba5694d5387a338d'
)
EXPECTED_MILESTONE = (
    COSH_ROOT / 'checkpoints' / EXPERIMENT / 'model' /
    'milestone-000020000.ckpt'
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def parse_log(path):
    text = path.read_text(encoding='utf-8', errors='replace')
    steps = [int(value) for value in re.findall(r'niter:\s*(\d+)', text)]
    expected_line = 'CoShMDM runtime contract: ' + json.dumps(
        EXPECTED_RUNTIME, sort_keys=True
    )
    require(expected_line in text, f'correct runtime contract missing: {path}')
    require('Seed set to 123' in text, f'seed proof missing: {path}')
    require(str(COSH_ROOT) in text, f'working-directory proof missing: {path}')
    restore_starts = text.count('Restoring states from the checkpoint path at ')
    restore_completions = text.count('Restored all states from the checkpoint')
    resumed = restore_starts > 0
    if resumed:
        require(
            restore_completions == restore_starts,
            f'full Lightning restore proof missing for an attempt: {path}',
        )
    return {
        'artifact': artifact(path),
        'minimum_logged_update': min(steps) if steps else None,
        'maximum_logged_update': max(steps) if steps else None,
        'logged_update_rows': len(steps),
        'resumed_full_lightning_state': resumed,
        'full_lightning_resume_count': restore_completions,
        'segmentation_fault': 'Segmentation fault' in text,
        'trainer_finished': (
            'Trainer.fit stopped' in text or 'max_epochs=2000' in text
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--training-manifest', type=Path,
        default=COSH_ROOT / 'checkpoints' / EXPERIMENT / 'meta' /
        'training_manifest.json',
    )
    parser.add_argument(
        '--runtime-manifest', type=Path,
        default=COSH_ROOT / 'checkpoints' / EXPERIMENT / 'meta' /
        'runtime_contract.json',
    )
    parser.add_argument(
        '--log', type=Path, action='append',
        default=[ROOT / 'output/queue_logs/gpu0_cosh_statsfix_20260830.log'],
    )
    parser.add_argument(
        '--excluded-run-diagnostic', type=Path,
        default=ROOT / 'output/coshmdm_mixed_stats_aborted_diagnostic.json',
    )
    parser.add_argument(
        '--final-checkpoint-audit', type=Path,
        default=ROOT / 'output/coshmdm_final_checkpoint_audit.json',
    )
    parser.add_argument(
        '--budget-cap-manifest', type=Path,
        default=ROOT / 'output/coshmdm_training_budget_cap.json',
    )
    parser.add_argument(
        '--validation-selection-manifest', type=Path,
        default=ROOT / 'output/coshmdm_interx_val_selection.json',
    )
    parser.add_argument(
        '--milestone-20k-audit', type=Path,
        default=ROOT / 'output/coshmdm_statsfix_20k_preserved_checkpoint_audit.json',
    )
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'output/coshmdm_resume_provenance.json',
    )
    args = parser.parse_args()

    training_manifest_path = args.training_manifest.resolve()
    runtime_manifest_path = args.runtime_manifest.resolve()
    excluded_path = args.excluded_run_diagnostic.resolve()
    milestone_audit_path = args.milestone_20k_audit.resolve()
    for label, path in (
        ('training manifest', training_manifest_path),
        ('runtime manifest', runtime_manifest_path),
        ('excluded-run diagnostic', excluded_path),
        ('20k milestone audit', milestone_audit_path),
    ):
        require(path.is_file(), f'missing {label}: {path}')

    training = json.loads(training_manifest_path.read_text(encoding='utf-8'))
    runtime = json.loads(runtime_manifest_path.read_text(encoding='utf-8'))
    excluded = json.loads(excluded_path.read_text(encoding='utf-8'))
    milestone_audit = json.loads(
        milestone_audit_path.read_text(encoding='utf-8')
    )
    require(
        training.get('schema') == 'coshmdm_interx_fair_training_manifest_v2',
        'corrected training manifest schema changed',
    )
    command = training.get('command', [])
    require(command[-len(EXPECTED_INITIAL_ARGUMENTS):] == EXPECTED_INITIAL_ARGUMENTS,
            'corrected initial command changed')
    require(training.get('runtime_contract', {}).get('stats_prefix')
            == 'interx_global', 'training manifest stats prefix changed')
    require(
        runtime.get('schema') == 'coshmdm_training_runtime_contract_v1'
        and runtime.get('experiment') == EXPERIMENT
        and runtime.get('runtime_contract') == EXPECTED_RUNTIME,
        'live runtime manifest does not bind corrected Inter-X statistics',
    )
    require(
        runtime.get('sources', {}).get('tools/train.py')
        == sha256_file(COSH_ROOT / 'tools/train.py'),
        'runtime manifest training source changed after launch',
    )
    require(
        excluded.get('status') == 'excluded_non_reusable'
        and excluded.get('normalization_resolution', {}).get(
            'mean_byte_equal') is False
        and excluded.get('normalization_resolution', {}).get(
            'std_byte_equal') is False,
        'mixed-stat predecessor is not explicitly excluded',
    )
    milestone_path = Path(
        milestone_audit.get('checkpoint', '')
    ).resolve()
    require(
        milestone_audit.get('status') == 'pass'
        and milestone_audit.get('global_step') == 20000
        and milestone_audit.get('checkpoint_sha256')
        == EXPECTED_MILESTONE_SHA256
        and milestone_audit.get('coshmdm_runtime_contract') == EXPECTED_RUNTIME
        and milestone_path == EXPECTED_MILESTONE.resolve()
        and milestone_path.is_file()
        and milestone_path.stat().st_size
        == milestone_audit.get('checkpoint_bytes'),
        'preserved stats-fixed 20k milestone contract changed',
    )

    attempts = [parse_log(path.resolve()) for path in args.log]
    require(attempts, 'no corrected CoShMDM training logs supplied')
    require(not any(item['segmentation_fault'] for item in attempts),
            'corrected CoShMDM attempt contains a segmentation fault')
    logged_steps = [
        item['maximum_logged_update'] for item in attempts
        if item['maximum_logged_update'] is not None
    ]
    maximum_step = max(logged_steps, default=0)

    final = None
    complete = False
    capped_complete = False
    budget_cap = None
    validation_selection = None
    final_path = args.final_checkpoint_audit.resolve()
    if final_path.is_file():
        final = json.loads(final_path.read_text(encoding='utf-8'))
        complete = (
            final.get('status') == 'pass'
            and final.get('global_step') == 568000
            and final.get('coshmdm_runtime_contract') == EXPECTED_RUNTIME
            and maximum_step == 568000
            and any(item['trainer_finished'] for item in attempts)
        )
    budget_cap_path = args.budget_cap_manifest.resolve()
    selection_path = args.validation_selection_manifest.resolve()
    if budget_cap_path.is_file():
        budget_cap = json.loads(budget_cap_path.read_text(encoding='utf-8'))
    if selection_path.is_file():
        validation_selection = json.loads(
            selection_path.read_text(encoding='utf-8')
        )
    if final is not None and budget_cap is not None and validation_selection is not None:
        selected_path = Path(
            validation_selection.get('selected_checkpoint', '')
        ).resolve()
        audited_path = Path(final.get('checkpoint', '')).resolve()
        capped_complete = (
            budget_cap.get('schema') == 'hhc_coshmdm_user_training_cap_v1'
            and budget_cap.get('status') in {
                'stopped_gracefully', 'sigterm_sent_after_180s_grace',
                'trainer_exited_before_monitor_signal',
            }
            and budget_cap.get('preferred_terminal_optimizer_step') == 200000
            and validation_selection.get('schema')
            == 'hhc_coshmdm_capped_validation_selection_v1'
            and validation_selection.get('status') == 'complete'
            and validation_selection.get('test_metrics_used_for_selection') is False
            and final.get('status') == 'pass'
            and final.get('coshmdm_runtime_contract') == EXPECTED_RUNTIME
            and int(final.get('global_step', -1))
            == int(validation_selection.get('selected_step', -2))
            and selected_path == audited_path
            and selected_path.is_file()
            and sha256_file(selected_path)
            == validation_selection.get('selected_checkpoint_sha256')
            == final.get('checkpoint_sha256')
            and maximum_step >= int(final.get('global_step', -1))
        )
    if args.require_complete:
        require(
            sha256_file(milestone_path) == EXPECTED_MILESTONE_SHA256,
            'preserved stats-fixed 20k milestone bytes changed',
        )
        require(complete, 'corrected CoShMDM training is not complete at 568k')

    payload = {
        'schema': 'hhc_coshmdm_training_provenance_v2',
        'status': (
            'pass' if complete
            else ('pass_user_capped' if capped_complete else 'active')
        ),
        'training_classification': (
            ('fresh Inter-X retraining with checkpoint-bound interx_global '
             'normalization; user-capped to two additional days; validation-'
             'selected milestone; mixed-stat predecessor excluded')
            if capped_complete else
            ('fresh Inter-X retraining with checkpoint-bound interx_global '
             'normalization; mixed-stat predecessor excluded')
        ),
        'initial_global_step': 0,
        'expected_final_global_step': 568000,
        'original_author_budget_completed': complete,
        'user_capped_run_complete': capped_complete,
        'maximum_logged_update': maximum_step,
        'runtime_contract': EXPECTED_RUNTIME,
        'training_manifest': artifact(training_manifest_path),
        'runtime_manifest': artifact(runtime_manifest_path),
        'excluded_mixed_stats_run': artifact(excluded_path),
        'milestone_20k_checkpoint_audit': artifact(milestone_audit_path),
        'milestone_20k_checkpoint': {
            'path': str(milestone_path),
            'bytes': milestone_path.stat().st_size,
            'sha256': EXPECTED_MILESTONE_SHA256,
        },
        'queue_source': artifact(ROOT / 'tools/queue_gpu0_cosh_timotion.sh'),
        'training_attempts': attempts,
        'resume_policy': (
            'A resume is accepted only when Lightning restores complete state '
            'and the checkpoint-embedded runtime contract exactly equals the '
            'current Inter-X statistics hashes.'
        ),
        'reproducibility_limitation': (
            'Two-worker caption/crop/person-order RNG streams are not '
            'serialized, so any OS-level resume is distributionally but not '
            'bitwise equivalent to an uninterrupted run.'
        ),
        'final_checkpoint_audit': (
            artifact(final_path) if final_path.is_file() else None
        ),
        'budget_cap_manifest': (
            artifact(budget_cap_path) if budget_cap_path.is_file() else None
        ),
        'validation_selection_manifest': (
            artifact(selection_path) if selection_path.is_file() else None
        ),
        'selected_global_step': (
            int(validation_selection['selected_step'])
            if validation_selection is not None
            and validation_selection.get('status') == 'complete'
            else None
        ),
        'implementation': artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
