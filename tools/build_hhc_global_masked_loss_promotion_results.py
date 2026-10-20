#!/usr/bin/env python3
"""Freeze the promoted 300k global-masked-loss control and common-50 results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'no_contact': 'hhc_global_masked_loss_control_300k_common50_no_contact',
    'contact': 'hhc_global_masked_loss_control_300k_common50_contact',
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        'path': str(path), 'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def load(path: Path) -> tuple[dict, dict]:
    record = artifact(path)
    return json.loads(Path(record['path']).read_text(encoding='utf-8')), record


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def same_path(left, right) -> bool:
    return bool(left and right) and Path(left).resolve() == Path(right).resolve()


def validate_bound_record(record: dict, label: str) -> None:
    path = Path(record.get('path', ''))
    require(path.is_file(), f'{label}: bound file is missing: {path}')
    require(record.get('sha256') == sha256_file(path),
            f'{label}: bound file hash is stale')


def finite_metric_summary(summary: dict, label: str) -> dict:
    require(summary.get('replications') == 20, f'{label}: expected 20 replications')
    require(summary.get('samples_per_replication') == 1696,
            f'{label}: expected 1696 samples per replication')
    metrics = summary.get('metrics', {})
    expected = {'fid', 'top1', 'top2', 'top3', 'matching', 'diversity',
                'multimodality'}
    require(expected <= set(metrics), f'{label}: semantic metric set is incomplete')
    for name in expected:
        record = metrics[name]
        require(len(record.get('values', [])) == 20,
                f'{label}: {name} does not have 20 values')
        require(all(math.isfinite(float(value)) for value in record['values']),
                f'{label}: {name} contains non-finite values')
        require(math.isfinite(float(record.get('mean', math.nan)))
                and math.isfinite(float(record.get('ci95', math.nan))),
                f'{label}: {name} aggregate is non-finite')
    provenance = summary.get('provenance', {})
    require(provenance.get('legacy_provenance_allowed') is False
            and provenance.get('generation_files_sha256_verified') is True
            and provenance.get('unique_requests_verified') is True,
            f'{label}: generation provenance is incomplete')
    return {
        name: {'mean': metrics[name]['mean'], 'ci95': metrics[name]['ci95']}
        for name in sorted(expected)
    }


def validate_method(directory: Path, label: str, guidance: str,
                    selection: dict, selection_record: dict,
                    training_record: dict) -> dict:
    run, run_record = load(directory / 'run_manifest.json')
    summary, summary_record = load(directory / 'summary.json')
    mm, mm_record = load(directory / 'multimodality_audit_summary.json')
    physical, physical_record = load(directory / 'replication_00_physical.json')
    timing, timing_record = load(directory / 'generation_timing_summary.json')
    require(
        run.get('method') == 'hhc'
        and run.get('label') == label
        and run.get('replication_range') == [0, 20]
        and run.get('samples_per_replication') == 1696
        and run.get('sampling_steps') == 50
        and run.get('temperature') == 0.5
        and run.get('cfg_scale') == 3.5
        and run.get('contact_guidance_scale') == 1.0
        and run.get('hhc_vq_contract') == 'precleanup_raw12288'
        and run.get('hhc_mgt_contract') == 'current'
        and run.get('hhc_stats_prefix') == 'Interx'
        and same_path(run.get('hhc_stats_dir'), ROOT / 'dataset/interx')
        and same_path(run.get('selection_manifest'), selection_record['path']),
        f'{label}: run contract differs from the promoted common-50 protocol',
    )
    expected_contact = guidance == 'contact'
    require(
        run.get('contact_guidance') is expected_contact
        and run.get('contact_timing_source')
        == ('request_oracle' if expected_contact else 'none')
        and run.get('contact_timing_source_requested')
        == ('request_oracle' if expected_contact else 'none'),
        f'{label}: timing/guidance contract changed',
    )
    provenance = run.get('provenance', {}).get('artifacts', {})
    for name in ('transformer', 'mgt_selection_manifest',
                 'mgt_training_manifest', 'vq'):
        validate_bound_record(provenance.get(name, {}), f'{label}/{name}')
    require(
        provenance['transformer']['sha256']
        == selection.get('selected_checkpoint_sha256')
        and same_path(provenance['transformer']['path'],
                      selection.get('selected_checkpoint'))
        and provenance['mgt_selection_manifest']['sha256']
        == selection_record['sha256']
        and provenance['mgt_training_manifest']['sha256']
        == training_record['sha256']
        and provenance['vq']['sha256']
        == 'fdf5afaa5d3513f933254fafc831f6046f7c38b5e649db08fb6be7e0ce1006bb',
        f'{label}: generation checkpoint/VQ/selection binding changed',
    )
    sampler = summary.get('provenance', {}).get('sampler_identity', {})
    require(
        sampler.get('sampler') == 'sample_intra_cas'
        and sampler.get('cascade_order') == 'root,body,hands'
        and sampler.get('sampling_steps') == 50
        and sampler.get('sampling_steps_unit') == 'outer_maskgit_iterations'
        and sampler.get('temperature') == 0.5
        and sampler.get('cfg_scale') == 3.5
        and sampler.get('contact_guidance') is expected_contact
        and sampler.get('contact_guidance_scale') == 1.0
        and sampler.get('checkpoint_sha256')
        == selection.get('selected_checkpoint_sha256'),
        f'{label}: aggregate sampler identity changed',
    )
    require(mm.get('replications') == 20
            and mm.get('legacy_provenance_allowed') is False,
            f'{label}: exhaustive multimodality summary is incomplete')
    require(physical.get('samples') == 1696 and physical.get('repeat') == 0,
            f'{label}: physical replication-0 result is incomplete')
    require(timing.get('replications') == 20 and timing.get('label') == label,
            f'{label}: timing summary is incomplete')
    return {
        'label': label,
        'metrics': finite_metric_summary(summary, label),
        'artifacts': {
            'run_manifest': run_record, 'summary': summary_record,
            'multimodality': mm_record, 'physical_replication_0': physical_record,
            'generation_timing': timing_record,
        },
    }


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatch', type=Path,
                        default=Path('output/hhc_post_50k_training_dispatch.json'))
    parser.add_argument('--selection', type=Path, default=Path(
        'output/hhc_global_masked_loss_control_300k_selection.json'))
    parser.add_argument('--wandb-audit', type=Path, default=Path(
        'output/hhc_global_masked_loss_control_300k_wandb_history_audit.json'))
    parser.add_argument('--fair-root', type=Path, default=Path('output/fair_eval'))
    parser.add_argument('--output', type=Path, default=Path(
        'output/hhc_global_masked_loss_control_300k_results.json'))
    args = parser.parse_args()

    dispatch, dispatch_record = load(args.dispatch)
    require(dispatch.get('status') == 'ready'
            and dispatch.get('action')
            == 'continue_global_control_to_300k_then_shape'
            and dispatch.get('continue_global_control') is True,
            '300k promotion results require the positive 50k dispatch')
    selection, selection_record = load(args.selection)
    require(selection.get('run_complete') is True
            and selection.get('expected_final_step') == 300000
            and selection.get('test_metrics_used_for_selection') is False
            and selection.get('final_state_audit', {}).get('status') == 'pass',
            '300k selection is incomplete or test-selected')
    for key in ('selected_checkpoint', 'final_latest_checkpoint'):
        bound = Path(selection[key])
        require(bound.is_file(), f'selection is missing {key}')
        require(selection[f'{key}_sha256'] == sha256_file(bound),
                f'selection {key} hash changed')
    training_path = Path(selection['training_manifest'])
    training_record = artifact(training_path)
    require(training_record['sha256'] == selection['training_manifest_sha256'],
            'selection training-manifest hash changed')
    training = json.loads(training_path.read_text(encoding='utf-8'))
    resolved = training.get('resolved_arguments', {})
    observability = training.get('observability', {})
    require(resolved.get('total_iter') == 300000
            and resolved.get('training_loss_contract')
            == 'paper_masked_global_token_mean'
            and observability.get('wandb_mode') == 'online'
            and observability.get('metric_profile')
            == 'masked_token_classification',
            'promoted training manifest has the wrong loss/budget/W&B contract')

    wandb, wandb_record = load(args.wandb_audit)
    required_norms = {
        'train/gradient_l2_norm', 'train/gradient_max_abs',
        'train/weight_l2_norm', 'train/weight_max_abs',
        'train/nonfinite_gradients', 'train/nonfinite_weights',
    }
    require(wandb.get('status') == 'pass'
            and wandb.get('remote_state') == 'finished'
            and wandb.get('expected_final_step') == 300000
            and wandb.get('run_id') == observability.get('wandb_run_id')
            and all(wandb.get('checks', {}).values())
            and required_norms <= set(wandb.get('metrics', {}))
            and wandb['metrics']['train/nonfinite_gradients']['maximum'] == 0
            and wandb['metrics']['train/nonfinite_weights']['maximum'] == 0,
            'promoted online W&B history is incomplete or non-finite')
    require(wandb.get('manifest', {}).get('sha256') == training_record['sha256'],
            'W&B history audit is not bound to the selected training manifest')

    methods = {
        guidance: validate_method(
            (ROOT / args.fair_root / label).resolve(), label, guidance,
            selection, selection_record, training_record,
        )
        for guidance, label in LABELS.items()
    }
    payload = {
        'schema': 'hhc_global_masked_loss_control_300k_results_v1',
        'status': 'complete',
        'training_loss_contract': 'paper_masked_global_token_mean',
        'training_budget': 300000,
        'dispatch': dispatch_record,
        'selection': selection_record,
        'training_manifest': training_record,
        'wandb_history_audit': wandb_record,
        'methods': methods,
        'implementation': artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
