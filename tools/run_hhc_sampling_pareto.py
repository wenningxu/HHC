#!/usr/bin/env python
"""Run a predeclared one-factor HHC quality/diversity sampling grid."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID = (
    {'temperature': 0.5, 'cfg_scale': 3.5, 'sampling_steps': 50},
    {'temperature': 0.3, 'cfg_scale': 3.5, 'sampling_steps': 50},
    {'temperature': 0.8, 'cfg_scale': 3.5, 'sampling_steps': 50},
    {'temperature': 0.5, 'cfg_scale': 2.5, 'sampling_steps': 50},
    {'temperature': 0.5, 'cfg_scale': 4.5, 'sampling_steps': 50},
    {'temperature': 0.5, 'cfg_scale': 3.5, 'sampling_steps': 10},
    {'temperature': 0.5, 'cfg_scale': 3.5, 'sampling_steps': 20},
    {
        'temperature': 1.0, 'cfg_scale': 3.5, 'sampling_steps': 3,
        'audit_scope': (
            'mmm_pre_cleanup_git_snapshot_function_defaults_on_'
            'paper_sampler_semantics_not_historical_runtime'
        ),
    },
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
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


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def label(config):
    return (
        f"temp{config['temperature']:.1f}_cfg{config['cfg_scale']:.1f}_"
        f"steps{config['sampling_steps']}"
    ).replace('.', 'p')


def run(command, dry_run=False):
    print(json.dumps({'cwd': str(ROOT), 'command': command}), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def validate_headline_generation(path, requests, checkpoint_hash, seed, repeats):
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = {
        'requests_sha256': sha256_file(requests),
        'checkpoint_sha256': checkpoint_hash,
        'vq_contract': 'precleanup_raw12288',
        'mgt_contract': 'historical_mmm',
        'inference_conditioning_contract': 'paper_text_cfg',
        'sampler': 'sample_intra_cas',
        'sampling_steps': 50,
        'temperature': 0.5,
        'cfg_scale': 3.5,
        'contact_guidance': False,
        'contact_timing_source': 'none',
        'seed': seed,
    }
    with h5py.File(path, 'r') as source:
        mismatches = {
            key: {'expected': value, 'actual': source.attrs.get(key)}
            for key, value in expected.items()
            if source.attrs.get(key) != value
        }
        completed = source.get('completed_repeats')
        if (
            mismatches
            or completed is None
            or completed.shape != (repeats,)
            or not completed[:].astype(bool).all()
        ):
            raise ValueError(
                f'headline generation is incomplete or belongs to another '
                f'contract: {path}: {mismatches}'
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--selection-manifest',
        default='output/hhc_historical_plus_paper_mask_ratio_selection.json',
    )
    parser.add_argument(
        '--vq-checkpoint',
        default='checkpoints/vq_precleanup_raw12288/net_175000.pth',
    )
    parser.add_argument(
        '--requests',
        default='output/fair_interx_requests/replication_00.npz',
    )
    parser.add_argument(
        '--output-root', default=(
            'output/fair_eval/'
            'hhc_historical_paper_mask_sampling_pareto_no_timing_rep0'
        )
    )
    parser.add_argument(
        '--headline-root', type=Path,
        default=Path(
            'output/fair_eval/'
            'hhc_historical_plus_paper_mask_ratio_no_contact_common50'
        ),
        help='Completed current 50-step no-timing row reused for the grid centre.',
    )
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    selection_path = Path(args.selection_manifest).resolve()
    with open(selection_path, encoding='utf-8') as handle:
        selection = json.load(handle)
    if (
        selection.get('schema') != 'hhc_official_interx_validation_selection_v1'
        or selection.get('status') != 'complete'
        or selection.get('test_metrics_used_for_selection') is not False
    ):
        raise ValueError('current official validation selection is incomplete or invalid')
    checkpoint = Path(selection['selected_checkpoint']).resolve()
    expected_hash = selection.get('selected_checkpoint_sha256')
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected_hash:
        raise ValueError('selected MGT checkpoint is missing or has changed')
    vq_checkpoint = Path(args.vq_checkpoint).resolve()
    requests = Path(args.requests).resolve()
    request = np.load(requests, allow_pickle=False)
    seed = int(request['seed'])
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / 'grid_manifest.json'
    manifest = {
        'schema': 'hhc_current_sampling_pareto_grid_v1',
        'status': 'running',
        'design': (
            'current_historical_architecture_plus_paper_mask_ratio_checkpoint;_'
            'no_contact_timing;_one_factor_at_a_time_around_requested_common_'
            '50_step_sampler_plus_mmm_pre_cleanup_parameter_audit'
        ),
        'sampler_semantics': (
            'paper_high_confidence_tokens_are_fixed; the pre-cleanup Git '
            'snapshot three-step row transfers only function-default '
            'step/temperature parameters and is neither a historical-runtime '
            'nor source-code-behavior equivalence claim'
        ),
        'replication': 0,
        'main_seed': seed,
        'multimodality_seed': seed + 100_000,
        'multimodality_prompts': 100,
        'generations_per_prompt': 30,
        'official_pairs_per_prompt': 10,
        'all_pairs_per_prompt': 435,
        'grid': list(DEFAULT_GRID),
        'vq_contract': 'precleanup_raw12288',
        'mgt_contract': 'historical_mmm',
        'training_conditioning_contract': 'historical_mmm_shape_dropout',
        'inference_conditioning_contract': 'paper_text_cfg',
        'contact_timing_source': 'none',
        'contact_guidance': False,
        'artifacts': {
            'selection_manifest': artifact(selection_path),
            'checkpoint': artifact(checkpoint),
            'vq_checkpoint': artifact(vq_checkpoint),
            'requests': artifact(requests),
            'exporter': artifact(ROOT / 'tools/export_hhc_official_interx.py'),
            'semantic_scorer': artifact(
                ROOT / 'tools/score_fair_interx_generations.py'
            ),
            'multimodality_analyzer': artifact(
                ROOT / 'tools/analyze_fair_interx_multimodality.py'
            ),
            'physical_scorer': artifact(
                ROOT / 'tools/score_fair_interx_physical.py'
            ),
        },
        'outputs': {},
    }
    if not args.dry_run:
        atomic_json(manifest_path, manifest)

    gpu = args.device.split(':', 1)[-1]
    for config in DEFAULT_GRID:
        run_label = label(config)
        run_dir = output_root / run_label
        run_dir.mkdir(parents=True, exist_ok=True)
        is_headline = config == DEFAULT_GRID[0]
        generation_root = args.headline_root.resolve() if is_headline else run_dir
        main_h5 = generation_root / 'replication_00_main.h5'
        mm_h5 = generation_root / 'replication_00_mm.h5'
        metrics = run_dir / 'replication_00.json'
        mm_audit = run_dir / 'replication_00_multimodality_audit.json'
        physical = run_dir / 'replication_00_physical.json'
        sampler_args = [
            '--sampling-steps', str(config['sampling_steps']),
            '--temperature', str(config['temperature']),
            '--cfg-scale', str(config['cfg_scale']),
        ]
        common = [
            sys.executable, 'tools/export_hhc_official_interx.py',
            '--checkpoint', str(checkpoint),
            '--vq-checkpoint', str(vq_checkpoint),
            '--vq-contract', 'precleanup_raw12288',
            '--mgt-contract', 'historical_mmm',
            '--inference-conditioning-contract', 'paper_text_cfg',
            '--requests', str(requests),
            '--device', args.device,
            '--batch-size', str(args.batch_size),
            '--no-contact-guidance',
            '--contact-timing-source', 'none',
            '--resume',
        ] + sampler_args
        if is_headline:
            if not args.dry_run:
                validate_headline_generation(
                    main_h5, requests, expected_hash, seed, 1,
                )
                validate_headline_generation(
                    mm_h5, requests, expected_hash, seed + 100_000, 30,
                )
        else:
            run(common + [
                '--output', str(main_h5), '--repeats', '1', '--seed', str(seed),
            ], args.dry_run)
            run(common + [
                '--output', str(mm_h5), '--indices-key', 'mm_indices',
                '--repeats', '30', '--seed', str(seed + 100_000),
            ], args.dry_run)
        run([
            sys.executable, 'tools/score_fair_interx_generations.py',
            '--requests', str(requests),
            '--main-generations', str(main_h5),
            '--mm-generations', str(mm_h5),
            '--output', str(metrics), '--device', args.device,
            '--batch-size', '32',
        ], args.dry_run)
        run([
            sys.executable, 'tools/analyze_fair_interx_multimodality.py',
            '--requests', str(requests), '--mm-generations', str(mm_h5),
            '--output', str(mm_audit), '--device', args.device,
            '--batch-size', '32',
        ], args.dry_run)
        run([
            sys.executable, 'tools/score_fair_interx_physical.py',
            '--requests', str(requests), '--generations', str(main_h5),
            '--output', str(physical), '--device', args.device,
        ], args.dry_run)
        manifest['outputs'][run_label] = {
            'config': config,
            'reused_current_headline_generation': is_headline,
            'main': str(main_h5),
            'multimodality': str(mm_h5),
            'metrics': str(metrics),
            'all_pairs_audit': str(mm_audit),
            'physical': str(physical),
        }
        if not args.dry_run:
            atomic_json(manifest_path, manifest)

    manifest['status'] = 'complete'
    if not args.dry_run:
        atomic_json(manifest_path, manifest)
    print(json.dumps({'manifest': str(manifest_path), 'status': manifest['status']}))


if __name__ == '__main__':
    main()
