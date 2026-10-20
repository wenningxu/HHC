#!/usr/bin/env python
"""Run the preregistered replication-0 HHC contact-timing ablation."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    {
        'label': 'text_predicted',
        'contact_timing_source': 'external_text_predictor',
        'contact_guidance_scale': 1.0,
    },
    {
        'label': 'full_valid_weak',
        'contact_timing_source': 'full_valid_weak',
        # Predeclared weak bias: one quarter of the paper oracle-timing bias.
        'contact_guidance_scale': 0.25,
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
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': sha256_file(path)}


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def run(command, dry_run):
    print(json.dumps({'cwd': str(ROOT), 'command': command}), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-manifest', type=Path,
                        default=Path(
                            'output/hhc_historical_plus_paper_mask_ratio_selection.json'
                        ))
    parser.add_argument('--vq-checkpoint', type=Path,
                        default=Path(
                            'checkpoints/vq_precleanup_raw12288/net_175000.pth'
                        ))
    parser.add_argument('--requests', type=Path,
                        default=Path('output/fair_interx_requests/replication_00.npz'))
    parser.add_argument('--contact-map', type=Path,
                        default=Path(
                            'checkpoints/vq_precleanup_raw12288_historical_mmm_cache/'
                            'code_contact_probs.npy'
                        ))
    parser.add_argument('--text-timing', type=Path,
                        default=Path('output/contact_timing/text_knn_replication_00.npz'))
    parser.add_argument('--baseline-root', type=Path,
                        default=Path('output/fair_eval'))
    parser.add_argument('--output-root', type=Path,
                        default=Path(
                            'output/fair_eval/'
                            'hhc_historical_paper_mask_contact_timing_ablation_rep0'
                        ))
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    selection_path = args.selection_manifest.resolve()
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    checkpoint = Path(selection['selected_checkpoint']).resolve()
    if sha256_file(checkpoint) != selection['selected_checkpoint_sha256']:
        raise ValueError('selected HHC checkpoint hash differs from selection')
    vq_checkpoint = args.vq_checkpoint.resolve()
    contact_map = args.contact_map.resolve()
    requests = args.requests.resolve()
    text_timing = args.text_timing.resolve()
    with np.load(requests, allow_pickle=False) as request:
        seed = int(request['seed'])
        request_hash = sha256_file(requests)
    with np.load(text_timing, allow_pickle=False) as timing:
        if str(timing['requests_sha256'].item()) != request_hash:
            raise ValueError('text timing is bound to another frozen request')

    baseline_root = args.baseline_root.resolve()
    baselines = {
        'request_oracle': baseline_root / (
            'hhc_historical_plus_paper_mask_ratio_contact_common50'
        ),
        'none': baseline_root / (
            'hhc_historical_plus_paper_mask_ratio_no_contact_common50'
        ),
    }
    required_baselines = [
        directory / filename
        for directory in baselines.values()
        for filename in (
            'replication_00.json',
            'replication_00_multimodality_audit.json',
            'replication_00_physical.json',
        )
    ]
    missing = [str(path) for path in required_baselines if not path.is_file()]
    if missing and not args.dry_run:
        raise FileNotFoundError(
            'strict HHC baseline outputs must finish first: ' + ', '.join(missing)
        )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / 'ablation_manifest.json'
    manifest = {
        'schema': 'hhc_contact_timing_ablation_rep0_v1',
        'status': 'running',
        'design': (
            'same checkpoint/request/seeds/sampler; only inference contact '
            'timing source and declared guidance strength differ'
        ),
        'replication': 0,
        'main_seed': seed,
        'multimodality_seed': seed + 100_000,
        'fixed_sampler': {
            'temperature': 0.5, 'cfg_scale': 3.5, 'sampling_steps': 50,
        },
        'conditions': {
            'request_oracle': {
                'timing': 'frozen test GT contact (oracle)',
                'contact_guidance_scale': 1.0,
                'existing_output': str(baselines['request_oracle']),
            },
            'none': {
                'timing': 'no contact logit bias',
                'contact_guidance_scale': 0.0,
                'existing_output': str(baselines['none']),
            },
            **{config['label']: config for config in CONFIGS},
        },
        'artifacts': {
            'selection_manifest': artifact(selection_path),
            'checkpoint': artifact(checkpoint),
            'vq_checkpoint': artifact(vq_checkpoint),
            'contact_map': artifact(contact_map),
            'requests': artifact(requests),
            'text_timing': artifact(text_timing),
            'text_timing_manifest': artifact(
                text_timing.with_suffix(text_timing.suffix + '.manifest.json')
            ),
            'exporter': artifact(ROOT / 'tools/export_hhc_official_interx.py'),
            'runner': artifact(Path(__file__)),
        },
        'outputs': {},
    }
    if not args.dry_run:
        atomic_json(manifest_path, manifest)

    for config in CONFIGS:
        run_dir = output_root / config['label']
        run_dir.mkdir(parents=True, exist_ok=True)
        main_h5 = run_dir / 'replication_00_main.h5'
        mm_h5 = run_dir / 'replication_00_mm.h5'
        metrics = run_dir / 'replication_00.json'
        mm_audit = run_dir / 'replication_00_multimodality_audit.json'
        physical = run_dir / 'replication_00_physical.json'
        timing_args = [
            '--contact-guidance',
            '--contact-timing-source', config['contact_timing_source'],
            '--contact-guidance-scale', str(config['contact_guidance_scale']),
        ]
        if config['contact_timing_source'] == 'external_text_predictor':
            timing_args += ['--contact-timing-file', str(text_timing)]
        common = [
            sys.executable, 'tools/export_hhc_official_interx.py',
            '--checkpoint', str(checkpoint),
            '--vq-checkpoint', str(vq_checkpoint),
            '--vq-contract', 'precleanup_raw12288',
            '--mgt-contract', 'historical_mmm',
            '--inference-conditioning-contract', 'paper_text_cfg',
            '--contact-map', str(contact_map),
            '--contact-map-contract', 'paper_formula_train_only',
            '--requests', str(requests), '--device', args.device,
            '--batch-size', str(args.batch_size), '--sampling-steps', '50',
            '--temperature', '0.5', '--cfg-scale', '3.5', '--resume',
        ] + timing_args
        run(common + [
            '--output', str(main_h5), '--repeats', '1', '--seed', str(seed),
        ], args.dry_run)
        run(common + [
            '--output', str(mm_h5), '--indices-key', 'mm_indices',
            '--repeats', '30', '--seed', str(seed + 100_000),
        ], args.dry_run)
        run([
            sys.executable, 'tools/score_fair_interx_generations.py',
            '--requests', str(requests), '--main-generations', str(main_h5),
            '--mm-generations', str(mm_h5), '--output', str(metrics),
            '--device', args.device, '--batch-size', '32',
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
        manifest['outputs'][config['label']] = {
            'main': str(main_h5), 'multimodality': str(mm_h5),
            'metrics': str(metrics), 'all_pairs_audit': str(mm_audit),
            'physical': str(physical),
        }
        if not args.dry_run:
            atomic_json(manifest_path, manifest)

    manifest['status'] = 'complete'
    if not args.dry_run:
        atomic_json(manifest_path, manifest)
    print(json.dumps({'manifest': str(manifest_path), 'status': 'complete'}))


if __name__ == '__main__':
    main()
