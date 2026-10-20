#!/usr/bin/env python
"""Run the predeclared paired paper-optimizer pilot ablation matrix."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = (
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
    parser.add_argument('--motion-dir', required=True, type=Path)
    parser.add_argument('--generation-manifest', required=True, type=Path)
    parser.add_argument(
        '--contact-dir', type=Path,
        default=Path('dataset/interx/contact_pairs_codex_luna_strict_v3_test'),
    )
    parser.add_argument(
        '--expected-split', type=Path,
        default=Path('dataset/interx/splits/test.txt'),
    )
    parser.add_argument(
        '--output-root', type=Path,
        default=Path('output/optimization_ablation_pilot'),
    )
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--limit', type=int, default=25)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--query-chunk', type=int, default=512)
    parser.add_argument(
        '--signed-distance-backend', choices=('nearest_vertex', 'volumetric_smpl'),
        default='nearest_vertex',
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.limit < 2:
        raise ValueError('paired pilot requires at least two samples')

    motion_dir = args.motion_dir.resolve()
    generation_manifest = args.generation_manifest.resolve()
    contact_dir = args.contact_dir.resolve()
    expected_split = args.expected_split.resolve()
    extraction_manifest = contact_dir / '_extraction_manifest.json'
    required = [
        generation_manifest, expected_split, extraction_manifest,
        ROOT / 'tools/run_paper_contact_optimization.py',
        ROOT / 'tools/score_volumetric_smpl_npz.py',
        ROOT / 'tools/compare_paired_physical_metrics.py',
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'strict optimization ablation prerequisites are missing: '
            + ', '.join(missing)
        )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / 'ablation_manifest.json'
    manifest = {
        'schema': 'hhc_paper_optimizer_ablation_pilot_v1',
        'status': 'running',
        'design': 'paired_same_sample_one_factor_and_iteration_weight_scan',
        'pilot_limit': args.limit,
        'seed': args.seed,
        'device': args.device,
        'query_chunk': args.query_chunk,
        'signed_distance_backend': args.signed_distance_backend,
        'ablations': list(ABLATIONS),
        'artifacts': {
            'ablation_runner': artifact(Path(__file__)),
            'generation_manifest': artifact(generation_manifest),
            'expected_split': artifact(expected_split),
            'contact_extraction_manifest': artifact(extraction_manifest),
            'optimizer_runner': artifact(
                ROOT / 'tools/run_paper_contact_optimization.py'
            ),
            'physical_scorer': artifact(
                ROOT / 'tools/score_volumetric_smpl_npz.py'
            ),
            'paired_comparator': artifact(
                ROOT / 'tools/compare_paired_physical_metrics.py'
            ),
        },
        'outputs': {},
    }
    if not args.dry_run:
        atomic_json(manifest_path, manifest)

    before_metrics = output_root / 'before_metrics'
    scorer_common = [
        sys.executable, 'tools/score_volumetric_smpl_npz.py',
        '--device', args.device,
        '--query-chunk', str(args.query_chunk),
        '--seed', str(args.seed),
        '--repeat-id', '0', '--limit', str(args.limit),
    ]
    if args.overwrite:
        scorer_common.append('--overwrite')
    run(scorer_common + [
        '--folder', str(motion_dir), '--output-dir', str(before_metrics),
    ], args.dry_run)

    for ablation in ABLATIONS:
        run_dir = output_root / ablation['label']
        optimized_dir = run_dir / 'optimized'
        after_metrics = run_dir / 'after_metrics'
        comparison = run_dir / 'paired_comparison.json'
        optimize_command = [
            sys.executable, 'tools/run_paper_contact_optimization.py',
            '--motion-dir', str(motion_dir),
            '--generation-manifest', str(generation_manifest),
            '--contact-dir', str(contact_dir),
            '--expected-split', str(expected_split),
            '--output-dir', str(optimized_dir),
            '--device', args.device,
            '--repeat-id', '0', '--limit', str(args.limit),
            '--seed', str(args.seed), '--query-chunk', str(args.query_chunk),
            '--num-epochs', str(ablation['iterations']),
            '--learning-rate', '0.02',
            '--fine-stage-start', str(ablation['fine_start']),
            '--signed-distance-backend', args.signed_distance_backend,
            '--loss-weights-json', json.dumps(
                ablation['weights'], sort_keys=True, separators=(',', ':')
            ),
        ]
        if args.overwrite:
            optimize_command.append('--overwrite')
        run(optimize_command, args.dry_run)
        run(scorer_common + [
            '--folder', str(optimized_dir),
            '--output-dir', str(after_metrics),
        ], args.dry_run)
        run([
            sys.executable, 'tools/compare_paired_physical_metrics.py',
            '--before', str(before_metrics), '--after', str(after_metrics),
            '--output', str(comparison), '--seed', str(args.seed),
            '--bootstrap-samples', '10000',
        ], args.dry_run)
        output_record = {
            'optimized': str(optimized_dir),
            'after_metrics': str(after_metrics),
            'paired_comparison': str(comparison),
        }
        if not args.dry_run:
            output_record.update({
                'optimizer_completion_artifact': artifact(
                    optimized_dir / '_completion_manifest.json'
                ),
                'after_metrics_artifact': artifact(
                    after_metrics / 'metrics.json'
                ),
                'paired_comparison_artifact': artifact(comparison),
            })
        manifest['outputs'][ablation['label']] = output_record
        if not args.dry_run:
            atomic_json(manifest_path, manifest)

    manifest['status'] = 'complete'
    manifest['before_metrics'] = str(before_metrics)
    if not args.dry_run:
        manifest['before_metrics_artifact'] = artifact(
            before_metrics / 'metrics.json'
        )
        atomic_json(manifest_path, manifest)
    print(json.dumps({'manifest': str(manifest_path), 'status': manifest['status']}))


if __name__ == '__main__':
    main()
