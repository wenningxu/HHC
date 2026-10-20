#!/usr/bin/env python3
"""Gate the unmodified original-MMM output without post-processing it."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MMM = Path('/home/wenning/projects/MMM')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def function_ast(path, name):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.dump(node, include_attributes=False)


def artifact(path):
    path = Path(path).resolve()
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assessment', required=True, type=Path)
    parser.add_argument('--alignment-diagnostic', required=True, type=Path)
    parser.add_argument('--optimizer-completion', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--maximum-projected-hours', type=float, default=24.0)
    parser.add_argument('--maximum-skate-increase', type=float, default=1e-6)
    args = parser.parse_args()

    assessment = json.loads(args.assessment.read_text(encoding='utf-8'))
    alignment = json.loads(
        args.alignment_diagnostic.read_text(encoding='utf-8')
    )
    completion = json.loads(
        args.optimizer_completion.read_text(encoding='utf-8')
    )
    optimizer = completion.get('optimizer', {})
    runtime = completion.get('runtime_summary', {})
    sample_count = int(assessment.get('samples', -1))

    fidelity = {
        'loss_byte_identical_to_original_mmm': (
            sha256_file(ROOT / 'optimization/losses.py')
            == sha256_file(MMM / 'optimization/losses.py')
        ),
        'optimizer_function_ast_identical_to_original_mmm': (
            function_ast(
                ROOT / 'optimize/optimize_full_body.py', 'optimize_full_body'
            ) == function_ast(MMM / 'optimize_full_body.py', 'optimize_full_body')
        ),
        'run_complete': (
            completion.get('complete') is True
            and completion.get('completed_samples') == sample_count
            and completion.get('requested_samples') == sample_count
        ),
        'original_mmm_hyperparameters': (
            optimizer.get('iterations') == 200
            and optimizer.get('learning_rate') == 0.01
            and optimizer.get('fine_stage_start') == 100
            and optimizer.get('signed_distance_backend') == 'nearest_vertex'
            and optimizer.get('contact_loss_reduction_policy') == 'original_mmm'
            and optimizer.get('optimization_variable_policy')
            == 'released_all_per_frame'
        ),
    }
    if not all(fidelity.values()):
        raise ValueError(f'original-MMM fidelity check failed: {fidelity}')

    baseline = assessment['aggregate']['random_shape_mismatch']
    optimized = assessment['aggregate']['original_mmm_raw_output']
    metrics = {
        metric: {
            'random_mean': float(baseline[metric]['mean']),
            'optimized_mean': float(optimized[metric]['mean']),
            'mean_delta': float(optimized[metric]['mean_delta_vs_random']),
            'improved_samples': int(optimized[metric]['improved_samples']),
            'degraded_samples': int(optimized[metric]['degraded_samples']),
        }
        for metric in ('mismatch', 'pene', 'skate')
    }
    projected_hours = float(
        runtime['projected_full_5868_optimizer_hours_single_gpu']
    )
    checks = {
        'contact_mismatch_mean_improves': metrics['mismatch']['mean_delta'] < 0,
        'foot_skate_does_not_increase': (
            metrics['skate']['mean_delta'] <= args.maximum_skate_increase
        ),
        'single_gpu_runtime_within_one_day': (
            projected_hours <= args.maximum_projected_hours
        ),
    }
    result = {
        'schema': 'hhc_original_mmm_shape_pilot_direct_gate_v1',
        'status': 'pass' if all(checks.values()) else 'fail',
        'gate_pass': bool(all(checks.values())),
        'scope': (
            'unmodified original-MMM output only; no upper-body projection, '
            'role reinterpretation, or sample-wise output selection'
        ),
        'samples': sample_count,
        'fidelity': fidelity,
        'checks': checks,
        'metrics': metrics,
        'aligned_coordinate_mismatch_mean_delta': float(
            alignment['aggregate']['delta_optimized_vs_random_aligned']
        ),
        'runtime': runtime,
        'thresholds': {
            'maximum_projected_hours_single_gpu': args.maximum_projected_hours,
            'maximum_skate_increase': args.maximum_skate_increase,
        },
        'artifacts': {
            'assessment': artifact(args.assessment),
            'alignment_diagnostic': artifact(args.alignment_diagnostic),
            'optimizer_completion': artifact(args.optimizer_completion),
            'gate_implementation': artifact(Path(__file__)),
            'production_loss': artifact(ROOT / 'optimization/losses.py'),
            'original_mmm_loss': artifact(MMM / 'optimization/losses.py'),
            'production_optimizer': artifact(
                ROOT / 'optimize/optimize_full_body.py'
            ),
            'original_mmm_optimizer': artifact(MMM / 'optimize_full_body.py'),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
