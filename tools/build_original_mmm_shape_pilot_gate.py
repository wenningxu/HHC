#!/usr/bin/env python
"""Gate a safe shape augmentation candidate without changing original-MMM loss."""

import argparse
import ast
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def function_ast(path, name):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.dump(node, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assessment', required=True, type=Path)
    parser.add_argument('--alignment-diagnostic', required=True, type=Path)
    parser.add_argument('--role-diagnostic', required=True, type=Path)
    parser.add_argument('--optimizer-completion', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--skate-tolerance', type=float, default=1e-6)
    parser.add_argument('--maximum-projected-hours', type=float, default=24.0)
    args = parser.parse_args()
    physical = json.loads(args.assessment.read_text())
    alignment = json.loads(args.alignment_diagnostic.read_text())
    role_diagnostic = json.loads(args.role_diagnostic.read_text())
    optimizer_completion = json.loads(
        args.optimizer_completion.read_text(encoding='utf-8')
    )
    roles = {
        row['augmented_name']: row['original_mmm_choice']
        for row in role_diagnostic['rows']
    }
    physical_by_name = {row['augmented_name']: row for row in physical['rows']}
    alignment_names = {row['augmented_name'] for row in alignment['rows']}
    if set(physical_by_name) != alignment_names or alignment_names != set(roles):
        raise ValueError('assessment/alignment/role sample sets differ')
    sample_count = len(alignment_names)
    optimizer = optimizer_completion.get('optimizer', {})
    runtime = optimizer_completion.get('runtime_summary', {})
    expected_implementation = {
        str(path): sha256_file(path)
        for path in (
            ROOT / 'tools/run_interx_shape_augmentation.py',
            ROOT / 'optimize/optimize_full_body.py',
            ROOT / 'optimization/losses.py',
            ROOT / 'utils/interx_ground_alignment.py',
        )
    }
    original_mmm_loss = Path('/home/wenning/projects/MMM/optimization/losses.py')
    original_mmm_optimizer = Path('/home/wenning/projects/MMM/optimize_full_body.py')
    if not (
        optimizer_completion.get('schema')
        == 'hhc_interx_shape_augmentation_run_v1'
        and optimizer_completion.get('complete') is True
        and optimizer_completion.get('completed_samples') == sample_count
        and set(optimizer_completion.get('completed_names', []))
        == alignment_names
        and optimizer.get('implementation_contract')
        == 'original_mmm_unmodified_numeric_path'
        and optimizer.get('iterations') == 200
        and optimizer.get('learning_rate') == 0.01
        and optimizer.get('fine_stage_start') == 100
        and optimizer.get('signed_distance_backend') == 'nearest_vertex'
        and optimizer.get('contact_loss_reduction_policy') == 'original_mmm'
        and optimizer.get('optimization_variable_policy')
        == 'released_all_per_frame'
        and optimizer_completion.get('implementation')
        == expected_implementation
        and original_mmm_loss.is_file()
        and sha256_file(ROOT / 'optimization/losses.py')
        == sha256_file(original_mmm_loss)
        and original_mmm_optimizer.is_file()
        and function_ast(
            ROOT / 'optimize/optimize_full_body.py', 'optimize_full_body'
        ) == function_ast(original_mmm_optimizer, 'optimize_full_body')
        and runtime.get('measured_samples') == sample_count
        and runtime.get('optimizer_wall_seconds_measured_samples') == sample_count
    ):
        raise ValueError('optimizer completion/runtime contract is incomplete')
    projected_hours = float(
        runtime['projected_full_5868_optimizer_hours_single_gpu']
    )
    rows = []
    for diagnostic in alignment['rows']:
        name = diagnostic['augmented_name']
        metrics = diagnostic['timing_weighted_mismatch_aligned']
        role = roles[name]
        if role != diagnostic['initial_role_preferred_by_timing_weighted_mismatch']:
            raise ValueError(f'{name}: original-MMM role and metric role disagree')
        suffix = 'swapped_role' if role == 'swapped' else 'fixed_role'
        initial_mismatch = metrics[f'random_{suffix}']
        projected_mismatch = metrics[f'foot_projection_{suffix}']
        p = physical_by_name[name]['conditions']
        initial_skate = p['random_shape_mismatch']['skate']
        projected_skate = p['foot_preserving_projection']['skate']
        accept = (
            projected_mismatch < initial_mismatch and
            abs(projected_skate - initial_skate) <= args.skate_tolerance
        )
        selected = p['foot_preserving_projection'] if accept else p['random_shape_mismatch']
        rows.append({
            'augmented_name': name,
            'role_fixed_at_initialization': role,
            'accepted_projection': accept,
            'timing_mismatch_random': initial_mismatch,
            'timing_mismatch_projection': projected_mismatch,
            'timing_mismatch_selected': projected_mismatch if accept else initial_mismatch,
            'all_frame_metrics_selected': selected,
        })
    result = {
        'schema': 'hhc_original_mmm_shape_pilot_gate_v1',
        'status': 'diagnostic_not_production',
        'contract': {
            'optimizer_and_loss': 'unmodified original MMM',
            'post_projection': 'preserve input root/lower-body motion; retain mean translation delta and optimized upper body',
            'acceptance': 'strict timing-weighted mismatch improvement in the initial auto-selected role and abs skate delta <= tolerance; otherwise retain random-shape input',
            'skate_tolerance': args.skate_tolerance,
            'maximum_projected_hours_single_gpu': args.maximum_projected_hours,
        },
        'artifacts': {
            'assessment': artifact(args.assessment),
            'alignment_diagnostic': artifact(args.alignment_diagnostic),
            'role_diagnostic': artifact(args.role_diagnostic),
            'optimizer_completion': artifact(args.optimizer_completion),
            'gate_implementation': artifact(Path(__file__)),
            'projection_implementation': artifact(
                ROOT / 'tools/diagnose_original_mmm_upper_body_projection.py'
            ),
            'original_mmm_loss': artifact(ROOT / 'optimization/losses.py'),
            'original_mmm_optimizer': artifact(
                ROOT / 'optimize/optimize_full_body.py'
            ),
        },
        'samples': len(rows),
        'accepted': sum(row['accepted_projection'] for row in rows),
        'rows': rows,
        'aggregate': {
            'runtime': runtime,
            'timing_mismatch_random_mean': statistics.fmean(
                row['timing_mismatch_random'] for row in rows
            ),
            'timing_mismatch_selected_mean': statistics.fmean(
                row['timing_mismatch_selected'] for row in rows
            ),
            **{
                f'{metric}_random_mean': statistics.fmean(
                    physical_by_name[row['augmented_name']]['conditions'][
                        'random_shape_mismatch'
                    ][metric]
                    for row in rows
                )
                for metric in ('mismatch', 'skate', 'pene')
            },
            **{
                f'{metric}_selected_mean': statistics.fmean(
                    row['all_frame_metrics_selected'][metric] for row in rows
                )
                for metric in ('mismatch', 'skate', 'pene')
            },
        },
    }
    result['aggregate'].update({
        'timing_mismatch_selected_delta': (
            result['aggregate']['timing_mismatch_selected_mean']
            - result['aggregate']['timing_mismatch_random_mean']
        ),
        **{
            f'{metric}_selected_delta': (
                result['aggregate'][f'{metric}_selected_mean']
                - result['aggregate'][f'{metric}_random_mean']
            )
            for metric in ('mismatch', 'skate', 'pene')
        },
    })
    result['gate_pass'] = bool(
        result['accepted'] > 0
        and result['aggregate']['timing_mismatch_selected_delta'] < 0
        and abs(result['aggregate']['skate_selected_delta'])
        <= args.skate_tolerance
        and projected_hours <= args.maximum_projected_hours
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result['aggregate'] | {'accepted': result['accepted']}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
