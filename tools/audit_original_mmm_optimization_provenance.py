#!/usr/bin/env python3
"""Audit the incompatible optimization contracts preserved around MMM."""

import ast
import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MMM = Path('/home/wenning/projects/MMM')
LOSS_HISTORY = ROOT / 'output/historical_mmm_local_history_optimization_losses'
OPTIMIZER_HISTORY = ROOT / 'output/historical_mmm_local_history_optimize_full_body'
OUTPUT = ROOT / 'output/original_mmm_optimization_provenance_audit.json'


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


def function_node(path, function_name, class_name=None):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    body = tree.body
    if class_name is not None:
        owner = next(
            node for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = owner.body
    return next(
        node for node in body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def function_ast(path, function_name, class_name=None):
    return ast.dump(
        function_node(path, function_name, class_name),
        include_attributes=False,
    )


def function_defaults(path, function_name):
    node = function_node(path, function_name)
    names = [argument.arg for argument in node.args.args]
    values = [ast.literal_eval(value) for value in node.args.defaults]
    return dict(zip(names[-len(values):], values))


def actual_call_count(path, function_name):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    return sum(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == function_name
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == function_name
        )
        for node in ast.walk(tree)
    )


def history_summary(directory, kind):
    rows = []
    for path in sorted(directory.glob('*_ContentChange.py')):
        source = path.read_text(encoding='utf-8')
        timestamp_ms = int(path.name.split('_', 1)[0])
        row = {
            'timestamp_ms': timestamp_ms,
            'path': str(path.resolve()),
            'sha256': sha256_file(path),
        }
        if kind == 'loss':
            row.update({
                'uses_abs_signed_contact_distance': (
                    'dist_abs_A = torch.abs(dist_signed_A)' in source
                ),
                'uses_relu_signed_contact_distance': (
                    'dist_abs_A = torch.relu(dist_signed_A)' in source
                ),
                'imports_volumetric_smpl': 'VolumetricSMPL' in source,
            })
        else:
            try:
                defaults = function_defaults(path, 'optimize_full_body')
            except (StopIteration, SyntaxError, ValueError):
                defaults = {}
            row.update({
                'num_epochs_default': defaults.get('num_epochs'),
                'learning_rate_default': defaults.get('lr'),
                'imports_losses_volume': 'losses_volume' in source,
                'freeze_epoch_literals': sorted({
                    int(value) for value in re.findall(
                        r'freeze_epoch\s*=\s*(\d+)', source
                    )
                }),
            })
        rows.append(row)
    if not rows:
        raise ValueError(f'empty recovered history: {directory}')
    return {
        'snapshots': len(rows),
        'earliest': rows[0],
        'latest': rows[-1],
        'distinct_observed_contracts': sorted({
            json.dumps(
                {key: value for key, value in row.items()
                 if key not in {'timestamp_ms', 'path', 'sha256'}},
                sort_keys=True,
            )
            for row in rows
        }),
        'recovery_manifest': artifact(directory / 'recovery_manifest.tsv'),
    }


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def main():
    production_loss = ROOT / 'optimization/losses.py'
    mmm_loss = MMM / 'optimization/losses.py'
    production_optimizer = ROOT / 'optimize/optimize_full_body.py'
    mmm_optimizer = MMM / 'optimize_full_body.py'
    production_scorer = ROOT / 'eval/calc_pene_from_npz.py'
    mmm_scorer = MMM / 'calc_pene_from_npz.py'
    volume_loss = MMM / 'optimization/losses_volume.py'
    paper = ROOT / 'HHC__Generating_Human_Human_Close_Interactions_with_Contacts.pdf'

    optimizer_defaults = function_defaults(mmm_optimizer, 'optimize_full_body')
    checks = {
        'production_loss_byte_identical_to_current_mmm': (
            sha256_file(production_loss) == sha256_file(mmm_loss)
        ),
        'production_optimizer_function_identical_to_current_mmm': (
            function_ast(production_optimizer, 'optimize_full_body')
            == function_ast(mmm_optimizer, 'optimize_full_body')
        ),
        'production_mismatch_function_identical_to_current_mmm': (
            function_ast(
                production_scorer, '_calc_contact_mismatch_per_frame',
                'PeneCalculator',
            ) == function_ast(
                mmm_scorer, '_calc_contact_mismatch_per_frame',
                'PeneCalculator',
            )
        ),
        'current_mmm_defaults_are_200_and_0_01': (
            optimizer_defaults.get('num_epochs') == 200
            and optimizer_defaults.get('lr') == 0.01
        ),
        'losses_volume_calls_attach_volume': (
            actual_call_count(volume_loss, 'attach_volume') > 0
        ),
    }
    payload = {
        'schema': 'hhc_original_mmm_optimization_provenance_audit_v1',
        'status': 'conflict_confirmed',
        'conclusion': (
            'The current executable MMM contract and the paper contract are '
            'different, and no recovered implementation realizes all paper '
            'settings. Do not label one as the other.'
        ),
        'paper_contract': {
            'iterations': 100,
            'learning_rate': 0.02,
            'coarse_iterations': 50,
            'fine_iterations': 50,
            'signed_distance_backend': 'VolumetricSMPL',
            'contact_distance_operation_equation_4': 'absolute_signed_distance',
            'source_location': 'supplementary lines 511-514 and Eq.4',
        },
        'current_mmm_contract': {
            'iterations_default': optimizer_defaults['num_epochs'],
            'learning_rate_default': optimizer_defaults['lr'],
            'fine_stage_start_literal': 100,
            'signed_distance_backend': 'nearest_target_vertex_normal_sign',
            'contact_distance_operation': 'relu_signed_distance',
            'optimizer_import': 'optimization.losses.DyadicInteractionLoss',
        },
        'checks': checks,
        'loss_history': history_summary(LOSS_HISTORY, 'loss'),
        'optimizer_history': history_summary(OPTIMIZER_HISTORY, 'optimizer'),
        'losses_volume_observation': {
            'imports_attach_volume': True,
            'actual_attach_volume_call_count': actual_call_count(
                volume_loss, 'attach_volume'
            ),
            'actual_point2point_signed_call_count': actual_call_count(
                volume_loss, 'point2point_signed'
            ),
            'usable_as_complete_paper_implementation': False,
        },
        'artifacts': {
            'paper': artifact(paper),
            'production_loss': artifact(production_loss),
            'current_mmm_loss': artifact(mmm_loss),
            'production_optimizer': artifact(production_optimizer),
            'current_mmm_optimizer': artifact(mmm_optimizer),
            'production_scorer': artifact(production_scorer),
            'current_mmm_scorer': artifact(mmm_scorer),
            'mmm_losses_volume': artifact(volume_loss),
            'audit_implementation': artifact(Path(__file__)),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
