#!/usr/bin/env python3
"""Freeze current random-shape and optimized-shape 20-rep comparisons."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('fid', 'matching', 'top1', 'top2', 'top3', 'diversity', 'multimodality')
CONDITIONS = {
    'no_contact': {
        'baseline': 'hhc_historical_plus_paper_mask_ratio_no_contact_common50',
        'random_shape_only': 'hhc_historical_paper_mask_shapeaug_random_no_contact_common50',
        'optimized_shape': 'hhc_historical_paper_mask_shapeaug_optimized_no_contact_common50',
    },
    'oracle_contact': {
        'baseline': 'hhc_historical_plus_paper_mask_ratio_contact_common50',
        'random_shape_only': 'hhc_historical_paper_mask_shapeaug_random_contact_common50',
        'optimized_shape': 'hhc_historical_paper_mask_shapeaug_optimized_contact_common50',
    },
}
ARM_ARTIFACTS = {
    'random': {
        'materialization': Path(
            'dataset/interx_shape_aug_random_seed123_v3/materialization_manifest.json'
        ),
        'selection': Path(
            'output/hhc_historical_plus_paper_mask_ratio_shapeaug_random_'
            'raw12288_300k_seed114514_v1_selection.json'
        ),
        'wandb': Path(
            'output/hhc_historical_plus_paper_mask_ratio_shapeaug_random_'
            'raw12288_300k_seed114514_v1_wandb_history_audit.json'
        ),
    },
    'optimized': {
        'materialization': Path(
            'dataset/interx_shape_aug_optimized_nearest_vertex_seed123_v3/'
            'materialization_manifest.json'
        ),
        'selection': Path(
            'output/hhc_historical_plus_paper_mask_ratio_shapeaug_optimized_'
            'raw12288_300k_seed114514_v1_selection.json'
        ),
        'wandb': Path(
            'output/hhc_historical_plus_paper_mask_ratio_shapeaug_optimized_'
            'raw12288_300k_seed114514_v1_wandb_history_audit.json'
        ),
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': sha256_file(path)}


def validate_artifact_record(record, label):
    if not isinstance(record, dict):
        raise ValueError(f'{label}: artifact record is missing')
    path = Path(record.get('path', '')).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get('bytes', -1))
        or sha256_file(path) != record.get('sha256')
    ):
        raise ValueError(f'{label}: artifact is absent or stale: {path}')
    return path


def read_json(path):
    record = artifact(path)
    return json.loads(Path(record['path']).read_text(encoding='utf-8')), record


def finite(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f'{label} must be finite')
    return float(value)


def validate_summary(path):
    payload, record = read_json(path)
    if payload.get('replications') != 20 or payload.get('samples_per_replication') != 1696:
        raise ValueError(f'{path}: expected 20 x 1696 result')
    compact = {}
    for name in METRICS:
        metric = payload.get('metrics', {}).get(name, {})
        values = metric.get('values')
        if not isinstance(values, list) or len(values) != 20:
            raise ValueError(f'{path}: {name} does not contain 20 values')
        values = [finite(value, f'{path}/{name}') for value in values]
        mean = finite(metric.get('mean'), f'{path}/{name}/mean')
        ci95 = finite(metric.get('ci95'), f'{path}/{name}/ci95')
        if not math.isclose(mean, sum(values) / 20, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f'{path}: {name} mean differs from values')
        compact[name] = {'mean': mean, 'ci95': ci95}
    return compact, record


def validate_paired(path, labels, fair_root):
    payload, record = read_json(path)
    expected_comparators = {'random_shape_only', 'optimized_shape'}
    if (
        payload.get('schema') != 'fair_interx_paired_method_comparison_v1'
        or payload.get('replications') != 20
        or payload.get('replication_ids') != list(range(20))
        or payload.get('reference', {}).get('label') != 'baseline'
        or Path(payload.get('reference', {}).get('directory', '')).resolve()
        != (fair_root / labels['baseline'])
        or set(payload.get('comparators', {})) != expected_comparators
    ):
        raise ValueError(f'{path}: paired comparison contract changed')
    for role in expected_comparators:
        comparator = payload['comparators'][role]
        if Path(comparator.get('directory', '')).resolve() != fair_root / labels[role]:
            raise ValueError(f'{path}: {role} directory changed')
        for metric in comparator.get('metrics', {}).values():
            finite(metric.get('holm_familywise_p'), f'{path}/{role}/holm_p')
    return payload, record


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fair-root', type=Path, default=Path('output/fair_eval'))
    parser.add_argument('--paired-no-contact', type=Path, required=True)
    parser.add_argument('--paired-contact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    fair_root = args.fair_root.resolve()

    contracts = {}
    for name, path in {
        'last_stage': Path('output/interx_shape_augmentation_last_stage_contract.json'),
        'training_launch': Path('output/hhc_shape_aug_training_launch_contract.json'),
        'optimization_backend': Path(
            'output/current_interx_optimization_backend_contract.json'
        ),
    }.items():
        payload, record = read_json(ROOT / path)
        if payload.get('status') != 'pass' or not all(payload.get('checks', {}).values()):
            raise ValueError(f'{path}: shape contract is not passing')
        internal_records = payload.get('artifacts')
        if internal_records is None:
            internal_records = {
                key: payload.get(key) for key in ('launcher', 'implementation')
            }
        if not internal_records or any(value is None for value in internal_records.values()):
            raise ValueError(f'{path}: contract does not bind its implementation inputs')
        for artifact_name, artifact_record in internal_records.items():
            validate_artifact_record(
                artifact_record, f'{path}/{artifact_name}'
            )
        if name == 'optimization_backend' and payload.get('contract', {}).get(
            'optimization_signed_distance_backend'
        ) != 'nearest_vertex':
            raise ValueError(f'{path}: optimizer is not the original MMM nearest-vertex path')
        contracts[name] = record

    arms = {}
    for arm, paths in ARM_ARTIFACTS.items():
        arms[arm] = {}
        for name, path in paths.items():
            payload, record = read_json(ROOT / path)
            if name == 'wandb' and (
                payload.get('status') != 'pass' or payload.get('remote_state') != 'finished'
            ):
                raise ValueError(f'{path}: completed online W&B audit is required')
            if name == 'selection' and payload.get('status') != 'complete':
                raise ValueError(f'{path}: official validation selection is incomplete')
            arms[arm][name] = record

    results = {}
    summary_artifacts = {}
    paired_paths = {
        'no_contact': args.paired_no_contact.resolve(),
        'oracle_contact': args.paired_contact.resolve(),
    }
    paired = {}
    paired_artifacts = {}
    for condition, labels in CONDITIONS.items():
        results[condition] = {}
        summary_artifacts[condition] = {}
        for role, label in labels.items():
            metrics, record = validate_summary(fair_root / label / 'summary.json')
            results[condition][role] = {'label': label, 'metrics': metrics}
            summary_artifacts[condition][role] = record
        paired[condition], paired_artifacts[condition] = validate_paired(
            paired_paths[condition], labels, fair_root
        )

    output = {
        'schema': 'hhc_current_shape_augmentation_results_v1',
        'status': 'complete',
        'design': (
            'same historical HHC + paper mask-ratio contract; official train-only '
            'baseline versus random-shape-only versus contact-optimized random shape; '
            'official unchanged validation/test; 20 paired frozen test replications'
        ),
        'conditions': results,
        'paired_comparisons': paired,
        'artifacts': {
            'contracts': contracts, 'arms': arms, 'summaries': summary_artifacts,
            'paired': paired_artifacts, 'implementation': artifact(Path(__file__)),
        },
    }
    atomic_write(args.output.resolve(), json.dumps(output, indent=2, sort_keys=True) + '\n')
    rows = []
    for condition, condition_results in results.items():
        for role, result in condition_results.items():
            metrics = result['metrics']
            rows.append(
                f"| {condition} | {role} | {metrics['fid']['mean']:.4f} ± "
                f"{metrics['fid']['ci95']:.4f} | {metrics['top1']['mean']:.4f} ± "
                f"{metrics['top1']['ci95']:.4f} | {metrics['matching']['mean']:.4f} ± "
                f"{metrics['matching']['ci95']:.4f} |\n"
            )
    markdown = (
        '# Current Inter-X shape augmentation results\n\n'
        'All entries use 20 paired frozen-request replications; shape augmentation '
        'changes official-train data only. Values are mean ± 95% CI.\n\n'
        '| Timing | Training arm | FID ↓ | R@1 ↑ | Matching ↓ |\n'
        '|---|---|---:|---:|---:|\n' + ''.join(rows)
    )
    atomic_write(args.markdown.resolve(), markdown)
    print(json.dumps({'status': 'complete', 'output': str(args.output.resolve()),
                      'markdown': str(args.markdown.resolve())}, indent=2))


if __name__ == '__main__':
    main()
