#!/usr/bin/env python
"""Paired replication-0 mesh statistics against deployable no-timing HHC."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'hhc_historical_plus_paper_mask_ratio_no_contact_common50'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve(strict=True)
    return {
        'path': str(path), 'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_artifact(record, label):
    require(isinstance(record, dict), f'{label}: artifact record is missing')
    path = Path(record.get('path', ''))
    require(
        path.is_file()
        and record.get('bytes') == path.stat().st_size
        and record.get('sha256') == sha256_file(path),
        f'{label}: artifact is absent or stale',
    )
    return path.resolve()


def load_sample_metrics(score_dir, expected_keys, expected_count):
    rows = {}
    for path in sorted((score_dir / 'samples').glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        key = record.get('sample_key')
        require(key not in rows, f'duplicate sample key {key}')
        metrics = record.get('metrics', {})
        require(
            set(expected_keys) <= set(metrics)
            and all(math.isfinite(float(metrics[name])) for name in expected_keys),
            f'{path}: incomplete/non-finite physical metrics',
        )
        rows[key] = {name: float(metrics[name]) for name in expected_keys}
    require(len(rows) == expected_count, f'{score_dir}: sample count differs')
    return rows


def bootstrap_mean_ci(values, rng, samples, chunk=512):
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        indices = rng.integers(
            0, len(values), size=(stop - start, len(values)),
        )
        means[start:stop] = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, (0.025, 0.975))]


def holm_adjust(records):
    ordered = sorted(records, key=lambda row: row['wilcoxon_two_sided_p'])
    running = 0.0
    total = len(ordered)
    for rank, row in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * row['wilcoxon_two_sided_p'])
        running = max(running, adjusted)
        row['holm_adjusted_p'] = running
        row['holm_reject_0p05'] = running <= 0.05


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--mesh-root', type=Path,
        default=Path('output/current_interx_mesh_physical'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path(
            'output/current_interx_mesh_physical/'
            'paired_vs_hhc_no_timing.json'
        ),
    )
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    mesh_root = args.mesh_root.resolve()
    manifest_path = mesh_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(
        manifest.get('schema') == 'hhc_final_interx_mesh_physical_main_table_v1'
        and manifest.get('status') == 'complete'
        and manifest.get('samples_per_method') == 1696,
        'final six-method mesh manifest is incomplete',
    )
    methods = manifest.get('method_order', [])
    require(
        methods and methods[0] == REFERENCE and len(methods) == 6
        and len(set(methods)) == 6,
        'mesh method order/reference changed',
    )
    metric_keys = manifest.get('metric_keys', [])
    require(len(metric_keys) == 7, 'mesh metric set changed')
    sample_tables = {}
    method_artifacts = {}
    for method in methods:
        method_record = manifest['methods'][method]
        metrics_path = validate_artifact(
            method_record.get('artifacts', {}).get('metrics'),
            f'{method}/metrics',
        )
        score_dir = metrics_path.parent
        sample_tables[method] = load_sample_metrics(
            score_dir, metric_keys, 1696,
        )
        method_artifacts[method] = {
            'method_summary': method_record['artifacts']['method_summary'],
            'metrics': method_record['artifacts']['metrics'],
        }
        validate_artifact(
            method_artifacts[method]['method_summary'],
            f'{method}/method_summary',
        )
    reference_keys = set(sample_tables[REFERENCE])
    require(
        all(set(table) == reference_keys for table in sample_tables.values()),
        'mesh sample-key sets are not paired across methods',
    )
    keys = sorted(reference_keys)
    rng = np.random.default_rng(args.seed)
    comparisons = {}
    family = []
    for method in methods[1:]:
        comparisons[method] = {}
        for metric in metric_keys:
            before = np.asarray([
                sample_tables[REFERENCE][key][metric] for key in keys
            ], dtype=np.float64)
            after = np.asarray([
                sample_tables[method][key][metric] for key in keys
            ], dtype=np.float64)
            delta = after - before
            p_value = 1.0 if np.all(delta == 0) else float(
                wilcoxon(delta).pvalue
            )
            row = {
                'direction': 'lower_is_better',
                'reference_mean': float(before.mean()),
                'comparator_mean': float(after.mean()),
                'paired_delta_comparator_minus_reference_mean': float(
                    delta.mean()
                ),
                'paired_delta_median': float(np.median(delta)),
                'paired_delta_bootstrap_95ci': bootstrap_mean_ci(
                    delta, rng, args.bootstrap_samples,
                ),
                'reference_better_samples': int((delta > 0).sum()),
                'equal_samples': int((delta == 0).sum()),
                'comparator_better_samples': int((delta < 0).sum()),
                'wilcoxon_two_sided_p': p_value,
            }
            comparisons[method][metric] = row
            family.append(row)
    holm_adjust(family)
    payload = {
        'schema': 'hhc_current_interx_mesh_paired_methods_v1',
        'status': 'complete',
        'scope': (
            'replication-0 paired sample-level physical sensitivity; '
            'not a 20-replication semantic metric claim'
        ),
        'reference': REFERENCE,
        'comparators': methods[1:],
        'oracle_disclosure': (
            'the HHC contact comparator consumes frozen test timing and is an '
            'oracle ablation, not a deployable baseline'
        ),
        'sample_count': len(keys),
        'metric_keys': metric_keys,
        'bootstrap_samples': args.bootstrap_samples,
        'seed': args.seed,
        'multiple_testing': 'Holm-Bonferroni over all 5x7 method-metric tests',
        'comparisons': comparisons,
        'artifacts': {
            'mesh_manifest': artifact(manifest_path),
            'methods': method_artifacts,
            'implementation': artifact(__file__),
        },
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({
        'status': payload['status'], 'sample_count': len(keys),
        'tests': len(family), 'output': str(args.output.resolve()),
    }, sort_keys=True))


if __name__ == '__main__':
    main()
