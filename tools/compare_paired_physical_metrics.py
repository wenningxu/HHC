#!/usr/bin/env python
"""Paired delta, bootstrap CI, and Wilcoxon test for physical metric runs."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


DEFAULT_METRICS = (
    'penetrating_vertex_fraction',
    'collision_frame_rate',
    'penetration_depth_sum_m_per_frame',
    'paper_eq6_squared_sum_m2_per_frame',
    'mean_depth_m_on_penetrating_vertices',
    'max_depth_m',
    'mean_foot_sliding_m_per_frame',
)


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def load_records(run_dir):
    records = {}
    for path in sorted((run_dir / 'samples').glob('*.json')):
        with open(path, encoding='utf-8') as handle:
            record = json.load(handle)
        key = record['sample_key']
        if key in records:
            raise ValueError(f'duplicate sample key {key} in {run_dir}')
        records[key] = record
    if not records:
        raise FileNotFoundError(f'no sample metrics in {run_dir}')
    return records


def bootstrap_mean_ci(values, rng, samples):
    if len(values) == 1:
        value = float(values[0])
        return [value, value]
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument(
        '--allow-before-superset', action='store_true',
        help=(
            'For a preregistered smaller pilot only: select the after-run '
            'keys from a complete before-run control. Missing before keys '
            'remain an error.'
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    before = load_records(args.before.resolve())
    after = load_records(args.after.resolve())
    before_keys = set(before)
    after_keys = set(after)
    if args.allow_before_superset:
        if not after_keys.issubset(before_keys):
            raise ValueError(
                'after keys must be a subset of before keys; '
                f'missing-before={sorted(after_keys - before_keys)[:10]}'
            )
        before = {key: before[key] for key in after_keys}
    elif before_keys != after_keys:
        raise ValueError(
            'paired comparison requires identical sample keys; '
            f'before-only={sorted(set(before) - set(after))[:10]}, '
            f'after-only={sorted(set(after) - set(before))[:10]}'
        )
    keys = sorted(before)
    rng = np.random.default_rng(args.seed)
    comparisons = {}
    for metric in DEFAULT_METRICS:
        before_values = np.asarray([
            before[key]['metrics'][metric] for key in keys
        ], dtype=np.float64)
        after_values = np.asarray([
            after[key]['metrics'][metric] for key in keys
        ], dtype=np.float64)
        delta = after_values - before_values
        if np.all(delta == 0):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(delta).pvalue)
        comparisons[metric] = {
            'direction': 'lower_is_better',
            'before_mean': float(before_values.mean()),
            'after_mean': float(after_values.mean()),
            'paired_delta_after_minus_before_mean': float(delta.mean()),
            'paired_delta_median': float(np.median(delta)),
            'paired_delta_bootstrap_95ci': bootstrap_mean_ci(
                delta, rng, args.bootstrap_samples
            ),
            'improved_samples': int((delta < 0).sum()),
            'unchanged_samples': int((delta == 0).sum()),
            'degraded_samples': int((delta > 0).sum()),
            'wilcoxon_two_sided_p': p_value,
        }
    report = {
        'schema': 'hhc_paired_physical_comparison_v1',
        'before': str(args.before.resolve()),
        'after': str(args.after.resolve()),
        'sample_count': len(keys),
        'sample_keys': keys,
        'bootstrap_samples': args.bootstrap_samples,
        'seed': args.seed,
        'pair_selection_policy': (
            'after_keys_selected_from_before_superset'
            if args.allow_before_superset else 'identical_complete_key_sets'
        ),
        'comparisons': comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
