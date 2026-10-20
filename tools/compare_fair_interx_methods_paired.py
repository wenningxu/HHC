#!/usr/bin/env python
"""Paired tests for strict Inter-X methods on identical frozen replications."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


REPLICATION_PATTERN = re.compile(r'replication_(\d{2})\.json$')
METRICS = {
    'fid': ('lower', None),
    'top1': ('higher', None),
    'top2': ('higher', None),
    'top3': ('higher', None),
    'matching': ('lower', None),
    'multimodality': ('higher', None),
    # Diversity has no monotonic direction.  Compare its absolute error from
    # the same-replication GT diversity instead of rewarding larger values.
    'diversity_absolute_error': ('lower', ('diversity', 'diversity_gt')),
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def load_strict_replications(directory, expected_replications, replication_ids=None):
    directory = directory.resolve()
    paths = sorted(
        path for path in directory.glob('replication_*.json')
        if REPLICATION_PATTERN.fullmatch(path.name)
    )
    expected_ids = (
        list(range(expected_replications))
        if replication_ids is None else list(replication_ids)
    )
    if len(expected_ids) != expected_replications or len(set(expected_ids)) != len(expected_ids):
        raise ValueError(
            'replication IDs must be unique and match --expected-replications'
        )
    if any(index < 0 or index > 19 for index in expected_ids):
        raise ValueError('replication IDs must be within the frozen range 0..19')
    expected_id_set = set(expected_ids)
    records = {}
    for path in paths:
        replication = int(REPLICATION_PATTERN.fullmatch(path.name).group(1))
        if replication_ids is not None and replication not in expected_id_set:
            continue
        if replication in records:
            raise ValueError(f'{directory}: duplicate replication {replication}')
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('samples') != 1696:
            raise ValueError(f'{path}: expected 1696 main samples')
        if record.get('mm_samples') != 100 or record.get('mm_repeats') != 30:
            raise ValueError(f'{path}: expected 100 x 30 multimodality samples')
        request_path = record.get('requests')
        if not request_path or not Path(request_path).is_file():
            raise ValueError(f'{path}: missing frozen request file {request_path!r}')
        request_hash = sha256_file(request_path)
        if record.get('requests_sha256') != request_hash:
            raise ValueError(f'{path}: frozen request hash is absent or stale')
        if record.get('generation_request_hash_verified') is not True:
            raise ValueError(f'{path}: generation request hash was not verified')
        record['_request_sha256_verified'] = request_hash
        records[replication] = record
    if sorted(records) != expected_ids:
        raise ValueError(
            f'{directory}: expected replication IDs {expected_ids}, '
            f'got {sorted(records)}'
        )
    return records


def metric_value(record, metric):
    _, components = METRICS[metric]
    if components is None:
        return float(record[metric])
    generated, ground_truth = components
    return abs(float(record[generated]) - float(record[ground_truth]))


def bootstrap_mean_ci(values, rng, samples):
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return [
        float(value)
        for value in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    ]


def holm_bonferroni(p_values):
    """Return family-wise-error controlled p-values in original order."""
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values, kind='stable')
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, float(values[index]) * (count - rank))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def compare_methods(reference, comparator, rng, bootstrap_samples):
    if set(reference) != set(comparator):
        raise ValueError('paired methods have different replication IDs')
    for replication in sorted(reference):
        reference_hash = reference[replication]['_request_sha256_verified']
        comparator_hash = comparator[replication]['_request_sha256_verified']
        if reference_hash != comparator_hash:
            raise ValueError(
                f'replication {replication}: methods use different frozen requests '
                f'({reference_hash} != {comparator_hash})'
            )

    comparisons = {}
    for metric, (direction, _) in METRICS.items():
        reference_values = np.asarray([
            metric_value(reference[index], metric) for index in sorted(reference)
        ], dtype=np.float64)
        comparator_values = np.asarray([
            metric_value(comparator[index], metric) for index in sorted(comparator)
        ], dtype=np.float64)
        raw_delta = reference_values - comparator_values
        advantage = raw_delta if direction == 'higher' else -raw_delta
        if np.all(advantage == 0):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(advantage, alternative='two-sided').pvalue)
        comparisons[metric] = {
            'direction': direction,
            'reference_mean': float(reference_values.mean()),
            'comparator_mean': float(comparator_values.mean()),
            'paired_delta_reference_minus_comparator_mean': float(raw_delta.mean()),
            'reference_advantage_mean': float(advantage.mean()),
            'reference_advantage_bootstrap_95ci': bootstrap_mean_ci(
                advantage, rng, bootstrap_samples
            ),
            'reference_wins': int((advantage > 0).sum()),
            'ties': int((advantage == 0).sum()),
            'reference_losses': int((advantage < 0).sum()),
            'wilcoxon_two_sided_p': p_value,
        }
    return comparisons


def parse_method(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('method must be LABEL=DIRECTORY')
    label, directory = value.split('=', 1)
    if not label or not directory:
        raise argparse.ArgumentTypeError('method must be LABEL=DIRECTORY')
    return label, Path(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=parse_method, required=True)
    parser.add_argument('--comparators', type=parse_method, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-replications', type=int, default=20)
    parser.add_argument(
        '--replication-ids', type=int, nargs='+', default=None,
        help=(
            'Explicit frozen replication IDs to compare. This permits a '
            'replication-0 sensitivity against a completed 20-rep reference.'
        ),
    )
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    reference_label, reference_path = args.reference
    reference = load_strict_replications(
        reference_path, args.expected_replications, args.replication_ids
    )
    rng = np.random.default_rng(args.seed)
    results = {}
    hypotheses = []
    for comparator_label, comparator_path in args.comparators:
        if comparator_label in results or comparator_label == reference_label:
            raise ValueError(f'duplicate method label {comparator_label!r}')
        comparisons = compare_methods(
            reference,
            load_strict_replications(
                comparator_path, args.expected_replications,
                args.replication_ids,
            ),
            rng,
            args.bootstrap_samples,
        )
        results[comparator_label] = {
            'directory': str(comparator_path.resolve()),
            'metrics': comparisons,
        }
        for metric, record in comparisons.items():
            hypotheses.append((
                comparator_label, metric, record['wilcoxon_two_sided_p']
            ))

    adjusted = holm_bonferroni([item[2] for item in hypotheses])
    for (comparator_label, metric, _), value in zip(hypotheses, adjusted):
        record = results[comparator_label]['metrics'][metric]
        record['holm_familywise_p'] = value
        record['holm_significant_at_0.05'] = bool(value < 0.05)

    payload = {
        'schema': 'fair_interx_paired_method_comparison_v1',
        'reference': {
            'label': reference_label,
            'directory': str(reference_path.resolve()),
        },
        'replications': args.expected_replications,
        'replication_ids': (
            list(range(args.expected_replications))
            if args.replication_ids is None else args.replication_ids
        ),
        'pairing': 'same replication ID and byte-verified frozen request NPZ',
        'test': 'two-sided paired Wilcoxon signed-rank',
        'multiplicity_correction': (
            'Holm-Bonferroni across every comparator-metric hypothesis'
        ),
        'bootstrap_samples': args.bootstrap_samples,
        'seed': args.seed,
        'comparators': results,
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
