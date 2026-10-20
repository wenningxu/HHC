#!/usr/bin/env python
"""Score a frozen, request-bound contact-timing prediction without tuning it."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def counts_to_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        ),
        'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
    }


def bootstrap_mean(values, seed, samples):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = generator.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1)
    return {
        'mean': float(values.mean()),
        'ci95': np.quantile(means, (0.025, 0.975)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    args = parser.parse_args()
    requests = args.requests.resolve()
    predictions = args.predictions.resolve()
    request_hash = sha256_file(requests)
    with np.load(requests, allow_pickle=False) as truth_file:
        names = truth_file['names'].astype(str)
        lengths = truth_file['lengths'].astype(np.int64)
        truth = truth_file['contact'].astype(bool)
    with np.load(predictions, allow_pickle=False) as predicted_file:
        predicted_names = predicted_file['names'].astype(str)
        prediction = predicted_file['contact'].astype(bool)
        bound_hash = str(predicted_file['requests_sha256'].item())
    if bound_hash != request_hash:
        raise ValueError('prediction is bound to another request file')
    if not np.array_equal(names, predicted_names):
        raise ValueError('prediction names/order differ from requests')
    if prediction.shape != truth.shape or truth.shape != (len(names), 150):
        raise ValueError('unexpected contact-mask shapes')
    valid = np.arange(150)[None] < lengths[:, None]
    if prediction[~valid].any() or truth[~valid].any():
        raise ValueError('contact masks are nonzero after valid length')
    tp = int(np.logical_and(prediction, truth)[valid].sum())
    fp = int(np.logical_and(prediction, ~truth)[valid].sum())
    fn = int(np.logical_and(~prediction, truth)[valid].sum())
    per_sample_f1, per_sample_iou = [], []
    onset_errors, offset_errors = [], []
    truth_positive = truth.any(axis=1)
    prediction_positive = prediction.any(axis=1)
    for index, length in enumerate(lengths):
        target = truth[index, :int(length)]
        guess = prediction[index, :int(length)]
        sample_tp = int(np.logical_and(target, guess).sum())
        sample_fp = int(np.logical_and(~target, guess).sum())
        sample_fn = int(np.logical_and(target, ~guess).sum())
        metrics = counts_to_metrics(sample_tp, sample_fp, sample_fn)
        per_sample_f1.append(metrics['f1'])
        per_sample_iou.append(metrics['iou'])
        if target.any() and guess.any():
            target_indices = np.flatnonzero(target)
            guess_indices = np.flatnonzero(guess)
            onset_errors.append(abs(int(guess_indices[0]) - int(target_indices[0])))
            offset_errors.append(abs(int(guess_indices[-1]) - int(target_indices[-1])))
    result = {
        'schema': 'hhc_interx_contact_timing_prediction_score_v1',
        'protocol': 'held-out frozen test; no post-test hyperparameter tuning',
        'requests': str(requests),
        'requests_sha256': request_hash,
        'predictions': str(predictions),
        'predictions_sha256': sha256_file(predictions),
        'samples': len(names),
        'valid_frames': int(valid.sum()),
        'micro': counts_to_metrics(tp, fp, fn),
        'per_sample_f1': bootstrap_mean(
            per_sample_f1, args.seed, args.bootstrap_samples
        ),
        'per_sample_iou': bootstrap_mean(
            per_sample_iou, args.seed + 1, args.bootstrap_samples
        ),
        'true_positive_sequence_rate': float(truth_positive.mean()),
        'predicted_positive_sequence_rate': float(prediction_positive.mean()),
        'missed_contact_sequence_rate': float(
            np.logical_and(truth_positive, ~prediction_positive).mean()
        ),
        'false_contact_sequence_rate': float(
            np.logical_and(~truth_positive, prediction_positive).mean()
        ),
        'both_positive_sequences': len(onset_errors),
        'onset_absolute_error_frames': {
            'mean': float(np.mean(onset_errors)) if onset_errors else None,
            'median': float(np.median(onset_errors)) if onset_errors else None,
            'p95': float(np.quantile(onset_errors, 0.95)) if onset_errors else None,
        },
        'offset_absolute_error_frames': {
            'mean': float(np.mean(offset_errors)) if offset_errors else None,
            'median': float(np.median(offset_errors)) if offset_errors else None,
            'p95': float(np.quantile(offset_errors, 0.95)) if offset_errors else None,
        },
        'implementation_sha256': sha256_file(__file__),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
