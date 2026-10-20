#!/usr/bin/env python
"""Summarize the replication-0 HHC sampling grid and mark FID/MM Pareto rows."""

import argparse
import hashlib
import json
import os
from pathlib import Path


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


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    if manifest.get('status') != 'complete':
        raise ValueError('sampling-grid manifest is not complete')

    rows = []
    input_records = {'manifest': sha256_file(manifest_path)}
    evaluator_runtime = None
    for label, paths in sorted(manifest['outputs'].items()):
        metrics_path = Path(paths['metrics']).resolve()
        mm_path = Path(paths['all_pairs_audit']).resolve()
        physical_path = Path(paths['physical']).resolve()
        metrics = load_json(metrics_path)
        mm_audit = load_json(mm_path)
        physical = load_json(physical_path)
        if metrics.get('samples') != 1696:
            raise ValueError(f'{label}: expected 1696 scored samples')
        if metrics.get('mm_samples') != 100 or metrics.get('mm_repeats') != 30:
            raise ValueError(f'{label}: unexpected multimodality protocol')
        if mm_audit.get('all_pairs_per_prompt') != 435:
            raise ValueError(f'{label}: all-pairs audit is incomplete')
        row_runtime = metrics.get('evaluator_runtime')
        if not isinstance(row_runtime, dict):
            raise ValueError(f'{label}: metrics lack evaluator runtime')
        if evaluator_runtime is None:
            evaluator_runtime = row_runtime
        elif row_runtime != evaluator_runtime:
            raise ValueError(
                f'{label}: evaluator runtime differs across sampling rows: '
                f'{row_runtime!r} != {evaluator_runtime!r}'
            )
        rows.append({
            'label': label,
            **paths['config'],
            'fid': metrics['fid'],
            'top1': metrics['top1'],
            'top2': metrics['top2'],
            'top3': metrics['top3'],
            'matching': metrics['matching'],
            'diversity': metrics['diversity'],
            'multimodality_official_10_pairs': metrics['multimodality'],
            'multimodality_all_435_pairs': mm_audit['mean_all_prompt_pairs'],
            'contact_f1_10cm': physical['contact_f1'],
            'contact_temporal_iou_10cm': physical['contact_temporal_iou'],
            'false_contact_rate_10cm': physical['false_contact_rate'],
            'foot_sliding_mps': physical['foot_sliding_mps'],
        })
        input_records[label] = {
            'metrics_sha256': sha256_file(metrics_path),
            'all_pairs_audit_sha256': sha256_file(mm_path),
            'physical_sha256': sha256_file(physical_path),
        }

    for candidate in rows:
        candidate['fid_all_pairs_mm_pareto'] = not any(
            other is not candidate
            and other['fid'] <= candidate['fid']
            and other['multimodality_all_435_pairs']
                >= candidate['multimodality_all_435_pairs']
            and (
                other['fid'] < candidate['fid']
                or other['multimodality_all_435_pairs']
                    > candidate['multimodality_all_435_pairs']
            )
            for other in rows
        )

    result = {
        'schema': 'hhc_sampling_pareto_summary_v1',
        'design': manifest['design'],
        'replication': 0,
        'evaluator_runtime': evaluator_runtime,
        'interpretation': (
            'One frozen request/seed sensitivity analysis; it is not a '
            'replacement for the 20-replication paper sampler result.'
        ),
        'pareto_objectives': {
            'minimize': 'fid',
            'maximize': 'multimodality_all_435_pairs',
        },
        'rows': rows,
        'inputs': input_records,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output.resolve(), result)
    print(json.dumps({'output': str(args.output.resolve()), 'rows': len(rows)}))


if __name__ == '__main__':
    main()
