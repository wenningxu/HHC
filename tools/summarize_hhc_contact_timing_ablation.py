#!/usr/bin/env python
"""Collect the four preregistered HHC timing conditions into one table JSON."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fair-root', type=Path,
                        default=Path('output/fair_eval'))
    parser.add_argument('--ablation-root', type=Path,
                        default=Path(
                            'output/fair_eval/'
                            'hhc_historical_paper_mask_contact_timing_ablation_rep0'
                        ))
    parser.add_argument('--output', type=Path,
                        default=Path(
                            'output/fair_eval/'
                            'hhc_historical_paper_mask_contact_timing_ablation_rep0/'
                            'summary.json'
                        ))
    args = parser.parse_args()
    directories = {
        'request_oracle': args.fair_root / (
            'hhc_historical_plus_paper_mask_ratio_contact_common50'
        ),
        'none': args.fair_root / (
            'hhc_historical_plus_paper_mask_ratio_no_contact_common50'
        ),
        'text_predicted': args.ablation_root / 'text_predicted',
        'full_valid_weak': args.ablation_root / 'full_valid_weak',
    }
    rows = {}
    artifacts = {}
    evaluator_runtime = None
    for label, directory in directories.items():
        paths = {
            'semantic': directory / 'replication_00.json',
            'all_pairs': directory / 'replication_00_multimodality_audit.json',
            'physical': directory / 'replication_00_physical.json',
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(', '.join(missing))
        records = {
            key: json.loads(path.read_text(encoding='utf-8'))
            for key, path in paths.items()
        }
        semantic, all_pairs, physical = (
            records['semantic'], records['all_pairs'], records['physical']
        )
        row_runtime = semantic.get('evaluator_runtime')
        if not isinstance(row_runtime, dict):
            raise ValueError(f'{label}: semantic metrics lack evaluator runtime')
        if evaluator_runtime is None:
            evaluator_runtime = row_runtime
        elif row_runtime != evaluator_runtime:
            raise ValueError(
                f'{label}: evaluator runtime differs across timing conditions: '
                f'{row_runtime!r} != {evaluator_runtime!r}'
            )
        rows[label] = {
            'fid': semantic['fid'],
            'top1': semantic['top1'], 'top2': semantic['top2'],
            'top3': semantic['top3'], 'matching': semantic['matching'],
            'diversity': semantic['diversity'],
            'multimodality_official_10_pairs': semantic['multimodality'],
            'multimodality_all_435_pairs': all_pairs['mean_all_prompt_pairs'],
            'contact_f1_10cm': physical['contact_f1'],
            'contact_temporal_iou_10cm': physical['contact_temporal_iou'],
            'false_contact_rate_10cm': physical['false_contact_rate'],
            'minimum_interperson_joint_distance_m': (
                physical['minimum_interperson_joint_distance_m']
            ),
            'foot_sliding_mps': physical['foot_sliding_mps'],
        }
        artifacts[label] = {
            key: {'path': str(path.resolve()), 'sha256': sha256_file(path)}
            for key, path in paths.items()
        }
    output = {
        'schema': 'hhc_contact_timing_ablation_summary_v1',
        'scope': (
            'replication-0 sensitivity; request_oracle/none also have separate '
            '20-replication strict estimates'
        ),
        'evaluator_runtime': evaluator_runtime,
        'rows': rows,
        'artifacts': artifacts,
        'implementation_sha256': sha256_file(__file__),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
