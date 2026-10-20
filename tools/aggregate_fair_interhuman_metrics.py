#!/usr/bin/env python3
"""Aggregate strict 20-rep InterHuman main and multimodality scores."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from dataset.interhuman_generated_h5 import sha256_file


MAIN_METRICS = ('fid', 'matching', 'top1', 'top2', 'top3', 'diversity', 'diversity_gt')
MODEL_IDENTITY_KEYS = (
    'checkpoint_sha256',
    'vq_checkpoint_sha256',
    'model_config_sha256',
    'generation_exporter_sha256',
    'interhuman_generation_bridge_sha256',
    'sampler',
    'sampling_steps',
    'temperature',
    'cfg_scale',
    'mgt_shape_condition',
    'contact_conditioning',
)
EXPECTED_HHC_IDENTITY = {
    'sampler': 'sample_intra_cas',
    'sampling_steps': 50,
    'temperature': 0.5,
    'cfg_scale': 3.5,
    'mgt_shape_condition': False,
    'contact_conditioning': False,
}


def _load(path: Path):
    return json.loads(path.read_text())


def _artifact_hash_matches(record, path_key, hash_key):
    path = Path(record[path_key]).resolve()
    if not path.is_file():
        raise ValueError(f'missing artifact {path}')
    actual = sha256_file(path)
    if record.get(hash_key) != actual:
        raise ValueError(f'{path_key} hash mismatch for {path}')


def _identity(metadata, require_hhc_common50):
    missing = [key for key in MODEL_IDENTITY_KEYS if key not in metadata]
    if missing:
        raise ValueError(f'generation metadata lacks identity keys {missing}')
    result = {key: metadata[key] for key in MODEL_IDENTITY_KEYS}
    if require_hhc_common50:
        for key, expected in EXPECTED_HHC_IDENTITY.items():
            if result[key] != expected:
                raise ValueError(
                    f'HHC InterHuman identity {key} mismatch: {result[key]!r} != {expected!r}'
                )
    return result


def summarize_records(
    main_records,
    mm_records,
    main_manifest,
    mm_manifest,
    *,
    require_hhc_common50=True,
    verify_generation_hashes=True,
):
    if len(main_records) != 20 or len(mm_records) != 20:
        raise ValueError('formal InterHuman aggregation requires 20 main and 20 MM scores')
    main_by_rep = {int(row['replication']): row for row in main_records}
    mm_by_rep = {int(row['replication']): row for row in mm_records}
    expected = set(range(20))
    if set(main_by_rep) != expected or set(mm_by_rep) != expected:
        raise ValueError('InterHuman score replication IDs must each be exactly 0..19')
    main_entries = {int(row['replication']): row for row in main_manifest['replications']}
    mm_entries = {int(row['replication']): row for row in mm_manifest['replications']}
    if set(main_entries) != expected or set(mm_entries) != expected:
        raise ValueError('InterHuman request manifest replication IDs must each be exactly 0..19')

    identities = []
    evaluator_provenance = None
    for replication in range(20):
        main = main_by_rep[replication]
        mm = mm_by_rep[replication]
        if main.get('schema') != 'interhuman_fair_main_score_v1' or main.get('status') != 'complete':
            raise ValueError(f'replication {replication}: invalid main score schema/status')
        if mm.get('schema') != 'interhuman_fair_mm_score_v1' or mm.get('status') != 'complete':
            raise ValueError(f'replication {replication}: invalid MM score schema/status')
        if main.get('samples') != 1056:
            raise ValueError(f'replication {replication}: expected 1056 main samples')
        for name in MAIN_METRICS:
            if name not in main or not np.isfinite(float(main[name])):
                raise ValueError(f'replication {replication}: invalid main metric {name}')
        if not np.isfinite(float(mm.get('multimodality', np.nan))):
            raise ValueError(f'replication {replication}: invalid multimodality')

        main_entry = main_entries[replication]
        mm_entry = mm_entries[replication]
        expected_main_fields = {
            'request_file_sha256': main_entry['sha256'],
            'request_content_sha256': main_entry['request_sha256'],
        }
        expected_mm_fields = {
            'request_file_sha256': mm_entry['sha256'],
            'request_content_sha256': mm_entry['request_sha256'],
        }
        for key, expected_value in expected_main_fields.items():
            if main.get(key) != expected_value:
                raise ValueError(f'replication {replication}: main {key} mismatch')
        for key, expected_value in expected_mm_fields.items():
            if mm.get(key) != expected_value:
                raise ValueError(f'replication {replication}: MM {key} mismatch')

        if verify_generation_hashes:
            _artifact_hash_matches(main, 'main_generations', 'main_generations_sha256')
            _artifact_hash_matches(mm, 'mm_generations', 'mm_generations_sha256')

        main_identity = _identity(main.get('generation_metadata', {}), require_hhc_common50)
        mm_identity = _identity(mm.get('generation_metadata', {}), require_hhc_common50)
        if main_identity != mm_identity:
            raise ValueError(
                f'replication {replication}: main and MM model/sampler identities differ'
            )
        identities.append(main_identity)
        if main_identity != identities[0]:
            raise ValueError(f'replication {replication}: model/sampler identity differs')

        if main.get('evaluator_provenance') != mm.get('evaluator_provenance'):
            raise ValueError(f'replication {replication}: main and MM evaluator provenance differs')
        if evaluator_provenance is None:
            evaluator_provenance = main['evaluator_provenance']
        elif main.get('evaluator_provenance') != evaluator_provenance:
            raise ValueError(f'replication {replication}: evaluator provenance differs')

    result = {
        'schema': 'interhuman_fair_20rep_summary_v1',
        'status': 'complete',
        'replications': 20,
        'samples_per_replication': 1056,
        'mm_prompts': 100,
        'mm_repeats': 30,
        'mm_pairs': 10,
        'confidence': 0.95,
        'ci_definition': '1.96 * population_std / sqrt(20)',
        'generation_hashes_verified': bool(verify_generation_hashes),
        'model_sampler_identity': identities[0],
        'evaluator_provenance': evaluator_provenance,
        'metrics': {},
    }
    metric_sources = {
        **{name: [main_by_rep[index][name] for index in range(20)] for name in MAIN_METRICS},
        'multimodality': [mm_by_rep[index]['multimodality'] for index in range(20)],
    }
    for name, raw_values in metric_sources.items():
        values = np.asarray(raw_values, dtype=np.float64)
        population_std = float(np.std(values))
        result['metrics'][name] = {
            'mean': float(np.mean(values)),
            'ci95': float(1.96 * population_std / math.sqrt(20)),
            'population_std': population_std,
            'values': values.tolist(),
        }
    return result


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-dir', type=Path, required=True)
    parser.add_argument('--mm-dir', type=Path, required=True)
    parser.add_argument('--main-manifest', type=Path, default=Path('output/interhuman_fair_requests_v3/manifest.json'))
    parser.add_argument('--mm-manifest', type=Path, default=Path('output/interhuman_fair_mm_requests/manifest.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-non-hhc-protocol', action='store_true')
    parser.add_argument('--skip-generation-rehash', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    main_paths = sorted(args.main_dir.resolve().glob('replication_[0-9][0-9].json'))
    mm_paths = sorted(args.mm_dir.resolve().glob('replication_[0-9][0-9].json'))
    result = summarize_records(
        [_load(path) for path in main_paths],
        [_load(path) for path in mm_paths],
        _load(args.main_manifest.resolve()),
        _load(args.mm_manifest.resolve()),
        require_hhc_common50=not args.allow_non_hhc_protocol,
        verify_generation_hashes=not args.skip_generation_rehash,
    )
    result.update({
        'main_score_files': [str(path.resolve()) for path in main_paths],
        'mm_score_files': [str(path.resolve()) for path in mm_paths],
        'main_manifest': str(args.main_manifest.resolve()),
        'main_manifest_sha256': sha256_file(args.main_manifest.resolve()),
        'mm_manifest': str(args.mm_manifest.resolve()),
        'mm_manifest_sha256': sha256_file(args.mm_manifest.resolve()),
    })
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
