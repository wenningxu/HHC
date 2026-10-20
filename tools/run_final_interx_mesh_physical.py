#!/usr/bin/env python
"""Run and audit one common rep-0 VolumetricSMPL protocol for final methods."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'hhc_final_interx_mesh_physical_main_table_v1'
METHOD_LABELS = (
    'hhc_historical_plus_paper_mask_ratio_no_contact_common50',
    'hhc_historical_plus_paper_mask_ratio_contact_common50',
    'intermask_official_exact_common50',
    'interact2ar_official_fair_no_gt_first_frame',
    'coshmdm_interx_retrained_2daycap_seed123',
    'timotion_transformer_lpa_interx_retrained_fair',
)
METRIC_KEYS = (
    'penetrating_vertex_fraction',
    'collision_frame_rate',
    'penetration_depth_sum_m_per_frame',
    'paper_eq6_squared_sum_m2_per_frame',
    'mean_depth_m_on_penetrating_vertices',
    'max_depth_m',
    'mean_foot_sliding_m_per_frame',
)
VOLUMETRIC_CHECKPOINT_SHA256 = {
    'male': '96b22f6409e67dc950dafdd83bb8dc4779381ecc2b2c752e1c6fa3a205b2e18c',
    'female': '2b582f009c2daeec50c90b5f39689a248bac96a4cc9983548ad4be677b72ccdb',
}
CONVERSION_RESUME_CONTRACT = (
    'existing NPZ arrays must exactly equal the deterministic current '
    'request/generation conversion; missing files are atomically created; '
    'stale files fail closed'
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def artifact_record(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def validate_artifact(record, label):
    path = Path(record.get('path', ''))
    require(path.is_file(), f'{label}: missing artifact {path}')
    require(
        record.get('sha256') == sha256_file(path),
        f'{label}: stale artifact hash for {path}',
    )
    return path.resolve()


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def run(command):
    print('$ ' + ' '.join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_conversion(conversion, requests, generations, expected_samples):
    label = 'conversion'
    require(
        conversion.get('schema') == 'hhc_fair_h5_to_optimizer_npz_v1'
        and conversion.get('status') == 'complete'
        and conversion.get('repeat_id') == 0
        and conversion.get('samples') == expected_samples
        and conversion.get('shape_gender_source') == 'frozen request arrays'
        and conversion.get('mocap_framerate') == 30,
        f'{label}: conversion contract changed or is incomplete',
    )
    require(
        conversion.get('resume_contract') == CONVERSION_RESUME_CONTRACT,
        f'{label}: strict per-array resume contract changed',
    )
    require(
        Path(conversion.get('requests', '')).resolve() == requests.resolve()
        and conversion.get('requests_sha256') == sha256_file(requests)
        and Path(conversion.get('generations', '')).resolve()
        == generations.resolve()
        and conversion.get('generations_sha256') == sha256_file(generations),
        f'{label}: request/generation binding differs',
    )
    records_path = Path(conversion.get('generation_records', ''))
    require(
        records_path.is_file()
        and conversion.get('generation_records_sha256')
        == sha256_file(records_path),
        f'{label}: generation record ledger is absent or stale',
    )
    for name, record in conversion.get('implementation', {}).items():
        validate_artifact(record, f'{label}/implementation/{name}')
    records = read_jsonl(records_path)
    require(len(records) == expected_samples, f'{label}: record count differs')
    require(
        [record.get('request_index') for record in records]
        == list(range(expected_samples)),
        f'{label}: request indices are incomplete or reordered',
    )
    for record in records:
        require(
            record.get('repeat_id') == 0
            and record.get('requests_sha256') == conversion['requests_sha256'],
            f'{label}: per-sample request/repeat binding differs',
        )
        for person in ('p1', 'p2'):
            path = Path(record.get(person, ''))
            require(
                path.is_file()
                and record.get(f'{person}_sha256') == sha256_file(path),
                f'{label}: stale {person} artifact for '
                f'{record.get("sample_key")}',
            )
    return records, records_path


def validate_metrics(metrics, conversion_records, folder, output_dir,
                     expected_samples, seed, query_chunk, device):
    label = 'mesh_metrics'
    require(
        metrics.get('schema') == 'hhc_volumetric_smpl_physical_metrics_v1'
        and metrics.get('status') == 'complete'
        and metrics.get('sample_count') == expected_samples
        and metrics.get('base_seed') == seed
        and metrics.get('max_frames') == 150
        and metrics.get('query_chunk') == query_chunk
        and metrics.get('device') == device
        and Path(metrics.get('folder', '')).resolve() == folder.resolve(),
        f'{label}: scoring contract changed or is incomplete',
    )
    checkpoint_records = metrics.get('volumetric_smpl_checkpoints', {})
    require(
        set(checkpoint_records) == {'male', 'female'}
        and {
            gender: record.get('sha256')
            for gender, record in checkpoint_records.items()
        } == VOLUMETRIC_CHECKPOINT_SHA256,
        f'{label}: released male/female VolumetricSMPL weights changed',
    )
    for gender, record in checkpoint_records.items():
        validate_artifact(record, f'{label}/checkpoint/{gender}')
    for name, record in metrics.get('implementation', {}).items():
        validate_artifact(record, f'{label}/implementation/{name}')
    require(
        metrics.get('scoring_contract', {}).get('implementation')
        == metrics.get('implementation')
        and metrics.get('scoring_contract', {}).get(
            'volumetric_smpl_checkpoints'
        ) == checkpoint_records
        and metrics.get('scoring_contract', {}).get('base_seed') == seed
        and metrics.get('scoring_contract', {}).get('device') == device
        and metrics.get('scoring_contract', {}).get('max_frames') == 150
        and metrics.get('scoring_contract', {}).get('query_chunk')
        == query_chunk
        and metrics.get('scoring_contract_sha256')
        == canonical_sha256(metrics.get('scoring_contract', {})),
        f'{label}: embedded scoring contract differs from the manifest',
    )
    aggregate = metrics.get('aggregate', {})
    require(set(aggregate) == set(METRIC_KEYS), f'{label}: metric set changed')
    for key, record in aggregate.items():
        require(
            set(record) == {'mean', 'median'}
            and all(math.isfinite(float(value)) for value in record.values()),
            f'{label}: non-finite or incomplete aggregate {key}',
        )
    sample_paths = [Path(path) for path in metrics.get('sample_records', [])]
    require(
        len(sample_paths) == expected_samples
        and all(path.parent.resolve() == (output_dir / 'samples').resolve()
                for path in sample_paths),
        f'{label}: sample record set differs',
    )
    conversion_by_key = {
        record['sample_key']: record for record in conversion_records
    }
    require(
        len(conversion_by_key) == expected_samples,
        f'{label}: duplicate conversion sample keys',
    )
    seen = set()
    for path in sample_paths:
        record = load_json(path)
        sample_key = record.get('sample_key')
        require(
            record.get('schema') == metrics['schema']
            and record.get('repeat_id') == 0
            and record.get('scoring_contract_sha256')
            == metrics.get('scoring_contract_sha256')
            and sample_key in conversion_by_key
            and sample_key not in seen,
            f'{label}: stale/duplicate sample record {path}',
        )
        conversion = conversion_by_key[sample_key]
        require(
            record.get('input_sha256') == {
                'p1': conversion['p1_sha256'],
                'p2': conversion['p2_sha256'],
            }
            and Path(record.get('input_p1', '')).resolve()
            == Path(conversion['p1']).resolve()
            and Path(record.get('input_p2', '')).resolve()
            == Path(conversion['p2']).resolve(),
            f'{label}: score input differs for {sample_key}',
        )
        sample_metrics = record.get('metrics', {})
        require(
            set(METRIC_KEYS) <= set(sample_metrics)
            and all(math.isfinite(float(sample_metrics[key]))
                    for key in METRIC_KEYS),
            f'{label}: non-finite sample metrics for {sample_key}',
        )
        seen.add(sample_key)
    require(seen == set(conversion_by_key), f'{label}: sample coverage differs')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--fair-root', type=Path, default=Path('output/fair_eval')
    )
    parser.add_argument(
        '--requests', type=Path,
        default=Path('output/fair_interx_requests/replication_00.npz'),
    )
    parser.add_argument(
        '--output-root', type=Path,
        default=Path('output/fair_eval_mesh_physical'),
    )
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--query-chunk', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--expected-samples', type=int, default=1696)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--labels', nargs='+', choices=METHOD_LABELS)
    return parser.parse_args()


def main():
    args = parse_args()
    fair_root = args.fair_root.resolve()
    requests = args.requests.resolve()
    output_root = args.output_root.resolve()
    labels = tuple(args.labels or METHOD_LABELS)
    expected_samples = args.limit or args.expected_samples
    require(args.query_chunk > 0, 'query chunk must be positive')
    require(args.seed >= 0, 'seed must be non-negative')
    require(
        len(labels) == len(set(labels)), 'method labels must be unique'
    )
    output_root.mkdir(parents=True, exist_ok=True)
    methods = {}
    for index, label in enumerate(labels, 1):
        print(f'[{index}/{len(labels)}] {label}', flush=True)
        method_dir = fair_root / label
        generation = method_dir / 'replication_00_main.h5'
        run_manifest = method_dir / 'run_manifest.json'
        if not generation.is_file() or not run_manifest.is_file():
            raise FileNotFoundError(
                f'{label}: final generation/run manifest is incomplete'
            )
        method_output = output_root / label
        npz_dir = method_output / 'paired_smplx_npz'
        score_dir = method_output / 'volumetric_smpl'
        conversion_command = [
            sys.executable, 'tools/export_fair_h5_to_optimizer_npz.py',
            '--requests', requests,
            '--generations', generation,
            '--output-dir', npz_dir,
            '--repeat-id', '0',
            '--resume',
        ]
        score_command = [
            sys.executable, 'tools/score_volumetric_smpl_npz.py',
            '--folder', npz_dir,
            '--output-dir', score_dir,
            '--device', args.device,
            '--query-chunk', str(args.query_chunk),
            '--seed', str(args.seed),
            '--repeat-id', '0',
        ]
        if args.limit is not None:
            conversion_command.extend(('--limit', str(args.limit)))
            score_command.extend(('--limit', str(args.limit)))
        run(conversion_command)
        run(score_command)
        conversion_path = npz_dir / '_conversion_manifest.json'
        metrics_path = score_dir / 'metrics.json'
        conversion = load_json(conversion_path)
        conversion_records, records_path = validate_conversion(
            conversion, requests, generation, expected_samples
        )
        metrics = load_json(metrics_path)
        validate_metrics(
            metrics, conversion_records, npz_dir, score_dir,
            expected_samples, args.seed, args.query_chunk, args.device,
        )
        method_payload = {
            'label': label,
            'samples': expected_samples,
            'repeat_id': 0,
            'aggregate': metrics['aggregate'],
            'bindings_verified': {
                'same_frozen_requests': True,
                'exact_generation_hdf5': True,
                'request_betas_and_gender': True,
                'complete_generation_record_ledger': True,
                'complete_per_sample_score_ledger': True,
                'released_volumetric_smpl_weights': True,
                'scoring_contract': True,
            },
            'artifacts': {
                'requests': artifact_record(requests),
                'generation': artifact_record(generation),
                'method_run_manifest': artifact_record(run_manifest),
                'conversion_manifest': artifact_record(conversion_path),
                'conversion_records': artifact_record(records_path),
                'metrics': artifact_record(metrics_path),
            },
        }
        atomic_json(method_output / 'method_summary.json', method_payload)
        methods[label] = {
            **method_payload,
            'artifacts': {
                **method_payload['artifacts'],
                'method_summary': artifact_record(
                    method_output / 'method_summary.json'
                ),
            },
        }

    full_main_table = labels == METHOD_LABELS and args.limit is None
    payload = {
        'schema': SCHEMA,
        'status': 'complete' if full_main_table else 'complete_subset',
        'protocol': (
            'same frozen Inter-X replication-00 requests and generated HDF5; '
            'request beta/gender; bidirectional released VolumetricSMPL SDF; '
            'all valid requested frames up to 150'
        ),
        'replication_id': 0,
        'samples_per_method': expected_samples,
        'method_order': list(labels),
        'metric_keys': list(METRIC_KEYS),
        'base_seed': args.seed,
        'query_chunk': args.query_chunk,
        'device': args.device,
        'statistical_scope': (
            'deterministic replication-0 descriptive main-table geometry; '
            'no 20-replication confidence interval or significance claim'
        ),
        'requests': artifact_record(requests),
        'methods': methods,
        'implementation': {
            'orchestrator': artifact_record(__file__),
            'converter': artifact_record(
                ROOT / 'tools/export_fair_h5_to_optimizer_npz.py'
            ),
            'scorer': artifact_record(
                ROOT / 'tools/score_volumetric_smpl_npz.py'
            ),
            'volumetric_backend': artifact_record(
                ROOT / 'optimization/volumetric_smpl_backend.py'
            ),
            'npz_parameter_loader': artifact_record(
                ROOT / 'eval/calc_pene_from_npz.py'
            ),
            'rotation_conversions': artifact_record(
                ROOT / 'utils/rotation_conversions.py'
            ),
        },
    }
    atomic_json(output_root / 'manifest.json', payload)
    print(json.dumps({
        'status': payload['status'],
        'methods': len(methods),
        'samples_per_method': expected_samples,
        'output': str(output_root / 'manifest.json'),
    }, indent=2))


if __name__ == '__main__':
    main()
