#!/usr/bin/env python
"""Audit every prerequisite for the strict paper-optimizer ablation."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


STRICT_SCHEMA = 'hhc_interx_contact_pairs_codex_strict_v3'
CONVERSION_RESUME_CONTRACT = (
    'existing NPZ arrays must exactly equal the deterministic current '
    'request/generation conversion; missing files are atomically created; '
    'stale files fail closed'
)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = path.resolve()
    if not path.is_file():
        return {'path': str(path), 'exists': False}
    return {
        'path': str(path), 'exists': True, 'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--motion-dir', type=Path,
        default=Path(
            'output/optimization_inputs/'
            'hhc_formal_raw12288_repaired_mgt_common50_no_contact_rep0'
        ),
    )
    parser.add_argument(
        '--generation-manifest', type=Path,
        default=Path(
            'output/optimization_inputs/'
            'hhc_formal_raw12288_repaired_mgt_common50_no_contact_rep0/'
            'generation_records.jsonl'
        ),
    )
    parser.add_argument(
        '--conversion-manifest', type=Path,
        default=Path(
            'output/optimization_inputs/'
            'hhc_formal_raw12288_repaired_mgt_common50_no_contact_rep0/'
            '_conversion_manifest.json'
        ),
    )
    parser.add_argument(
        '--contact-dir', type=Path,
        default=Path('dataset/interx/contact_pairs_codex_luna_strict_v3_test'),
    )
    parser.add_argument(
        '--requests', type=Path,
        default=Path('output/fair_interx_requests/replication_00.npz'),
    )
    parser.add_argument(
        '--expected-split', type=Path,
        default=Path('dataset/interx/splits/test.txt'),
    )
    parser.add_argument(
        '--contact-cache-audit', type=Path,
        default=Path('output/interx_luna_test_frozen_cache_audit.json'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/optimization_ablation_readiness.json'),
    )
    parser.add_argument('--expected-samples', type=int, default=1696)
    parser.add_argument('--require-ready', action='store_true')
    args = parser.parse_args()

    motion_dir = args.motion_dir.resolve()
    generation_manifest = args.generation_manifest.resolve()
    conversion_manifest = args.conversion_manifest.resolve()
    contact_dir = args.contact_dir.resolve()
    contact_manifest_path = contact_dir / '_extraction_manifest.json'
    requests = args.requests.resolve()
    expected_split = args.expected_split.resolve()
    contact_cache_audit_path = args.contact_cache_audit.resolve()
    blockers = []

    for label, path in (
        ('frozen_requests_missing', requests),
        ('official_test_split_missing', expected_split),
        ('generation_records_missing', generation_manifest),
        ('conversion_manifest_missing', conversion_manifest),
        ('strict_test_contact_audit_missing', contact_cache_audit_path),
    ):
        if not path.is_file():
            blockers.append(label)

    paired_npz_files = sorted(motion_dir.glob('*_pred_p?.npz')) if motion_dir.is_dir() else []
    expected_npz_files = args.expected_samples * 2
    if len(paired_npz_files) != expected_npz_files:
        blockers.append('converted_generation_npz_incomplete')

    conversion = None
    conversion_manifest_mismatches = {}
    generation_record_integrity = {
        'records': 0,
        'unique_sample_keys': 0,
        'request_indices_complete': False,
        'all_npz_paths_and_hashes_verified': False,
        'errors': [],
    }
    if conversion_manifest.is_file():
        try:
            conversion = json.loads(
                conversion_manifest.read_text(encoding='utf-8')
            )
        except (OSError, json.JSONDecodeError):
            blockers.append('conversion_manifest_invalid_json')
        if conversion is not None:
            request_hash = sha256_file(requests) if requests.is_file() else None
            generation_value = conversion.get('generations')
            generation_path = (
                Path(generation_value) if isinstance(generation_value, str)
                and generation_value else Path('__missing_generation__')
            )
            generation_attrs = conversion.get(
                'generation_hdf5_attributes', {}
            )
            expected_values = {
                'schema': 'hhc_fair_h5_to_optimizer_npz_v1',
                'status': 'complete',
                'samples': args.expected_samples,
                'repeat_id': 0,
                'requests_sha256': request_hash,
                'generation_records': str(generation_manifest),
                'resume_contract': CONVERSION_RESUME_CONTRACT,
            }
            for key, expected in expected_values.items():
                actual = conversion.get(key)
                if key == 'generation_records' and isinstance(actual, str):
                    actual = str(Path(actual).resolve())
                if actual != expected:
                    conversion_manifest_mismatches[key] = {
                        'actual': actual, 'expected': expected,
                    }
            expected_hdf_attrs = {
                'contact_guidance': False,
                'contact_timing_source': 'none',
                'contact_timing_file': '',
                'contact_timing_file_sha256': '',
                'contact_map': '',
                'contact_map_sha256': '',
                'sampler': 'sample_intra_cas',
                'sampling_steps': 50,
                'requests_sha256': request_hash,
            }
            for key, expected in expected_hdf_attrs.items():
                actual = generation_attrs.get(key)
                if actual != expected:
                    conversion_manifest_mismatches[f'hdf5/{key}'] = {
                        'actual': actual, 'expected': expected,
                    }
            bound_files = {
                'generation_records_sha256': generation_manifest,
                'generations_sha256': generation_path,
            }
            for hash_key, path in bound_files.items():
                actual = conversion.get(hash_key)
                expected = sha256_file(path) if path.is_file() else None
                if actual != expected:
                    conversion_manifest_mismatches[hash_key] = {
                        'actual': actual, 'expected': expected,
                    }
            expected_implementation = {
                'converter': Path(
                    'tools/export_fair_h5_to_optimizer_npz.py'
                ).resolve(),
                'rotation_conversions': Path(
                    'utils/rotation_conversions.py'
                ).resolve(),
            }
            implementation = conversion.get('implementation', {})
            if set(implementation) != set(expected_implementation):
                conversion_manifest_mismatches['implementation/set'] = {
                    'actual': sorted(implementation),
                    'expected': sorted(expected_implementation),
                }
            for name, path in expected_implementation.items():
                record = implementation.get(name, {})
                expected_record = {
                    'path': str(path),
                    'bytes': path.stat().st_size,
                    'sha256': sha256_file(path),
                }
                if record != expected_record:
                    conversion_manifest_mismatches[
                        f'implementation/{name}'
                    ] = {'actual': record, 'expected': expected_record}
            if conversion_manifest_mismatches:
                blockers.append('conversion_manifest_contract_mismatch')

    if generation_manifest.is_file():
        records = []
        try:
            with open(generation_manifest, encoding='utf-8') as handle:
                records = [
                    json.loads(line) for line in handle if line.strip()
                ]
        except (OSError, json.JSONDecodeError) as error:
            generation_record_integrity['errors'].append(
                f'invalid generation ledger: {error}'
            )
        generation_record_integrity['records'] = len(records)
        sample_keys = [record.get('sample_key') for record in records]
        generation_record_integrity['unique_sample_keys'] = len(set(sample_keys))
        request_indices = [record.get('request_index') for record in records]
        generation_record_integrity['request_indices_complete'] = (
            request_indices == list(range(args.expected_samples))
        )
        request_hash = sha256_file(requests) if requests.is_file() else None
        for index, record in enumerate(records):
            sample_key = record.get('sample_key')
            if (
                record.get('repeat_id') != 0
                or record.get('requests_sha256') != request_hash
                or not isinstance(sample_key, str)
            ):
                generation_record_integrity['errors'].append(
                    f'record {index}: identity/request mismatch'
                )
                continue
            for person in ('p1', 'p2'):
                path = Path(record.get(person, ''))
                expected_name = f'{sample_key}_pred_{person}.npz'
                if (
                    not path.is_file()
                    or path.resolve().parent != motion_dir
                    or path.name != expected_name
                    or record.get(f'{person}_sha256') != sha256_file(path)
                ):
                    generation_record_integrity['errors'].append(
                        f'record {index}: stale {person}'
                    )
            if len(generation_record_integrity['errors']) >= 20:
                break
        generation_record_integrity[
            'all_npz_paths_and_hashes_verified'
        ] = (
            len(records) == args.expected_samples
            and len(set(sample_keys)) == args.expected_samples
            and generation_record_integrity['request_indices_complete']
            and not generation_record_integrity['errors']
        )
        if not generation_record_integrity[
            'all_npz_paths_and_hashes_verified'
        ]:
            blockers.append('generation_record_npz_integrity_mismatch')

    contact_manifest = None
    mismatches = {}
    if not contact_manifest_path.is_file():
        blockers.append('strict_test_contact_manifest_missing')
    else:
        try:
            contact_manifest = json.loads(
                contact_manifest_path.read_text(encoding='utf-8')
            )
        except (OSError, json.JSONDecodeError):
            blockers.append('strict_test_contact_manifest_invalid_json')
        if contact_manifest is not None:
            expected_contact = {
                'schema': STRICT_SCHEMA,
                'status': 'complete',
                'sample_count': args.expected_samples,
                'completed_count': args.expected_samples,
            }
            mismatches = {
                key: {'actual': contact_manifest.get(key), 'expected': value}
                for key, value in expected_contact.items()
                if contact_manifest.get(key) != value
            }
            if mismatches:
                blockers.append('strict_test_contact_manifest_contract_mismatch')

    contact_pickles = sorted(contact_dir.glob('*.pkl')) if contact_dir.is_dir() else []
    contact_metadata = sorted((contact_dir / '_metadata').glob('*.json')) \
        if (contact_dir / '_metadata').is_dir() else []
    if contact_manifest_path.is_file() and (
        len(contact_pickles) != args.expected_samples
        or len(contact_metadata) != args.expected_samples
    ):
        blockers.append('strict_test_contact_records_incomplete')

    contact_cache_audit = None
    contact_cache_audit_mismatches = {}
    if contact_cache_audit_path.is_file():
        try:
            contact_cache_audit = json.loads(
                contact_cache_audit_path.read_text(encoding='utf-8')
            )
        except (OSError, json.JSONDecodeError):
            blockers.append('strict_test_contact_audit_invalid_json')
        if contact_cache_audit is not None:
            audit_inputs = contact_cache_audit.get('inputs', {})
            audit_counts = contact_cache_audit.get('counts', {})
            expected_audit = {
                'schema': 'hhc_interx_codex_contact_cache_audit_v1',
                'status': 'pass',
                'complete': True,
                'integrity_passed': True,
                'caption_binding': 'frozen_fair_evaluation_requests',
            }
            contact_cache_audit_mismatches = {
                key: {
                    'actual': contact_cache_audit.get(key),
                    'expected': value,
                }
                for key, value in expected_audit.items()
                if contact_cache_audit.get(key) != value
            }
            binding_checks = {
                'pairs_dir': (
                    audit_inputs.get('pairs_dir'), str(contact_dir)
                ),
                'split_sha256': (
                    audit_inputs.get('split_sha256'),
                    sha256_file(expected_split) if expected_split.is_file() else None,
                ),
                'requests_sha256': (
                    audit_inputs.get('requests_sha256'),
                    sha256_file(requests) if requests.is_file() else None,
                ),
                'manifest_sha256': (
                    audit_inputs.get('manifest_sha256'),
                    sha256_file(contact_manifest_path)
                    if contact_manifest_path.is_file() else None,
                ),
                'expected_samples': (
                    audit_counts.get('expected_samples'), args.expected_samples,
                ),
                'completed_samples': (
                    audit_counts.get('completed_samples'), args.expected_samples,
                ),
            }
            for key, (actual, expected) in binding_checks.items():
                if actual != expected:
                    contact_cache_audit_mismatches[key] = {
                        'actual': actual, 'expected': expected,
                    }
            if contact_cache_audit_mismatches:
                blockers.append('strict_test_contact_audit_contract_mismatch')

    codex_login_available = False
    if shutil.which('codex'):
        status = subprocess.run(
            ['codex', 'login', 'status'], capture_output=True, text=True,
        )
        codex_login_available = (
            status.returncode == 0
            and 'Logged in using ChatGPT' in (status.stdout + status.stderr)
        )

    payload = {
        'schema': 'hhc_optimization_ablation_readiness_v1',
        'status': 'ready' if not blockers else 'pending',
        'ready': not blockers,
        'blockers': sorted(set(blockers)),
        'expected_samples': args.expected_samples,
        'converted_npz_files': len(paired_npz_files),
        'expected_converted_npz_files': expected_npz_files,
        'contact_pickle_files': len(contact_pickles),
        'contact_metadata_files': len(contact_metadata),
        'codex_chatgpt_login_available': codex_login_available,
        'authentication_secret_recorded': False,
        'contact_manifest_mismatches': mismatches,
        'contact_cache_audit_mismatches': contact_cache_audit_mismatches,
        'conversion_manifest_mismatches': conversion_manifest_mismatches,
        'generation_record_integrity': generation_record_integrity,
        'implementation': artifact(Path(__file__)),
        'artifacts': {
            'requests': artifact(requests),
            'expected_split': artifact(expected_split),
            'generation_records': artifact(generation_manifest),
            'conversion_manifest': artifact(conversion_manifest),
            'contact_extraction_manifest': artifact(contact_manifest_path),
            'contact_cache_audit': artifact(contact_cache_audit_path),
        },
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_ready and blockers:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
