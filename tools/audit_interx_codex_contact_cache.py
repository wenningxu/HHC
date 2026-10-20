#!/usr/bin/env python
"""Strictly audit a Codex-labelled Inter-X contact cache and its captions."""

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.extract_val_contacts import (  # noqa: E402
    canonical_sha256,
    load_caption_requests,
    sha256_file,
    sha256_text,
)
from tools.audit_interx_contact_pairs import validate_record  # noqa: E402
from utils.paramUtil import CANDIDATES  # noqa: E402


SCHEMA = 'hhc_interx_contact_pairs_codex_strict_v3'
AUDIT_SCHEMA = 'hhc_interx_codex_contact_cache_audit_v1'


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs-dir', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument(
        '--text-dir', type=Path,
        default=Path('dataset/interx/texts_processed'),
    )
    parser.add_argument(
        '--requests', type=Path,
        help='Frozen fair-evaluation requests used to select exact captions.',
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--allow-incomplete', action='store_true',
        help='Accept a structurally valid running prefix as partial.',
    )
    parser.add_argument('--print-full', action='store_true')
    return parser.parse_args()


def compare_fields(actual, expected):
    return {
        key: {'actual': actual.get(key), 'expected': value}
        for key, value in expected.items()
        if actual.get(key) != value
    }


def resolve_hashed_artifact(record):
    if not isinstance(record, dict):
        return {'status': 'invalid_record', 'record': record}
    path_value = record.get('path')
    expected_hash = record.get('sha256')
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        return {'status': 'invalid_record', 'record': record}
    path = Path(path_value)
    if path.is_file() and sha256_file(path) == expected_hash:
        return {
            'status': 'exact_path_match', 'path': str(path.resolve()),
            'sha256': expected_hash,
        }
    # A versioned sibling may preserve the exact source after the active entry
    # point evolves.  Accept it only by content hash, never by filename alone.
    candidates = sorted(path.parent.glob(f'{path.stem}*{path.suffix}'))
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == expected_hash:
            return {
                'status': 'content_addressed_archive_match',
                'path': str(candidate.resolve()), 'sha256': expected_hash,
                'recorded_path': str(path),
            }
    return {
        'status': 'missing_or_hash_mismatch', 'recorded_path': str(path),
        'sha256': expected_hash,
    }


def summarize_values(values):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = 0.95 * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    p95 = values[lower] + (values[upper] - values[lower]) * (
        position - lower
    )
    return {
        'total': math.fsum(values),
        'mean_per_sample': math.fsum(values) / len(values),
        'median_per_sample': (
            values[len(values) // 2]
            if len(values) % 2 else
            (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
        ),
        'p95_per_sample': p95,
        'samples': len(values),
    }


def manifest_wall_clock(manifest, batch_paths):
    """Recover the end-to-end annotation span from extractor timestamps."""
    created_text = manifest.get('created_at_utc')
    completed_text = manifest.get('completed_at_utc')
    if not isinstance(created_text, str) or not isinstance(completed_text, str):
        return None
    created = datetime.fromisoformat(created_text)
    completed = datetime.fromisoformat(completed_text)
    if created.tzinfo is None or completed.tzinfo is None:
        raise ValueError('annotation manifest timestamps must be timezone-aware')
    created = created.astimezone(timezone.utc)
    completed = completed.astimezone(timezone.utc)
    total_seconds = (completed - created).total_seconds()
    if not math.isfinite(total_seconds) or total_seconds < 0:
        raise ValueError('annotation manifest wall-clock span is invalid')
    mtimes = [path.stat().st_mtime for path in batch_paths]
    tolerance_seconds = 2.0
    batch_window_inside_manifest = bool(mtimes) and (
        min(mtimes) >= created.timestamp() - tolerance_seconds
        and max(mtimes) <= completed.timestamp() + tolerance_seconds
    )
    return {
        'source': 'extraction_manifest_created_at_utc_to_completed_at_utc',
        'created_at_utc': created.isoformat(),
        'completed_at_utc': completed.isoformat(),
        'total_seconds': total_seconds,
        'total_hours': total_seconds / 3600.0,
        'batch_files': len(batch_paths),
        'batch_mtime_window_inside_manifest': batch_window_inside_manifest,
        'interpretation': (
            'end-to-end elapsed span including retries, resume/pause overhead '
            'and orchestration; not a per-request model latency estimate'
        ),
    }


def failed_attempt_usage(failure):
    """Recover usage recorded directly or inside the timed wrapper error."""
    metadata = failure.get('cli_metadata') or {}
    wall_seconds = metadata.get('wall_seconds')
    tokens_used = metadata.get('tokens_used')
    message = failure.get('error_message')
    try:
        wrapper = json.loads(message) if isinstance(message, str) else {}
    except json.JSONDecodeError:
        wrapper = {}
    if wall_seconds is None:
        wall_seconds = wrapper.get('wall_seconds')
    if tokens_used is None:
        tokens_used = wrapper.get('tokens_used')
    underlying = wrapper.get('underlying_error_message')
    try:
        underlying = json.loads(underlying) \
            if isinstance(underlying, str) else {}
    except json.JSONDecodeError:
        underlying = {}
    if tokens_used is None:
        tokens_used = underlying.get('tokens_used')
    return wall_seconds, tokens_used


def main():
    args = parse_args()
    pairs_dir = args.pairs_dir.resolve()
    split = args.split.resolve()
    text_dir = args.text_dir.resolve()
    requests = args.requests.resolve() if args.requests else None
    records, caption_source = load_caption_requests(
        split, text_dir, requests,
    )
    expected = {row['sample_name']: row for row in records}
    expected_names = list(expected)
    expected_set = set(expected)

    manifest_path = pairs_dir / '_extraction_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    split_names = [
        line.strip() for line in split.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    expected_manifest = {
        'schema': SCHEMA,
        'split_sha256': sha256_file(split),
        'split_count': len(split_names),
        'selection_offset': 0,
        'selection_limit': None,
        'sample_count': len(records),
        'selected_names_sha256': canonical_sha256(expected_names),
        'selected_captions_sha256': canonical_sha256(
            [row['caption'] for row in records]
        ),
        'caption_source': caption_source['caption_source'],
        'requests': caption_source.get('requests'),
        'requests_sha256': caption_source.get('requests_sha256'),
        'provider': 'openai_codex_cli_chatgpt_login',
        'strict_schema': True,
        'api_key_recorded': False,
        'authentication_secret_recorded': False,
    }
    if requests is not None:
        expected_manifest.update({
            'request_names_sha256': caption_source['request_names_sha256'],
            'request_captions_sha256': caption_source[
                'request_captions_sha256'
            ],
        })
    manifest_mismatches = compare_fields(manifest, expected_manifest)
    if not isinstance(manifest.get('requested_model'), str):
        manifest_mismatches['requested_model'] = {
            'actual': manifest.get('requested_model'), 'expected': 'nonempty string',
        }
    if manifest.get('reasoning_effort') not in {'low', 'medium', 'high'}:
        manifest_mismatches['reasoning_effort'] = {
            'actual': manifest.get('reasoning_effort'),
            'expected': 'low|medium|high',
        }

    runtime_manifest_keys = {
        'created_at_utc', 'status', 'completed_count',
        'completed_batches_this_run', 'total_tokens_used', 'updated_at_utc',
        'completed_at_utc', 'failed_batch',
    }
    recomputed_contract = canonical_sha256({
        key: manifest[key] for key in manifest
        if key not in runtime_manifest_keys and key != 'contract_sha256'
    })
    contract_hash_matches = (
        manifest.get('contract_sha256') == recomputed_contract
    )
    implementation_resolution = {
        label: resolve_hashed_artifact(record)
        for label, record in manifest.get('implementation', {}).items()
    }
    implementation_resolved = bool(implementation_resolution) and all(
        record['status'] in {
            'exact_path_match', 'content_addressed_archive_match',
        }
        for record in implementation_resolution.values()
    )
    output_schema_path = Path(str(manifest.get('output_schema', '')))
    output_schema_resolved = (
        output_schema_path.is_file()
        and sha256_file(output_schema_path)
        == manifest.get('output_schema_sha256')
    )

    pair_paths = {path.stem: path for path in pairs_dir.glob('*.pkl')}
    metadata_paths = {
        path.stem: path for path in (pairs_dir / '_metadata').glob('*.json')
    }
    completed = expected_set & set(pair_paths) & set(metadata_paths)
    missing_pairs = sorted(expected_set - set(pair_paths))
    missing_metadata = sorted(expected_set - set(metadata_paths))
    extra_pairs = sorted(set(pair_paths) - expected_set)
    extra_metadata = sorted(set(metadata_paths) - expected_set)

    valid_region_ids = {
        int(region_id) for ids in CANDIDATES.values() for region_id in ids
    }
    metadata_errors = {}
    payload_errors = {}
    batch_errors = {}
    structured_rows = 0
    empty_samples = 0
    for name in sorted(completed):
        row = expected[name]
        metadata_path = metadata_paths[name]
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        expected_metadata = {
            'schema': SCHEMA,
            'sample_name': name,
            'caption': row['caption'],
            'caption_sha256': sha256_text(row['caption']),
            'caption_source': caption_source['caption_source'],
            'request_index': row['request_index'],
            'provider': manifest.get('provider'),
            'requested_model': manifest.get('requested_model'),
            'response_model_version': manifest.get('requested_model'),
            'reasoning_effort': manifest.get('reasoning_effort'),
            'prompt_template_sha256': manifest.get('prompt_template_sha256'),
            'status': 'success',
            'api_key_recorded': False,
            'authentication_secret_recorded': False,
        }
        if requests is None:
            expected_metadata.update({
                'text_path': row['text_path'],
                'text_sha256': row['text_sha256'],
            })
        else:
            expected_metadata.update({
                'requests': caption_source['requests'],
                'requests_sha256': caption_source['requests_sha256'],
            })
        mismatches = compare_fields(metadata, expected_metadata)
        if requests is not None:
            writer = resolve_hashed_artifact(
                metadata.get('request_binding_writer')
            )
            if writer['status'] not in {
                'exact_path_match', 'content_addressed_archive_match',
            }:
                mismatches['request_binding_writer'] = {
                    'actual': metadata.get('request_binding_writer'),
                    'expected': 'hash-resolved binding writer',
                }

        batch_path_value = metadata.get('batch_metadata')
        batch_path = Path(batch_path_value).resolve() \
            if isinstance(batch_path_value, str) else None
        if batch_path is None or not batch_path.is_file():
            mismatches['batch_metadata'] = {
                'actual': batch_path_value, 'expected': 'existing batch sidecar',
            }
        else:
            batch_hash = sha256_file(batch_path)
            if metadata.get('batch_metadata_sha256') != batch_hash:
                mismatches['batch_metadata_sha256'] = {
                    'actual': metadata.get('batch_metadata_sha256'),
                    'expected': batch_hash,
                }
            batch = json.loads(batch_path.read_text(encoding='utf-8'))
            problems = []
            if batch.get('status') != 'success':
                problems.append('batch_not_success')
            if batch.get('contract_sha256') != manifest.get('contract_sha256'):
                problems.append('batch_contract_mismatch')
            if name not in batch.get('sample_names', []):
                problems.append('sample_missing_from_batch')
            if problems:
                batch_errors[name] = problems

        if mismatches:
            metadata_errors[name] = mismatches
        try:
            with open(pair_paths[name], 'rb') as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, list):
                raise TypeError('payload_not_list')
            errors = sorted({
                error for contact in payload
                for error in validate_record(contact, valid_region_ids)
            })
            if errors:
                payload_errors[name] = errors
            else:
                structured_rows += len(payload)
                empty_samples += int(not payload)
                if metadata.get('structured_contact_rows') != len(payload):
                    metadata_errors.setdefault(name, {})[
                        'structured_contact_rows'
                    ] = {
                        'actual': metadata.get('structured_contact_rows'),
                        'expected': len(payload),
                    }
        except Exception as error:
            payload_errors[name] = [
                f'load_failed:{type(error).__name__}:{error}'
            ]

    batch_paths = sorted((pairs_dir / '_batches').glob('*.json'))
    end_to_end_wall_clock = manifest_wall_clock(manifest, batch_paths)
    failed_attempts = 0
    retry_batches = 0
    successful_batch_tokens = 0
    successful_batches = 0
    sample_wall_seconds = []
    sample_tokens_with_failures = []
    wall_timed_attempts = 0
    wall_untimed_attempts = 0
    token_reported_attempts = 0
    invalid_batch_sidecars = {}
    for path in batch_paths:
        batch = json.loads(path.read_text(encoding='utf-8'))
        failures = batch.get('attempt_failures', [])
        failed_attempts += len(failures)
        retry_batches += int(bool(failures) or int(batch.get('attempts', 0)) > 1)
        if batch.get('status') != 'success':
            invalid_batch_sidecars[path.name] = ['batch_not_success']
        else:
            successful_batches += 1
        successful_batch_tokens += int(
            batch.get('cli_metadata', {}).get('tokens_used') or 0
        )
        batch_size = len(batch.get('sample_names', []))
        if batch_size < 1:
            invalid_batch_sidecars.setdefault(path.name, []).append(
                'empty_sample_names'
            )
            continue
        final_metadata = batch.get('cli_metadata') or {}
        attempt_usage = [(
            final_metadata.get('wall_seconds'),
            final_metadata.get('tokens_used'),
        )]
        attempt_usage.extend(failed_attempt_usage(failure) for failure in failures)
        batch_wall_seconds = 0.0
        batch_tokens = 0.0
        has_wall = False
        has_tokens = False
        for wall_seconds, tokens_used in attempt_usage:
            if wall_seconds is None:
                wall_untimed_attempts += 1
            else:
                wall_seconds = float(wall_seconds)
                if not math.isfinite(wall_seconds) or wall_seconds < 0:
                    invalid_batch_sidecars.setdefault(path.name, []).append(
                        'invalid_attempt_wall_seconds'
                    )
                else:
                    batch_wall_seconds += wall_seconds
                    wall_timed_attempts += 1
                    has_wall = True
            if tokens_used is not None:
                tokens_used = float(tokens_used)
                if not math.isfinite(tokens_used) or tokens_used < 0:
                    invalid_batch_sidecars.setdefault(path.name, []).append(
                        'invalid_attempt_tokens'
                    )
                else:
                    batch_tokens += tokens_used
                    token_reported_attempts += 1
                    has_tokens = True
        if has_wall:
            sample_wall_seconds.extend(
                [batch_wall_seconds / batch_size] * batch_size
            )
        if has_tokens:
            sample_tokens_with_failures.extend(
                [batch_tokens / batch_size] * batch_size
            )

    complete = (
        manifest.get('status') == 'complete'
        and manifest.get('completed_count') == len(records)
        and len(completed) == len(records)
        and not missing_pairs
        and not missing_metadata
    )
    integrity_passed = not any((
        manifest_mismatches,
        extra_pairs,
        extra_metadata,
        metadata_errors,
        payload_errors,
        batch_errors,
        invalid_batch_sidecars,
    )) and contract_hash_matches and implementation_resolved \
        and output_schema_resolved
    token_count_matches = (
        successful_batch_tokens == int(manifest.get('total_tokens_used') or 0)
    )
    if not token_count_matches:
        integrity_passed = False
    if complete and integrity_passed:
        status = 'pass'
    elif args.allow_incomplete and integrity_passed:
        status = 'partial'
    else:
        status = 'fail'

    report = {
        'schema': AUDIT_SCHEMA,
        'status': status,
        'complete': complete,
        'integrity_passed': integrity_passed,
        'caption_binding': caption_source['caption_source'],
        'inputs': {
            'pairs_dir': str(pairs_dir),
            'manifest': str(manifest_path),
            'manifest_sha256': sha256_file(manifest_path),
            'split': str(split),
            'split_sha256': sha256_file(split),
            'requests': str(requests) if requests else None,
            'requests_sha256': sha256_file(requests) if requests else None,
        },
        'counts': {
            'expected_samples': len(records),
            'completed_samples': len(completed),
            'structured_contact_rows': structured_rows,
            'empty_samples': empty_samples,
            'batch_files': len(batch_paths),
            'retry_batches': retry_batches,
            'failed_attempts': failed_attempts,
            'successful_batch_tokens': successful_batch_tokens,
            'manifest_total_tokens_used': manifest.get('total_tokens_used'),
        },
        'resource_usage': {
            'scope': (
                'amortized share of each Codex CLI batch across its samples; '
                'includes failed attempts when their timing/token metadata is '
                'available'
            ),
            'invocations': {
                'successful_attempts': successful_batches,
                'failed_attempts': failed_attempts,
                'wall_timed_attempts': wall_timed_attempts,
                'wall_untimed_attempts': wall_untimed_attempts,
                'token_reported_attempts': token_reported_attempts,
            },
            'wall_seconds': summarize_values(sample_wall_seconds),
            'end_to_end_wall_clock': end_to_end_wall_clock,
            'tokens': summarize_values(sample_tokens_with_failures),
        },
        'implementation': {
            'path': str(Path(__file__).resolve()),
            'bytes': Path(__file__).resolve().stat().st_size,
            'sha256': sha256_file(Path(__file__).resolve()),
        },
        'token_count_matches': token_count_matches,
        'contract_hash_matches': contract_hash_matches,
        'recomputed_contract_sha256': recomputed_contract,
        'implementation_resolved': implementation_resolved,
        'implementation_resolution': implementation_resolution,
        'output_schema_resolved': output_schema_resolved,
        'manifest_mismatches': manifest_mismatches,
        'missing_pairs': missing_pairs,
        'missing_metadata': missing_metadata,
        'extra_pairs': extra_pairs,
        'extra_metadata': extra_metadata,
        'metadata_errors': metadata_errors,
        'payload_errors': payload_errors,
        'batch_errors': batch_errors,
        'invalid_batch_sidecars': invalid_batch_sidecars,
    }
    output = args.output.resolve()
    atomic_json(output, report)
    if args.print_full:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps({
            'output': str(output),
            'status': status,
            'complete': complete,
            'integrity_passed': integrity_passed,
            **report['counts'],
        }, indent=2, sort_keys=True))
    if status == 'fail':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
