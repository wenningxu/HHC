#!/usr/bin/env python
"""Audit cached Inter-X LLM contact pairs without making external API calls."""

import argparse
import collections
import hashlib
import json
import math
import os
import pickle
import re
from pathlib import Path

from utils.paramUtil import CANDIDATES


ALLOWED_CONTACT_TYPES = {'point', 'flat', 'wrap', 'hug'}
ALLOWED_ALIGNMENT_TYPES = {'strict', 'adaptive', 'none'}
REQUIRED_KEYS = {
    'p1_part', 'p1_ids', 'p2_part', 'p2_ids',
    'contact_type', 'align_type', 'w_p1', 'w_p2',
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_named_hash(paths):
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode('utf-8'))
        digest.update(b'\0')
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def read_split(path):
    with open(path, encoding='utf-8') as handle:
        names = [line.strip() for line in handle if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f'duplicate names in split: {path}')
    return names


def validate_record(record, valid_region_ids):
    errors = []
    if not isinstance(record, dict):
        return ['record_not_dict']
    missing = sorted(REQUIRED_KEYS - set(record))
    if missing:
        errors.append('missing_keys:' + ','.join(missing))
    for person in ('p1', 'p2'):
        ids = record.get(f'{person}_ids')
        if not isinstance(ids, list) or not ids:
            errors.append(f'{person}_ids_invalid_or_empty')
        elif any(not isinstance(value, int) or value not in valid_region_ids for value in ids):
            errors.append(f'{person}_ids_out_of_dictionary')
    if record.get('contact_type') not in ALLOWED_CONTACT_TYPES:
        errors.append('contact_type_invalid')
    if record.get('align_type') not in ALLOWED_ALIGNMENT_TYPES:
        errors.append('align_type_invalid')
    weights = (record.get('w_p1'), record.get('w_p2'))
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
           for value in weights):
        errors.append('role_weights_invalid')
    elif not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=2e-6):
        errors.append('role_weights_not_normalized')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs-dir', default='dataset/interx/contact_pairs')
    parser.add_argument('--split', default='dataset/interx/splits/val.txt')
    parser.add_argument(
        '--output', default='output/interx_contact_pair_audit.json'
    )
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir).resolve()
    split_path = Path(args.split).resolve()
    expected_names = read_split(split_path)
    expected_set = set(expected_names)
    pair_paths = sorted(pairs_dir.glob('*.pkl'))
    actual_names = {path.stem for path in pair_paths}
    valid_region_ids = {
        int(region_id)
        for region_ids in CANDIDATES.values()
        for region_id in region_ids
    }

    invalid_files = {}
    invalid_error_file_counts = collections.Counter()
    invalid_error_record_counts = collections.Counter()
    empty_names = []
    pair_counts = collections.Counter()
    contact_types = collections.Counter()
    alignment_types = collections.Counter()
    role_weights = collections.Counter()
    action_counts = collections.Counter()
    action_empty_counts = collections.Counter()
    record_count = 0

    for path in pair_paths:
        try:
            with open(path, 'rb') as handle:
                payload = pickle.load(handle)
        except Exception as error:  # trusted local cache, but audit malformed files
            invalid_files[path.name] = [f'unpickle_failed:{type(error).__name__}']
            continue
        if not isinstance(payload, list):
            invalid_files[path.name] = ['payload_not_list']
            continue
        pair_counts[len(payload)] += 1
        action_match = re.search(r'A(\d{3})', path.stem)
        action = action_match.group(1) if action_match else 'unparsed'
        action_counts[action] += 1
        if not payload:
            empty_names.append(path.stem)
            action_empty_counts[action] += 1
        file_errors = []
        for record in payload:
            record_count += 1
            record_errors = validate_record(record, valid_region_ids)
            file_errors.extend(record_errors)
            invalid_error_record_counts.update(record_errors)
            if isinstance(record, dict):
                contact_types[str(record.get('contact_type'))] += 1
                alignment_types[str(record.get('align_type'))] += 1
                if isinstance(record.get('w_p1'), (int, float)) and isinstance(
                    record.get('w_p2'), (int, float)
                ):
                    role_weights[
                        f'{float(record["w_p1"]):.6f}:{float(record["w_p2"]):.6f}'
                    ] += 1
        if file_errors:
            unique_errors = sorted(set(file_errors))
            invalid_files[path.name] = unique_errors
            invalid_error_file_counts.update(unique_errors)

    report = {
        'pairs_dir': str(pairs_dir),
        'split': str(split_path),
        'split_sha256': sha256_file(split_path),
        'expected_samples': len(expected_names),
        'cache_files': len(pair_paths),
        'cache_aggregate_sha256': aggregate_named_hash(pair_paths),
        'missing_names': sorted(expected_set - actual_names),
        'extra_names': sorted(actual_names - expected_set),
        'record_count': record_count,
        'empty_sample_count': len(empty_names),
        'empty_names': sorted(empty_names),
        'pair_count_distribution': dict(sorted(pair_counts.items())),
        'contact_type_distribution': dict(sorted(contact_types.items())),
        'alignment_type_distribution': dict(sorted(alignment_types.items())),
        'role_weight_distribution': dict(sorted(role_weights.items())),
        'action_sample_distribution': dict(sorted(action_counts.items())),
        'action_empty_distribution': dict(sorted(action_empty_counts.items())),
        'invalid_file_count': len(invalid_files),
        'invalid_error_file_counts': dict(sorted(invalid_error_file_counts.items())),
        'invalid_error_record_counts': dict(sorted(invalid_error_record_counts.items())),
        'invalid_files': invalid_files,
        'structural_validation_passed': (
            actual_names == expected_set and not invalid_files
        ),
        'provenance_limitations': {
            'raw_llm_response_saved': False,
            'request_caption_saved_with_cache': False,
            'model_name_or_version_saved_with_cache': False,
            'prompt_hash_saved_with_cache': False,
            'parse_status_saved_with_cache': False,
            'retry_count_saved_with_cache': False,
            'latency_or_token_usage_saved_with_cache': False,
            'invalid_or_hallucination_rate_recoverable': False,
        },
    }
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    os.replace(temporary, target)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
