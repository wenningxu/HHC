#!/usr/bin/env python
"""Audit Luna contact regions against frozen Inter-X contact timing labels."""

import argparse
import collections
import hashlib
import json
import os
import pickle
import re
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_interx_contact_pairs import validate_record
from utils.paramUtil import CANDIDATES


SCHEMA = 'hhc_interx_contact_timing_consistency_audit_v1'
CONTACT_SCHEMA = 'hhc_interx_contact_pairs_codex_strict_v3'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_names(path):
    names = [
        line.strip() for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(names) != len(set(names)):
        raise ValueError(f'duplicate names in split: {path}')
    return names


def action_id(name):
    match = re.search(r'A(\d{3})', name)
    if not match:
        raise ValueError(f'cannot parse action from {name}')
    return match.group(1)


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--pairs-dir', type=Path,
        default=Path('dataset/interx/contact_pairs_codex_luna_strict_v3_train'),
    )
    parser.add_argument('--split', type=Path, default=Path('dataset/interx/splits/train.txt'))
    parser.add_argument('--timing-dir', type=Path, default=Path('dataset/interx/contact'))
    parser.add_argument(
        '--motion-file', type=Path, default=Path('dataset/interx/motions/train.h5'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/interx_luna_train_timing_consistency_audit.json'),
    )
    parser.add_argument(
        '--allow-incomplete', action='store_true',
        help='Audit the completed prefix while extraction is still running.',
    )
    parser.add_argument('--print-full', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    pairs_dir = args.pairs_dir.resolve()
    split = args.split.resolve()
    timing_dir = args.timing_dir.resolve()
    motion_file = args.motion_file.resolve()
    manifest_path = pairs_dir / '_extraction_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    names = read_names(split)
    expected = set(names)

    manifest_errors = []
    if manifest.get('schema') != CONTACT_SCHEMA:
        manifest_errors.append('unsupported_contact_schema')
    if not manifest.get('strict_schema'):
        manifest_errors.append('contact_schema_not_strict')
    if manifest.get('split_sha256') != sha256_file(split):
        manifest_errors.append('split_hash_mismatch')
    if manifest.get('sample_count') != len(names):
        manifest_errors.append('manifest_sample_count_mismatch')
    if not args.allow_incomplete and manifest.get('status') != 'complete':
        manifest_errors.append('extraction_not_complete')

    pair_paths = {path.stem: path for path in pairs_dir.glob('*.pkl')}
    metadata_paths = {
        path.stem: path for path in (pairs_dir / '_metadata').glob('*.json')
    }
    completed = expected & set(pair_paths) & set(metadata_paths)
    missing_pairs = sorted(expected - set(pair_paths))
    missing_metadata = sorted(expected - set(metadata_paths))
    extra_pairs = sorted(set(pair_paths) - expected)
    extra_metadata = sorted(set(metadata_paths) - expected)
    if not args.allow_incomplete and (missing_pairs or missing_metadata):
        manifest_errors.append('completed_cache_missing_samples')
    if extra_pairs or extra_metadata:
        manifest_errors.append('cache_contains_out_of_split_samples')

    action_presence = collections.defaultdict(lambda: [0, 0])
    for name in names:
        counts = action_presence[action_id(name)]
        counts[0] += 1
        counts[1] += int((timing_dir / f'{name}.npy').is_file())
    timing_coverage_by_action = {
        action: {'samples': counts[0], 'timing_files': counts[1]}
        for action, counts in sorted(action_presence.items())
    }

    valid_region_ids = {
        int(region_id) for ids in CANDIDATES.values() for region_id in ids
    }
    metadata_errors = {}
    payload_errors = {}
    timing_errors = {}
    timing_uncovered_names = []
    positive_timing_empty_regions = []
    zero_timing_nonempty_regions = []
    timing_covered_samples = 0
    timing_positive_samples = 0
    region_positive_samples = 0
    structured_contact_rows = 0
    action_summary = collections.defaultdict(collections.Counter)

    with h5py.File(motion_file, 'r') as motions:
        for name in sorted(completed):
            action = action_id(name)
            summary = action_summary[action]
            summary['audited_samples'] += 1
            try:
                with open(pair_paths[name], 'rb') as handle:
                    pairs = pickle.load(handle)
            except Exception as error:
                payload_errors[name] = [f'unpickle_failed:{type(error).__name__}']
                continue
            if not isinstance(pairs, list):
                payload_errors[name] = ['payload_not_list']
                continue
            errors = sorted({
                error for record in pairs
                for error in validate_record(record, valid_region_ids)
            })
            if errors:
                payload_errors[name] = errors
                continue
            region_has_contact = bool(pairs)
            region_positive_samples += int(region_has_contact)
            structured_contact_rows += len(pairs)
            summary['region_positive_samples'] += int(region_has_contact)
            summary['structured_contact_rows'] += len(pairs)

            metadata = json.loads(metadata_paths[name].read_text(encoding='utf-8'))
            expected_metadata = {
                'schema': CONTACT_SCHEMA,
                'sample_name': name,
                'status': 'success',
                'provider': manifest.get('provider'),
                'requested_model': manifest.get('requested_model'),
                'reasoning_effort': manifest.get('reasoning_effort'),
                'prompt_template_sha256': manifest.get('prompt_template_sha256'),
                'structured_contact_rows': len(pairs),
            }
            mismatches = {
                key: {'actual': metadata.get(key), 'expected': value}
                for key, value in expected_metadata.items()
                if metadata.get(key) != value
            }
            caption = metadata.get('caption')
            if not isinstance(caption, str) or metadata.get('caption_sha256') != hashlib.sha256(
                (caption or '').encode('utf-8')
            ).hexdigest():
                mismatches['caption_sha256'] = {'actual': metadata.get('caption_sha256')}
            if mismatches:
                metadata_errors[name] = mismatches

            if name not in motions:
                timing_errors[name] = ['motion_missing']
                continue
            motion_frames = int(motions[name].shape[0])
            timing_path = timing_dir / f'{name}.npy'
            if not timing_path.is_file():
                # Missing action-wide files are annotation coverage gaps, not
                # negative labels.  The v2 augmentation plan excludes them.
                timing_uncovered_names.append(name)
                summary['timing_uncovered_samples'] += 1
                continue
            timing = np.load(timing_path, allow_pickle=False).reshape(-1)
            errors = []
            if len(timing) != motion_frames:
                errors.append(f'timing_length_{len(timing)}_motion_length_{motion_frames}')
            if not set(np.unique(timing).tolist()) <= {0, 1}:
                errors.append('timing_not_binary')
            if errors:
                timing_errors[name] = errors
                continue
            timing_covered_samples += 1
            timing_has_contact = bool(np.any(timing))
            timing_positive_samples += int(timing_has_contact)
            summary['timing_positive_samples'] += int(timing_has_contact)
            summary['timing_zero_samples'] += int(not timing_has_contact)

            if timing_has_contact and not region_has_contact:
                positive_timing_empty_regions.append(name)
                summary['positive_timing_empty_regions'] += 1
            if not timing_has_contact and region_has_contact:
                zero_timing_nonempty_regions.append(name)
                summary['zero_timing_nonempty_regions'] += 1

    batch_paths = sorted((pairs_dir / '_batches').glob('*.json'))
    retry_batches = []
    failed_attempts = 0
    batch_tokens = 0
    for path in batch_paths:
        payload = json.loads(path.read_text(encoding='utf-8'))
        attempts = int(payload.get('attempts', 0))
        failures = payload.get('attempt_failures', [])
        failed_attempts += len(failures)
        batch_tokens += int(payload.get('cli_metadata', {}).get('tokens_used', 0) or 0)
        if attempts > 1 or failures:
            retry_batches.append({
                'batch': path.name,
                'attempts': attempts,
                'failed_attempts': len(failures),
                'failures': failures,
            })

    complete_snapshot = (
        len(completed) == len(names)
        and not missing_pairs and not missing_metadata
        and manifest.get('status') == 'complete'
    )
    contradictions = (
        len(positive_timing_empty_regions) + len(zero_timing_nonempty_regions)
    )
    fusion_readiness_passed = (
        complete_snapshot
        and not manifest_errors
        and not metadata_errors
        and not payload_errors
        and not timing_errors
    )
    exact_modal_agreement_passed = contradictions == 0
    report = {
        'schema': SCHEMA,
        'status': 'complete' if complete_snapshot else 'partial',
        'fusion_readiness_passed': fusion_readiness_passed,
        'exact_modal_agreement_passed': exact_modal_agreement_passed,
        'fusion_policy': {
            'timing_positive_and_regions_nonempty': 'timed_semantic_contact',
            'timing_positive_and_regions_empty': 'physical_losses_only',
            'timing_zero_and_regions_nonempty': 'semantic_regions_inactive',
            'timing_annotation_missing': 'excluded_from_shape_augmentation',
            'rationale': (
                'frame timing encodes observed geometric proximity/contact, '
                'while LLM regions encode caption-described inter-human body '
                'parts; disagreement is audited but must not invent contact'
            ),
        },
        'inputs': {
            'pairs_dir': str(pairs_dir),
            'extraction_manifest': str(manifest_path),
            'extraction_manifest_sha256': sha256_file(manifest_path),
            'split': str(split),
            'split_sha256': sha256_file(split),
            'timing_dir': str(timing_dir),
            'motion_file': str(motion_file),
            'motion_file_sha256': sha256_file(motion_file),
        },
        'implementation': {
            'path': str(Path(__file__).resolve()),
            'bytes': Path(__file__).resolve().stat().st_size,
            'sha256': sha256_file(Path(__file__).resolve()),
        },
        'counts': {
            'expected_samples': len(names),
            'audited_samples': len(completed),
            'missing_pairs': len(missing_pairs),
            'missing_metadata': len(missing_metadata),
            'timing_covered_samples': timing_covered_samples,
            'timing_uncovered_samples': len(timing_uncovered_names),
            'timing_positive_samples': timing_positive_samples,
            'region_positive_samples': region_positive_samples,
            'structured_contact_rows': structured_contact_rows,
            'positive_timing_empty_regions': len(positive_timing_empty_regions),
            'zero_timing_nonempty_regions': len(zero_timing_nonempty_regions),
            'batch_files': len(batch_paths),
            'retry_batches': len(retry_batches),
            'failed_attempts': failed_attempts,
            'batch_tokens_from_sidecars': batch_tokens,
            'manifest_total_tokens_used': manifest.get('total_tokens_used'),
        },
        'manifest_errors': manifest_errors,
        'timing_coverage_by_action': timing_coverage_by_action,
        'missing_pairs': missing_pairs,
        'missing_metadata': missing_metadata,
        'extra_pairs': extra_pairs,
        'extra_metadata': extra_metadata,
        'metadata_errors': metadata_errors,
        'payload_errors': payload_errors,
        'timing_errors': timing_errors,
        'timing_uncovered_names': timing_uncovered_names,
        'positive_timing_empty_region_names': positive_timing_empty_regions,
        'zero_timing_nonempty_region_names': zero_timing_nonempty_regions,
        'retry_batches': retry_batches,
        'action_summary': {
            action: dict(sorted(counts.items()))
            for action, counts in sorted(action_summary.items())
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    if args.print_full:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps({
            'output': str(output),
            'status': report['status'],
            'fusion_readiness_passed': fusion_readiness_passed,
            'exact_modal_agreement_passed': exact_modal_agreement_passed,
            **report['counts'],
        }, indent=2, sort_keys=True))
    if not args.allow_incomplete and not fusion_readiness_passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
