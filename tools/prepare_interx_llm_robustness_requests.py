#!/usr/bin/env python
"""Freeze a timing-covered, stratified Inter-X LLM robustness request set."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SCHEMA = 'hhc_interx_llm_robustness_requests_v1'
CONTACT_CUES = (
    'hug', 'hold', 'grab', 'touch', 'tap', 'pat', 'push', 'pull', 'punch',
    'hit', 'kick', 'shake', 'high-five', 'high five', 'carry', 'lift', 'kiss',
    'embrace', 'lean', 'slap', 'collide', 'clap', 'support',
)
ROLE_CUES = (
    'first person', 'second person', 'one person', 'other person',
    'first individual', 'second individual', 'initial individual',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def atomic_npz(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def validate_or_write_npz(path, payload):
    """Keep a frozen request archive byte-stable across harmless reruns."""
    if not path.is_file():
        atomic_npz(path, payload)
        return False
    with np.load(path, allow_pickle=False) as existing:
        if set(existing.files) != set(payload):
            raise ValueError(
                f'existing frozen request keys differ; refusing overwrite: {path}'
            )
        for key, expected in payload.items():
            actual = existing[key]
            if (
                actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or not np.array_equal(actual, expected)
            ):
                raise ValueError(
                    f'existing frozen request array {key!r} differs; '
                    f'refusing overwrite: {path}'
                )
    return True


def caption_flags(caption):
    text = caption.lower()
    cue_count = sum(text.count(cue) for cue in CONTACT_CUES)
    return {
        'laterality': 'left' in text or 'right' in text,
        'role_reference': any(cue in text for cue in ROLE_CUES),
        'multiple_contact_cues': cue_count >= 2,
        'contact_cue_count': cue_count,
    }


def select_strata(records, per_stratum, seed):
    rng = np.random.default_rng(seed)
    definitions = (
        ('timing_negative', lambda row: not row['timing_positive']),
        ('laterality_positive', lambda row: (
            row['timing_positive'] and row['flags']['laterality']
        )),
        ('multiple_contact_cues_positive', lambda row: (
            row['timing_positive'] and row['flags']['multiple_contact_cues']
        )),
        ('role_reference_positive', lambda row: (
            row['timing_positive'] and row['flags']['role_reference']
        )),
    )
    selected = []
    used = set()
    availability = {}
    for label, predicate in definitions:
        candidates = [
            row for row in records
            if row['index'] not in used and predicate(row)
        ]
        availability[label] = len(candidates)
        if len(candidates) < per_stratum:
            raise ValueError(
                f'{label} has {len(candidates)} candidates, needs {per_stratum}'
            )
        order = rng.permutation(len(candidates))[:per_stratum]
        for candidate_index in order:
            row = dict(candidates[int(candidate_index)])
            row['stratum'] = label
            selected.append(row)
            used.add(row['index'])
    return selected, availability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-requests', type=Path,
        default=Path('output/fair_interx_requests/replication_00.npz'),
    )
    parser.add_argument(
        '--timing-dir', type=Path, default=Path('dataset/interx/contact'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/interx_llm_robustness/requests.npz'),
    )
    parser.add_argument('--per-stratum', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20260828)
    args = parser.parse_args()
    if args.per_stratum < 1:
        raise ValueError('--per-stratum must be positive')

    source_path = args.source_requests.resolve()
    timing_dir = args.timing_dir.resolve()
    with np.load(source_path, allow_pickle=False) as source:
        source_payload = {key: source[key] for key in source.files}
    names = source_payload['names'].astype(str)
    captions = source_payload['captions'].astype(str)
    if len(names) != len(captions) or len(names) != len(set(names.tolist())):
        raise ValueError('source request names/captions are not uniquely aligned')

    records = []
    missing_timing = []
    for index, (name, caption) in enumerate(zip(names, captions)):
        timing_path = timing_dir / f'{name}.npy'
        if not timing_path.is_file():
            missing_timing.append(name)
            continue
        timing = np.load(timing_path, allow_pickle=False).reshape(-1)
        if not set(np.unique(timing).tolist()) <= {0, 1}:
            raise ValueError(f'{name}: timing annotation is not binary')
        records.append({
            'index': index,
            'name': name,
            'caption': caption,
            'timing_positive': bool(np.any(timing)),
            'timing_path': str(timing_path),
            'timing_sha256': sha256_file(timing_path),
            'flags': caption_flags(caption),
        })

    selected, availability = select_strata(
        records, args.per_stratum, args.seed,
    )
    indices = np.asarray([row['index'] for row in selected], dtype=np.int64)
    output_payload = {}
    for key, value in source_payload.items():
        if value.ndim > 0 and value.shape[0] == len(names):
            output_payload[key] = value[indices]
        else:
            output_payload[key] = value
    output_payload.update({
        'source_request_indices': indices,
        'strata': np.asarray([row['stratum'] for row in selected]),
        'robustness_selection_seed': np.asarray(args.seed, dtype=np.int64),
    })

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reused_frozen_output = validate_or_write_npz(output, output_payload)
    manifest = {
        'schema': SCHEMA,
        'status': 'complete',
        'source_requests': str(source_path),
        'source_requests_sha256': sha256_file(source_path),
        'timing_dir': str(timing_dir),
        'seed': args.seed,
        'per_stratum': args.per_stratum,
        'sample_count': len(selected),
        'stratum_counts': {
            label: sum(row['stratum'] == label for row in selected)
            for label in sorted({row['stratum'] for row in selected})
        },
        'availability_before_disjoint_selection': availability,
        'timing_covered_source_samples': len(records),
        'timing_missing_source_samples': len(missing_timing),
        'selected_records': selected,
        'selected_records_sha256': canonical_sha256(selected),
        'output': str(output),
        'output_sha256': sha256_file(output),
        'reused_byte_identical_frozen_output': reused_frozen_output,
        'implementation': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(__file__),
        },
    }
    manifest_path = output.with_suffix(output.suffix + '.manifest.json')
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        'output': str(output), 'manifest': str(manifest_path),
        'sample_count': len(selected),
        'stratum_counts': manifest['stratum_counts'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
