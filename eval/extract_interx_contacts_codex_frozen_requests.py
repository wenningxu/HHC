#!/usr/bin/env python
"""Run the Codex contact extractor with per-sample frozen-request binding.

The base extractor is left byte-identical while a train annotation is active.
This entry point is for fair-evaluation request files and adds an independently
hash-addressed writer record to every per-sample metadata sidecar.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import extract_interx_contacts_codex as base  # noqa: E402
from eval.resume_interx_contacts_codex_preserve_failures import (  # noqa: E402
    merge_prior_failures,
)


class FrozenRequestRecords(list):
    """Expose the official split size until the extractor takes its slice.

    The base extractor intentionally treats ``len(all_records)`` as the split
    count and then slices it into the selected records.  Frozen evaluation
    requests are a subset of the official split (1,696 of 1,708 for Inter-X),
    so a plain list would incorrectly collapse both counts to 1,696.  A slice
    returns a normal list, keeping ``sample_count`` equal to the request count.
    """

    def __init__(self, records, split_count):
        super().__init__(records)
        self.split_count = int(split_count)
        if self.split_count < list.__len__(self):
            raise ValueError('frozen requests exceed the official split size')

    def __len__(self):
        return self.split_count

    def __getitem__(self, key):
        value = list.__getitem__(self, key)
        return list(value) if isinstance(key, slice) else value


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_binding(requests_path):
    requests_path = requests_path.resolve()
    with np.load(requests_path, allow_pickle=False) as request:
        names = request['names'].astype(str).tolist()
        captions = request['captions'].astype(str).tolist()
    if len(names) != len(captions) or len(names) != len(set(names)):
        raise ValueError('frozen request names/captions are not uniquely aligned')
    return {
        'requests': str(requests_path),
        'requests_sha256': sha256_file(requests_path),
        'names': names,
        'captions': captions,
        'writer': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(__file__),
        },
    }


def enrich_metadata(payload, binding):
    if payload.get('caption_source') != 'frozen_fair_evaluation_requests':
        raise ValueError('binding writer received non-frozen caption metadata')
    index = payload.get('request_index')
    if not isinstance(index, int) or not 0 <= index < len(binding['names']):
        raise ValueError('metadata has an invalid frozen request index')
    expected_name = binding['names'][index]
    expected_caption = binding['captions'][index]
    if (
        payload.get('sample_name') != expected_name
        or payload.get('caption') != expected_caption
    ):
        raise ValueError('metadata differs from its indexed frozen request')
    return {
        **payload,
        'requests': binding['requests'],
        'requests_sha256': binding['requests_sha256'],
        'request_binding_writer': binding['writer'],
    }


class TimedInvocationError(RuntimeError):
    """Preserve failed-call wall time in the base extractor's sidecar."""


def invoke_with_timing(invoke, args, prompt):
    """Measure one Codex CLI attempt without changing its response contract."""
    started = time.perf_counter()
    try:
        response, metadata = invoke(args, prompt)
    except Exception as error:
        elapsed = time.perf_counter() - started
        raise TimedInvocationError(json.dumps({
            'wall_seconds': elapsed,
            'underlying_error_type': type(error).__name__,
            'underlying_error_message': str(error),
        }, sort_keys=True)) from error
    metadata = dict(metadata)
    metadata['wall_seconds'] = time.perf_counter() - started
    metadata['wall_clock_scope'] = 'single_codex_cli_attempt'
    return response, metadata


def parse_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--requests', required=True, type=Path)
    parser.add_argument(
        '--split-file', type=Path,
        default=Path('dataset/interx/splits/train.txt'),
    )
    parsed, _ = parser.parse_known_args(argv)
    return parsed


def bind_official_split_count(records, split_file):
    split_names = [
        line.strip()
        for line in split_file.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(split_names) != len(set(split_names)):
        raise ValueError(f'duplicate names in split: {split_file}')
    return FrozenRequestRecords(records, len(split_names))


def main():
    wrapper_args = parse_wrapper_args(sys.argv[1:])
    binding = load_binding(wrapper_args.requests)
    original_atomic_json = base.atomic_json
    original_load_caption_requests = base.load_caption_requests
    original_invoke_codex = base.invoke_codex

    def split_count_bound_loader(split_file, text_dir, requests_path=None):
        records, source = original_load_caption_requests(
            split_file, text_dir, requests_path,
        )
        if requests_path is None:
            return records, source
        if Path(requests_path).resolve() != wrapper_args.requests.resolve():
            raise ValueError('base extractor received a different request file')
        return bind_official_split_count(records, Path(split_file)), source

    def binding_atomic_json(path, payload):
        path = Path(path)
        payload = merge_prior_failures(path, payload)
        if path.parent.name == '_metadata':
            payload = enrich_metadata(payload, binding)
        original_atomic_json(path, payload)

    def timed_invoke_codex(args, prompt):
        return invoke_with_timing(original_invoke_codex, args, prompt)

    base.load_caption_requests = split_count_bound_loader
    base.atomic_json = binding_atomic_json
    base.invoke_codex = timed_invoke_codex
    try:
        base.main()
    finally:
        base.load_caption_requests = original_load_caption_requests
        base.atomic_json = original_atomic_json
        base.invoke_codex = original_invoke_codex


if __name__ == '__main__':
    main()
