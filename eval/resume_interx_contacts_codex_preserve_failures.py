#!/usr/bin/env python
"""Resume a Codex contact cache without discarding prior failed attempts."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import extract_interx_contacts_codex as base  # noqa: E402


def merge_prior_failures(path, payload):
    path = Path(path)
    if path.parent.name != '_batches' or not path.is_file():
        return payload
    previous = json.loads(path.read_text(encoding='utf-8'))
    if previous.get('status') != 'failed':
        return payload
    for key in ('sample_names', 'captions_sha256', 'prompt_sha256', 'contract_sha256'):
        if previous.get(key) != payload.get(key):
            raise ValueError(f'{path}: prior failed batch identity changed at {key}')
    previous_failures = previous.get('attempt_failures', [])
    current_failures = payload.get('attempt_failures', [])
    return {
        **payload,
        'attempts': int(previous.get('attempts', 0))
        + int(payload.get('attempts', 0)),
        'attempt_failures': previous_failures + current_failures,
        'resume_provenance': {
            'prior_failed_sidecar_sha256': base.sha256_file(path),
            'writer': str(Path(__file__).resolve()),
            'writer_sha256': base.sha256_file(__file__),
            'preserved_prior_failed_attempts': len(previous_failures),
        },
    }


def main():
    original_atomic_json = base.atomic_json

    def preserving_atomic_json(path, payload):
        original_atomic_json(path, merge_prior_failures(path, payload))

    base.atomic_json = preserving_atomic_json
    base.main()


if __name__ == '__main__':
    main()
