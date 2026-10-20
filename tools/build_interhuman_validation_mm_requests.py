#!/usr/bin/env python3
"""Freeze validation-only prompts for periodic InterHuman MModality."""

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'output/interhuman_validation_requests/validation.json'
DESTINATION = ROOT / 'output/interhuman_validation_mm_requests/validation_mm.json'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def build_payload():
    source = json.loads(SOURCE.read_text())
    if source.get('evaluated_entries') != 480 or len(source.get('requests', [])) != 480:
        raise ValueError('frozen InterHuman validation request is not the 480-sample set')
    requests = []
    for prompt_index, request in enumerate(source['requests'][:100]):
        row = dict(request)
        row['prompt_index'] = prompt_index
        requests.append(row)
    if len({row['identifier'] for row in requests}) != 100:
        raise ValueError('validation MModality prompts are not unique interactions')
    payload = {
        'schema': 'interhuman_validation_mm_requests_v1',
        'split': 'val',
        'purpose': 'periodic diagnostics only; never formal test reporting',
        'selection': 'first 100 prompts of the already-seeded frozen 480-request validation order',
        'source_validation_request': str(SOURCE),
        'source_validation_request_sha256': sha256_file(SOURCE),
        'prompts': 100,
        'repeats': 11,
        'pairs': 10,
        'requests': requests,
    }
    payload['request_sha256'] = canonical_sha256(requests)
    return payload


def main():
    payload = build_payload()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists():
        if json.loads(DESTINATION.read_text()) != payload:
            raise ValueError(f'refusing to overwrite changed request: {DESTINATION}')
    else:
        temporary = DESTINATION.with_name(DESTINATION.name + f'.tmp.{os.getpid()}')
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        os.replace(temporary, DESTINATION)
    print(json.dumps({
        'status': 'complete', 'path': str(DESTINATION),
        'sha256': sha256_file(DESTINATION), 'prompts': 100, 'repeats': 11,
    }, indent=2))


if __name__ == '__main__':
    main()
