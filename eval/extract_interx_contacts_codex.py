#!/usr/bin/env python
"""Batch-label Inter-X contact regions through ChatGPT-authenticated Codex.

This is a no-API-key alternative to the Gemini extractor.  It uses the local
Codex CLI login, validates a closed JSON schema, and stores batch and
per-sample provenance so interrupted full-train extraction is resumable.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from eval.extract_val_contacts import (
    atomic_json,
    canonical_sha256,
    load_caption_requests,
    sha256_file,
    sha256_text,
)
from optimization.text_2_regions import ContactExtractor
from utils.paramUtil import CANDIDATES


SCHEMA = 'hhc_interx_contact_pairs_codex_strict_v3'
ALLOWED_PARTS = tuple(CANDIDATES)

PROMPT_TEMPLATE = """You annotate physical body contacts in dyadic motion descriptions.

Return one record for EVERY input sample, preserving sample_name exactly and
preserving input order. Return no additional samples. For each description,
list all physical contacts between Person 1 and Person 2. If they never touch,
return an empty contacts array.

Person numbering follows the description: "the first person" is Person 1 and
"the second person" is Person 2. If the description says "one person" and
"the other person", the first-mentioned person is Person 1. Do not include
self-contact or foot-to-ground contact.

Allowed body-part strings (use no others):
{allowed_parts}

If a hand/arm/leg/foot side is truly ambiguous, put both allowed alternatives
in one string separated by comma and one space, for example
"left hand (palm), right hand (palm)". Use palm for ordinary touching,
holding, grabbing, pushing, hitting, handshaking, or high-fiving unless the
description explicitly implies the back of the hand.
Map cheek/face to head, forearm/elbow to arm, thigh/knee/calf to leg, and hip
to waist. The output schema enumerates every permitted single or bilateral
body-part string; never copy a more specific synonym from the description.

contact_type must be exactly one of:
- point: tiny contact such as poke or tap
- flat: planar/medium contact such as palm contact, pat, push, punch, high-five
- wrap: grasping, holding, or an arm/hand wrapping a body part
- hug: broad full-body embrace or carrying contact

role_ratio must be exactly one of:
- 5:5 symmetric interaction
- 9:1 Person 1 is active and Person 2 passive
- 1:9 Person 2 is active and Person 1 passive
- 7:3 Person 1 mostly active and Person 2 reacts

Do not invent contact merely because people approach, dodge, gesture, or face
each other. Output only the JSON required by the supplied schema.

INPUT SAMPLES:
{samples_json}
"""


def atomic_pickle(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'wb') as handle:
        pickle.dump(payload, handle)
    os.replace(temporary, path)


def codex_version(codex_binary):
    result = subprocess.run(
        [codex_binary, '--version'], check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def build_prompt(records):
    samples = [
        {'sample_name': row['sample_name'], 'description': row['caption']}
        for row in records
    ]
    return PROMPT_TEMPLATE.format(
        allowed_parts='\n'.join(f'- {part}' for part in ALLOWED_PARTS),
        samples_json=json.dumps(samples, ensure_ascii=False, indent=2),
    )


def parse_cli_stats(stdout):
    session = re.search(r'^session id:\s*(\S+)', stdout, flags=re.MULTILINE)
    tokens = re.search(r'tokens used\s*\n([0-9,]+)', stdout)
    return {
        'session_id': session.group(1) if session else None,
        'tokens_used': int(tokens.group(1).replace(',', '')) if tokens else None,
    }


def invoke_codex(args, prompt):
    with tempfile.TemporaryDirectory(prefix='hhc_codex_contact_') as temp_dir:
        response_path = Path(temp_dir) / 'response.json'
        command = [
            args.codex_binary,
            'exec',
            '--ephemeral',
            '--ignore-user-config',
            '--ignore-rules',
            '--skip-git-repo-check',
            '-C', temp_dir,
            '-s', 'read-only',
            '-m', args.model,
            '-c', f'model_reasoning_effort="{args.reasoning_effort}"',
            '--output-schema', str(args.output_schema.resolve()),
            '--color', 'never',
            '-o', str(response_path),
            '-',
        ]
        result = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
        )
        response = response_path.read_text(encoding='utf-8') \
            if response_path.is_file() else None
    metadata = {
        'returncode': result.returncode,
        'stdout_tail': result.stdout[-4000:],
        'stderr_tail': result.stderr[-4000:],
        **parse_cli_stats(result.stdout + '\n' + result.stderr),
    }
    if result.returncode != 0 or response is None:
        raise RuntimeError(json.dumps(metadata, sort_keys=True))
    return response, metadata


def map_contact_rows(rows):
    markdown_style = []
    for row in rows:
        markdown_style.append({
            'person 1 body part': row['p1_part'],
            'person 2 body part': row['p2_part'],
            'contact area': row['contact_type'],
            'role ratio (p1:p2)': row['role_ratio'],
        })
    extractor = ContactExtractor.__new__(ContactExtractor)
    return extractor.map_contacts_to_ids(markdown_style, CANDIDATES)


def validate_response(raw_response, batch):
    payload = json.loads(raw_response)
    if set(payload) != {'records'} or not isinstance(payload['records'], list):
        raise ValueError('response root must contain only a records array')
    expected = [row['sample_name'] for row in batch]
    actual = [row.get('sample_name') for row in payload['records']]
    if actual != expected:
        raise ValueError(
            f'response names/order differ: expected {expected}, got {actual}'
        )
    mapped = {}
    for record in payload['records']:
        if set(record) != {'sample_name', 'contacts'}:
            raise ValueError(f'{record.get("sample_name")}: unexpected record keys')
        if not isinstance(record['contacts'], list):
            raise ValueError(f'{record["sample_name"]}: contacts is not a list')
        mapped[record['sample_name']] = map_contact_rows(record['contacts'])
    return payload, mapped


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-file', type=Path, default=Path('dataset/interx/splits/train.txt'))
    parser.add_argument('--text-dir', type=Path, default=Path('dataset/interx/texts_processed'))
    parser.add_argument('--requests', type=Path)
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path('dataset/interx/contact_pairs_codex_luna_strict_v3_train'),
    )
    parser.add_argument('--codex-binary', default='codex')
    parser.add_argument('--model', default='gpt-5.6-luna')
    parser.add_argument('--reasoning-effort', default='low', choices=('low', 'medium', 'high'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument(
        '--output-schema', type=Path,
        default=Path('configs/contact_annotation_batch_schema.json'),
    )
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_retries < 0:
        raise ValueError('batch-size must be positive and max-retries non-negative')
    if args.offset < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError('invalid offset/limit')
    if not args.output_schema.is_file():
        raise FileNotFoundError(args.output_schema)
    if importlib.util.find_spec('jsonschema') is None:
        # Codex enforces the supplied schema server-side; this script performs
        # a second semantic/closed-dictionary validation below.
        jsonschema_version = None
    else:
        jsonschema_version = importlib_metadata.version('jsonschema')

    all_records, caption_source = load_caption_requests(
        args.split_file.resolve(), args.text_dir.resolve(), args.requests,
    )
    records = all_records[args.offset:]
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        raise ValueError('selected extraction range is empty')

    out_dir = args.out_dir.resolve()
    metadata_dir = out_dir / '_metadata'
    batches_dir = out_dir / '_batches'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)
    cli_version = codex_version(args.codex_binary)
    prompt_template_sha256 = sha256_text(PROMPT_TEMPLATE)
    manifest_path = out_dir / '_extraction_manifest.json'
    manifest = {
        'schema': SCHEMA,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'split': str(args.split_file.resolve()),
        'split_sha256': sha256_file(args.split_file),
        'split_count': len(all_records),
        'selection_offset': args.offset,
        'selection_limit': args.limit,
        'sample_count': len(records),
        'selected_names_sha256': canonical_sha256(
            [row['sample_name'] for row in records]
        ),
        'selected_captions_sha256': canonical_sha256(
            [row['caption'] for row in records]
        ),
        **caption_source,
        'provider': 'openai_codex_cli_chatgpt_login',
        'requested_model': args.model,
        'reasoning_effort': args.reasoning_effort,
        'codex_cli_version': cli_version,
        'jsonschema_version': jsonschema_version,
        'batch_size': args.batch_size,
        'max_retries': args.max_retries,
        'timeout_seconds': args.timeout_seconds,
        'prompt_template_sha256': prompt_template_sha256,
        'output_schema': str(args.output_schema.resolve()),
        'output_schema_sha256': sha256_file(args.output_schema),
        'implementation': {
            'extractor': {
                'path': str(Path(__file__).resolve()),
                'sha256': sha256_file(__file__),
            },
            'contact_parser': {
                'path': str((ROOT / 'optimization/text_2_regions.py').resolve()),
                'sha256': sha256_file(ROOT / 'optimization/text_2_regions.py'),
            },
            'region_dictionary': {
                'path': str((ROOT / 'utils/paramUtil.py').resolve()),
                'sha256': sha256_file(ROOT / 'utils/paramUtil.py'),
            },
        },
        'strict_schema': True,
        'api_key_recorded': False,
        'authentication_secret_recorded': False,
        'status': 'running',
        'completed_count': 0,
    }
    contract_keys = [
        key for key in manifest
        if key not in {'created_at_utc', 'status', 'completed_count'}
    ]
    manifest['contract_sha256'] = canonical_sha256({
        key: manifest[key] for key in contract_keys
    })
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('contract_sha256') != manifest['contract_sha256']:
            raise ValueError('existing cache has a different extraction contract')
        manifest = previous
        manifest['status'] = 'running'
    atomic_json(manifest_path, manifest)

    expected_common = {
        'schema': SCHEMA,
        'provider': 'openai_codex_cli_chatgpt_login',
        'requested_model': args.model,
        'reasoning_effort': args.reasoning_effort,
        'prompt_template_sha256': prompt_template_sha256,
    }
    pending = []
    completed_count = 0
    for row in records:
        pair_path = out_dir / f'{row["sample_name"]}.pkl'
        metadata_path = metadata_dir / f'{row["sample_name"]}.json'
        if pair_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            expected = {
                **expected_common,
                'sample_name': row['sample_name'],
                'caption': row['caption'],
                'caption_sha256': sha256_text(row['caption']),
                'status': 'success',
            }
            mismatch = {
                key: (metadata.get(key), value) for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatch:
                raise ValueError(f'{row["sample_name"]}: cache mismatch {mismatch}')
            completed_count += 1
        elif pair_path.exists() or metadata_path.exists():
            raise RuntimeError(f'{row["sample_name"]}: partial cached record')
        else:
            pending.append(row)

    batch_counter = 0
    total_tokens = int(manifest.get('total_tokens_used') or 0)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        prompt = build_prompt(batch)
        batch_identity = {
            'sample_names': [row['sample_name'] for row in batch],
            'captions_sha256': canonical_sha256([row['caption'] for row in batch]),
            'prompt_sha256': sha256_text(prompt),
            'contract_sha256': manifest['contract_sha256'],
        }
        batch_id = canonical_sha256(batch_identity)[:16]
        batch_path = batches_dir / f'{batch_id}.json'
        last_error = None
        attempt_failures = []
        for attempt in range(args.max_retries + 1):
            raw_response = None
            cli_metadata = None
            try:
                raw_response, cli_metadata = invoke_codex(args, prompt)
                parsed, mapped = validate_response(raw_response, batch)
                last_error = None
                break
            except Exception as error:
                last_error = error
                attempt_failures.append({
                    'attempt': attempt + 1,
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    'raw_response': raw_response,
                    'cli_metadata': cli_metadata,
                })
        if last_error is not None:
            atomic_json(batch_path, {
                'schema': 'hhc_codex_contact_batch_v1',
                **batch_identity,
                'status': 'failed',
                'attempts': args.max_retries + 1,
                'attempt_failures': attempt_failures,
                'error_type': type(last_error).__name__,
                'error_message': str(last_error),
            })
            manifest.update({
                'status': 'incomplete',
                'completed_count': completed_count,
                'failed_batch': batch_id,
                'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            })
            atomic_json(manifest_path, manifest)
            raise RuntimeError(f'batch {batch_id} failed closed: {last_error}')

        atomic_json(batch_path, {
            'schema': 'hhc_codex_contact_batch_v1',
            **batch_identity,
            'status': 'success',
            'attempts': attempt + 1,
            'attempt_failures': attempt_failures,
            'raw_response': raw_response,
            'parsed_response': parsed,
            'cli_metadata': cli_metadata,
            'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        })
        batch_sha256 = sha256_file(batch_path)
        for row in batch:
            sample_name = row['sample_name']
            contacts = mapped[sample_name]
            pair_path = out_dir / f'{sample_name}.pkl'
            metadata_path = metadata_dir / f'{sample_name}.json'
            atomic_pickle(pair_path, contacts)
            atomic_json(metadata_path, {
                **expected_common,
                'sample_name': sample_name,
                'caption': row['caption'],
                'caption_sha256': sha256_text(row['caption']),
                'caption_source': caption_source['caption_source'],
                'text_path': row['text_path'],
                'text_sha256': row['text_sha256'],
                'request_index': row['request_index'],
                'status': 'success',
                'structured_contact_rows': len(contacts),
                'batch_id': batch_id,
                'batch_metadata': str(batch_path),
                'batch_metadata_sha256': batch_sha256,
                'response_model_version': args.model,
                'session_id': cli_metadata['session_id'],
                'batch_tokens_used': cli_metadata['tokens_used'],
                'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                'api_key_recorded': False,
                'authentication_secret_recorded': False,
            })
            completed_count += 1
        batch_counter += 1
        total_tokens += int(cli_metadata['tokens_used'] or 0)
        manifest.update({
            'status': 'running',
            'completed_count': completed_count,
            'completed_batches_this_run': batch_counter,
            'total_tokens_used': total_tokens,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        })
        atomic_json(manifest_path, manifest)

    manifest.update({
        'status': 'complete',
        'completed_count': completed_count,
        'total_tokens_used': total_tokens,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    })
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        'out_dir': str(out_dir),
        'model': args.model,
        'completed_count': completed_count,
        'batches_this_run': batch_counter,
        'total_tokens_used': total_tokens,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
