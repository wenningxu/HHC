import argparse
import codecs as cs
import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import pickle as pkl
from pathlib import Path

import numpy as np
from tqdm import tqdm

from optimization.text_2_regions import ContactExtractor
from utils.paramUtil import CANDIDATES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and cache strict contact pairs for an Inter-X split."
    )
    parser.add_argument(
        "--split-file", "--val-file", dest="split_file", type=Path,
        default=Path("dataset/interx/splits/val.txt"),
        help="Frozen split to extract; --val-file remains a compatibility alias.",
    )
    parser.add_argument("--text-dir", type=Path, default=Path("dataset/interx/texts_processed"))
    parser.add_argument(
        "--requests", type=Path,
        help=(
            "Frozen fair-evaluation NPZ. When supplied, extract the exact "
            "caption selected by the evaluator for each request instead of "
            "silently using the first line of the source text file."
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("dataset/interx/contact_pairs_gemini3_strict_v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""), help="Gemini API key")
    parser.add_argument("--model-name", default="gemini-3-flash-preview")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help=(
            "Record failed samples and continue through the split. The final "
            "manifest remains incomplete and the command exits non-zero until "
            "all failed records have been successfully resumed."
        ),
    )
    parser.add_argument(
        "--manifest-update-interval", type=int, default=25,
        help="Atomically checkpoint progress after this many attempted samples.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-legacy-cache", action="store_true",
        help="Skip existing pickle files without required provenance sidecars.",
    )
    return parser.parse_args()


def read_first_caption(text_path):
    with cs.open(text_path, "r") as f:
        for line in f:
            return line.strip().split("#")[0]
    return ""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_caption_requests(split_file, text_dir, requests_path=None):
    split_names = [
        line.strip() for line in split_file.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(split_names) != len(set(split_names)):
        raise ValueError(f'duplicate names in split: {split_file}')
    split_set = set(split_names)
    if requests_path is None:
        records = []
        for sample_name in split_names:
            text_path = text_dir / f'{sample_name}.txt'
            caption = read_first_caption(text_path)
            records.append({
                'sample_name': sample_name,
                'caption': caption,
                'request_index': None,
                'text_path': str(text_path.resolve()),
                'text_sha256': sha256_file(text_path),
            })
        source = {
            'caption_source': 'first_caption_from_split_text_file',
            'requests': None,
            'requests_sha256': None,
        }
        return records, source

    requests_path = requests_path.resolve()
    with np.load(requests_path, allow_pickle=False) as request:
        missing = sorted({'names', 'captions'} - set(request.files))
        if missing:
            raise ValueError(
                f'frozen requests lack required arrays: {missing}'
            )
        names = request['names'].astype(str)
        captions = request['captions'].astype(str)
    if names.ndim != 1 or captions.ndim != 1 or len(names) != len(captions):
        raise ValueError('frozen request names/captions must be aligned 1D arrays')
    names_list = names.tolist()
    captions_list = captions.tolist()
    if len(names_list) != len(set(names_list)):
        raise ValueError('frozen requests contain duplicate sample names')
    outside_split = sorted(set(names_list) - split_set)
    if outside_split:
        raise ValueError(
            'frozen requests contain names outside the requested split: '
            f'{outside_split[:5]}'
        )
    if any(not caption.strip() for caption in captions_list):
        raise ValueError('frozen requests contain an empty caption')
    records = [
        {
            'sample_name': sample_name,
            'caption': caption,
            'request_index': index,
            'text_path': None,
            'text_sha256': None,
        }
        for index, (sample_name, caption) in enumerate(
            zip(names_list, captions_list)
        )
    ]
    source = {
        'caption_source': 'frozen_fair_evaluation_requests',
        'requests': str(requests_path),
        'requests_sha256': sha256_file(requests_path),
        'request_names_sha256': canonical_sha256(names_list),
        'request_captions_sha256': canonical_sha256(captions_list),
    }
    return records, source


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def atomic_pickle(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'wb') as handle:
        pkl.dump(payload, handle)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing Gemini API key. Pass --api-key or set GEMINI_API_KEY.")
    if args.manifest_update_interval < 1:
        raise ValueError('--manifest-update-interval must be positive')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extractor = ContactExtractor(
        api_key=args.api_key,
        model_name=args.model_name,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )

    records, caption_source = load_caption_requests(
        args.split_file.resolve(), args.text_dir.resolve(), args.requests
    )

    metadata_dir = args.out_dir / '_metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / '_extraction_manifest.json'
    manifest = {
        'schema': 'hhc_interx_contact_pairs_gemini3_strict_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'split': str(args.split_file.resolve()),
        'split_sha256': sha256_file(args.split_file),
        'split_count': len([
            line for line in args.split_file.read_text().splitlines()
            if line.strip()
        ]),
        'sample_count': len(records),
        **caption_source,
        'requested_model': args.model_name,
        'google_genai_version': importlib.metadata.version('google-genai'),
        'prompt_sha256': extractor.prompt_sha256,
        'system_instruction': extractor.system_instruction,
        'max_retries': args.max_retries,
        'retry_delay_seconds': args.retry_delay_seconds,
        'strict_schema': True,
        'api_key_recorded': False,
        'status': 'running',
        'completed_count': 0,
    }
    contract_keys = [
        'schema', 'split_sha256', 'split_count', 'sample_count',
        'caption_source', 'requests_sha256', 'request_names_sha256',
        'request_captions_sha256', 'requested_model', 'google_genai_version',
        'prompt_sha256', 'system_instruction', 'max_retries',
        'retry_delay_seconds', 'strict_schema', 'api_key_recorded',
    ]
    manifest['contract_sha256'] = canonical_sha256({
        key: manifest.get(key) for key in contract_keys
    })
    if manifest_path.is_file() and not args.overwrite:
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('contract_sha256') != manifest['contract_sha256']:
            raise ValueError(
                'contact extraction contract differs from the existing cache; '
                'use a new --out-dir or explicit --overwrite'
            )
        manifest = previous
        manifest['status'] = 'running'
    atomic_json(manifest_path, manifest)

    completed_count = 0
    failed_samples = []
    attempted_count = 0

    def checkpoint_progress(status='running'):
        manifest.update({
            'status': status,
            'completed_count': completed_count,
            'failed_count': len(failed_samples),
            'failed_samples': failed_samples,
            'attempted_count': attempted_count,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        })
        atomic_json(manifest_path, manifest)

    for record in tqdm(records, desc="Extracting contacts", unit="sample"):
        sample_name = record['sample_name']
        caption = record['caption']
        if not caption.strip():
            raise ValueError(f'{sample_name}: empty caption')
        out_path = args.out_dir / f"{sample_name}.pkl"
        metadata_path = metadata_dir / f'{sample_name}.json'
        if out_path.exists() and not args.overwrite:
            if metadata_path.is_file():
                existing = json.loads(metadata_path.read_text(encoding='utf-8'))
                expected = {
                    'schema': 'hhc_interx_contact_pairs_gemini3_strict_v1',
                    'sample_name': sample_name,
                    'caption': caption,
                    'caption_sha256': sha256_text(caption),
                    'caption_source': caption_source['caption_source'],
                    'requests_sha256': caption_source['requests_sha256'],
                    'request_index': record['request_index'],
                    'status': 'success',
                    'prompt_sha256': extractor.prompt_sha256,
                    'requested_model': args.model_name,
                }
                mismatches = {
                    key: (existing.get(key), value)
                    for key, value in expected.items()
                    if existing.get(key) != value
                }
                if mismatches:
                    raise ValueError(
                        f'{sample_name}: cached metadata differs from the '
                        f'frozen extraction contract: {mismatches}'
                    )
                completed_count += 1
                attempted_count += 1
                if attempted_count % args.manifest_update_interval == 0:
                    checkpoint_progress()
                continue
            if args.allow_legacy_cache:
                completed_count += 1
                attempted_count += 1
                if attempted_count % args.manifest_update_interval == 0:
                    checkpoint_progress()
                continue
            raise RuntimeError(
                f'legacy cache lacks extraction metadata: {out_path}; use a '
                'new --out-dir, --overwrite, or explicit --allow-legacy-cache'
            )

        contacts, metadata = extractor.get_contacts_with_metadata(
            caption, CANDIDATES
        )
        metadata.update({
            'schema': 'hhc_interx_contact_pairs_gemini3_strict_v1',
            'sample_name': sample_name,
            'caption_sha256': sha256_text(caption),
            'caption_source': caption_source['caption_source'],
            'requests': caption_source['requests'],
            'requests_sha256': caption_source['requests_sha256'],
            'request_index': record['request_index'],
            'text_path': record['text_path'],
            'text_sha256': record['text_sha256'],
            'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        })
        atomic_json(metadata_path, metadata)
        attempted_count += 1
        if metadata['status'] != 'success':
            failed_samples.append({
                'sample_name': sample_name,
                'status': metadata['status'],
                'error_type': metadata['error_type'],
                'error_message': metadata['error_message'],
            })
            checkpoint_progress(status='incomplete')
            if args.continue_on_error:
                continue
            raise RuntimeError(
                f'contact extraction failed closed for {sample_name}: '
                f'{metadata["status"]}: {metadata["error_message"]}'
            )
        atomic_pickle(out_path, contacts)
        completed_count += 1
        if attempted_count % args.manifest_update_interval == 0:
            checkpoint_progress()

    if failed_samples:
        checkpoint_progress(status='incomplete')
        raise RuntimeError(
            f'contact extraction incomplete: {len(failed_samples)} of '
            f'{len(records)} samples failed; rerun the same command to resume'
        )
    manifest['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    checkpoint_progress(status='complete')


if __name__ == "__main__":
    main()
