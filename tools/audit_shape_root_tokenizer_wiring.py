#!/usr/bin/env python
"""Fail closed until the shape-cache training entrypoint uses root tokenization."""

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPTIONS = ROOT / 'options/option_transformer.py'
TRAIN = ROOT / 'train/train_t2i_trans.py'
TOKENIZER = ROOT / 'dataset/dataset_interx_tokenize_historical_root.py'
PATCH = ROOT / 'tools/provenance_snapshots/post_formal_shape_root_tokenizer_wiring.patch'
APPLIER = ROOT / 'tools/apply_post_formal_shape_root_tokenizer_wiring.sh'
DEFAULT_OUTPUT = ROOT / 'output/shape_root_tokenizer_wiring_audit.json'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    option_source = OPTIONS.read_text(encoding='utf-8')
    train_source = TRAIN.read_text(encoding='utf-8')
    tokenizer_source = TOKENIZER.read_text(encoding='utf-8')
    contract = 'deterministic_historical_root_v1'
    checks = {
        'option_accepts_contract': contract in option_source,
        'training_imports_root_tokenizer': (
            'dataset_interx_tokenize_historical_root' in train_source
        ),
        'training_manifest_hashes_root_tokenizer': (
            'dataset/dataset_interx_tokenize_historical_root.py'
            in train_source
        ),
        'training_selects_contract_explicitly': (
            f"args.token_cache_contract == '{contract}'" in train_source
        ),
        'cache_manifest_binds_tokenizer_implementation': (
            'tokenizer_implementation_sha256' in train_source
        ),
        'root_cache_binds_root_mean': (
            "f'{args.stats_prefix}_Mean_root.npy'" in train_source
        ),
        'root_cache_binds_root_std': (
            "f'{args.stats_prefix}_Std_root.npy'" in train_source
        ),
        'selected_loader_is_used': 'tokenizer_dataset.DATALoader(' in train_source,
        'selected_loader_receives_stats_dir': 'stats_dir=args.stats_dir' in train_source,
        'selected_loader_receives_stats_prefix': (
            'stats_prefix=args.stats_prefix' in train_source
        ),
        'tokenizer_declares_exact_contract': (
            f"TOKENIZER_CONTRACT = '{contract}'" in tokenizer_source
        ),
        'tokenizer_uses_root_statistics': "motion_rep='vel'" in tokenizer_source,
        'tokenizer_has_no_extra_coordinate_preprocessing': (
            'prepare_interx' not in tokenizer_source
        ),
    }
    status = 'pass' if all(checks.values()) else 'pending'
    payload = {
        'schema': 'hhc_shape_root_tokenizer_wiring_audit_v1',
        'status': status,
        'contract': contract,
        'checks': checks,
        'missing_checks': [key for key, value in checks.items() if not value],
        'artifacts': {
            'options': artifact(OPTIONS),
            'training_entrypoint': artifact(TRAIN),
            'root_tokenizer': artifact(TOKENIZER),
            'wiring_patch': artifact(PATCH),
            'guarded_applier': artifact(APPLIER),
            'implementation': artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if status != 'pass':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
