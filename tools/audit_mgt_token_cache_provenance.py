#!/usr/bin/env python
"""Audit exact MMM token-cache import and quantify the rebuilt-cache drift."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_names(path):
    return {
        line.strip() for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    }


def npy_names(directory):
    return {path.stem for path in directory.glob('*.npy')}


def eligible(names, motion_h5):
    with h5py.File(motion_h5, 'r') as motions:
        return {
            name for name in names
            if name in motions and 30 <= motions[name].shape[0] < 1000
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-split', type=Path, default=Path('dataset/interx/splits/train.txt'))
    parser.add_argument('--val-split', type=Path, default=Path('dataset/interx/splits/val.txt'))
    parser.add_argument('--test-split', type=Path, default=Path('dataset/interx/splits/test.txt'))
    parser.add_argument('--motion-h5', type=Path, default=Path('dataset/interx/motions/all.h5'))
    parser.add_argument('--historical-codebook', type=Path, required=True)
    parser.add_argument('--historical-contact', type=Path, required=True)
    parser.add_argument('--imported-codebook', type=Path, required=True)
    parser.add_argument('--imported-contact', type=Path, required=True)
    parser.add_argument('--rebuilt-codebook', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    train = eligible(split_names(args.train_split), args.motion_h5)
    val = eligible(split_names(args.val_split), args.motion_h5)
    test = eligible(split_names(args.test_split), args.motion_h5)
    expected = train | val
    if train & val:
        raise RuntimeError('official train and validation splits overlap')

    historical_names = npy_names(args.historical_codebook)
    imported_names = npy_names(args.imported_codebook)
    imported_contact_names = npy_names(args.imported_contact)
    rebuilt_names = npy_names(args.rebuilt_codebook)
    if imported_names != expected or imported_contact_names != expected:
        raise RuntimeError('isolated import is not exactly official eligible train+val')
    if expected - historical_names:
        raise RuntimeError('historical source lacks official train/val tokens')

    exact_import_code = 0
    exact_import_contact = 0
    rebuilt_exact_files = 0
    rebuilt_exact_arrays = 0
    rebuilt_same_shape = 0
    aligned_equal = np.zeros(3, dtype=np.int64)
    aligned_total = np.zeros(3, dtype=np.int64)
    historical_lengths = []
    rebuilt_lengths = []
    for name in sorted(expected):
        historical_path = args.historical_codebook / f'{name}.npy'
        imported_path = args.imported_codebook / f'{name}.npy'
        if sha256_file(historical_path) == sha256_file(imported_path):
            exact_import_code += 1
        historical_contact_path = args.historical_contact / f'{name}.npy'
        imported_contact_path = args.imported_contact / f'{name}.npy'
        if sha256_file(historical_contact_path) == sha256_file(imported_contact_path):
            exact_import_contact += 1

        if name not in rebuilt_names:
            continue
        rebuilt_path = args.rebuilt_codebook / f'{name}.npy'
        if sha256_file(historical_path) == sha256_file(rebuilt_path):
            rebuilt_exact_files += 1
        historical = np.load(historical_path).reshape(-1, 3)
        rebuilt = np.load(rebuilt_path).reshape(-1, 3)
        historical_lengths.append(len(historical))
        rebuilt_lengths.append(len(rebuilt))
        if historical.shape == rebuilt.shape:
            rebuilt_same_shape += 1
            if np.array_equal(historical, rebuilt):
                rebuilt_exact_arrays += 1
        common = min(len(historical), len(rebuilt))
        aligned_equal += (historical[:common] == rebuilt[:common]).sum(axis=0)
        aligned_total += common

    imported_manifest = json.loads(
        (args.imported_codebook / '_cache_manifest.json').read_text(encoding='utf-8')
    )
    result = {
        'schema': 'hhc_mgt_token_cache_provenance_audit_v1',
        'status': 'pass',
        'official_split': {
            'eligible_train': len(train),
            'eligible_validation': len(val),
            'eligible_test': len(test),
            'train_validation_overlap': len(train & val),
        },
        'source_scope': {
            'historical_code_files': len(historical_names),
            'historical_contains_test_files': len(historical_names & test),
            'imported_code_files': len(imported_names),
            'imported_contact_files': len(imported_contact_names),
            'imported_test_files': len(imported_names & test),
            'imported_equals_official_train_validation': imported_names == expected,
        },
        'byte_exact_import': {
            'code_files_equal': exact_import_code,
            'contact_files_equal': exact_import_contact,
            'expected_files': len(expected),
            'all_code_files_equal': exact_import_code == len(expected),
            'all_contact_files_equal': exact_import_contact == len(expected),
        },
        'rebuilt_cache_drift_from_historical': {
            'compared_files': len(expected & rebuilt_names),
            'byte_equal_files': rebuilt_exact_files,
            'array_equal_files': rebuilt_exact_arrays,
            'same_shape_files': rebuilt_same_shape,
            'mean_historical_token_length': float(np.mean(historical_lengths)),
            'mean_rebuilt_token_length': float(np.mean(rebuilt_lengths)),
            'aligned_channel_token_agreement': {
                part: float(aligned_equal[index] / max(aligned_total[index], 1))
                for index, part in enumerate(('body', 'hand', 'root'))
            },
        },
        'normalization': {
            'historical_tokenizer_default': 'Interx_Mean_root.npy/Interx_Std_root.npy',
            'failed_rebuild': 'Interx_Mean.npy/Interx_Std.npy',
            'historical_mean_sha256': sha256_file(Path('dataset/interx/Interx_Mean_root.npy')),
            'historical_std_sha256': sha256_file(Path('dataset/interx/Interx_Std_root.npy')),
            'rebuilt_mean_sha256': sha256_file(Path('dataset/interx/Interx_Mean.npy')),
            'rebuilt_std_sha256': sha256_file(Path('dataset/interx/Interx_Std.npy')),
        },
        'import_manifest': {
            'path': str((args.imported_codebook / '_cache_manifest.json').resolve()),
            'sha256': sha256_file(args.imported_codebook / '_cache_manifest.json'),
            'token_cache_contract': imported_manifest['token_cache_contract'],
            'contact_map_scope': imported_manifest['contact_map_scope'],
            'contact_map_samples': imported_manifest['contact_map_samples'],
            'contact_map_sha256': imported_manifest['contact_map_sha256'],
        },
    }
    if not (
        result['source_scope']['imported_equals_official_train_validation']
        and result['source_scope']['imported_test_files'] == 0
        and result['byte_exact_import']['all_code_files_equal']
        and result['byte_exact_import']['all_contact_files_equal']
        and imported_manifest['contact_map_scope'] == 'train_only'
        and imported_manifest['contact_map_samples'] == len(train)
    ):
        raise RuntimeError('historical cache import provenance gate failed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
