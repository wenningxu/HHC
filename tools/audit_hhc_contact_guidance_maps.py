#!/usr/bin/env python
"""Audit the fair HHC contact map and replay MMM's legacy map construction."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MMM_ROOT = Path('/home/wenning/projects/MMM')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.contact_map import build_code_contact_map


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def split_names(split_root):
    output = {}
    for split in ('train', 'val', 'test'):
        output[split] = {
            line.strip() for line in
            (split_root / f'{split}.txt').read_text(encoding='utf-8').splitlines()
            if line.strip()
        }
    return output


def cache_names(cache_dir):
    return {path.stem for path in cache_dir.glob('*.npy')}


def map_stats(array):
    records = []
    for channel in array:
        records.append({
            'minimum': float(channel.min()),
            'maximum': float(channel.max()),
            'mean': float(channel.mean()),
            'zero_codes': int(np.count_nonzero(channel == 0.0)),
            'neutral_0p5_codes': int(np.count_nonzero(channel == 0.5)),
            'one_codes': int(np.count_nonzero(channel == 1.0)),
            'positive_codes': int(np.count_nonzero(channel > 0.0)),
        })
    return records


def replay_legacy_axis_bug(indices_dir, contact_dir, codebook_size):
    """Replay the exact `(1,T,3)` axis error in MMM/test_data.py."""
    total = np.zeros((3, codebook_size), dtype=np.float64)
    positive = np.zeros_like(total)
    used = 0
    for index_path in sorted(indices_dir.glob('*.npy')):
        contact_path = contact_dir / index_path.name
        if not contact_path.is_file():
            continue
        indices = np.load(index_path).astype(np.int32)
        contact = np.load(contact_path).astype(np.float32)
        if contact.ndim > 1:
            contact = contact.flatten()
        cached_batch_axis = indices.ndim == 3 and indices.shape[0] == 1
        token_count = indices.shape[0]
        if contact.shape[0] != token_count:
            length = min(token_count, contact.shape[0])
            indices = indices[:length]
            contact = contact[:length]
        for part_idx in range(3):
            # This intentionally preserves the old bug: with [1,T,3], this
            # selects time step `part_idx` and all three channels.
            part_codes = indices[:, part_idx]
            np.add.at(total[part_idx], part_codes, 1)
            np.add.at(positive[part_idx], part_codes, contact)
        used += 1
        require(cached_batch_axis,
                f'legacy replay expected [1,T,3]: {index_path}: {indices.shape}')
    probabilities = positive / (total + 1e-8)
    return probabilities, total, positive, used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--current-cache-root', type=Path,
        default=ROOT / 'checkpoints/vq_precleanup_raw12288',
    )
    parser.add_argument(
        '--legacy-cache-root', type=Path,
        default=MMM_ROOT / 'checkpoints/vq',
    )
    parser.add_argument(
        '--legacy-map-copy', type=Path,
        default=ROOT / 'checkpoints/vq_precleanup_raw12288/'
        'code_contact_probs_mmm_legacy_full_axis_bug.npy',
    )
    parser.add_argument(
        '--vq-checkpoint', type=Path,
        default=ROOT / 'checkpoints/vq_precleanup_raw12288/net_175000.pth',
    )
    parser.add_argument(
        '--split-root', type=Path,
        default=ROOT / 'dataset/interx/splits',
    )
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'output/hhc_contact_guidance_map_audit.json',
    )
    args = parser.parse_args()

    current_root = args.current_cache_root.resolve()
    legacy_root = args.legacy_cache_root.resolve()
    current_map_path = current_root / 'code_contact_probs.npy'
    legacy_source_path = legacy_root / 'code_contact_probs.npy'
    legacy_copy_path = args.legacy_map_copy.resolve()
    vq_checkpoint_path = args.vq_checkpoint.resolve()
    current_manifest_path = current_root / 'codebook/_cache_manifest.json'
    for path in (
        current_map_path, legacy_source_path, legacy_copy_path,
        current_manifest_path, vq_checkpoint_path,
    ):
        require(path.is_file(), f'missing contact-map input: {path}')

    current_map = np.load(current_map_path)
    legacy_source = np.load(legacy_source_path)
    legacy_copy = np.load(legacy_copy_path)
    for label, array in (
        ('current', current_map), ('legacy_source', legacy_source),
        ('legacy_copy', legacy_copy),
    ):
        require(array.shape == (3, 12288),
                f'{label} map shape changed: {array.shape}')
        require(np.isfinite(array).all(), f'{label} map is not finite')
    require(np.array_equal(legacy_source, legacy_copy),
            'legacy HHC copy differs from the MMM source map')
    require(sha256_file(legacy_source_path) == sha256_file(legacy_copy_path),
            'legacy source/copy file hashes differ')

    splits = split_names(args.split_root.resolve())
    require({key: len(value) for key, value in splits.items()}
            == {'train': 9110, 'val': 570, 'test': 1708},
            'official Inter-X split counts changed')
    current_codes = cache_names(current_root / 'codebook')
    current_contacts = cache_names(current_root / 'contact_token_label')
    require(current_codes == current_contacts,
            'current code/contact cache names differ')
    current_manifest = json.loads(
        current_manifest_path.read_text(encoding='utf-8')
    )
    require(current_manifest.get('contact_map_scope') == 'train_only'
            and current_manifest.get('contact_map_samples') == 9095
            and current_manifest.get('contact_map_estimator')
            == 'empirical_frequency_unseen_neutral_0.5'
            and current_manifest.get('contact_map_sha256')
            == sha256_file(current_map_path),
            'current cache manifest does not bind the fair contact map')
    require(
        current_manifest.get('vq_checkpoint_sha256')
        == sha256_file(vq_checkpoint_path),
        'current cache manifest does not bind the requested VQ checkpoint',
    )
    included_train = splits['train'] & current_codes
    recomputed_current, current_counts, current_used = build_code_contact_map(
        current_root / 'codebook', current_root / 'contact_token_label',
        12288, alpha=0.0, included_names=included_train,
    )
    require(current_used == 9095, 'fair map did not use 9,095 train samples')
    require(np.array_equal(recomputed_current, current_map),
            'fair train-only contact map is not exactly reproducible')

    legacy_codes = cache_names(legacy_root / 'codebook')
    legacy_contacts = cache_names(legacy_root / 'contact_token_label')
    require(legacy_codes == legacy_contacts,
            'legacy code/contact cache names differ')
    legacy_scope = {
        split: len(names & legacy_codes) for split, names in splits.items()
    }
    require(legacy_scope == {'train': 9095, 'val': 569, 'test': 1706},
            f'unexpected legacy cache split scope: {legacy_scope}')
    replayed, replay_counts, replay_positive, replay_used = (
        replay_legacy_axis_bug(
            legacy_root / 'codebook', legacy_root / 'contact_token_label', 12288
        )
    )
    require(replay_used == 11370, 'legacy replay sample count changed')
    require(np.array_equal(replayed, legacy_source),
            'legacy axis-bug replay does not exactly reconstruct MMM map')
    require(replay_counts.sum(axis=1).tolist()
            == [34110.0, 34110.0, 34110.0],
            'legacy replay no longer observes exactly three codes/sample/row')
    require(replay_positive.sum(axis=1).tolist()
            == [2718.0, 2718.0, 2718.0],
            'legacy replay contact sums changed')

    result = {
        'schema': 'hhc_contact_guidance_map_audit_v1',
        'status': 'pass',
        'audit_script': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(Path(__file__).resolve()),
        },
        'vq_checkpoint': {
            'path': str(vq_checkpoint_path),
            'sha256': sha256_file(vq_checkpoint_path),
        },
        'paper_formula_fair_map': {
            'path': str(current_map_path),
            'sha256': sha256_file(current_map_path),
            'dtype': str(current_map.dtype),
            'shape': list(current_map.shape),
            'scope': 'official_train_only',
            'samples': current_used,
            'axis_semantics': '[Body, Hand, Root] over every temporal token',
            'unseen_code_probability': 0.5,
            'exact_recomputation_match': True,
            'dead_codes': (current_counts == 0).sum(axis=1).astype(int).tolist(),
            'channel_statistics': map_stats(current_map),
            'cache_manifest': {
                'path': str(current_manifest_path),
                'sha256': sha256_file(current_manifest_path),
            },
        },
        'mmm_legacy_full_axis_bug_map': {
            'source_path': str(legacy_source_path),
            'copied_path': str(legacy_copy_path),
            'sha256': sha256_file(legacy_source_path),
            'copy_byte_identical': True,
            'dtype': str(legacy_source.dtype),
            'shape': list(legacy_source.shape),
            'scope_by_official_split': legacy_scope,
            'samples': replay_used,
            'uses_test_split': True,
            'test_samples_used': legacy_scope['test'],
            'axis_bug': (
                'cached indices are [1,T,3], but legacy code uses '
                'indices[:, part_idx], selecting temporal rows 0/1/2 and all '
                'three code channels instead of Body/Hand/Root across time'
            ),
            'codes_counted_per_sample_per_output_row': 3,
            'total_counts_per_output_row': replay_counts.sum(axis=1).tolist(),
            'positive_counts_per_output_row': replay_positive.sum(axis=1).tolist(),
            'exact_bug_replay_match': True,
            'channel_statistics': map_stats(legacy_source),
            'source_implementation': {
                'path': str(MMM_ROOT / 'test_data.py'),
                'sha256': sha256_file(MMM_ROOT / 'test_data.py'),
            },
        },
        'policy': {
            'fair_main_table': 'paper_formula_fair_map',
            'legacy_map_role': (
                'author-implementation reproduction sensitivity only; excluded '
                'from the fair main table because it consumes 1,706 test '
                'samples and exactly reproduces an axis bug'
            ),
            'active_training_or_checkpoint_mutated': False,
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
