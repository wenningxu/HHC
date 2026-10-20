#!/usr/bin/env python
"""Predict Inter-X contact timing from text using train-only kNN priors.

This deliberately simple, auditable baseline answers whether HHC requires an
oracle frame-level contact schedule.  It selects k and a probability threshold
on validation only, then predicts the frozen test requests without reading their
contact labels.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.word_vectorizer import WordVectorizer


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def atomic_npz(path, **payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'wb') as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def parse_text_records(path):
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding='utf-8').splitlines(), start=1
    ):
        fields = line.strip().split('#')
        if len(fields) < 2 or not fields[0].strip():
            raise ValueError(f'{path}:{line_number}: malformed text record')
        records.append((fields[0].strip(), fields[1].strip().split()))
    if not records:
        raise ValueError(f'{path}: no captions')
    return records


def marginal_crop_contact(raw_contact, width=150):
    """Expected fixed-width mask under an unknown uniformly sampled crop."""
    contact = np.asarray(raw_contact).reshape(-1).astype(np.float32)
    if not np.isin(contact, (0, 1)).all():
        raise ValueError('contact annotation is not binary')
    if len(contact) <= width:
        result = np.zeros(width, dtype=np.float32)
        result[:len(contact)] = contact
        return result
    # Mean over every legal crop represents timing uncertainty without looking
    # at the held-out request's randomly selected crop.
    windows = np.lib.stride_tricks.sliding_window_view(contact, width)
    return windows.mean(axis=0, dtype=np.float64).astype(np.float32)


def encode_tokens(tokens, vectorizer):
    vectors = []
    pos_vectors = []
    for token in tokens:
        if '/' not in token:
            token = token.lower() + '/OTHER'
        word, pos = vectorizer[token]
        vectors.append(word)
        pos_vectors.append(pos)
    if not vectors:
        word, pos = vectorizer['unk/OTHER']
        vectors, pos_vectors = [word], [pos]
    feature = np.concatenate((
        np.mean(vectors, axis=0), np.mean(pos_vectors, axis=0)
    )).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('non-finite or zero text feature')
    return feature / norm


def load_split_bank(
    split_path, motion_path, text_dir, contact_dir, vectorizer
):
    names = [
        line.strip() for line in split_path.read_text().splitlines()
        if line.strip()
    ]
    if len(names) != len(set(names)):
        raise ValueError(f'{split_path}: duplicate sample names')
    features, targets, sample_names = [], [], []
    artifacts = []
    missing_contact_count = 0
    with h5py.File(motion_path, 'r') as motions:
        for name in names:
            text_path = text_dir / f'{name}.txt'
            contact_path = contact_dir / f'{name}.npy'
            if not text_path.is_file() or name not in motions:
                raise FileNotFoundError(
                    f'{name}: missing text or split motion'
                )
            if contact_path.is_file():
                raw_contact = np.load(contact_path, allow_pickle=False)
                contact_hash = sha256_file(contact_path)
                contact_source = 'annotation_file'
            else:
                # Match MotionDatasetV2HHI exactly: an absent annotation means
                # a zero contact vector with the motion's native frame count.
                raw_contact = np.zeros(motions[name].shape[0], dtype=np.float32)
                contact_hash = None
                contact_source = 'dataset_zero_fallback'
                missing_contact_count += 1
            target = marginal_crop_contact(raw_contact)
            for _, tokens in parse_text_records(text_path):
                features.append(encode_tokens(tokens, vectorizer))
                targets.append(target)
                sample_names.append(name)
            artifacts.append({
                'name': name,
                'text_sha256': sha256_file(text_path),
                'contact_sha256': contact_hash,
                'contact_source': contact_source,
                'native_frames': int(motions[name].shape[0]),
            })
    return (
        np.stack(features), np.stack(targets), np.asarray(sample_names),
        canonical_sha256(artifacts), len(names), missing_contact_count,
    )


def topk_probabilities(query, bank, targets, k, batch_size=128):
    if not 1 <= k <= len(bank):
        raise ValueError(f'k={k} outside training bank size {len(bank)}')
    outputs = []
    for start in range(0, len(query), batch_size):
        similarities = query[start:start + batch_size] @ bank.T
        indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
        outputs.append(targets[indices].mean(axis=1))
    return np.concatenate(outputs).astype(np.float32)


def binary_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = tp + fp + fn
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': tp / union if union else 1.0,
        'predicted_positive_rate': float(prediction.mean()),
        'target_positive_rate': float(target.mean()),
    }


def request_features(requests_path, text_dir, vectorizer):
    with np.load(requests_path, allow_pickle=False) as request:
        names = request['names'].astype(str)
        captions = request['captions'].astype(str)
        lengths = request['lengths'].astype(np.int64)
    if len(names) != len(set(names.tolist())):
        raise ValueError('frozen requests contain duplicate names')
    features = []
    duplicate_caption_records = 0
    for name, caption in zip(names, captions):
        records = parse_text_records(text_dir / f'{name}.txt')
        matching = [tokens for candidate, tokens in records if candidate == caption]
        if not matching:
            raise ValueError(
                f'{name}: frozen caption has no exact source match'
            )
        encoded = np.stack([
            encode_tokens(tokens, vectorizer) for tokens in matching
        ])
        feature = encoded.mean(axis=0)
        feature /= np.linalg.norm(feature)
        features.append(feature.astype(np.float32))
        duplicate_caption_records += len(matching) - 1
    return (
        names, captions, lengths, np.stack(features),
        duplicate_caption_records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-split', type=Path,
                        default=Path('dataset/interx/splits/train.txt'))
    parser.add_argument('--val-split', type=Path,
                        default=Path('dataset/interx/splits/val.txt'))
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--train-motion', type=Path,
                        default=Path('dataset/interx/motions/train.h5'))
    parser.add_argument('--val-motion', type=Path,
                        default=Path('dataset/interx/motions/val.h5'))
    parser.add_argument('--text-dir', type=Path,
                        default=Path('dataset/interx/texts_processed'))
    parser.add_argument('--contact-dir', type=Path,
                        default=Path('dataset/interx/contact'))
    parser.add_argument('--glove-dir', type=Path,
                        default=Path('dataset/interx/glove'))
    parser.add_argument('--glove-prefix', default='hhi_vab')
    parser.add_argument('--k-values', default='1,3,5,10,20,50')
    parser.add_argument('--thresholds', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    paths = [
        args.train_split.resolve(), args.val_split.resolve(),
        args.requests.resolve(), args.train_motion.resolve(),
        args.val_motion.resolve(), args.text_dir.resolve(),
        args.contact_dir.resolve(), args.glove_dir.resolve(),
    ]
    (
        train_split, val_split, requests, train_motion, val_motion,
        text_dir, contact_dir, glove_dir,
    ) = paths
    train_names = set(train_split.read_text().split())
    val_names = set(val_split.read_text().split())
    with np.load(requests, allow_pickle=False) as frozen:
        request_names = set(frozen['names'].astype(str).tolist())
    if train_names & val_names or train_names & request_names or val_names & request_names:
        raise ValueError('train/validation/frozen-test names are not disjoint')

    vectorizer = WordVectorizer(str(glove_dir), args.glove_prefix)
    (
        train_x, train_y, _, train_artifacts_hash, train_sequence_count,
        train_missing_contact_count,
    ) = (
        load_split_bank(
            train_split, train_motion, text_dir, contact_dir, vectorizer
        )
    )
    (
        val_x, val_y_prob, _, val_artifacts_hash, val_sequence_count,
        val_missing_contact_count,
    ) = (
        load_split_bank(
            val_split, val_motion, text_dir, contact_dir, vectorizer
        )
    )
    val_y = val_y_prob >= 0.5
    k_values = [int(value) for value in args.k_values.split(',')]
    thresholds = [float(value) for value in args.thresholds.split(',')]
    if len(k_values) != len(set(k_values)) or len(thresholds) != len(set(thresholds)):
        raise ValueError('hyperparameter grids contain duplicates')
    grid = []
    for k in k_values:
        probability = topk_probabilities(val_x, train_x, train_y, k)
        for threshold in thresholds:
            metrics = binary_metrics(probability >= threshold, val_y)
            grid.append({'k': k, 'threshold': threshold, **metrics})
    # Maximize validation F1 only; stable deterministic ties prefer smaller k
    # and then the larger threshold (the more conservative schedule).
    selected = sorted(
        grid, key=lambda row: (-row['f1'], row['k'], -row['threshold'])
    )[0]

    (
        names, captions, lengths, test_x, duplicate_request_caption_records,
    ) = request_features(requests, text_dir, vectorizer)
    probabilities = topk_probabilities(
        test_x, train_x, train_y, selected['k']
    )
    contact = (probabilities >= selected['threshold']).astype(np.int8)
    for index, length in enumerate(lengths):
        if not 1 <= int(length) <= 150:
            raise ValueError(f'invalid request length: {length}')
        contact[index, int(length):] = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    request_hash = sha256_file(requests)
    atomic_npz(
        args.output,
        names=names,
        contact=contact,
        probabilities=probabilities,
        requests_sha256=np.asarray(request_hash),
        selected_k=np.asarray(selected['k']),
        selected_threshold=np.asarray(selected['threshold']),
    )
    manifest = {
        'schema': 'hhc_interx_text_only_contact_timing_knn_v1',
        'status': 'complete',
        'method': (
            'cosine kNN over mean Inter-X GloVe+POS caption features; '
            'train contact target is the marginal over all legal 150-frame crops'
        ),
        'test_contact_labels_read': False,
        'selection_metric': 'validation frame-level micro F1',
        'tie_break': 'smaller k, then larger threshold',
        'selected': selected,
        'validation_grid': grid,
        'train': {
            'split': str(train_split),
            'split_sha256': sha256_file(train_split),
            'sequence_count': train_sequence_count,
            'caption_count': len(train_x),
            'motion_file': str(train_motion),
            'missing_contact_zero_fallback_count': train_missing_contact_count,
            'source_artifacts_sha256': train_artifacts_hash,
        },
        'validation': {
            'split': str(val_split),
            'split_sha256': sha256_file(val_split),
            'sequence_count': val_sequence_count,
            'caption_count': len(val_x),
            'motion_file': str(val_motion),
            'missing_contact_zero_fallback_count': val_missing_contact_count,
            'source_artifacts_sha256': val_artifacts_hash,
        },
        'requests': str(requests),
        'requests_sha256': request_hash,
        'request_count': len(names),
        'duplicate_exact_caption_records_averaged': (
            duplicate_request_caption_records
        ),
        'output': str(args.output.resolve()),
        'output_sha256': sha256_file(args.output),
        'glove': {
            'directory': str(glove_dir),
            'prefix': args.glove_prefix,
            'data_sha256': sha256_file(glove_dir / f'{args.glove_prefix}_data.npy'),
            'words_sha256': sha256_file(glove_dir / f'{args.glove_prefix}_words.pkl'),
            'index_sha256': sha256_file(glove_dir / f'{args.glove_prefix}_idx.pkl'),
        },
        'implementation_sha256': sha256_file(__file__),
        'elapsed_seconds': time.time() - started,
    }
    atomic_json(args.output.with_suffix(args.output.suffix + '.manifest.json'), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
