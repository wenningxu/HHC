#!/usr/bin/env python
"""Freeze the exact stochastic validation inputs used for MGT selection."""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import dataset_interx_TM
from utils.word_vectorizer import WordVectorizer


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def seed_evaluation(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def update_array(digest, label, value):
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    header = json.dumps({
        'label': label, 'shape': list(array.shape), 'dtype': str(array.dtype),
    }, sort_keys=True).encode('utf-8')
    digest.update(len(header).to_bytes(8, 'big'))
    digest.update(header)
    digest.update(array.tobytes())


def collect(loader, seed):
    seed_evaluation(seed)
    records = []
    inputs = hashlib.sha256()
    for batch_index, batch in enumerate(loader):
        word_embeddings, pos_one_hots, captions, sent_len, motions, lengths, _, names, _, _, contact = batch
        names = [str(name) for name in names]
        captions = [str(caption) for caption in captions]
        lengths_list = [int(value) for value in lengths.tolist()]
        records.extend({
            'batch': batch_index,
            'name': name,
            'caption': caption,
            'length': length,
        } for name, caption, length in zip(names, captions, lengths_list))
        for label, value in (
            ('word_embeddings', word_embeddings),
            ('pos_one_hots', pos_one_hots),
            ('sent_len', sent_len),
            ('motions', motions),
            ('lengths', lengths),
            ('contact', contact),
        ):
            update_array(inputs, f'{batch_index}:{label}', value)
    serialized = json.dumps(
        records, ensure_ascii=False, separators=(',', ':'), sort_keys=True
    ).encode('utf-8')
    return records, hashlib.sha256(serialized).hexdigest(), inputs.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', default='dataset/interx/splits/val.txt')
    parser.add_argument('--motions', default='dataset/interx/motions/val.h5')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=500123)
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/mgt_validation_selection_manifest.json'),
    )
    args = parser.parse_args()

    vectorizer = WordVectorizer('./dataset/interx/glove', 'hhi_vab')
    loader = dataset_interx_TM.DATALoader(
        'hhi', args.batch_size, args.split, vectorizer, args.motions,
        num_workers=args.num_workers, unit_length=4, shuffle=True,
    )
    first, records_sha256, inputs_sha256 = collect(loader, args.seed)
    second, second_records_sha256, second_inputs_sha256 = collect(
        loader, args.seed
    )
    if first != second or records_sha256 != second_records_sha256:
        raise RuntimeError('fixed validation seed did not reproduce request records')
    if inputs_sha256 != second_inputs_sha256:
        raise RuntimeError('fixed validation seed did not reproduce evaluator inputs')

    eligible_names = set(loader.dataset.data_dict)
    evaluated_names = [record['name'] for record in first]
    if len(evaluated_names) != len(set(evaluated_names)):
        raise RuntimeError('validation selection contains duplicate sample IDs')
    result = {
        'schema': 'hhc_mgt_validation_selection_v1',
        'status': 'pass',
        'implementation': {
            'path': str(Path(__file__).resolve()),
            'bytes': Path(__file__).stat().st_size,
            'sha256': sha256_file(__file__),
        },
        'seed': args.seed,
        'seed_semantics': 'training_seed + 500000',
        'split': str(Path(args.split).resolve()),
        'split_sha256': sha256_file(args.split),
        'motions': str(Path(args.motions).resolve()),
        'motions_sha256': sha256_file(args.motions),
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'drop_last': True,
        'shuffle': True,
        'eligible_samples': len(eligible_names),
        'evaluated_samples': len(first),
        'batches': len(loader),
        'omitted_by_drop_last': sorted(eligible_names - set(evaluated_names)),
        'ordered_request_records_sha256': records_sha256,
        'ordered_evaluator_inputs_sha256': inputs_sha256,
        'repeat_deterministic': True,
        'records': first,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(
        args.output.suffix + f'.tmp.{os.getpid()}'
    )
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, args.output)
    print(json.dumps({
        key: result[key] for key in (
            'status', 'eligible_samples', 'evaluated_samples', 'batches',
            'ordered_request_records_sha256',
            'ordered_evaluator_inputs_sha256', 'repeat_deterministic',
        )
    }, indent=2))


if __name__ == '__main__':
    main()
