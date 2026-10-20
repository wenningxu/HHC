#!/usr/bin/env python3
"""Export request-bound HHC InterHuman main or MM generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import clip
import h5py
import numpy as np
import torch

from dataset.interhuman_generated_h5 import (
    MEAN_SHA256,
    STD_SHA256,
    mm_repeat_seed,
    sha256_file,
)
from utils.interhuman_artifacts import load_hhc_interhuman_mgt
from utils.interhuman_mgt_evaluation import generate_interhuman_pair


class TextCLIP(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, text):
        word = self.model.token_embedding(text).type(self.model.dtype)
        word = word + self.model.positional_embedding.type(self.model.dtype)
        word = self.model.transformer(word.permute(1, 0, 2))
        word = self.model.ln_final(word).permute(1, 0, 2).float()
        return self.model.encode_text(text).float(), word


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mgt-checkpoint', required=True)
    parser.add_argument('--mgt-training-manifest', required=True)
    parser.add_argument('--request-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('main', 'mm'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--sampling-steps', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--cfg-scale', type=float, default=3.5)
    return parser.parse_args()


def expected_metadata(args, request_path, request, provenance):
    model_config_sha256 = hashlib.sha256(
        json.dumps(
            provenance['model_contract'], sort_keys=True, separators=(',', ':')
        ).encode()
    ).hexdigest()
    return {
        'request_file_sha256': sha256_file(request_path),
        'request_content_sha256': request['request_sha256'],
        'normalization_mean_sha256': MEAN_SHA256,
        'normalization_std_sha256': STD_SHA256,
        'sampler': 'sample_intra_cas',
        'sampling_steps': int(args.sampling_steps),
        'temperature': float(args.temperature),
        'cfg_scale': float(args.cfg_scale),
        'cascade_order': 'root,body,hands_arms',
        'generation_length_policy': 'floor_processed_length_div4_then_zero_pad',
        'per_sample_seed_contract': 'sha256(sample_seed:repeat:repeat_index), batch_invariant',
        'generation_exporter_sha256': sha256_file(Path(__file__).resolve()),
        'interhuman_generation_bridge_sha256': sha256_file(
            ROOT / 'utils/interhuman_mgt_evaluation.py'
        ),
        'checkpoint': provenance['checkpoint'],
        'checkpoint_sha256': provenance['checkpoint_sha256'],
        'training_manifest': provenance['training_manifest'],
        'training_manifest_sha256': provenance['training_manifest_sha256'],
        'training_identity_sha256': provenance['training_identity_sha256'],
        'model_config_sha256': model_config_sha256,
        'mgt_shape_condition': False,
        'contact_conditioning': False,
        'vq_checkpoint': provenance['vq']['checkpoint'],
        'vq_checkpoint_sha256': provenance['vq']['checkpoint_sha256'],
        'generation_batch_size': int(args.batch_size),
    }


def _strings(handle, name, values):
    handle.create_dataset(
        name, data=np.asarray(values, dtype=object),
        dtype=h5py.string_dtype('utf-8'),
    )


def initialize_or_validate(path, args, request_path, request, provenance):
    rows = request['requests']
    count = len(rows)
    if args.mode == 'main':
        if count != 1056:
            raise ValueError(f'formal main export requires 1056 requests, got {count}')
        schema = 'interhuman_fair_generation_v1'
    else:
        if count != 100 or int(request.get('mm_repeats', 0)) != 30:
            raise ValueError('formal MM export requires 100 prompts x 30 repeats')
        schema = 'interhuman_fair_mm_generation_v1'
    metadata = expected_metadata(args, request_path, request, provenance)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, 'w') as handle:
            handle.attrs['schema'] = schema
            handle.attrs['status'] = 'running'
            for key, value in metadata.items():
                handle.attrs[key] = value
            _strings(handle, 'names', [row['identifier'] for row in rows])
            _strings(handle, 'texts', [row['caption'] for row in rows])
            handle.create_dataset(
                'lengths', data=np.asarray(
                    [row['processed_length'] for row in rows], dtype=np.int64
                ),
            )
            if args.mode == 'main':
                shape = (count, 300, 262)
                chunks = (1, 300, 262)
                handle.create_dataset('completed', shape=(count,), dtype=np.bool_)
            else:
                repeats = int(request['mm_repeats'])
                shape = (count, repeats, 300, 262)
                chunks = (1, 1, 300, 262)
                repeat_seeds = np.asarray([
                    [mm_repeat_seed(row['sample_seed'], repeat) for repeat in range(repeats)]
                    for row in rows
                ], dtype=np.uint64)
                handle.create_dataset('repeat_seeds', data=repeat_seeds)
                handle.create_dataset(
                    'completed', shape=(count, repeats), dtype=np.bool_
                )
            handle.create_dataset(
                'motion1', shape=shape, dtype=np.float32,
                chunks=chunks, compression='lzf', fillvalue=0,
            )
            handle.create_dataset(
                'motion2', shape=shape, dtype=np.float32,
                chunks=chunks, compression='lzf', fillvalue=0,
            )
            handle.flush()
    handle = h5py.File(path, 'r+')
    if handle.attrs.get('schema') != schema:
        handle.close()
        raise ValueError('existing InterHuman generation has another schema')
    for key, expected in metadata.items():
        actual = handle.attrs.get(key)
        if actual != expected:
            handle.close()
            raise ValueError(
                f'existing InterHuman generation {key} changed: {actual!r} != {expected!r}'
            )
    return handle


def main():
    args = parse_args()
    if (args.sampling_steps, args.temperature, args.cfg_scale) != (50, 0.5, 3.5):
        raise ValueError('formal HHC InterHuman export is locked to common50')
    if args.batch_size <= 0:
        raise ValueError('generation batch size must be positive')
    device = torch.device(args.device)
    transformer, vq, provenance = load_hhc_interhuman_mgt(
        args.mgt_checkpoint, args.mgt_training_manifest, device=device,
        inference_conditioning=True,
    )
    request_path = Path(args.request_file).resolve()
    request = json.loads(request_path.read_text())
    output = Path(args.output).resolve()
    handle = initialize_or_validate(
        output, args, request_path, request, provenance
    )
    clip_base, _ = clip.load('ViT-B/32', device=device, jit=False)
    clip.model.convert_weights(clip_base)
    text_encoder = TextCLIP(clip_base.eval())
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    rows = request['requests']

    if args.mode == 'main':
        tasks = [
            (index, None) for index in range(len(rows))
            if not bool(handle['completed'][index])
        ]
    else:
        tasks = [
            (index, repeat)
            for repeat in range(int(request['mm_repeats']))
            for index in range(len(rows))
            if not bool(handle['completed'][index, repeat])
        ]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    for start in range(0, len(tasks), args.batch_size):
        batch_tasks = tasks[start:start + args.batch_size]
        batch_rows = [rows[index] for index, _ in batch_tasks]
        texts = [row['caption'] for row in batch_rows]
        lengths = torch.tensor(
            [row['processed_length'] for row in batch_rows],
            dtype=torch.long, device=device,
        )
        seeds = [
            mm_repeat_seed(row['sample_seed'], 0 if repeat is None else repeat)
            for row, (_, repeat) in zip(batch_rows, batch_tasks)
        ]
        tokens = clip.tokenize(texts, truncate=True).to(device)
        clip_feature, words = text_encoder(tokens)
        first, second = generate_interhuman_pair(
            transformer, vq, clip_feature, words, lengths,
            sampling_steps=50, temperature=0.5, cfg_scale=3.5,
            sample_seeds=seeds,
        )
        first = first.cpu().numpy().astype(np.float32, copy=False)
        second = second.cpu().numpy().astype(np.float32, copy=False)
        for local, (index, repeat) in enumerate(batch_tasks):
            if repeat is None:
                handle['motion1'][index] = first[local]
                handle['motion2'][index] = second[local]
            else:
                handle['motion1'][index, repeat] = first[local]
                handle['motion2'][index, repeat] = second[local]
        # A completion marker is a recovery boundary: motion chunks must be
        # durable before the marker can make a resumed export skip them.
        handle.flush()
        for index, repeat in batch_tasks:
            if repeat is None:
                handle['completed'][index] = True
            else:
                handle['completed'][index, repeat] = True
        handle.flush()
        print(json.dumps({
            'mode': args.mode, 'generated': start + len(batch_tasks),
            'pending_at_start': len(tasks),
        }), flush=True)
    if not bool(np.asarray(handle['completed'][:]).all()):
        handle.close()
        raise RuntimeError('InterHuman generation ended with incomplete records')
    handle.attrs['status'] = 'complete'
    handle.flush()
    handle.close()
    print(json.dumps({
        'status': 'complete', 'mode': args.mode, 'output': str(output),
        'sha256': sha256_file(output),
    }, indent=2))


if __name__ == '__main__':
    main()
