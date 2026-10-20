#!/usr/bin/env python3
"""Export released InterMask VQ reconstructions on frozen InterHuman requests.

This process intentionally resolves ``models`` and ``data`` from the released
InterMask tree before importing HHC's separate ``dataset`` package.  Keeping
the namespaces isolated prevents accidentally evaluating an architectural
rewrite under the official-baseline label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
INTERMASK_ROOT = Path('/mnt/8T-HDD/wenning/InterMask').resolve()
sys.path.insert(0, str(INTERMASK_ROOT))
sys.path.insert(1, str(ROOT))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.vq.model import RVQVAE
from dataset.interhuman_fair_requests import (
    InterHumanFrozenRequestDataset,
    InterHumanValidationRequestDataset,
)
from dataset.interhuman_generated_h5 import MEAN_SHA256, STD_SHA256, sha256_file


CHECKPOINT = INTERMASK_ROOT / 'checkpoints/interhuman/vq_default/model/best_fid.tar'
CHECKPOINT_SHA256 = 'af57bbdde94ed4d4a01f3ba5f9fe24f0012c2a37b3e38c2fa57a2a91ebcc1272'
OPT_PATH = INTERMASK_ROOT / 'checkpoints/interhuman/vq_default/opt.txt'
SOURCE_HASHES = {
    'models/vq/model.py': 'db3f33b367d0f18cbf46816ab426fe75ba25ac92aafbe1baa04e9e7b17de55c5',
    'models/vq/encdec.py': '555a67b9ee78dd67a69100b733b5dbf9e323627719cee44b751110b7d18442b5',
    'models/vq/residual_vq.py': '164b8a2ce129b382d285448f51341f19068db61a1efae431b74db83371bd4be8',
    'models/vq/quantizer.py': '230dd5919cf1aaaf8911d1d848cb65d46c7de2197159a04131f2b22d0c8b8426',
}


def require_official_sources():
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError('released InterMask InterHuman VQ checkpoint hash changed')
    for relative, expected in SOURCE_HASHES.items():
        path = INTERMASK_ROOT / relative
        if sha256_file(path) != expected:
            raise ValueError(f'released InterMask VQ source changed: {path}')
    imported = Path(sys.modules[RVQVAE.__module__].__file__).resolve()
    if INTERMASK_ROOT not in imported.parents:
        raise RuntimeError(f'RVQVAE resolved outside released InterMask tree: {imported}')


def build_official_vq(device):
    options = SimpleNamespace(
        joints_num=22,
        dataset_name='interhuman',
        num_quantizers=1,
        shared_codebook=False,
        quantize_dropout_prob=0.2,
        mu=0.99,
    )
    model = RVQVAE(
        options,
        input_dim=12,
        nb_code=1024,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=2,
        dilation_growth_rate=3,
        activation='relu',
        norm=None,
    )
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    if set(checkpoint) != {'vq_model', 'opt_vq_model', 'scheduler', 'ep', 'total_it'}:
        raise ValueError(f'unexpected official VQ checkpoint fields {sorted(checkpoint)}')
    model.load_state_dict(checkpoint['vq_model'], strict=True)
    return model.to(device).eval(), {
        'checkpoint_epoch': int(checkpoint['ep']),
        'checkpoint_total_iteration': int(checkpoint['total_it']),
    }


@torch.no_grad()
def reconstruct_person(model, motion, lengths):
    output = torch.zeros_like(motion)
    for length in lengths.unique(sorted=True).tolist():
        selected = lengths == int(length)
        padded = int(math.ceil(int(length) / 4) * 4)
        if padded > motion.shape[1]:
            raise ValueError(f'official VQ length {length} needs unavailable {padded} frames')
        model_input = torch.zeros(
            int(selected.sum()), padded, 262,
            dtype=motion.dtype, device=motion.device,
        )
        model_input[:, :int(length)] = motion[selected, :int(length)]
        prediction, _, _ = model(model_input)
        if prediction.shape != model_input.shape:
            raise ValueError(f'official VQ output shape {prediction.shape} != {model_input.shape}')
        output[selected, :int(length)] = prediction[:, :int(length)]
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=str(ROOT / 'output/interhuman_fair_requests_v3/manifest.json'))
    parser.add_argument('--replication', type=int)
    parser.add_argument('--validation-manifest')
    parser.add_argument('--output')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--preflight-report')
    return parser.parse_args()


def main():
    args = parse_args()
    require_official_sources()
    if args.preflight_only:
        model, checkpoint_metadata = build_official_vq('cpu')
        report = {
            'schema': 'official_intermask_interhuman_vq_preflight_v1',
            'status': 'pass',
            'checkpoint_sha256': CHECKPOINT_SHA256,
            'strict_state_load': True,
            'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'official_source_sha256': SOURCE_HASHES,
            'official_opt_path': str(OPT_PATH),
            'official_opt_sha256': sha256_file(OPT_PATH),
            **checkpoint_metadata,
        }
        if args.preflight_report:
            report_path = Path(args.preflight_report).resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = report_path.with_name(report_path.name + f'.tmp.{os.getpid()}')
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
            os.replace(temporary, report_path)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.output is None or (args.replication is None) == (args.validation_manifest is None):
        raise ValueError(
            'choose exactly one of --replication or --validation-manifest and provide --output'
        )
    if args.validation_manifest is not None:
        manifest_path = Path(args.validation_manifest).resolve()
        manifest = json.loads(manifest_path.read_text())
        entry = manifest['request']
        dataset = InterHumanValidationRequestDataset(manifest_path)
        request_path = dataset.request_path
        split = 'val'
        expected_count = 480
        expected_batches = 5
    else:
        manifest_path = Path(args.manifest).resolve()
        manifest = json.loads(manifest_path.read_text())
        entries = {int(row['replication']): row for row in manifest['replications']}
        if args.replication not in entries:
            raise ValueError(f'replication {args.replication} absent from frozen requests')
        entry = entries[args.replication]
        request_path = Path(entry['path']).resolve()
        dataset = InterHumanFrozenRequestDataset(args.replication, manifest_path)
        split = 'test'
        expected_count = 1056
        expected_batches = 11
    loader = DataLoader(dataset, batch_size=96, shuffle=False, drop_last=True, num_workers=0)
    if len(dataset) != expected_count or len(loader) != expected_batches:
        raise ValueError(
            f'official VQ export requires {expected_count} {split} requests in '
            f'{expected_batches} batches'
        )
    device = torch.device(args.device)
    model, checkpoint_metadata = build_official_vq(device)

    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f'.tmp.{os.getpid()}')
    if destination.exists() or temporary.exists():
        raise FileExistsError(f'refusing to overwrite VQ reconstruction {destination}')
    strings = h5py.string_dtype('utf-8')
    with h5py.File(temporary, 'w') as handle:
        handle.attrs['schema'] = 'interhuman_fair_generation_v1'
        handle.attrs['status'] = 'running'
        handle.attrs['method'] = 'intermask_official_vq_reconstruction'
        handle.attrs['split'] = split
        handle.attrs['exporter_sha256'] = sha256_file(Path(__file__).resolve())
        handle.attrs['official_source_contract_sha256'] = hashlib.sha256(
            json.dumps(SOURCE_HASHES, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        handle.attrs['request_manifest_sha256'] = sha256_file(manifest_path)
        handle.attrs['checkpoint_sha256'] = CHECKPOINT_SHA256
        handle.attrs['vq_checkpoint_sha256'] = CHECKPOINT_SHA256
        handle.attrs['model_config_sha256'] = sha256_file(OPT_PATH)
        handle.attrs['request_file_sha256'] = entry['sha256']
        handle.attrs['request_content_sha256'] = entry['request_sha256']
        handle.attrs['normalization_mean_sha256'] = MEAN_SHA256
        handle.attrs['normalization_std_sha256'] = STD_SHA256
        for key, value in checkpoint_metadata.items():
            handle.attrs[key] = value
        names_ds = handle.create_dataset('names', (expected_count,), dtype=strings)
        texts_ds = handle.create_dataset('texts', (expected_count,), dtype=strings)
        lengths_ds = handle.create_dataset('lengths', (expected_count,), dtype=np.int64)
        motion1_ds = handle.create_dataset(
            'motion1', (expected_count, 300, 262), dtype=np.float32,
            chunks=(1, 300, 262), compression='lzf',
        )
        motion2_ds = handle.create_dataset(
            'motion2', (expected_count, 300, 262), dtype=np.float32,
            chunks=(1, 300, 262), compression='lzf',
        )
        offset = 0
        for names, texts, motion1, motion2, lengths in loader:
            count = len(names)
            motion1 = motion1.to(device).float()
            motion2 = motion2.to(device).float()
            device_lengths = lengths.to(device).long()
            prediction1 = reconstruct_person(model, motion1, device_lengths)
            prediction2 = reconstruct_person(model, motion2, device_lengths)
            target = slice(offset, offset + count)
            names_ds[target] = list(names)
            texts_ds[target] = list(texts)
            lengths_ds[target] = lengths.numpy()
            motion1_ds[target] = prediction1.cpu().numpy()
            motion2_ds[target] = prediction2.cpu().numpy()
            offset += count
        if offset != expected_count:
            raise RuntimeError(
                f'official VQ export wrote {offset}/{expected_count} samples'
            )
        handle.attrs['status'] = 'complete'
        handle.flush()
    os.replace(temporary, destination)
    print(json.dumps({
        'status': 'complete',
        'output': str(destination),
        'bytes': destination.stat().st_size,
        'sha256': sha256_file(destination),
        'replication': args.replication,
        'split': split,
        'checkpoint_sha256': CHECKPOINT_SHA256,
    }, indent=2))


if __name__ == '__main__':
    main()
