#!/usr/bin/env python3
"""Paired, no-update train/eval-mode diagnostic on train and frozen val data."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clip
import numpy as np
import torch
from torch.utils.data import default_collate

from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from dataset.interhuman_mgt import InterHumanMGTTrainDataset, InterHumanMGTDrawBatchSampler
from tools.aggregate_fair_interhuman_metrics import atomic_json
from tools.export_hhc_interhuman import TextCLIP
from utils.interhuman_artifacts import load_hhc_interhuman_mgt
from utils.interhuman_mgt_training import prepare_interhuman_mgt_batch, interhuman_mgt_objective


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    model, vq, provenance = load_hhc_interhuman_mgt(
        args.checkpoint, args.manifest, device=device, inference_conditioning=False)
    before = state_digest(model)
    clip_base, _ = clip.load('ViT-B/32', device=device, jit=False)
    clip.model.convert_weights(clip_base)
    text_encoder = TextCLIP(clip_base.eval())
    train = InterHumanMGTTrainDataset(base_seed=114514)
    val = InterHumanValidationRequestDataset(adapter=train.adapter)
    train_draws = next(iter(InterHumanMGTDrawBatchSampler(train, 64, 1, 114514)))
    results = {}
    for split, dataset, indices in [('train', train, train_draws), ('val', val, list(range(64)))]:
        generator = torch.Generator(device=device).manual_seed(20260131)
        rows = {'eval': [], 'train': []}
        names, lengths = [], []
        for start in range(0, 64, 8):
            batch = default_collate([dataset[index] for index in indices[start:start + 8]])
            names.extend(batch[0])
            lengths.extend(batch[4].tolist())
            prepared = prepare_interhuman_mgt_batch(
                vq, batch, device, num_codes=vq.num_code, generator=generator)
            feature, words = text_encoder(clip.tokenize(prepared['captions'], truncate=True).to(device))
            for mode in rows:
                model.train(mode == 'train')
                vq.eval()  # parent train() must never enable VQ EMA updates
                torch.manual_seed(20260133 + start)
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(20260133 + start)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    logits = model(prepared['masked_input'], feature,
                                   src_mask=prepared['attention_mask'], word_emb=words)
                values = interhuman_mgt_objective(logits.float(), prepared)
                assert all(torch.isfinite(value).all() for value in values.values())
                rows[mode].append({key: float(value) for key, value in values.items()})
        results[split] = {
            'names': names, 'processed_lengths': lengths,
            'metrics': {mode: {key: float(np.mean([row[key] for row in values]))
                               for key in values[0]} for mode, values in rows.items()},
        }
    model.eval()
    after = state_digest(model)
    assert before == after, 'diagnostic mutated model/VQ state'
    result = {'status': 'complete', 'optimizer_step': provenance['optimizer_step'],
              'checkpoint_sha256': provenance['checkpoint_sha256'],
              'samples_per_split': 64, 'micro_batch_size': 8,
              'conditioning_contract': model.conditioning_contract,
              'scope': 'same prepared tokens/masks per train/eval-mode pair; first seeded train draw vs first 64 frozen val requests; diagnostic, not formal quality scoring',
              'model_state_unchanged': True, 'state_sha256': after, 'results': results}
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'results'}))
    print(json.dumps({split: value['metrics'] for split, value in results.items()}, indent=2))


if __name__ == '__main__':
    main()
