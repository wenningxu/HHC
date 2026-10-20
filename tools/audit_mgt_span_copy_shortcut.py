#!/usr/bin/env python
"""CPU cache diagnostic only: equal-count random versus circular-span masking."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import utils.motion_masks as masks
from tools.audit_mgt_mask_exposure_gap import make_batch, Text2MotionDatasetV2HHI_Token, sha256_file


def circular_mask(valid, counts, generator):
    lengths = valid.sum(1)
    positions = torch.arange(valid.shape[1])[None, :, None]
    if not torch.equal(valid, positions < lengths[:, None]):
        raise ValueError('only contiguous valid prefixes are supported')
    starts = (torch.rand(lengths.shape, generator=generator) * lengths).long()
    return ((positions - starts[:, None]) % lengths[:, None].clamp(min=1) < counts[:, None]) & valid


def copyable(target, inputs, selected, valid, radius):
    visible = valid & ~selected
    found = torch.zeros_like(selected)
    for offset in range(1, radius + 1):
        found[:, offset:] |= visible[:, :-offset] & (inputs[:, :-offset] == target[:, offset:])
        found[:, :-offset] |= visible[:, offset:] & (inputs[:, offset:] == target[:, :-offset])
    return (found & selected).sum((0, 1))


def self_check():
    target = torch.tensor([[[7], [7], [9], [7]]]).expand(-1, -1, 3)
    valid = torch.ones_like(target, dtype=torch.bool)
    selected = torch.zeros_like(valid); selected[:, 1] = True
    inputs = target.clone(); inputs[:, 1] = 99
    assert copyable(target, inputs, selected, valid, 1).tolist() == [1, 1, 1]
    inputs[:, 0] = 3
    assert copyable(target, inputs, selected, valid, 1).tolist() == [0, 0, 0]
    assert copyable(target, inputs, selected, valid, 2).tolist() == [1, 1, 1]
    count = torch.tensor([[0, 2, 4]])
    span = circular_mask(valid, count, torch.Generator().manual_seed(1))
    assert torch.equal(span.sum(1), count)
    print('self-check PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--draws', type=int, default=64)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    if not args.audit or not args.output or args.draws <= 0:
        parser.error('--audit, --output and positive --draws required')
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads(args.audit.read_text())
    manifest = Path(audit['training_manifest'])
    assert sha256_file(manifest) == audit['training_manifest_sha256']
    resolved = json.loads(manifest.read_text())['resolved_arguments']
    assert sha256_file(resolved['val_split_file']) == audit['split_file_sha256']
    dataset = Text2MotionDatasetV2HHI_Token(
        resolved['val_split_file'], resolved['codebook_dir'],
        unit_length=2 ** resolved['down_t'], codebook_size=resolved['nb_code'],
        text_dir='dataset/interx/texts_processed', max_motion_length=resolved['block_size'] - 1,
        contact_dir=str(Path(resolved['cache_root']) / 'contact_token_label'),
        beta_dir='dataset/interx/betas', strict_assets=True)
    active = sorted(dataset.name_list[dataset.pointer:])
    names = sorted(np.random.default_rng(audit['seed']).choice(active, audit['samples'], replace=False).tolist())
    digest = hashlib.sha256('\n'.join(names).encode()).hexdigest()
    assert digest == audit['sample_names_sha256']
    _, target, lengths, _, _ = make_batch(dataset, names, resolved['block_size'])
    valid = (torch.arange(target.shape[1])[None, :] < lengths[:, None]).unsqueeze(-1).expand_as(target)
    totals = torch.zeros(3, dtype=torch.long)
    hits = {arm: {r: torch.zeros(3, dtype=torch.long) for r in (1, 2)} for arm in ('original', 'span')}
    original_sampler = masks.sample_kinematic_cascade_mask
    for draw in range(args.draws):
        seed = audit['seed'] + 10000 + draw
        original, selected, _, _ = masks.sample_historical_mmm_corruption_with_kinematic_ratios(
            target, valid, resolved['nb_code'], pkeep=.5, generator=torch.Generator().manual_seed(seed))
        def span_sampler(valid_mask, generator=None):
            mask, ratios, taus = original_sampler(valid_mask, generator)
            assert torch.equal(mask, selected)
            return circular_mask(valid_mask, mask.sum(1), torch.Generator().manual_seed(seed + 1000000)), ratios, taus
        with patch.object(masks, 'sample_kinematic_cascade_mask', span_sampler):
            span_input, span, _, _ = masks.sample_historical_mmm_corruption_with_kinematic_ratios(
                target, valid, resolved['nb_code'], pkeep=.5, generator=torch.Generator().manual_seed(seed))
        assert torch.equal(selected.sum(1), span.sum(1))
        count = selected.sum(1)
        assert (count[:, 1] >= count[:, 0]).all() and (count[:, 0] >= count[:, 2]).all()
        shared_visible = valid & ~selected & ~span
        assert torch.equal(original[shared_visible], span_input[shared_visible])
        assert torch.equal(original[~valid], target[~valid]) and torch.equal(span_input[~valid], target[~valid])
        totals += selected.sum((0, 1))
        for arm, inputs, mask in [('original', original, selected), ('span', span_input, span)]:
            for radius in (1, 2):
                hits[arm][radius] += copyable(target, inputs, mask, valid, radius)
    result = {'purpose': 'mask-copy feasibility only; no model, MM or quality claim',
              'audit_sha256': sha256_file(args.audit), 'source_sha256': sha256_file(__file__),
              'mask_helper_sha256': sha256_file(masks.__file__), 'names': names,
              'sample_names_sha256': digest, 'draws': args.draws, 'parts': ['body', 'hand', 'root'],
              'target_sha256': hashlib.sha256(target.numpy().tobytes()).hexdigest(),
              'masked_counts_both_arms': totals.tolist(),
              'copyable_counts': {a: {r: v.tolist() for r, v in h.items()} for a, h in hits.items()},
              'copyable_fraction': {a: {r: (v / totals).tolist() for r, v in h.items()} for a, h in hits.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'names'}, indent=2))


if __name__ == '__main__':
    main()
