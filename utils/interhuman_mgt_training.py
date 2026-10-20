"""One-batch training contract for HHC MGT on official InterHuman."""

from __future__ import annotations

import torch

from utils.interhuman_mgt_batch import encode_interhuman_mgt_batch
from utils.motion_masks import (
    all_valid_token_nll,
    sample_historical_mmm_corruption_with_kinematic_ratios,
)
from utils.runtime import generate_src_mask


def freeze_interhuman_vq_for_mgt(vq):
    """Freeze the online post-crop tokenizer without detaching the MGT."""
    vq.eval()
    for parameter in vq.parameters():
        parameter.requires_grad_(False)
    return vq


@torch.no_grad()
def prepare_interhuman_mgt_batch(
    vq,
    batch,
    device,
    *,
    num_codes,
    pkeep=0.5,
    paper_architecture=False,
    generator=None,
):
    """Tokenize after preprocessing, then apply the recovered paper-ratio mask.

    This intentionally retains the known-good historical BERT-style random
    replacement and all-valid loss support; two sorted kinematic ratios adapt
    the HHC mask rule to Root -> Body, with no hand channel.
    """
    captions, target, token_lengths, _, _, _ = encode_interhuman_mgt_batch(
        vq, batch, device
    )
    if target.ndim != 3 or target.shape[1:] != (75, 2):
        raise ValueError(f'InterHuman MGT target must be [B,75,2], got {target.shape}')
    if int(num_codes) != int(getattr(vq, 'num_code', num_codes)):
        raise ValueError('MGT codebook size differs from its frozen InterHuman VQ')
    if torch.any(target < 0) or torch.any(target >= int(num_codes)):
        raise ValueError('online InterHuman VQ emitted an out-of-range code')
    valid_time = generate_src_mask(target.shape[1], token_lengths)
    valid = valid_time.unsqueeze(-1).expand_as(target)
    if paper_architecture:
        attention_mask = valid_time
    else:
        # The retained high-performing HHC topology embeds one explicit END
        # position and special PAD tokens. Online VQ encoding of the fixed
        # 300-frame buffer instead yields ordinary codes in that tail, so
        # replace it explicitly to match historical Inter-X training.
        target = target.masked_fill(~valid, int(num_codes) + 1)
        for batch_index, length in enumerate(token_lengths.tolist()):
            if length >= target.shape[1]:
                raise ValueError('historical InterHuman MGT has no room for END')
            target[batch_index, int(length), :] = int(num_codes)
        attention_mask = generate_src_mask(target.shape[1], token_lengths + 1)
    masked_input, masked, ratios, taus = (
        sample_historical_mmm_corruption_with_kinematic_ratios(
            target,
            valid,
            int(num_codes),
            pkeep=float(pkeep),
            generator=generator,
        )
    )
    return {
        'captions': captions,
        'target': target,
        'token_lengths': token_lengths,
        'attention_mask': attention_mask,
        'valid_mask': valid,
        'masked_input': masked_input,
        'masked_mask': masked,
        'mask_ratios': ratios,
        'mask_taus': taus,
    }


def interhuman_mgt_objective(logits, prepared):
    """Return historical all-valid NLL and transparent accuracy diagnostics."""
    target = prepared['target']
    valid = prepared['valid_mask'].bool()
    masked = prepared['masked_mask'].bool()
    if logits.shape[:-1] != target.shape:
        raise ValueError(
            f'InterHuman logits {tuple(logits.shape)} do not align with '
            f'target {tuple(target.shape)}'
        )
    loss = all_valid_token_nll(logits, target, valid)
    prediction = logits.argmax(dim=-1)

    def accuracy(support, channel=None):
        count = support.sum()
        if int(count) == 0:
            return logits.new_tensor(float('nan'))
        predicted = prediction if channel is None else prediction[..., channel]
        expected = target if channel is None else target[..., channel]
        return (predicted[support] == expected[support]).float().mean()

    return {
        'loss': loss,
        'accuracy_valid': accuracy(valid),
        'accuracy_masked': accuracy(masked),
        'accuracy_body': accuracy(valid[..., 0], 0),
        'accuracy_root': accuracy(valid[..., 1], 1),
        'masked_fraction': masked.sum().float() / valid.sum().clamp(min=1),
    }
