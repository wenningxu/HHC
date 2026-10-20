"""Bridge official InterHuman motion batches into HHC MaskGIT token batches."""

from __future__ import annotations

import torch

from utils.runtime import evaluating


@torch.no_grad()
def encode_interhuman_mgt_batch(vq, batch, device):
    """Encode post-crop paired motions with released floor-length semantics."""
    names, captions, motion1, motion2, processed_lengths = batch
    del names
    device = torch.device(device)
    motion1 = motion1.to(device).float()
    motion2 = motion2.to(device).float()
    if motion1.shape != motion2.shape or motion1.ndim != 3:
        raise ValueError(f'InterHuman MGT motions must be a matching pair, got {motion1.shape}, {motion2.shape}')
    if motion1.shape[1:] != (300, 262):
        raise ValueError(f'InterHuman MGT expects padded [B,300,262], got {motion1.shape}')
    lengths = torch.as_tensor(processed_lengths, device=device, dtype=torch.long)
    if lengths.shape != (motion1.shape[0],) or torch.any(lengths < 14) or torch.any(lengths > 299):
        raise ValueError(f'invalid InterHuman processed lengths {lengths}')
    paired = torch.cat((motion1, motion2), dim=-1)
    with evaluating(vq):
        tokens = vq(paired, type='encode')
    if tokens.ndim != 3 or tokens.shape[0] != paired.shape[0] or tokens.shape[-1] != 2:
        raise ValueError(f'paired InterHuman VQ returned invalid tokens {tokens.shape}')
    if tokens.shape[1] != 75:
        raise ValueError(f'300-frame InterHuman VQ must return 75 token slots, got {tokens.shape[1]}')
    token_lengths = torch.div(lengths, 4, rounding_mode='floor')
    if torch.any(token_lengths < 3) or torch.any(token_lengths > 74):
        raise ValueError(f'invalid official InterHuman token lengths {token_lengths}')
    batch_size = paired.shape[0]
    # Existing HHC training_step keeps a shared batch signature. These values
    # are deliberately inert and make any accidental condition enablement easy
    # to reject in the InterHuman model/launch contract.
    dummy_betas = torch.zeros(batch_size, 20, device=device)
    dummy_gender = torch.zeros(batch_size, 2, dtype=torch.long, device=device)
    no_contact_condition = torch.zeros(batch_size, 75, dtype=torch.long, device=device)
    return (
        list(captions), tokens.long(), token_lengths,
        dummy_betas, dummy_gender, no_contact_condition,
    )
