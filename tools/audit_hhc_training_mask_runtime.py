#!/usr/bin/env python
"""Runtime-audit the paper HHC training mask and masked-only objective."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.motion_masks import (  # noqa: E402
    BODY_TOKEN_CHANNEL,
    HAND_TOKEN_CHANNEL,
    ROOT_TOKEN_CHANNEL,
    masked_token_nll,
    sample_kinematic_cascade_mask,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--tokens', type=int, default=38)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/hhc_training_mask_runtime_audit.json'),
    )
    args = parser.parse_args()
    require(args.batch_size >= 4, 'batch size must be at least four')
    require(args.tokens >= 1, 'token count must be positive')

    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    lengths = 1 + torch.arange(args.batch_size) % args.tokens
    positions = torch.arange(args.tokens).view(1, -1, 1)
    valid = positions < lengths.view(-1, 1, 1)
    valid = valid.expand(-1, -1, 3).clone()
    mask, ratios, taus = sample_kinematic_cascade_mask(valid, generator)

    valid_lengths = valid.sum(dim=1)
    expected_counts = torch.round(valid_lengths.float() * ratios).long()
    expected_counts = torch.minimum(expected_counts, valid_lengths)
    expected_counts = torch.where(
        valid_lengths > 0, expected_counts.clamp(min=1), expected_counts,
    )
    actual_counts = mask.sum(dim=1)

    root = ROOT_TOKEN_CHANNEL
    body = BODY_TOKEN_CHANNEL
    hand = HAND_TOKEN_CHANNEL
    checks = {
        'channel_order_is_body_hand_root': (body, hand, root) == (0, 1, 2),
        'tau_order_is_root_body_hand': bool(torch.all(
            (taus[:, root] >= taus[:, body])
            & (taus[:, body] >= taus[:, hand])
        )),
        'ratio_order_is_root_body_hand': bool(torch.all(
            (ratios[:, root] <= ratios[:, body])
            & (ratios[:, body] <= ratios[:, hand])
        )),
        'ratios_are_bounded': bool(torch.all((ratios >= 0) & (ratios <= 1))),
        'mask_is_subset_of_valid_tokens': bool(torch.all(~mask | valid)),
        'discrete_counts_match_round_min_one_contract': bool(torch.equal(
            actual_counts, expected_counts
        )),
        'every_nonempty_part_masks_at_least_one_token': bool(torch.all(
            actual_counts[valid_lengths > 0] >= 1
        )),
    }

    # Visible targets deliberately use an invalid class. masked_token_nll must
    # replace them before cross entropy, and visible-logit perturbations must
    # not affect the objective.
    loss_generator = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    num_codes = 17
    loss_batch, loss_tokens = 4, min(args.tokens, 8)
    loss_mask = mask[:loss_batch, :loss_tokens].clone()
    require(bool(loss_mask.any()), 'runtime loss audit sampled no masked tokens')
    targets = torch.randint(
        num_codes, (loss_batch, loss_tokens, 3), generator=loss_generator,
    )
    targets[~loss_mask] = num_codes + 99
    logits = torch.randn(
        loss_batch, loss_tokens, 3, num_codes, generator=loss_generator,
    )
    baseline_loss = masked_token_nll(logits, targets, loss_mask)
    visible_perturbed = logits.clone()
    visible_perturbed[..., 0][~loss_mask] += 1000.0
    visible_loss = masked_token_nll(
        visible_perturbed, targets, loss_mask
    )
    masked_perturbed = logits.clone()
    masked_perturbed[..., 0][loss_mask] += 10.0
    masked_loss = masked_token_nll(masked_perturbed, targets, loss_mask)
    checks.update({
        'visible_tokens_do_not_change_masked_only_loss': bool(torch.equal(
            baseline_loss, visible_loss
        )),
        'masked_logits_change_masked_only_loss': bool(not torch.equal(
            baseline_loss, masked_loss
        )),
        'losses_are_finite': bool(torch.isfinite(torch.stack([
            baseline_loss, visible_loss, masked_loss,
        ])).all()),
    })
    require(all(checks.values()), f'training-mask runtime checks failed: {checks}')

    payload = {
        'schema': 'hhc_training_mask_runtime_audit_v1',
        'status': 'pass',
        'contract': {
            'tau_order': 'tau_root >= tau_body >= tau_hand',
            'ratio_transform': 'cos(pi*tau/2)',
            'mask_ratio_order': 'root <= body <= hand',
            'discretization': 'round(valid_length * ratio), minimum one',
            'objective': 'mean per-sample NLL over masked tokens only',
        },
        'runtime': {
            'batch_size': args.batch_size,
            'maximum_tokens': args.tokens,
            'seed': args.seed,
            'mean_mask_ratio_body_hand_root': [
                float(value) for value in ratios.mean(dim=0)
            ],
            'mean_tau_body_hand_root': [
                float(value) for value in taus.mean(dim=0)
            ],
            'baseline_masked_nll': float(baseline_loss),
            'visible_perturbed_nll': float(visible_loss),
            'masked_perturbed_nll': float(masked_loss),
        },
        'checks': checks,
        'source': {
            'path': str((ROOT / 'utils/motion_masks.py').resolve()),
            'sha256': sha256_file(ROOT / 'utils/motion_masks.py'),
        },
        'implementation': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(__file__),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(
        args.output.suffix + f'.tmp.{os.getpid()}'
    )
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
