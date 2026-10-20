"""Focused regression tests for the one-factor HHC MGT training arm."""

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.motion_masks import (
    sample_kinematic_cascade_mask,
    sample_historical_mmm_corruption_with_kinematic_ratios,
)


def test_two_part_mask_uses_two_draws_and_exact_sorted_counts():
    valid = torch.ones(2, 7, 2, dtype=torch.bool)
    valid[0, 4:] = False
    valid[1, :, 1] = False  # empty root support must remain empty
    actual = torch.Generator().manual_seed(82)
    mask, ratios, taus = sample_kinematic_cascade_mask(valid, actual)
    expected = torch.Generator().manual_seed(82)
    expected_taus = torch.rand(2, 2, generator=expected).sort(dim=-1).values
    expected_ratios = torch.cos(expected_taus * (math.pi * 0.5))
    lengths = valid.sum(dim=1)
    counts = torch.minimum(torch.round(lengths.float() * expected_ratios).long(), lengths)
    counts = torch.where(lengths > 0, counts.clamp(min=1), counts)
    scores = torch.rand(2, 7, 2, generator=expected).masked_fill(~valid, float('inf'))
    expected_mask = (scores.argsort(dim=1).argsort(dim=1) < counts.unsqueeze(1)) & valid
    assert torch.equal(taus, expected_taus)
    assert torch.equal(ratios, expected_ratios)
    assert torch.equal(mask, expected_mask)
    assert torch.equal(mask.sum(dim=1), counts)
    assert torch.all(ratios[:, 1] <= ratios[:, 0])
    assert torch.equal(actual.get_state(), expected.get_state())


def test_historical_corruption_changes_only_channel_mask_ratio_rule():
    target = torch.arange(2 * 7 * 3, dtype=torch.long).view(2, 7, 3) % 97
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0, 5:] = False
    valid[1, 6:] = False
    seed = 314159

    actual_generator = torch.Generator().manual_seed(seed)
    masked, mask, ratios, taus = (
        sample_historical_mmm_corruption_with_kinematic_ratios(
            target, valid, num_codes=97, pkeep=0.5,
            generator=actual_generator,
        )
    )

    # Reconstruct the historical BERT-style pre-mask corruption and then the
    # paper's sorted per-channel cosine-ratio/count rule from the same RNG.
    expected_generator = torch.Generator().manual_seed(seed)
    keep = torch.bernoulli(
        torch.full(target.shape, 0.5), generator=expected_generator,
    ).bool() | ~valid
    random_indices = torch.randint(
        97, target.shape, dtype=target.dtype, generator=expected_generator,
    )
    corrupted = torch.where(keep, target, random_indices)

    raw_taus = torch.rand(2, 3, generator=expected_generator)
    tau_hand, tau_body, tau_root = raw_taus.sort(dim=-1).values.unbind(dim=-1)
    expected_taus = torch.stack((tau_body, tau_hand, tau_root), dim=-1)
    expected_ratios = torch.cos(expected_taus * (math.pi * 0.5))
    valid_lengths = valid.sum(dim=1)
    counts = torch.round(valid_lengths.float() * expected_ratios).long()
    counts = torch.minimum(counts, valid_lengths)
    counts = torch.where(valid_lengths > 0, counts.clamp(min=1), counts)
    scores = torch.rand(2, 7, 3, generator=expected_generator)
    scores = scores.masked_fill(~valid, float("inf"))
    ranks = scores.argsort(dim=1).argsort(dim=1)
    expected_mask = (ranks < counts.unsqueeze(1)) & valid
    expected = torch.where(expected_mask, 99, corrupted)

    assert torch.equal(masked, expected)
    assert torch.equal(mask, expected_mask)
    assert torch.equal(mask.sum(dim=1), counts)
    assert torch.equal(taus, expected_taus)
    assert torch.equal(ratios, expected_ratios)
    assert torch.all(taus[:, 2] >= taus[:, 0])  # tau_root >= tau_body
    assert torch.all(taus[:, 0] >= taus[:, 1])  # tau_body >= tau_hand
    assert torch.all(ratios[:, 2] <= ratios[:, 0])
    assert torch.all(ratios[:, 0] <= ratios[:, 1])
    assert torch.equal(masked[~mask], corrupted[~mask])
    assert torch.all(masked[mask] == 99)
    assert not mask[~valid].any()


def test_historical_paper_ratio_corruption_is_seed_deterministic():
    target = torch.arange(3 * 9 * 3, dtype=torch.long).view(3, 9, 3) % 101
    valid = torch.ones_like(target, dtype=torch.bool)
    calls = []
    for _ in range(2):
        generator = torch.Generator().manual_seed(2718)
        calls.append(
            sample_historical_mmm_corruption_with_kinematic_ratios(
                target, valid, num_codes=101, pkeep=0.5,
                generator=generator,
            )
        )
    for left, right in zip(*calls):
        assert torch.equal(left, right)


def test_finalizer_uses_method_neutral_val_and_user_fixed_sampling():
    source = (
        ROOT / "tools/finalize_hhc_historical_plus_paper_mask_ratio.sh"
    ).read_text(encoding="utf-8")
    required = (
        "tools/audit_lightning_checkpoint.py",
        "tools/audit_hhc_sampling_runtime.py",
        "tools/audit_hhc_historical_model_checkpoint_structure.py",
        "hhc_historical_plus_paper_mask_ratio_model_structure_audit.json",
        "hhc_historical_plus_paper_mask_ratio_final_checkpoint_audit.json",
        "--expected-step 300000 --expected-lr 0.00001",
        "tools/audit_wandb_online_metric_history.py",
        "--metric-profile masked_token_classification --expected-final-step 300000",
        "hhc_historical_plus_paper_mask_ratio_wandb_history_audit.json",
        "tools/audit_hhc_mask_ratio_runtime_history.py",
        "--through-step 300000 --telemetry-interval 50",
        "tools/promote_interx_validation_request_contract.py",
        "tools/select_hhc_checkpoint_on_official_interx_val.py",
        "--sampling-steps 50 --temperature 0.5 --cfg-scale 3.5",
        "--contact-guidance-scale 1.0 --start-replication 0 --end-replication 20",
        "--hhc-inference-conditioning-contract paper_text_cfg",
        "--contact-guidance --hhc-contact-timing-source request_oracle",
        "--no-contact-guidance --hhc-contact-timing-source none",
        "hhc_historical_plus_paper_mask_ratio_contact_common50",
        "hhc_historical_plus_paper_mask_ratio_no_contact_common50",
    )
    assert all(marker in source for marker in required)
    assert source.index("audit_lightning_checkpoint.py") < source.index(
        "promote_interx_validation_request_contract.py"
    )
    assert source.index("audit_wandb_online_metric_history.py") < source.index(
        "promote_interx_validation_request_contract.py"
    )
    assert source.index("promote_interx_validation_request_contract.py") < source.index(
        "select_hhc_checkpoint_on_official_interx_val.py"
    )
    assert source.index("select_hhc_checkpoint_on_official_interx_val.py") < source.index(
        "run_fair_interx_method.py"
    )


if __name__ == "__main__":
    test_historical_corruption_changes_only_channel_mask_ratio_rule()
    test_historical_paper_ratio_corruption_is_seed_deterministic()
    test_finalizer_uses_method_neutral_val_and_user_fixed_sampling()
    print("3 historical paper-mask/finalizer tests passed")
