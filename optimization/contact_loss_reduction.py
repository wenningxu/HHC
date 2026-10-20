"""Explicit contact-loss reductions used by shape-optimization ablations."""

from __future__ import annotations

import torch


CONTACT_LOSS_REDUCTION_POLICIES = (
    "paper_literal_raw_sum",
    "active_contact_mean",
)


def reduce_contact_pair_sums(
    pair_sums,
    active_weight_sum,
    policy,
    elements_per_pair=None,
):
    """Reduce differentiable per-contact-pair temporal sums.

    ``pair_sums[i]`` is already summed over active frames.  For frozen-anchor
    squared distance, pass the number of locked one-to-one vertex pairs as
    ``elements_per_pair[i]``.  For coarse region distance, omit it (one
    already-averaged spatial contact element per frame and semantic pair).
    """
    if policy not in CONTACT_LOSS_REDUCTION_POLICIES:
        raise ValueError(f"unknown contact-loss reduction policy: {policy!r}")
    if not pair_sums:
        raise ValueError("contact-pair reduction requires at least one pair")
    if elements_per_pair is None:
        elements_per_pair = [1] * len(pair_sums)
    if len(elements_per_pair) != len(pair_sums):
        raise ValueError("one element count is required for every pair sum")
    if any(int(count) < 1 for count in elements_per_pair):
        raise ValueError("contact element counts must be positive")

    if policy == "paper_literal_raw_sum":
        return torch.stack(tuple(pair_sums)).sum()

    active_weight_sum = torch.as_tensor(
        active_weight_sum,
        dtype=pair_sums[0].dtype,
        device=pair_sums[0].device,
    )
    if not bool(torch.isfinite(active_weight_sum)):
        raise ValueError("active contact weight sum must be finite")
    if float(active_weight_sum.detach()) <= 0:
        raise ValueError("active contact weight sum must be positive")
    per_pair_means = [
        pair_sum / (active_weight_sum * int(element_count))
        for pair_sum, element_count in zip(pair_sums, elements_per_pair)
    ]
    return torch.stack(tuple(per_pair_means)).mean()
