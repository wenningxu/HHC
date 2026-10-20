#!/usr/bin/env python
"""Audit exact floating-point semantics of the recovered historical MGT loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.motion_masks import all_valid_token_nll


HISTORICAL_SOURCE = ROOT / (
    "output/historical_mmm_local_history_train/"
    "1771033604763_set-424245_change-424244_ContentChange.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recovered_reference(logits, target, valid):
    weights = valid.float()
    weights = weights / valid.sum((1, 2), keepdim=True).clamp(min=1)
    losses = F.cross_entropy(
        logits[valid].reshape(-1, logits.shape[-1]),
        target[valid].reshape(-1), reduction="none",
    )
    return (losses * weights[valid].reshape(-1)).sum() / logits.shape[0]


def regrouped_alternative(logits, target, valid):
    safe_target = target.masked_fill(~valid, 0)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), safe_target.reshape(-1),
        reduction="none",
    ).view_as(target)
    return (
        losses.masked_fill(~valid, 0).sum((1, 2))
        / valid.sum((1, 2)).clamp(min=1)
    ).mean()


def evaluate(loss_function, base, target, valid):
    logits = base.clone().requires_grad_()
    loss = loss_function(logits, target, valid)
    loss.backward()
    return loss.detach(), logits.grad.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/historical_mgt_loss_reduction_audit.json",
    )
    args = parser.parse_args()

    torch.manual_seed(7)
    base = torch.randn(7, 13, 3, 97)
    target = torch.randint(97, (7, 13, 3))
    lengths = torch.tensor([1, 2, 4, 7, 9, 12, 13])
    valid = (
        torch.arange(13)[None, :, None] < lengths[:, None, None]
    ).expand(-1, -1, 3)

    reference_loss, reference_grad = evaluate(
        recovered_reference, base, target, valid
    )
    current_loss, current_grad = evaluate(
        all_valid_token_nll, base, target, valid
    )
    alternative_loss, alternative_grad = evaluate(
        regrouped_alternative, base, target, valid
    )
    historical_text = HISTORICAL_SOURCE.read_text(encoding="utf-8")
    checks = {
        "recovered_source_contains_weight_before_batch_sum": all(
            expression in historical_text for expression in (
                "weights = weights / sample_valid_tokens.clamp(min=1)",
                "loss_cls = (loss_cls * weights_flat).sum()",
                "loss_cls = loss_cls / batch_size",
            )
        ),
        "current_loss_bitwise_matches_recovered": bool(
            torch.equal(current_loss, reference_loss)
        ),
        "current_gradient_bitwise_matches_recovered": bool(
            torch.equal(current_grad, reference_grad)
        ),
        "regrouped_alternative_is_observably_different": bool(
            not torch.equal(alternative_loss, reference_loss)
            and not torch.equal(alternative_grad, reference_grad)
        ),
    }
    payload = {
        "schema": "hhc_historical_mgt_loss_reduction_audit_v1",
        "status": "pass" if all(checks.values()) else "mismatch",
        "checks": checks,
        "measurement": {
            "reference_loss": float(reference_loss),
            "current_loss": float(current_loss),
            "regrouped_alternative_loss": float(alternative_loss),
            "regrouped_minus_reference": float(
                alternative_loss - reference_loss
            ),
            "regrouped_gradient_max_abs_difference": float(
                (alternative_grad - reference_grad).abs().max()
            ),
        },
        "sources": {
            "historical_training": {
                "path": str(HISTORICAL_SOURCE),
                "sha256": sha256_file(HISTORICAL_SOURCE),
            },
            "current_loss": {
                "path": str(ROOT / "utils/motion_masks.py"),
                "sha256": sha256_file(ROOT / "utils/motion_masks.py"),
            },
        },
        "body_controlled_fid_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
