#!/usr/bin/env python3
"""Audit TiMotion one-GPU accumulation against one global-batch update."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


AUTHOR = Path("/mnt/8T-HDD/wenning/TIMotion/tools/train.py")
FAIR = Path("/mnt/8T-HDD/wenning/TIMotion/tools/train_interx_fair.py")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def gradients(model):
    return [parameter.grad.detach().clone() for parameter in model.parameters()]


def max_difference(left, right):
    return max(
        float(torch.max(torch.abs(a.double() - b.double())))
        for a, b in zip(left, right)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/timotion_gradient_accumulation_audit.json"),
    )
    args = parser.parse_args()

    author_text = AUTHOR.read_text(encoding="utf-8")
    fair_text = FAIR.read_text(encoding="utf-8")
    author_tokens = (
        "self.manual_backward(loss)",
        "torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)",
        "opt.step()",
    )
    fair_tokens = (
        "self.manual_backward(loss / self.accumulate)",
        "torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)",
        "optimizer.step()",
    )
    require(all(token in author_text for token in author_tokens),
            "released TiMotion gradient update source changed")
    require(all(token in fair_text for token in fair_tokens),
            "fair TiMotion gradient update source changed")
    require(
        fair_text.index(fair_tokens[0])
        < fair_text.index(fair_tokens[1])
        < fair_text.index(fair_tokens[2], fair_text.index(fair_tokens[1])),
        "fair backward/clip/step order changed",
    )

    # Mean-reduced losses under synchronous DDP average gradients across eight
    # ranks. Eight local mean losses divided by eight and accumulated on one
    # rank are the same global-batch gradient. Use float64 so the audit tests
    # the arithmetic rather than float32 reduction-order noise.
    torch.manual_seed(123)
    accumulated = torch.nn.Sequential(
        torch.nn.Linear(7, 11), torch.nn.GELU(), torch.nn.Linear(11, 5)
    ).double()
    global_batch = torch.nn.Sequential(
        torch.nn.Linear(7, 11), torch.nn.GELU(), torch.nn.Linear(11, 5)
    ).double()
    global_batch.load_state_dict(accumulated.state_dict())
    inputs = torch.randn(8, 32, 7, dtype=torch.float64)
    targets = torch.randn(8, 32, 5, dtype=torch.float64)
    for index in range(8):
        loss = torch.nn.functional.mse_loss(
            accumulated(inputs[index]), targets[index], reduction="mean"
        )
        (loss / 8.0).backward()
    torch.nn.functional.mse_loss(
        global_batch(inputs.reshape(256, 7)),
        targets.reshape(256, 5),
        reduction="mean",
    ).backward()
    before_accumulated = gradients(accumulated)
    before_global = gradients(global_batch)
    before_max = max_difference(before_accumulated, before_global)
    require(before_max <= 1e-14,
            f"accumulated/global gradient mismatch: {before_max}")

    accumulated_norm = float(torch.nn.utils.clip_grad_norm_(
        accumulated.parameters(), 0.5
    ))
    global_norm = float(torch.nn.utils.clip_grad_norm_(
        global_batch.parameters(), 0.5
    ))
    after_max = max_difference(gradients(accumulated), gradients(global_batch))
    require(abs(accumulated_norm - global_norm) <= 1e-14,
            "pre-clip global gradient norms differ")
    require(after_max <= 1e-14,
            f"post-clip gradients differ: {after_max}")

    optimizer_accumulated = torch.optim.AdamW(
        accumulated.parameters(), lr=1e-4, betas=(0.9, 0.999),
        weight_decay=2e-5,
    )
    optimizer_global = torch.optim.AdamW(
        global_batch.parameters(), lr=1e-4, betas=(0.9, 0.999),
        weight_decay=2e-5,
    )
    optimizer_accumulated.step()
    optimizer_global.step()
    weight_max = max_difference(
        [p.detach() for p in accumulated.parameters()],
        [p.detach() for p in global_batch.parameters()],
    )
    require(weight_max <= 1e-14,
            f"post-AdamW weights differ: {weight_max}")

    payload = {
        "schema": "timotion_gradient_accumulation_audit_v1",
        "status": "pass",
        "contract": {
            "released_per_gpu_batch": 32,
            "released_gpu_count": 8,
            "paper_global_batch": 256,
            "local_microbatch": 32,
            "accumulation_steps": 8,
            "loss_reduction": "mean",
            "gradient_clip_global_norm": 0.5,
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "betas": [0.9, 0.999],
            "weight_decay": 0.00002
        },
        "runtime_equivalence": {
            "dtype": "float64",
            "examples": 256,
            "pre_clip_gradient_max_abs_difference": before_max,
            "accumulated_pre_clip_global_l2": accumulated_norm,
            "global_batch_pre_clip_global_l2": global_norm,
            "post_clip_gradient_max_abs_difference": after_max,
            "post_adamw_weight_max_abs_difference": weight_max
        },
        "sources": {
            "released_runner": {
                "path": str(AUTHOR), "sha256": sha256_file(AUTHOR)
            },
            "fair_runner": {
                "path": str(FAIR), "sha256": sha256_file(FAIR)
            }
        },
        "interpretation": (
            "For a mean-reduced objective, eight 32-example losses divided "
            "by eight produce the same gradient, global-norm clipping and "
            "AdamW update as one 256-example global batch."
        )
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), "status": "pass"}))


if __name__ == "__main__":
    main()
