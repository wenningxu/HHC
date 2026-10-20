#!/usr/bin/env python
"""Behaviorally compare the recovered and isolated historical MGT samplers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.two_person_transformer import Text2Motion_Transformer as CurrentTransformer
from tools.audit_mgt_initialization_contract import (
    DEFAULT_HISTORICAL_TRANSFORMER,
    recovered_classes,
    sha256_file,
    transformer_namespace,
)


class Quantizer(nn.Module):
    def __init__(self, size: int, width: int):
        super().__init__()
        self.codebook = nn.Embedding(size, width)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook(indices)


class Tokenizer(nn.Module):
    code_dim = 32

    def __init__(self, size: int):
        super().__init__()
        self.quantizer_upper = Quantizer(size, 12)
        self.quantizer_lower = Quantizer(size, 12)
        self.quantizer_root = Quantizer(size, 8)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def rng_sha256() -> str:
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def build_pair(historical_class: type[nn.Module], seed: int):
    kwargs = dict(
        num_vq=16,
        embed_dim=32,
        clip_dim=16,
        block_size=5,
        num_layers=1,
        num_local_layer=1,
        n_head=4,
        drop_out_rate=0.0,
        fc_rate=2,
        contact=False,
        shape=False,
    )
    torch.manual_seed(seed)
    historical = historical_class(vqvae=Tokenizer(16), **kwargs).eval()
    historical_rng = rng_sha256()
    torch.manual_seed(seed)
    current = CurrentTransformer(
        vqvae=Tokenizer(16),
        paper_architecture=False,
        conditioning_contract="historical_mmm_shape_dropout",
        **kwargs,
    ).eval()
    current_rng = rng_sha256()
    return historical, current, historical_rng, current_rng


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-transformer", type=Path,
        default=DEFAULT_HISTORICAL_TRANSFORMER,
    )
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/historical_mgt_sampler_behavior_audit.json",
    )
    args = parser.parse_args()

    historical_class = recovered_classes(
        args.historical_transformer,
        {"CrossCondTransBase", "CrossCondTransHead", "Text2Motion_Transformer"},
        transformer_namespace(),
    )["Text2Motion_Transformer"]
    historical, current, historical_init_rng, current_init_rng = build_pair(
        historical_class, args.seed
    )

    input_generator = torch.Generator().manual_seed(20260829)
    indices = torch.randint(0, 16, (1, 4, 3), generator=input_generator)
    clip_feature = torch.randn(1, 16, generator=input_generator)
    word_embedding = torch.randn(1, 7, 16, generator=input_generator)
    source_mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        historical_logits = historical(
            indices, clip_feature, src_mask=source_mask,
            word_emb=word_embedding,
        )
        current_logits = current(
            indices, clip_feature, src_mask=source_mask,
            word_emb=word_embedding,
        )

        sampling_seed = 424245
        torch.manual_seed(sampling_seed)
        historical_tokens = historical(
            clip_feature, word_embedding, type="sample_intra_cas",
            m_length=torch.tensor([16]), rand_pos=False, CFG=3.5,
            max_steps=2,
        )
        historical_sampling_rng = rng_sha256()
        torch.manual_seed(sampling_seed)
        current_tokens = current(
            clip_feature, word_embedding, type="sample_intra_cas_historical",
            m_length=torch.tensor([16]), rand_pos=False, CFG=3.5,
            max_steps=2,
        )
        current_sampling_rng = rng_sha256()

    checks = {
        "same_constructor_rng": historical_init_rng == current_init_rng,
        "same_state": state_sha256(historical) == state_sha256(current),
        "same_forward_logits": torch.equal(historical_logits, current_logits),
        "same_sampled_tokens": torch.equal(historical_tokens, current_tokens),
        "same_sampling_rng_after": (
            historical_sampling_rng == current_sampling_rng
        ),
        "formal_sampler_is_separate_dispatch": hasattr(
            current, "sample_intra_cas_historical"
        ) and current.sample_intra_cas.__func__ is not current.sample_intra_cas_historical.__func__,
    }
    payload = {
        "schema": "historical_mgt_sampler_behavior_audit_v1",
        "status": "pass" if all(checks.values()) else "mismatch",
        "historical_source": {
            "path": str(args.historical_transformer),
            "sha256": sha256_file(args.historical_transformer),
        },
        "current_source": {
            "path": str(ROOT / "models/two_person_transformer.py"),
            "sha256": sha256_file(ROOT / "models/two_person_transformer.py"),
        },
        "configuration": {
            "constructor_seed": args.seed,
            "sampling_seed": sampling_seed,
            "max_steps": 2,
            "rand_pos": False,
            "cfg": 3.5,
        },
        "checks": checks,
        "evidence": {
            "historical_state_sha256": state_sha256(historical),
            "current_state_sha256": state_sha256(current),
            "historical_logits_sha256": tensor_sha256(historical_logits),
            "current_logits_sha256": tensor_sha256(current_logits),
            "maximum_absolute_logit_difference": float(
                (historical_logits - current_logits).abs().max()
            ),
            "historical_tokens_sha256": tensor_sha256(historical_tokens),
            "current_tokens_sha256": tensor_sha256(current_tokens),
            "historical_sampling_rng_after": historical_sampling_rng,
            "current_sampling_rng_after": current_sampling_rng,
        },
        "scope": (
            "Diagnostic historical sampler only; formal common-50 dispatch "
            "remains sample_intra_cas."
        ),
        "body_controlled_fid_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "status": payload["status"],
        "checks": checks,
    }, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
