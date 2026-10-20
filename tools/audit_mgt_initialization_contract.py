#!/usr/bin/env python
"""Compare the recovered 2026-02-13 MGT initialization with the current bridge.

This is intentionally a construction-only CPU audit.  It never loads a training
checkpoint and never touches CUDA RNG state.  The recovered Local History files
are parsed and only the relevant class definitions are executed against the
current dependency modules, whose provenance is reported separately.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.encdec import Decoder, Encoder
from models.motion_transformer import AdaBlock, Block, Block_crossatt
from models.pos_encoding import PositionEmbedding
from models.quantize_cnn import QuantizeEMAReset
from models.trans_encdec import PositionEmbeddingSine1D
from models.two_person_transformer import Text2Motion_Transformer as CurrentTransformer
from models.vqvae_sep3_shape import VQVAE_SEP3_shape as CurrentVQ
from utils.fk import FK_Pair_Layer_Batch
from utils.motion_masks import (
    HHI_BODY_MASK,
    HHI_HAND_MASK,
    HHI_ROOT_MASK,
    HHI_ROOT_MASK_OLD,
    HML_LOWER_BODY_MASK,
    HML_UPPER_BODY_MASK,
)
from utils.paramUtil import INTERX_SKELETONS, SMPLX_TREE
from utils.runtime import cosine_schedule, generate_src_mask, gumbel_sample


DEFAULT_HISTORICAL_VQ = ROOT / (
    "output/historical_mmm_local_history_vq/"
    "1772118531397_set-429297_change-429296_ContentChange.py"
)
DEFAULT_HISTORICAL_TRANSFORMER = ROOT / (
    "output/historical_mmm_local_history/"
    "1771299984324_set-427147_change-427146_ContentChange.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rng_sha256() -> str:
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def state_sha256(module: nn.Module, *, exclude_prefix: str | None = None) -> tuple[str, dict[str, list[int]]]:
    digest = hashlib.sha256()
    shapes: dict[str, list[int]] = {}
    for key, tensor in module.state_dict().items():
        if exclude_prefix is not None and key.startswith(exclude_prefix):
            continue
        value = tensor.detach().cpu().contiguous()
        shapes[key] = list(value.shape)
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest(), shapes


def recovered_classes(path: Path, names: set[str], namespace: dict) -> dict[str, type]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    found = {node.name for node in selected}
    if found != names:
        raise RuntimeError(f"missing recovered classes in {path}: {sorted(names - found)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def vq_namespace() -> dict:
    return {
        "torch": torch,
        "nn": nn,
        "Encoder": Encoder,
        "Decoder": Decoder,
        "QuantizeEMAReset": QuantizeEMAReset,
        "HML_UPPER_BODY_MASK": HML_UPPER_BODY_MASK,
        "HML_LOWER_BODY_MASK": HML_LOWER_BODY_MASK,
        "HHI_BODY_MASK": HHI_BODY_MASK,
        "HHI_HAND_MASK": HHI_HAND_MASK,
        "HHI_ROOT_MASK": HHI_ROOT_MASK,
        "HHI_ROOT_MASK_OLD": HHI_ROOT_MASK_OLD,
        "FK_Pair_Layer_Batch": FK_Pair_Layer_Batch,
        "SMPLX_TREE": SMPLX_TREE,
        "INTERX_SKELETONS": INTERX_SKELETONS,
    }


def transformer_namespace() -> dict:
    return {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "cosine_schedule": cosine_schedule,
        "gumbel_sample": gumbel_sample,
        "generate_src_mask": generate_src_mask,
        "PositionEmbedding": PositionEmbedding,
        "PositionEmbeddingSine1D": PositionEmbeddingSine1D,
        "Block": Block,
        "Block_crossatt": Block_crossatt,
        "AdaBlock": AdaBlock,
    }


def make_args() -> SimpleNamespace:
    return SimpleNamespace(
        dataname="hhi",
        motion_rep="raw",
        device="cpu",
        quantizer="ema_reset",
        mu=0.99,
        nb_code=12288,
        code_dim=256,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        vq_act="relu",
        vq_norm=None,
        shape_condition_dim=8,
        shape_mask_prob=0.1,
    )


def construct_record(vq_class: type[nn.Module], transformer_class: type[nn.Module], seed: int) -> dict:
    args = make_args()
    skeleton_copy = INTERX_SKELETONS.copy()
    torch.manual_seed(seed)
    rng_before = rng_sha256()
    vq = vq_class(
        args,
        args.nb_code,
        args.code_dim,
        args.output_emb_width,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
        args.vq_act,
        args.vq_norm,
    )
    rng_after_vq = rng_sha256()
    vq_hash, vq_shapes = state_sha256(vq)
    transformer_kwargs = dict(
        vqvae=vq,
        num_vq=12288,
        embed_dim=1024,
        clip_dim=512,
        block_size=51,
        num_layers=9,
        num_local_layer=2,
        n_head=16,
        drop_out_rate=0.1,
        fc_rate=4,
        contact=False,
        shape=True,
    )
    if transformer_class is CurrentTransformer:
        transformer_kwargs.update(
            paper_architecture=False,
            conditioning_contract="historical_mmm_shape_dropout",
        )
    transformer = transformer_class(**transformer_kwargs)
    rng_after_transformer = rng_sha256()
    transformer_hash, transformer_shapes = state_sha256(
        transformer, exclude_prefix="trans_base.vqvae."
    )
    parameter_count = sum(
        parameter.numel() for name, parameter in transformer.named_parameters()
        if not name.startswith("trans_base.vqvae.")
    )
    del transformer, vq
    gc.collect()
    INTERX_SKELETONS[...] = skeleton_copy
    return {
        "rng_before": rng_before,
        "rng_after_vq": rng_after_vq,
        "rng_after_transformer": rng_after_transformer,
        "vq_state_sha256": vq_hash,
        "vq_state_shapes": vq_shapes,
        "transformer_state_sha256_excluding_vq": transformer_hash,
        "transformer_state_shapes_excluding_vq": transformer_shapes,
        "transformer_parameter_count_excluding_vq": parameter_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-vq", type=Path, default=DEFAULT_HISTORICAL_VQ)
    parser.add_argument("--historical-transformer", type=Path, default=DEFAULT_HISTORICAL_TRANSFORMER)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/historical_mgt_initialization_contract_audit.json",
    )
    args = parser.parse_args()

    historical_vq = recovered_classes(
        args.historical_vq, {"VQVAE_SEP3_shape"}, vq_namespace()
    )["VQVAE_SEP3_shape"]
    historical_transformer = recovered_classes(
        args.historical_transformer,
        {"CrossCondTransBase", "CrossCondTransHead", "Text2Motion_Transformer"},
        transformer_namespace(),
    )["Text2Motion_Transformer"]

    historical = construct_record(historical_vq, historical_transformer, args.seed)
    current = construct_record(CurrentVQ, CurrentTransformer, args.seed)
    checks = {
        "same_rng_before": historical["rng_before"] == current["rng_before"],
        "same_rng_after_vq": historical["rng_after_vq"] == current["rng_after_vq"],
        "same_rng_after_transformer": (
            historical["rng_after_transformer"] == current["rng_after_transformer"]
        ),
        "same_vq_keys_and_shapes": (
            historical["vq_state_shapes"] == current["vq_state_shapes"]
        ),
        "same_vq_initial_state": (
            historical["vq_state_sha256"] == current["vq_state_sha256"]
        ),
        "same_transformer_keys_and_shapes_excluding_vq": (
            historical["transformer_state_shapes_excluding_vq"]
            == current["transformer_state_shapes_excluding_vq"]
        ),
        "same_transformer_initial_state_excluding_vq": (
            historical["transformer_state_sha256_excluding_vq"]
            == current["transformer_state_sha256_excluding_vq"]
        ),
        "same_transformer_parameter_count_excluding_vq": (
            historical["transformer_parameter_count_excluding_vq"]
            == current["transformer_parameter_count_excluding_vq"]
        ),
    }
    payload = {
        "schema": "historical_mgt_initialization_contract_audit_v1",
        "seed": args.seed,
        "scope": "CPU constructor RNG and initial state; frozen CLIP is constructed identically before this boundary",
        "sources": {
            "historical_vq": {
                "path": str(args.historical_vq),
                "sha256": sha256_file(args.historical_vq),
            },
            "historical_transformer": {
                "path": str(args.historical_transformer),
                "sha256": sha256_file(args.historical_transformer),
            },
            "current_vq": {
                "path": str(ROOT / "models/vqvae_sep3_shape.py"),
                "sha256": sha256_file(ROOT / "models/vqvae_sep3_shape.py"),
            },
            "current_transformer": {
                "path": str(ROOT / "models/two_person_transformer.py"),
                "sha256": sha256_file(ROOT / "models/two_person_transformer.py"),
            },
        },
        "historical": historical,
        "current": current,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "mismatch",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": payload["status"], "checks": checks}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
