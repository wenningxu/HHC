#!/usr/bin/env python3
"""Audit the deployed 1024-D historical HHC topology from a real checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def indexed_modules(keys, prefix: str, component: str) -> list[int]:
    pattern = re.compile(re.escape(prefix) + re.escape(component) + r"\.(\d+)\.")
    return sorted({
        int(match.group(1))
        for key in keys if (match := pattern.match(key)) is not None
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--sampling-runtime-audit", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    manifest_path = args.training_manifest.resolve()
    sampling_path = args.sampling_runtime_audit.resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state = checkpoint.get("state_dict", {})
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no non-empty Lightning state_dict")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = manifest.get("resolved_arguments", {})
    sampling = json.loads(sampling_path.read_text(encoding="utf-8"))

    keys = set(state)
    base_blocks = indexed_modules(
        keys, "trans_encoder.trans_base.", "blocks"
    )
    cross_blocks = indexed_modules(
        keys, "trans_encoder.trans_base.", "cross_att"
    )
    head_blocks = indexed_modules(
        keys, "trans_encoder.trans_head.", "blocks"
    )
    expected_shapes = {
        "trans_encoder.trans_base.part_emb.weight": [3, 1024],
        "trans_encoder.trans_base.pos_embed.pe": [51, 1, 1024],
        "trans_encoder.trans_head.head.weight": [12288, 1024],
        "trans_encoder.trans_base.gender_embedding.weight": [2, 4],
        "trans_encoder.trans_base.shape_projection.weight": [40, 20],
        "trans_encoder.trans_base.condition_mlp.0.weight": [1024, 48],
        "trans_encoder.trans_base.condition_mlp.2.weight": [1024, 1024],
    }
    observed_shapes = {
        key: list(state[key].shape) if key in state else None
        for key in expected_shapes
    }
    model_source = ROOT / "models/two_person_transformer.py"
    motion_source = ROOT / "models/motion_transformer.py"
    exporter = ROOT / "tools/export_hhc_official_interx.py"
    exporter_source = exporter.read_text(encoding="utf-8")
    checks = {
        "checkpoint_step_matches": int(checkpoint.get("global_step", -1))
        == args.expected_step,
        "manifest_declares_exact_historical_topology": all((
            resolved.get("embed_dim_gpt") == 1024,
            resolved.get("n_head_gpt") == 16,
            resolved.get("num_layers") == 9,
            resolved.get("num_local_layer") == 2,
            resolved.get("nb_code") == 12288,
            resolved.get("block_size") == 51,
            resolved.get("paper_architecture") is False,
            resolved.get("shape") is True,
        )),
        "seven_global_base_blocks_present": base_blocks == list(range(7)),
        "two_local_cross_attention_blocks_present": cross_blocks == [0, 1],
        "nine_output_blocks_present": head_blocks == list(range(9)),
        "paper_blocks_absent": not any(
            key.startswith("trans_encoder.trans_base.paper_blocks.")
            for key in keys
        ),
        "checkpoint_tensor_shapes_match": observed_shapes == expected_shapes,
        "live_model_sources_equal_launch_hashes": (
            manifest.get("sha256", {}).get("models/two_person_transformer.py")
            == sha256_file(model_source)
            and manifest.get("sha256", {}).get("models/motion_transformer.py")
            == sha256_file(motion_source)
        ),
        "sampler_audit_binds_same_model_source": (
            sampling.get("status") == "pass"
            and sampling.get("contract", {}).get("model_architecture_contract")
            == "historical_1024d_with_end_token"
            and sampling.get("source", {}).get("sha256")
            == sha256_file(model_source)
        ),
        "exporter_instantiates_exact_historical_topology": all(
            marker in exporter_source for marker in (
                "embed_dim_gpt=1024, clip_dim=512, block_size=51",
                "num_layers=9, num_local_layer=2, n_head_gpt=16",
                "shape=True, paper_architecture=False",
            )
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    payload = {
        "schema": "hhc_historical_model_checkpoint_structure_audit_v1",
        "status": status,
        "expected_step": args.expected_step,
        "checks": checks,
        "observed": {
            "base_block_indices": base_blocks,
            "local_cross_attention_indices": cross_blocks,
            "output_block_indices": head_blocks,
            "tensor_shapes": observed_shapes,
            "state_dict_tensor_count": len(state),
        },
        "contract": {
            "hidden_dimension": 1024,
            "attention_heads": 16,
            "global_base_blocks": 7,
            "local_cross_attention_blocks": 2,
            "output_blocks": 9,
            "vocabulary_per_part": 12288,
            "token_layout": "historical explicit end token",
            "paper_architecture": False,
        },
        "artifacts": {
            "checkpoint": artifact(checkpoint_path),
            "training_manifest": artifact(manifest_path),
            "sampling_runtime_audit": artifact(sampling_path),
            "model_source": artifact(model_source),
            "motion_source": artifact(motion_source),
            "exporter": artifact(exporter),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
