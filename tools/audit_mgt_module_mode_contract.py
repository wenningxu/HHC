#!/usr/bin/env python
"""Audit the MGT train/eval child-module mode contract around sanity validation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
from lightning.pytorch.loops.fit_loop import _FitLoop
from lightning.pytorch.utilities.model_helpers import _ModuleMode


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_TRAIN = ROOT / (
    "output/historical_mmm_local_history_train/"
    "1771033604763_set-424245_change-424244_ContentChange.py"
)
CURRENT_TRAIN = ROOT / "train/train_t2i_trans.py"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_method(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    klass = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node for node in klass.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def calls_self_method(node, attribute, method):
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        owner = child.func.value
        if (
            child.func.attr == method
            and isinstance(owner, ast.Attribute)
            and owner.attr == attribute
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "self"
        ):
            return True
    return False


def reads_training_flag(node, attribute):
    return any(
        isinstance(child, ast.Attribute)
        and child.attr == "training"
        and isinstance(child.value, ast.Attribute)
        and child.value.attr == attribute
        and isinstance(child.value.value, ast.Name)
        and child.value.value.id == "self"
        for child in ast.walk(node)
    )


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-train", type=Path, default=HISTORICAL_TRAIN)
    parser.add_argument("--current-train", type=Path, default=CURRENT_TRAIN)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/historical_mgt_module_mode_contract_audit.json",
    )
    args = parser.parse_args()
    for path in (args.historical_train, args.current_train):
        if not path.is_file():
            raise FileNotFoundError(path)

    historical_init = class_method(args.historical_train, "MaskGITLightning", "__init__")
    current_init = class_method(args.current_train, "MaskGITLightning", "__init__")
    current_train_start = class_method(
        args.current_train, "MaskGITLightning", "on_train_start"
    )

    # Demonstrate the exact Lightning semantics involved: restore writes each
    # captured .training flag directly, preserving a child that was eval even
    # when its parent was train.
    root = torch.nn.Sequential(torch.nn.Dropout(0.1)).train()
    child = root[0]
    child.eval()
    modes = _ModuleMode()
    modes.capture(root)
    root.eval()
    modes.restore(root)
    runtime_restore = {
        "parent_training": root.training,
        "child_training": child.training,
    }
    fit_on_run_start_source = inspect.getsource(_FitLoop.on_run_start)
    restore_source = inspect.getsource(_ModuleMode.restore)

    checks = {
        "historical_loaded_vq_initialized_eval": calls_self_method(
            historical_init, "net", "eval"
        ),
        "historical_transformer_not_initialized_eval": not calls_self_method(
            historical_init, "trans_encoder", "eval"
        ),
        "current_loaded_vq_initialized_eval": calls_self_method(
            current_init, "net", "eval"
        ),
        "current_transformer_not_initialized_eval": not calls_self_method(
            current_init, "trans_encoder", "eval"
        ),
        "current_train_start_checks_transformer_mode": reads_training_flag(
            current_train_start, "trans_encoder"
        ),
        "current_train_start_checks_vq_mode": reads_training_flag(
            current_train_start, "net"
        ),
        "lightning_restore_preserves_parent_train_child_eval": (
            runtime_restore == {"parent_training": True, "child_training": False}
        ),
        "lightning_restore_assigns_captured_training_flags": (
            ".training = self.mode[name]" in restore_source
        ),
        "fit_on_run_start_does_not_call_model_train": (
            ".train(" not in fit_on_run_start_source
        ),
    }
    payload = {
        "schema": "hhc_historical_mgt_module_mode_contract_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "effective_training_contract": {
            "transformer": "train",
            "loaded_vq": "eval",
            "formal_clip_and_vq_parameters": "frozen",
            "historical_optimizer_membership_includes_detached_vq": True,
        },
        "runtime_restore_demonstration": runtime_restore,
        "checks": checks,
        "sources": {
            "historical_train": {
                "path": str(args.historical_train.resolve()),
                "sha256": sha256_file(args.historical_train),
            },
            "current_train": {
                "path": str(args.current_train.resolve()),
                "sha256": sha256_file(args.current_train),
            },
            "lightning_fit_on_run_start": inspect.getsourcefile(_FitLoop),
            "lightning_module_mode": inspect.getsourcefile(_ModuleMode),
        },
        "interpretation": (
            "Calling trans_encoder.eval() before Trainer.fit is not repaired by "
            "the sanity loop and disables transformer dropout during fitting."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
