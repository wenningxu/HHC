#!/usr/bin/env python3
"""Prove TiMotion and CoShMDM use the same 262-D to 672-D adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
TIMOTION_ADAPTER = ROOT / "tools/export_timotion_official_interx.py"
COSH_ADAPTER = Path(
    "/mnt/8T-HDD/wenning/coshmdm-code/utils/interx_official_adapter.py"
)
TIMOTION_ROOT = Path("/mnt/8T-HDD/wenning/TIMotion")
COSH_ROOT = Path("/mnt/8T-HDD/wenning/coshmdm-code")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_module_isolated(name: str, path: Path, dependency_root: Path):
    """Load an adapter with its own repository's ``utils`` dependencies.

    Both repositories use the top-level package name ``utils``. Loading both
    adapters in one interpreter without isolation silently makes the second
    adapter reuse the first adapter's cached rotation-conversion module and
    can produce a false bitwise-equivalence pass.
    """
    saved_path = list(sys.path)
    saved_utils = {
        key: value for key, value in sys.modules.items()
        if key == "utils" or key.startswith("utils.")
    }
    for key in list(saved_utils):
        del sys.modules[key]
    sys.path.insert(0, str(dependency_root))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key in list(sys.modules):
            if key == "utils" or key.startswith("utils."):
                del sys.modules[key]
        sys.modules.update(saved_utils)
        sys.path[:] = saved_path


def difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    delta = (left - right).abs()
    return {
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "bitwise_equal": bool(torch.equal(left, right)),
        "max_abs": float(delta.max().item()),
        "rmse": float(torch.sqrt(torch.mean(delta.square())).item()),
        "finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
    }


def make_case(dtype: torch.dtype, flattened: bool) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260831)
    features = torch.zeros((3, 5, 2, 262), dtype=dtype)
    positions = torch.randn((3, 5, 2, 22, 3), generator=generator, dtype=dtype)
    rotations = torch.randn((3, 5, 2, 21, 6), generator=generator, dtype=dtype)
    features[..., :66] = positions.reshape(3, 5, 2, 66)
    features[..., 132:258] = rotations.reshape(3, 5, 2, 126)
    skeleton = torch.randn((3, 2, 55, 3), generator=generator, dtype=dtype)
    # Give the body a non-degenerate, reproducible articulated rest shape.
    skeleton[..., :22, :] = torch.cumsum(skeleton[..., :22, :] * 0.05, dim=-2)
    if flattened:
        features = features.reshape(3, 5, 524)
    return features, skeleton


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/timotion_cosh_adapter_equivalence_audit.json",
    )
    parser.add_argument(
        "--timotion-adapter", type=Path, default=TIMOTION_ADAPTER,
        help="TiMotion adapter implementation to audit.",
    )
    parser.add_argument(
        "--timotion-rotation-contract",
        choices=("repository", "standard"),
        default="repository",
        help=(
            "Use TiMotion's repository decoder to expose the current bug, or "
            "the standard row-concatenated decoder used by the repair."
        ),
    )
    args = parser.parse_args()

    timotion_adapter = args.timotion_adapter.resolve()
    timotion = load_module_isolated(
        "timotion_official_exporter_audit", timotion_adapter, TIMOTION_ROOT
    )
    cosh = load_module_isolated(
        "cosh_official_adapter_audit", COSH_ADAPTER, COSH_ROOT
    )
    if args.timotion_rotation_contract == "standard":
        timotion.rotation_6d_to_matrix = cosh.rotation_6d_to_matrix
    cases = {}
    with torch.inference_mode():
        for dtype in (torch.float32, torch.float64):
            for flattened in (False, True):
                features, skeleton = make_case(dtype, flattened)
                for shape_specific in (False, True):
                    active_skeleton = skeleton if shape_specific else None
                    left = timotion.timotion_to_official_interx(
                        features, skeleton=active_skeleton
                    )
                    right = cosh.cosh_to_official_interx(
                        features, skeleton=active_skeleton
                    )
                    label = (
                        f"{str(dtype).rsplit('.', 1)[-1]}_"
                        f"{'flat524' if flattened else 'pair262'}_"
                        f"{'shape_specific' if shape_specific else 'shape_free'}"
                    )
                    cases[label] = difference(left, right)

    checks = {
        "all_cases_finite": all(case["finite"] for case in cases.values()),
        "all_cases_bitwise_equal": all(
            case["bitwise_equal"] for case in cases.values()
        ),
        "all_cases_zero_max_abs": all(
            case["max_abs"] == 0.0 for case in cases.values()
        ),
        "both_input_layouts_covered": (
            any("flat524" in label for label in cases)
            and any("pair262" in label for label in cases)
        ),
        "both_root_paths_covered": (
            any("shape_specific" in label for label in cases)
            and any("shape_free" in label for label in cases)
        ),
        "both_float_precisions_covered": (
            any(label.startswith("float32") for label in cases)
            and any(label.startswith("float64") for label in cases)
        ),
    }
    payload = {
        "schema": "timotion_cosh_adapter_equivalence_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "contract": {
            "input": "released two-person 262-D representation",
            "output": "official raw Inter-X 672-D representation",
            "formal_path": "shape-specific all-joint root fit; identity hands",
            "ground_truth_motion_consumed": False,
            "dependency_loading": "isolated repository roots",
            "timotion_rotation_contract": args.timotion_rotation_contract,
            "timotion_adapter": str(timotion_adapter),
        },
        "cases": cases,
        "sources": {
            "timotion_exporter": {
                "path": str(timotion_adapter),
                "bytes": timotion_adapter.stat().st_size,
                "sha256": sha256_file(timotion_adapter),
            },
            "coshmdm_adapter": {
                "path": str(COSH_ADAPTER),
                "bytes": COSH_ADAPTER.stat().st_size,
                "sha256": sha256_file(COSH_ADAPTER),
            },
            "timotion_rotation_conversions": {
                "path": str(TIMOTION_ROOT / "utils/rotation_conversions.py"),
                "bytes": (TIMOTION_ROOT / "utils/rotation_conversions.py").stat().st_size,
                "sha256": sha256_file(
                    TIMOTION_ROOT / "utils/rotation_conversions.py"
                ),
            },
            "coshmdm_rotation_conversions": {
                "path": str(COSH_ROOT / "utils/rotation_conversions.py"),
                "bytes": (COSH_ROOT / "utils/rotation_conversions.py").stat().st_size,
                "sha256": sha256_file(
                    COSH_ROOT / "utils/rotation_conversions.py"
                ),
            },
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
