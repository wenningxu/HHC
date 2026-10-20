#!/usr/bin/env python3
"""Freeze the current Inter-X optimizer backend and its MMM provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MMM_LOSSES = Path("/home/wenning/projects/MMM/optimization/losses.py")
OUTPUT = ROOT / "output/current_interx_optimization_backend_contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    runner_path = ROOT / "tools/run_paper_contact_optimization.py"
    formal_path = ROOT / "tools/run_paper_optimization_formal.py"
    ablation_path = ROOT / "tools/run_paper_optimization_ablation.py"
    shape_path = ROOT / "tools/run_interx_shape_augmentation_nearest_last_stage.sh"
    runner = runner_path.read_text(encoding="utf-8")
    formal = formal_path.read_text(encoding="utf-8")
    ablation = ablation_path.read_text(encoding="utf-8")
    shape = shape_path.read_text(encoding="utf-8")
    mmm = MMM_LOSSES.read_text(encoding="utf-8")
    checks = {
        "mmm_uses_cuda_chamfer": (
            "import chamfer_distance as chd" in mmm
            and "chd.ChamferDistance()" in mmm
        ),
        "mmm_uses_nearest_target_vertex": (
            "xidx_near_expanded" in mmm and "y.gather" in mmm
        ),
        "mmm_uses_target_normal_sign": (
            "y_normals.gather" in mmm
            and "torch.sign(dot)" in mmm
            and "dist_unsigned * sign" in mmm
        ),
        "formal_runner_defaults_to_nearest_vertex": (
            "default='nearest_vertex'" in runner
            and "signed_distance_backend=args.signed_distance_backend" in runner
        ),
        "formal_experiment_passes_nearest_vertex_explicitly": (
            '"signed_distance_backend": "nearest_vertex"' in formal
            and '"--signed-distance-backend", "nearest_vertex"' in formal
        ),
        "ablation_defaults_to_nearest_vertex": (
            "default='nearest_vertex'" in ablation
            and "'--signed-distance-backend', args.signed_distance_backend" in ablation
        ),
        "shape_augmentation_passes_nearest_vertex_explicitly": (
            "--signed-distance-backend nearest_vertex" in shape
        ),
        "volumetric_smpl_not_claimed_as_paper_requirement": (
            "do not mandate" in runner
            and "read-only physical scorer" in formal
        ),
    }
    payload = {
        "schema": "hhc_current_interx_optimization_backend_contract_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "contract": {
            "optimization_signed_distance_backend": "nearest_vertex",
            "operation": (
                "CUDA Chamfer nearest target vertex with target vertex-normal sign"
            ),
            "paper_scope": (
                "Eq.4/Eq.6 specify signed distance but not a particular backend"
            ),
            "volumetric_smpl_scope": "read_only_common_physical_metric_or_diagnostic",
        },
        "artifacts": {
            "original_mmm_losses": artifact(MMM_LOSSES),
            "optimizer_runner": artifact(runner_path),
            "formal_runner": artifact(formal_path),
            "ablation_runner": artifact(ablation_path),
            "shape_launcher": artifact(shape_path),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
