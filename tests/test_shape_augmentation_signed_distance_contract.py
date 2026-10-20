#!/usr/bin/env python3
"""Regression checks for the fast original HHC signed-distance path."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_shape_augmentation_defaults_to_original_nearest_vertex_backend():
    runner = (ROOT / "tools/run_interx_shape_augmentation.py").read_text(
        encoding="utf-8"
    )
    optimizer = (ROOT / "optimize/optimize_full_body.py").read_text(
        encoding="utf-8"
    )
    ast.parse(runner)
    ast.parse(optimizer)
    assert "--signed-distance-backend" in runner
    assert "default='nearest_vertex'" in runner
    assert "signed_distance_backend=args.signed_distance_backend" in runner
    assert "signed_distance_backend=signed_distance_backend" in optimizer
    assert "args.signed_distance_backend == 'volumetric_smpl'" in runner


def test_original_development_source_uses_cuda_chamfer_normal_sign():
    source_path = Path("/home/wenning/projects/MMM/optimization/losses.py")
    source = source_path.read_text(encoding="utf-8")
    assert "import chamfer_distance as chd" in source
    assert "signed_dist = dist_unsigned * sign" in source
    assert "d_B2A = point2point_signed" in source
    assert "d_A2B = point2point_signed" in source
    assert "VolumetricSMPL" not in source


if __name__ == "__main__":
    test_shape_augmentation_defaults_to_original_nearest_vertex_backend()
    test_original_development_source_uses_cuda_chamfer_normal_sign()
    print("shape augmentation signed-distance contract: PASS")
