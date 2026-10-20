#!/usr/bin/env python3

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_backend_freezes_interx_flat_hand_convention():
    backend = (
        ROOT / "optimization/volumetric_smpl_backend.py"
    ).read_text(encoding="utf-8")
    ast.parse(backend, filename="optimization/volumetric_smpl_backend.py")
    assert "FLAT_HAND_MEAN = True" in backend
    assert "flat_hand_mean=self.FLAT_HAND_MEAN" in backend
    assert "'smplx_flat_hand_mean': True" in backend


def test_optimizer_and_shape_sidecars_bind_the_representation():
    optimizer = (ROOT / "optimize/optimize_full_body.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "tools/run_interx_shape_augmentation.py").read_text(
        encoding="utf-8"
    )
    ast.parse(optimizer, filename="optimize/optimize_full_body.py")
    ast.parse(runner, filename="tools/run_interx_shape_augmentation.py")
    assert "'volumetric_smpl_representation'" in optimizer
    assert "'volumetric_smpl_representation'" in runner
    assert "'hand_pose_convention': 'flat_hand_mean'" in runner


def test_physical_scorer_binds_the_same_representation_contract():
    scorer = (ROOT / "tools/score_volumetric_smpl_npz.py").read_text(
        encoding="utf-8"
    )
    ast.parse(scorer, filename="tools/score_volumetric_smpl_npz.py")
    assert scorer.count(
        "'volumetric_smpl_representation': backend.representation_contract"
    ) == 2


def test_shape_queue_and_result_builder_require_promoted_flat_hand_evidence():
    gate = (
        ROOT / "tools/require_interx_volumetric_smpl_contract.py"
    ).read_text(encoding="utf-8")
    queue = (
        ROOT / "tools/queue_interx_shape_augmentation_gpu1.sh"
    ).read_text(encoding="utf-8")
    builder = (
        ROOT / "tools/build_shape_augmentation_results.py"
    ).read_text(encoding="utf-8")
    ast.parse(gate, filename="tools/require_interx_volumetric_smpl_contract.py")
    ast.parse(builder, filename="tools/build_shape_augmentation_results.py")
    assert "query chunk {args.query_chunk} is not authorized" in gate
    assert "--query-chunk 10475" in queue
    assert "interx_shape_aug_pilot_optimized_flathand_query10475" in queue
    assert "VOLUMETRIC_CONTRACT_GATE" in builder
    assert "pilot_assessment.get('full_augmentation_authorized') is True" in builder


if __name__ == "__main__":
    test_production_backend_freezes_interx_flat_hand_convention()
    test_optimizer_and_shape_sidecars_bind_the_representation()
    test_physical_scorer_binds_the_same_representation_contract()
    test_shape_queue_and_result_builder_require_promoted_flat_hand_evidence()
