import importlib.util
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "collect_donors", ROOT / "tools/collect_humanml3d_amass_smplx_donors.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_deformation_matrix_matches_official_mtx_half_convention(tmp_path):
    path = tmp_path / "correspondence.pkl"
    original = np.arange(24, dtype=np.float64).reshape(3, 8)
    with path.open("wb") as handle:
        pickle.dump({"mtx": original}, handle)
    result = MODULE.deformation_matrix(path)
    np.testing.assert_array_equal(result, original[:, :4])


def test_shape_statistics_effective_rank_and_nearest_distances():
    betas = np.asarray([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    stats = MODULE.shape_statistics(betas)
    assert stats["effective_rank"] == 1.0
    distances = MODULE.nearest_distances(
        np.asarray([[0.0, 0.0], [2.0, 0.0]]),
        np.asarray([[1.0, 0.0], [4.0, 0.0]]),
    )
    np.testing.assert_allclose(distances, [1.0, 1.0])


def test_nearest_distances_are_gender_matched():
    source = np.asarray([[0.0], [10.0]], dtype=np.float64)
    target = np.asarray([[0.1], [9.9]], dtype=np.float64)
    # Unrestricted matching would be 0.1 for both. Swapping target genders
    # proves that the coverage report respects the SMPL-X model family.
    distances = MODULE.gender_matched_nearest_distances(
        source, ["female", "male"], target, ["male", "female"]
    )
    np.testing.assert_allclose(distances, [9.9, 9.9])


def test_obj_vertex_parser_rejects_nonfinite(tmp_path):
    path = tmp_path / "mesh.obj"
    path.write_text("v 0 0 0\nv nan 1 2\nf 1 1 1\n")
    try:
        MODULE.obj_vertices(path)
    except ValueError as error:
        assert "invalid OBJ vertices" in str(error)
    else:
        raise AssertionError("non-finite source mesh was accepted")


def test_target_shape_id_is_native_smplx_beta_and_gender_specific():
    betas = np.arange(10, dtype=np.float32)
    assert MODULE.canonical_shape_id("male", betas) == MODULE.canonical_shape_id(
        "male", betas.copy()
    )
    assert MODULE.canonical_shape_id("male", betas) != MODULE.canonical_shape_id(
        "female", betas
    )
