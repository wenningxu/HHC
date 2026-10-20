from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.score_volumetric_smpl_npz import foot_sliding
from utils.interx_ground_alignment import (
    align_person_for_mmm_optimization,
    restore_person_to_raw_coordinates,
)


def test_mmm_alignment_and_raw_coordinate_roundtrip():
    payload = {
        "trans": np.asarray([[0.0, 0.1, 0.0], [0.2, 0.2, 0.3]], np.float32),
        "betas": np.zeros(10, np.float32),
        "gender": "male",
    }

    def geometry(current):
        joints = np.zeros((2, 2, 3), np.float32)
        joints[..., 1] = np.asarray([[-1.2, -1.0], [-1.1, -0.9]])
        joints[..., 1] += current["trans"][:, None, 1]
        return None, joints

    aligned, floor = align_person_for_mmm_optimization(payload, geometry)
    assert abs(floor - (-1.1)) < 1e-6
    np.testing.assert_allclose(
        aligned["trans"][:, 1], payload["trans"][:, 1] + 1.1
    )
    optimized = dict(aligned)
    optimized["trans"] = aligned["trans"] + np.asarray([0.01, 0.02, 0.03])
    restored = restore_person_to_raw_coordinates(optimized, floor)
    np.testing.assert_allclose(
        restored["trans"], payload["trans"] + np.asarray([0.01, 0.02, 0.03]),
        atol=1e-7,
    )
    np.testing.assert_allclose(payload["trans"][:, 1], [0.1, 0.2])


def test_foot_sliding_uses_sequence_floor_not_pelvis_origin():
    joints = torch.zeros(3, 12, 3)
    joints[..., 1] = -1.0
    joints[:, 7, 1] = -0.95
    joints[:, 10, 1] = -0.90
    joints[:, 8, 1] = -0.70
    joints[:, 11, 1] = -0.60
    joints[1, 7, 0] = 0.2
    joints[2, 7, 0] = 0.6
    result = foot_sliding(joints)
    assert result["contact_observations"] == 2
    assert abs(result["mean_horizontal_speed_m_per_frame"] - 0.3) < 1e-6


def test_foot_sliding_floor_excludes_appended_landmark_joints():
    skeleton = torch.zeros(3, 55, 3)
    skeleton[..., 1] = -1.0
    skeleton[:, 7, 1] = -0.95
    skeleton[:, 10, 1] = -0.90
    skeleton[:, 8, 1] = -0.70
    skeleton[:, 11, 1] = -0.60
    skeleton[1, 7, 0] = 0.2
    skeleton[2, 7, 0] = 0.6
    expected = foot_sliding(skeleton)

    landmarks = torch.zeros(3, 72, 3)
    landmarks[..., 1] = -10.0
    landmarks[1:, :, 1] -= 1.0
    observed = foot_sliding(torch.cat((skeleton, landmarks), dim=1))
    assert observed == expected


if __name__ == "__main__":
    test_mmm_alignment_and_raw_coordinate_roundtrip()
    test_foot_sliding_uses_sequence_floor_not_pelvis_origin()
    test_foot_sliding_floor_excludes_appended_landmark_joints()
