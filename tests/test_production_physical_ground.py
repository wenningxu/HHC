import unittest

import numpy as np
import torch

from tools.run_interx_shape_augmentation import prepare_physical_ground


class FakeCalc:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def _compute_geometry(self, params, *, compute_normals=True):
        self.calls.append((params, compute_normals))
        y = float(params["betas"][0])
        return torch.tensor([[[0., y, 0.]]]), None, None


def record(length=5):
    return {
        "source_frames": length, "crop_start": 2, "crop_stop": 4,
        "source_betas": [[1.] + [0.] * 9, [2.] + [0.] * 9],
        "source_genders": ["male", "female"],
    }


class ProductionPhysicalGroundTest(unittest.TestCase):
    def test_full_source_floors_and_crop_payloads(self):
        raw = np.zeros((5, 56, 6), np.float32)
        raw[:, 55, 1] = np.arange(5)
        raw[:, 55, 4] = np.arange(5) + 10
        donors = [{"betas": [3.] + [0.] * 9, "gender": "male"},
                  {"betas": [4.] + [0.] * 9, "gender": "female"}]
        calc = FakeCalc()
        original, new, floors = prepare_physical_ground(raw, record(), donors, calc)
        self.assertEqual(floors, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(len(calc.calls), 4)
        self.assertTrue(all(len(params['trans']) == 5 for params, _ in calc.calls))
        self.assertTrue(all(not normals for _, normals in calc.calls))
        self.assertEqual(original[0]["trans"].shape[0], 2)
        self.assertEqual(new[1]["trans"].shape[0], 2)
        np.testing.assert_allclose(original[0]["trans"][:, 1], [1., 2.])
        np.testing.assert_allclose(new[1]["trans"][:, 1], [8., 9.])
        self.assertEqual(original[0]["gender"], "male")
        np.testing.assert_array_equal(new[1]["betas"], donors[1]["betas"])
        self.assertEqual(raw[:, 55, 1].tolist(), [0., 1., 2., 3., 4.])

    def test_source_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_physical_ground(np.zeros((4, 56, 6), np.float32), record(), [], FakeCalc())


if __name__ == "__main__":
    unittest.main()
