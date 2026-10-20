import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.diagnose_shape_support_transfer import paired_volumetric_metrics


def person(frames=151, gender="male"):
    return {
        "root_orient": np.zeros((frames, 3), np.float32),
        "pose_body": np.ones((frames, 21, 3), np.float32),
        "pose_lhand": np.zeros((frames, 15, 3), np.float32),
        "pose_rhand": np.zeros((frames, 15, 3), np.float32),
        "trans": np.full((frames, 3), 2, np.float32),
        "betas": np.arange(10, dtype=np.float32),
        "gender": gender,
    }


class FakeBackend:
    device = torch.device("cpu")


class PairedVolumetricMetricsTest(unittest.TestCase):
    def test_scores_both_pairs_full_length_and_preserves_inputs(self):
        original = [person(), person(gender="female")]
        optimized = [person(), person(gender="female")]
        optimized[0]['trans'] += 3
        optimized[1]['trans'] -= 1
        optimized[0]['betas'] += .5
        before = copy.deepcopy((original, optimized))
        calls = []

        def score(backend, first, second, seed):
            calls.append((backend, first, second, seed))
            return {"frames": first["trans"].shape[0]}

        with patch("tools.score_volumetric_smpl_npz.score_pair", score), \
             patch("tools.score_volumetric_smpl_npz.stable_sample_seed", return_value=77) as seed_fn:
            result = paired_volumetric_metrics(FakeBackend(), original, optimized, "sample")
        seed_fn.assert_called_once_with(1234, 'sample')
        self.assertEqual(set(result), {"original_shape", "optimized"})
        self.assertEqual([x[1]["trans"].shape[0] for x in calls], [151, 151])
        self.assertEqual([x[3] for x in calls], [77, 77])
        for actual_people, expected_people in zip((original, optimized), before):
            for actual, expected in zip(actual_people, expected_people):
                for key in actual:
                    if isinstance(actual[key], np.ndarray):
                        np.testing.assert_array_equal(actual[key], expected[key])
                    else:
                        self.assertEqual(actual[key], expected[key])
        for call, pair in zip(calls, [original, optimized]):
            for actual, expected in zip(call[1:3], pair):
                for key in expected:
                    if isinstance(expected[key], np.ndarray):
                        np.testing.assert_array_equal(actual[key].numpy(), expected[key])
                    else:
                        self.assertEqual(actual[key], expected[key])

    def test_rejects_unequal_or_empty_frames(self):
        with self.assertRaises(AssertionError):
            paired_volumetric_metrics(FakeBackend(), [person(), person(150)], [person(), person()], "x")
        with self.assertRaises(AssertionError):
            paired_volumetric_metrics(FakeBackend(), [person(0), person(0)], [person(0), person(0)], "x")


if __name__ == "__main__":
    unittest.main()
