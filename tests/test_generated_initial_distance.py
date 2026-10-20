import unittest
import numpy as np
from tools.diagnose_generated_initial_distance import align_initial_distance


class DistanceAnchorTest(unittest.TestCase):
    def test_only_constant_horizontal_translation_changes(self):
        motion = np.zeros((5, 672), dtype=np.float32)
        motion[:3, 666] = [1, 1.2, 1.4]
        reference = motion[0].copy()
        reference[666] = 2
        result, shift, skipped = align_initial_distance(motion, reference, 3)
        self.assertFalse(skipped)
        np.testing.assert_allclose(shift, [1, 0])
        np.testing.assert_allclose(result[0, 666], 2)
        np.testing.assert_allclose(np.diff(result[:3], axis=0),
                                   np.diff(motion[:3], axis=0), atol=2e-7)
        untouched = [i for i in range(672) if i not in (666, 668)]
        np.testing.assert_array_equal(result[:, untouched], motion[:, untouched])
        np.testing.assert_array_equal(result[3:], motion[3:])
        zero = np.zeros_like(motion)
        output, _, skipped = align_initial_distance(zero, reference, 3)
        self.assertTrue(skipped)
        np.testing.assert_array_equal(output, zero)
        with self.assertRaises(ValueError):
            align_initial_distance(motion, reference, 0)


if __name__ == '__main__':
    unittest.main()
