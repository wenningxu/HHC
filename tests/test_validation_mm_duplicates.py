import unittest
import numpy as np
from tools.score_hhc_validation_mm import duplicate_motion_summary, all_pair_mm_summary

class TestDuplicates(unittest.TestCase):
    def test_all_pairs_and_zero_diversity(self):
        result = all_pair_mm_summary(np.array([[[0.], [3.], [6.]]]))
        self.assertEqual(result['pairs_per_prompt'], 3)
        self.assertEqual(result['mean'], 4.)
        self.assertEqual(result['per_prompt_median'], [3.])
        self.assertEqual(all_pair_mm_summary(np.zeros((2, 3, 4)))[
            'top_5pct_prompts_fraction_of_total_distance'], 0.)
        with self.assertRaises(ValueError): all_pair_mm_summary(np.full((2, 3, 4), np.nan))
    def test_pairs_and_padding(self):
        x = np.zeros((1, 3, 4, 2), dtype=np.float32)
        x[0, 1, 2:] = 9
        self.assertEqual(duplicate_motion_summary(x, [2])['duplicate_pairs'], 3)
        x[0, 2, :2] = 1
        x[0, 1, :2] = 2
        self.assertEqual(duplicate_motion_summary(x, [2])['duplicate_pairs'], 0)
        self.assertEqual(duplicate_motion_summary(x, [2])['total_pairs'], 3)

if __name__ == '__main__':
    unittest.main()
