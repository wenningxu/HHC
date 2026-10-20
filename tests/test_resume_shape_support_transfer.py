import unittest

from tools.resume_shape_support_transfer import merged_rows, same_values


class ResumeTest(unittest.TestCase):
    def test_prefix_replay_and_full_coverage(self):
        saved = [{'name': 'a', 'metric': 1.}, {'name': 'b', 'metric': 2.}]
        fresh = [{'name': 'b', 'metric': 2.0000001}, {'name': 'c', 'metric': 3.}]
        result = merged_rows(saved, fresh, ['a', 'b', 'c'], complete=True)
        self.assertEqual(result[:2], saved)
        self.assertEqual(len(result), 3)
        for bad, complete in [([{'name': 'b', 'metric': 4.}], False),
                              ([fresh[1]], False), ([fresh[0]], True),
                              (fresh + [fresh[1]], False)]:
            with self.assertRaises(AssertionError):
                merged_rows(saved, bad, ['a', 'b', 'c'], complete)
        with self.assertRaises(AssertionError):
            same_values({'x': float('nan')}, {'x': float('nan')})
        with self.assertRaises(AssertionError):
            same_values({'x': 1}, {'x': 1.})


if __name__ == '__main__':
    unittest.main()
