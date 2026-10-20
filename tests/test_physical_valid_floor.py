import unittest
import torch
from tools.score_fair_interx_physical import foot_floor_height


class FloorTest(unittest.TestCase):
    def test_padding_does_not_change_valid_floor(self):
        feet = torch.zeros(2, 4, 4, 3)
        feet[..., 1] = 2
        valid = torch.tensor([[True, True, False, False], [True] * 4])
        expected = foot_floor_height(feet, valid)
        feet[0, 2:, :, 1] = -100
        self.assertTrue(torch.equal(foot_floor_height(feet, valid), expected))
        self.assertEqual(foot_floor_height(feet, valid, 'legacy_all_frames')[0].item(), -100)
        valid[0] = False
        with self.assertRaises(ValueError):
            foot_floor_height(feet, valid)


if __name__ == '__main__':
    unittest.main()
