import unittest
from unittest.mock import patch

import torch

import optimization.losses as losses


class FakeBodyModel(torch.nn.Module):
    calls = 0

    def __init__(self, **kwargs):
        super().__init__()
        type(self).calls += 1
        self.register_buffer("sentinel", torch.zeros(1))


def make_loss(static=None):
    return losses.DyadicInteractionLoss(
        device="cpu", static_body_models=static,
    )


class StaticBodyModelReuseTest(unittest.TestCase):
    def setUp(self):
        FakeBodyModel.calls = 0
        self.patches = [
            patch.object(losses, "BodyModel", FakeBodyModel),
            patch.object(losses.pkl, "load", return_value={1: [0], 2: [1]}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def test_shared_models_reused_but_loss_state_is_fresh(self):
        shared = {}
        first, second = make_loss(shared), make_loss(shared)
        self.assertEqual(FakeBodyModel.calls, 3)
        self.assertIs(first.male_bm, second.male_bm)
        self.assertIs(first.neutral_bm, second.neutral_bm)
        self.assertIs(first.female_bm, second.female_bm)
        self.assertIsNot(first.fixed_pairs, second.fixed_pairs)
        self.assertIsNot(first._initial_grounded_foot_masks, second._initial_grounded_foot_masks)
        self.assertIsNot(first.current_winner_regions, second.current_winner_regions)
        self.assertIsNot(first.weights, second.weights)

    def test_default_models_are_not_reused(self):
        make_loss()
        make_loss()
        self.assertEqual(FakeBodyModel.calls, 6)

    def test_partial_static_dict_is_rejected(self):
        with self.assertRaises(ValueError):
            make_loss({"male": FakeBodyModel()})


if __name__ == "__main__":
    unittest.main()
