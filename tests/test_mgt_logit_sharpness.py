import math
import unittest
from types import SimpleNamespace
import numpy as np

import torch

from tools.audit_mgt_mask_exposure_gap import accumulator, add_metrics, make_batch


class MGTLogitSharpnessTest(unittest.TestCase):
    def test_explicit_token_window_preserves_prefix_end_and_padding(self):
        tokens = np.arange(60 * 3).reshape(60, 3)
        common = dict(text=[{'caption': 'test'}], betas=np.zeros(20), gender=[0, 1])
        dataset = SimpleNamespace(codebook_size=200, data_dict={
            'long': dict(common, m_token_list=tokens),
            'short': dict(common, m_token_list=tokens[:20])})
        for window in (38, 50):
            _, target, lengths, _, _ = make_batch(dataset, ['long', 'short'], window)
            self.assertEqual(target.shape, (2, window, 3))
            self.assertEqual(lengths.tolist(), [window - 1, 20])
            self.assertTrue(np.array_equal(target[0, :-1].numpy(), tokens[:window - 1]))
            self.assertTrue((target[0, -1] == 200).all())
            self.assertTrue((target[1, 20] == 200).all())
            self.assertTrue((target[1, 21:] == 201).all())
        for invalid in (0, 1, 38.5, True):
            with self.assertRaises(ValueError):
                make_batch(dataset, ['long'], invalid)

    def test_uniform_logits_have_maximum_entropy_and_zero_margin(self) -> None:
        store = accumulator()
        logits = torch.zeros(1, 1, 3, 4)
        target = torch.zeros(1, 1, 3, dtype=torch.long)
        mask = torch.ones(1, 1, 3, dtype=torch.bool)
        add_metrics(
            store, "all_parts_fully_masked", logits, target, mask,
            sharpness_temperature=0.5,
        )
        for part in ("body", "hand", "root"):
            record = store["all_parts_fully_masked"][part]
            self.assertAlmostEqual(record["entropy_sum"], math.log(4), places=6)
            self.assertAlmostEqual(record["top1_probability_sum"], 0.25, places=6)
            self.assertAlmostEqual(record["top1_top2_margin_sum"], 0.0, places=6)
            self.assertAlmostEqual(record["target_probability_sum"], 0.25, places=6)
            self.assertEqual(record["class_count"], 4)
            self.assertEqual(record['ranking_saturated_tokens'], 0)
            self.assertEqual(record['ranking_logit_margin_sum'], 0.0)

    def test_peaked_logits_have_low_entropy_and_large_margin(self) -> None:
        store = accumulator()
        logits = torch.full((1, 1, 3, 4), -10.0)
        logits[..., 0] = 10.0
        target = torch.zeros(1, 1, 3, dtype=torch.long)
        mask = torch.ones(1, 1, 3, dtype=torch.bool)
        add_metrics(
            store, "paper_scheduled_mask", logits, target, mask,
            sharpness_temperature=0.5,
        )
        for part in ("body", "hand", "root"):
            record = store["paper_scheduled_mask"][part]
            self.assertLess(record["entropy_sum"], 1e-6)
            self.assertGreater(record["top1_probability_sum"], 0.999999)
            self.assertGreater(record["top1_top2_margin_sum"], 0.999999)
            self.assertGreater(record["target_probability_sum"], 0.999999)
            self.assertEqual(record['ranking_saturated_tokens'], 1)
            self.assertEqual(record['ranking_logit_margin_sum'], 20.0)

    def test_ranking_saturation_does_not_use_proposal_temperature(self):
        for temperature in (0.1, 2.0):
            store = accumulator()
            logits = torch.zeros(1, 1, 3, 4)
            logits[..., 0] = 10.0
            add_metrics(store, 'all_parts_fully_masked', logits,
                        torch.zeros(1, 1, 3, dtype=torch.long),
                        torch.ones(1, 1, 3, dtype=torch.bool),
                        sharpness_temperature=temperature)
            for record in store['all_parts_fully_masked'].values():
                self.assertEqual(record['ranking_saturated_tokens'], 0)
                self.assertEqual(record['ranking_logit_margin_sum'], 10.0)

    def test_class_count_drift_fails_closed(self) -> None:
        store = accumulator()
        target = torch.zeros(1, 1, 3, dtype=torch.long)
        mask = torch.ones(1, 1, 3, dtype=torch.bool)
        add_metrics(
            store, "paper_scheduled_mask", torch.zeros(1, 1, 3, 4),
            target, mask,
        )
        with self.assertRaisesRegex(ValueError, "class count changed"):
            add_metrics(
                store, "paper_scheduled_mask", torch.zeros(1, 1, 3, 5),
                target, mask,
            )


if __name__ == "__main__":
    unittest.main()
