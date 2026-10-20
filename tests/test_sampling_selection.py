import unittest
import torch
from tools.audit_sampling_selection import selection_counts, selected_log_odds


class SelectionTest(unittest.TestCase):
    def test_log_odds_stays_finite_when_probability_saturates(self):
        logits = torch.tensor([[1000.0, 0.0], [10.0, 0.0], [0.0, 0.0]])
        selected = torch.tensor([[0, 1], [0, 1], [0, 1]])
        odds = selected_log_odds(logits, selected)
        torch.testing.assert_close(odds, torch.tensor([[1000., -1000.], [10., -10.], [0., 0.]]))
        self.assertEqual(float(logits[0].softmax(-1)[0]), 1.0)
        uniform = selected_log_odds(torch.zeros(1, 3), torch.tensor([[0, 1, 2]]))
        torch.testing.assert_close(uniform, torch.full((1, 3), -torch.log(torch.tensor(2.))))
        moderate = torch.tensor([[2.0, -1.0, 0.5, 1.0]])
        probability = moderate.double().softmax(-1)
        expected = (probability.log() - torch.log1p(-probability)).float()
        torch.testing.assert_close(selected_log_odds(moderate, torch.arange(4)[None]), expected)

    def test_rejection_and_determinism(self):
        logits = torch.tensor([1.0, 0.0]).expand(2, 8, 3, 2)
        valid = torch.ones(2, 8, dtype=torch.bool)
        def run():
            return selection_counts(logits, valid, draws=1000,
                                    generator=torch.Generator().manual_seed(17))
        counts = run()
        self.assertEqual(counts, run())
        for part in counts.values():
            proposal = part['proposal_non_argmax'] / part['proposal_count']
            uniform = part['uniform_position_accepted_non_argmax'] / part['uniform_position_accepted_count']
            highest = part['highest_confidence_accepted_non_argmax'] / part['highest_confidence_accepted_count']
            self.assertLess(abs(proposal - uniform), 0.03)
            self.assertLess(highest, uniform)
        with self.assertRaises(ValueError):
            selection_counts(logits, valid, temperature=0)
        valid[0] = False
        with self.assertRaises(ValueError):
            selection_counts(logits, valid)


if __name__ == '__main__':
    unittest.main()
