import unittest
import torch
from tools.probe_vq_temporal_compression import thin_tokens, nearest_token_ids, part_neighbor_alternatives, part_thinning

class TestThinning(unittest.TestCase):
    def test_part_interventions_change_only_the_selected_stream(self):
        tokens = torch.arange(12).reshape(4, 3)
        original = tokens.clone()
        alternatives = part_neighbor_alternatives(tokens, tokens + 20)
        for channel, part in enumerate(('body', 'hands', 'root')):
            changed = alternatives[f'nearest_{part}']
            self.assertTrue(torch.equal(changed[:, channel], tokens[:, channel] + 20))
            other = [c for c in range(3) if c != channel]
            self.assertTrue(torch.equal(changed[:, other], tokens[:, other]))
        self.assertTrue(torch.equal(tokens, original))

    def test_odd_even_lengths(self):
        x=torch.arange(5)
        self.assertEqual(len(thin_tokens(x,'even')),3)
        self.assertEqual(len(thin_tokens(x,'odd')),3)
        self.assertTrue(torch.equal(thin_tokens(x,'odd'),torch.tensor([1,3,4])))
        for length in (1, 4, 5):
            for mode in ('even', 'odd'):
                restored = thin_tokens(torch.arange(length), mode).repeat_interleave(2)[:length]
                self.assertEqual(len(restored), length)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError): thin_tokens(torch.empty(0), 'even')
    def test_nearest_excludes_self_and_preserves_order(self):
        book = torch.tensor([[0.], [1.], [3.]])
        tokens = torch.tensor([[2], [0], [1], [2]])
        self.assertTrue(torch.equal(nearest_token_ids(tokens, [book]),
                                    torch.tensor([[1], [1], [0], [1]])))
        with self.assertRaises(ValueError): nearest_token_ids(tokens, [book[:1]])
        with self.assertRaises(ValueError): nearest_token_ids(torch.tensor([[3]]), [book])

    def test_part_thinning_phases_change_only_selected_stream(self):
        for length in (1, 4, 5):
            tokens = torch.arange(length * 3).reshape(length, 3)
            original = tokens.clone()
            for phase in ('even', 'odd'):
                alternatives = part_thinning(tokens, phase)
                for channel, part in enumerate(('body', 'hands', 'root')):
                    changed = alternatives[f'{phase}_{part}']
                    expected = thin_tokens(tokens[:, channel], phase).repeat_interleave(2)[:length]
                    self.assertTrue(torch.equal(changed[:, channel], expected))
                    other = [c for c in range(3) if c != channel]
                    self.assertTrue(torch.equal(changed[:, other], tokens[:, other]))
            self.assertTrue(torch.equal(tokens, original))

    def test_part_thinning_retained_ratio(self):
        self.assertEqual([(2 * n + (n + 1) // 2) / (3 * n) for n in (1, 4, 5)],
                         [1.0, 0.8333333333333334, 0.8666666666666667])

if __name__ == '__main__': unittest.main()
