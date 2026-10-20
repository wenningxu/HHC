import sys
import unittest
from types import SimpleNamespace
import torch
from models.two_person_transformer import Text2Motion_Transformer
from tools.trace_hhc_sampling import trace_cascade, summarize_records, select_indices


class TraceTest(unittest.TestCase):
    def test_length_selection_is_deterministic_and_covers_extremes(self):
        self.assertEqual(select_indices([150, 20, 100, 50, 120], 3, 'prefix').tolist(), [0, 1, 2])
        self.assertEqual(select_indices([150, 20, 100, 50, 120], 3, 'length_quantiles').tolist(), [1, 2, 0])
        with self.assertRaises(ValueError):
            select_indices([50], 2, 'length_quantiles')

    def test_trace_preserves_output_rng_and_records_real_commitments(self):
        logits = torch.arange(2 * 5 * 3 * 8).reshape(2, 5, 3, 8).float().sin()
        model = SimpleNamespace(num_vq=8, block_size=6, paper_architecture=False,
                                forward_function=lambda *a, **kw: logits)
        args = (torch.zeros(2, 1), None)
        for ranking, noise in [('probability', 0.), ('probability', 1.), ('log_odds', 1.), ('max_probability', 0.)]:
            options = dict(m_length=torch.tensor([8, 20]), max_steps=8,
                           confidence_ranking=ranking, confidence_noise_scale=noise)
            torch.manual_seed(912)
            plain = Text2Motion_Transformer.sample_intra_cas(model, *args, **options)
            plain_rng = torch.get_rng_state().clone()
            torch.manual_seed(912)
            traced, rows = trace_cascade(model, *args, **options)
            self.assertTrue(torch.equal(plain, traced))
            self.assertTrue(torch.equal(plain_rng, torch.get_rng_state()))
            self.assertIsNone(sys.gettrace())
            self.assertTrue(torch.equal(torch.from_numpy(rows[-1]['after_ids']), traced))
            for previous, following in zip(rows, rows[1:]):
                self.assertTrue((previous['after_ids'] == following['before_ids']).all())
            for row in rows:
                c, fixed = row['channel'], row['newly_fixed']
                self.assertTrue((row['before_ids'][..., c][fixed] == 10).all())
                self.assertTrue((row['after_ids'][..., c][fixed] == row['proposals'][fixed]).all())
            self.assertTrue(all(part['accepted'] == 7 for part in summarize_records(rows).values()))

    def test_trace_is_removed_on_sampler_error(self):
        with self.assertRaises(ValueError):
            trace_cascade(SimpleNamespace(), None, None, max_steps=0)
        self.assertIsNone(sys.gettrace())

    def test_cutoff_preserves_early_trace_of_full_noise(self):
        logits = torch.arange(2 * 5 * 3 * 8).reshape(2, 5, 3, 8).float().sin()
        model = SimpleNamespace(num_vq=8, block_size=6, paper_architecture=False,
                                forward_function=lambda *a, **kw: logits)
        traces = []
        for cutoff in (0, 2):
            torch.manual_seed(912)
            _, rows = trace_cascade(model, torch.zeros(2, 1), None,
                                    m_length=torch.tensor([8, 20]), max_steps=8,
                                    confidence_noise_scale=1., confidence_noise_until_step=cutoff)
            traces.append(rows)
        for first, second in zip(*traces):
            if first['step'] > 2:
                break
            for key in ('before_ids', 'after_ids', 'proposals', 'newly_fixed'):
                self.assertTrue((first[key] == second[key]).all())


if __name__ == '__main__':
    unittest.main()
