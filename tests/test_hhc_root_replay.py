import sys
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from tools.trace_hhc_sampling import trace_cascade
from tools.probe_hhc_root_replay import trace_root_replay, sample_with_control_root_replay


class RootReplayTest(unittest.TestCase):
    def test_two_pass_preserves_control_chain_across_batches(self):
        from models.two_person_transformer import Text2Motion_Transformer
        from tools.export_hhc_official_interx import validate_resume_attributes
        logits = torch.arange(2*5*3*8).reshape(2, 5, 3, 8).float().sin()
        model = SimpleNamespace(num_vq=8, block_size=6, paper_architecture=False,
                                forward_function=lambda *a, **kw: logits)
        args = (torch.zeros(2, 1), None)
        options = dict(m_length=torch.tensor([8, 20]), max_steps=8)
        torch.manual_seed(321)
        baseline = [Text2Motion_Transformer.sample_intra_cas(model, *args, **options) for _ in range(2)]
        state = torch.get_rng_state().clone()
        torch.manual_seed(321)
        for expected in baseline:
            candidate, control = sample_with_control_root_replay(
                model, *args, **options, type='sample_intra_cas',
                confidence_ranking='log_odds', confidence_noise_scale=1.)
            self.assertTrue(torch.equal(control, expected))
            self.assertTrue(torch.equal(candidate[..., 2], control[..., 2]))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for before, after in (({}, {'root_replay_control': True}), ({'root_replay_control': True}, {})):
            with self.assertRaises(ValueError):
                validate_resume_attributes(SimpleNamespace(attrs=before), after)

    def test_real_sampler_replay_and_fail_closed(self):
        logits = torch.arange(2*5*3*8).reshape(2, 5, 3, 8).float().sin()
        seen = []
        def forward(ids, *args, **kwargs):
            seen.append(ids.clone())
            return logits
        model = SimpleNamespace(num_vq=8, block_size=6, paper_architecture=False, forward_function=forward)
        args = (torch.zeros(2, 1), None)
        options = dict(m_length=torch.tensor([8, 20]), max_steps=8)
        torch.manual_seed(912)
        original, rows = trace_cascade(model, *args, **options)
        state = torch.get_rng_state().clone()
        original_contexts = [x[..., 2].clone() for x in seen]
        seen.clear()
        torch.manual_seed(912)
        result = trace_root_replay(model, rows, *args, **options)
        self.assertTrue(torch.equal(original, result))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        seen.clear()
        torch.manual_seed(913)
        noisy = trace_root_replay(model, rows, *args, **options,
                                  confidence_ranking='log_odds', confidence_noise_scale=1.)
        self.assertTrue(torch.equal(noisy[..., 2], original[..., 2]))
        self.assertTrue(all(torch.equal(x[..., 2], y) for x, y in zip(seen, original_contexts)))
        self.assertEqual(len(seen), len(original_contexts))
        self.assertFalse(torch.equal(noisy[..., :2], original[..., :2]))
        for invalid in ([], [r for r in rows if r['step'] != 1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                trace_root_replay(model, invalid, *args, **options)
            self.assertIsNone(sys.gettrace())
        malformed = [{**r, 'after_ids': r['after_ids'].astype(np.float32)} for r in rows]
        with self.assertRaises(ValueError):
            trace_root_replay(model, malformed, *args, **options)
        self.assertIsNone(sys.gettrace())


if __name__ == '__main__':
    unittest.main()
