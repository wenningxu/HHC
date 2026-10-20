"""Pre-change CPU golden output and RNG state for the common cascade sampler."""
import hashlib
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from models.two_person_transformer import Text2Motion_Transformer


def sample_reference(temperature=0.5, **kwargs):
    logits = torch.arange(2 * 5 * 3 * 8).reshape(2, 5, 3, 8).float().sin()
    model = SimpleNamespace(num_vq=8, block_size=6, paper_architecture=False,
                            forward_function=lambda *args, **kw: logits)
    torch.manual_seed(912)
    ids = Text2Motion_Transformer.sample_intra_cas(
        model, torch.zeros(2, 1), None, m_length=torch.tensor([8, 20]),
        max_steps=8, temperature=temperature, **kwargs,
    )
    return ids, hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


class ReferenceTest(unittest.TestCase):
    def test_new_policies_preserve_padding_and_are_reproducible(self):
        for ranking in ('probability', 'log_odds', 'max_probability'):
            for schedule in ('interleaved', 'partwise'):
                for noise in (0., 1.):
                    options = dict(confidence_ranking=ranking, channel_schedule=schedule,
                                   confidence_noise_scale=noise)
                    ids, rng = sample_reference(**options)
                    again, again_rng = sample_reference(**options)
                    self.assertTrue(torch.equal(ids, again))
                    self.assertEqual(rng, again_rng)
                    self.assertTrue((ids[0, 2] == 8).all())
                    self.assertTrue((ids[0, 3:] == 9).all())
                    self.assertTrue((ids[0, :2] < 8).all())
                    self.assertTrue((ids[1] < 8).all())
        for options in ({'confidence_ranking': 'bad'}, {'channel_schedule': 'bad'},
                        {'max_steps': 0}):
            with self.assertRaises(ValueError):
                model = SimpleNamespace()
                Text2Motion_Transformer.sample_intra_cas(model, None, None, **options)

    def test_partwise_finishes_each_part_before_starting_next(self):
        traces = []
        def forward(ids, *args, **kwargs):
            traces.append(ids.clone())
            return torch.zeros(1, 5, 3, 8)
        model = SimpleNamespace(num_vq=8, block_size=6, paper_architecture=False,
                                forward_function=forward)
        Text2Motion_Transformer.sample_intra_cas(
            model, torch.zeros(1, 1), None, m_length=torch.tensor([20]),
            max_steps=4, channel_schedule='partwise', CFG=3.5,
        )
        self.assertEqual(len(traces), 24)  # 4 steps * 3 parts * 2 CFG forwards
        for block, channel in enumerate((2, 0, 1)):
            for ids in traces[block * 8:(block + 1) * 8]:
                self.assertTrue((ids[:, :, channel] == 10).any())
                for finished in (2, 0, 1)[:block]:
                    self.assertTrue((ids[:, :, finished] < 8).all())
                for future in (2, 0, 1)[block + 1:]:
                    self.assertTrue((ids[:, :, future] == 10).all())

    def test_zero_proposal_temperature_with_noisy_order(self):
        ids, noise_rng = sample_reference(temperature=0, confidence_noise_scale=1)
        _, control_rng = sample_reference(temperature=0)
        expected = torch.arange(2 * 5 * 3 * 8).reshape(2, 5, 3, 8).float().sin().argmax(-1)
        self.assertTrue(torch.equal(ids[0, :2], expected[0, :2]))
        self.assertTrue(torch.equal(ids[1], expected[1]))
        self.assertTrue((ids[0, 2] == 8).all())
        self.assertTrue((ids[0, 3:] == 9).all())
        self.assertNotEqual(noise_rng, control_rng)

    def test_noise_is_reproducible_and_preserves_end_padding(self):
        for scope in ('all', 'root'):
            ids, _ = sample_reference(confidence_noise_scale=1.0, confidence_noise_scope=scope)
            again, _ = sample_reference(confidence_noise_scale=1.0, confidence_noise_scope=scope)
            self.assertTrue(torch.equal(ids, again))
            self.assertTrue((ids[0, 2] == 8).all())
            self.assertTrue((ids[0, 3:] == 9).all())
            self.assertTrue((ids[1] < 8).all())
        reference, reference_rng = sample_reference()
        zero, zero_rng = sample_reference(confidence_noise_scale=0.0)
        self.assertTrue(torch.equal(reference, zero))
        self.assertEqual(reference_rng, zero_rng)
        with self.assertRaises(ValueError):
            sample_reference(confidence_noise_scale=-1)

    def test_resume_cannot_mix_confidence_policies(self):
        from tools.export_hhc_official_interx import validate_resume_attributes
        validate_resume_attributes(SimpleNamespace(attrs={}), {})
        with self.assertRaisesRegex(ValueError, 'confidence_noise_scale'):
            validate_resume_attributes(SimpleNamespace(attrs={'confidence_noise_scale': 1.}), {})
        with self.assertRaisesRegex(ValueError, 'confidence_noise_scale'):
            validate_resume_attributes(SimpleNamespace(attrs={}), {'confidence_noise_scale': 1.})
        for key, value in (('confidence_ranking', 'log_odds'), ('confidence_ranking', 'max_probability'),
                           ('confidence_noise_until_step', 2), ('channel_schedule', 'partwise')):
            for old, new in (({}, {key: value}), ({key: value}, {})):
                with self.assertRaisesRegex(ValueError, key):
                    validate_resume_attributes(SimpleNamespace(attrs=old), new)

    def test_noise_cutoff_preserves_endpoints_and_stops_noise_draws(self):
        baseline, baseline_rng = sample_reference(confidence_noise_scale=1.)
        endpoint, endpoint_rng = sample_reference(confidence_noise_scale=1., confidence_noise_until_step=8)
        self.assertTrue(torch.equal(baseline, endpoint))
        self.assertEqual(baseline_rng, endpoint_rng)
        plain, plain_rng = sample_reference()
        zero, zero_rng = sample_reference(confidence_noise_until_step=2)
        self.assertTrue(torch.equal(plain, zero))
        self.assertEqual(plain_rng, zero_rng)
        calls = []
        original = torch.Tensor.exponential_
        def track(tensor, *args, **kwargs):
            calls.append(tensor.shape)
            return original(tensor, *args, **kwargs)
        # Remove proposal RNG so only ranking-noise draws are counted.
        with patch('models.two_person_transformer.gumbel_sample', side_effect=lambda logits, **kw: logits.argmax(-1)):
            with patch.object(torch.Tensor, 'exponential_', track):
                sample_reference(confidence_noise_scale=1., confidence_noise_until_step=2)
        self.assertEqual(len(calls), 6)  # two steps, three channels; no later noise draws
        for bad in (-1, 9, 1.5, True):
            with self.assertRaises(ValueError):
                sample_reference(confidence_noise_until_step=bad)

    def test_max_probability_retains_candidate_independent_position(self):
        logits = torch.tensor([[[[.9, .1]] * 3, [[.6, .4]] * 3]]).log()
        proposal = torch.tensor([[1, 0]])  # first position draws its minority candidate
        committed = {}
        for ranking in ('probability', 'max_probability'):
            traces = []
            def forward(ids, *args, **kwargs):
                traces.append(ids.clone())
                return logits
            model = SimpleNamespace(num_vq=2, block_size=3, paper_architecture=True,
                                    forward_function=forward)
            with patch('models.two_person_transformer.gumbel_sample', return_value=proposal):
                Text2Motion_Transformer.sample_intra_cas(
                    model, torch.zeros(1, 1), None, m_length=torch.tensor([8]),
                    max_steps=2, confidence_ranking=ranking)
            committed[ranking] = traces[1][0, :, 2].tolist()
        self.assertEqual(committed['probability'], [4, 0])
        self.assertEqual(committed['max_probability'], [1, 4])
        self.assertEqual(sample_reference()[1], sample_reference(confidence_ranking='max_probability')[1])

    def test_default_output_and_rng(self):
        ids, rng_hash = sample_reference()
        self.assertEqual(ids.tolist(), [
            [[3, 6, 5], [3, 2, 5], [8, 8, 8], [9, 9, 9], [9, 9, 9]],
            [[1, 6, 4], [2, 7, 5], [3, 1, 0], [4, 2, 7], [0, 3, 1]],
        ])
        self.assertEqual(rng_hash, '615803ddf04a6281f54d63b4142ead5ade45703fc6d515bf468c2b9a8b9151ab')


if __name__ == '__main__':
    unittest.main()
