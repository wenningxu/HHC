import unittest

from tools.score_hhc_validation_mm import validate_generation_metadata


BASE = {
    'sampling_steps': 50, 'temperature': 0.5, 'cfg_scale': 3.5,
    'sampler': 'sample_intra_cas', 'mgt_shape_conditioning': False,
    'contact_guidance': False, 'contact_timing_source': 'none',
    'vq_contract': 'precleanup_raw12288', 'mgt_contract': 'historical_mmm',
    'inference_conditioning_contract': 'paper_text_cfg', 'seed': 600123,
    'temperature_end': 'constant', 'temperature_schedule': 'constant',
}


class RootReplayContractTest(unittest.TestCase):
    def test_default_and_both_directions_are_checked(self):
        validate_generation_metadata(BASE)
        validate_generation_metadata(dict(BASE, root_replay_control=True,
                                          root_replay_control_ranking='probability',
                                          root_replay_control_noise_scale=0.0,
                                          root_replay_rng_policy='preserve_control_batch_chain',
                                          root_replay_passes_per_batch=2), root_replay_control=True)
        with self.assertRaises(ValueError):
            validate_generation_metadata(dict(BASE, root_replay_control=True), root_replay_control=True)
        with self.assertRaises(ValueError):
            validate_generation_metadata(dict(BASE, root_replay_control=True), root_replay_control=False)
        with self.assertRaises(ValueError):
            validate_generation_metadata(BASE, root_replay_control=True)


    def test_rejects_bad_policies(self):
        for key, value in (
            ('root_replay_control_ranking', 'log_odds'),
            ('root_replay_control_noise_scale', 1.0),
            ('root_replay_rng_policy', 'other'),
            ('root_replay_passes_per_batch', 1),
        ):
            metadata = dict(BASE, root_replay_control=True,
                            root_replay_control_ranking='probability',
                            root_replay_control_noise_scale=0.0,
                            root_replay_rng_policy='preserve_control_batch_chain',
                            root_replay_passes_per_batch=2)
            metadata[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_generation_metadata(metadata, root_replay_control=True)
