from types import SimpleNamespace

import torch

from models.interhuman_evaluator_wrapper import load_official_metrics_module
from models.interhuman_transformer import InterHumanText2MotionTransformer
from models.vqvae_sep3_interhuman import VQVAE_SEP3_InterHuman
from utils.interhuman_mgt_evaluation import (
    decode_interhuman_tokens,
    evaluate_interhuman_generator,
    evaluate_interhuman_validation_multimodality,
    generate_interhuman_pair,
    interhuman_token_lengths,
)


class FakeDecodeVQ(torch.nn.Module):
    temporal_downsample = 4
    num_code = 16

    def forward(self, tokens, type='decode'):
        assert type == 'decode'
        return torch.ones(tokens.shape[0], tokens.shape[1] * 4, 524)


class FakeSampler(torch.nn.Module):
    block_size = 75

    def __init__(self):
        super().__init__()
        self.call = None

    def forward(self, clip_feature, word_embedding, **kwargs):
        self.call = kwargs
        result = torch.full(
            (clip_feature.shape[0], 74, 2), 17,
            dtype=torch.long, device=clip_feature.device,
        )
        for index, length in enumerate((kwargs['m_length'] // 4).tolist()):
            result[index, :length] = index + 1
        return result


def test_interhuman_token_lengths_match_released_floor_division():
    assert interhuman_token_lengths([14, 15, 16, 297, 299]).tolist() == [3, 3, 4, 74, 74]
    for invalid in ([13], [300], [[16]]):
        try:
            interhuman_token_lengths(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid official length accepted: {invalid}')


def test_decode_uses_only_valid_codes_and_zero_pads_remainder():
    tokens = torch.full((2, 74, 2), 17, dtype=torch.long)
    tokens[0, :3] = 1
    tokens[1, :4] = 2
    motion1, motion2 = decode_interhuman_tokens(FakeDecodeVQ(), tokens, [3, 4])
    assert motion1.shape == motion2.shape == (2, 300, 262)
    assert torch.all(motion1[0, :12] == 1) and torch.count_nonzero(motion1[0, 12:]) == 0
    assert torch.all(motion2[1, :16] == 1) and torch.count_nonzero(motion2[1, 16:]) == 0


def test_generation_bridges_floor_lengths_without_changing_sampler():
    sampler = FakeSampler()
    motion1, motion2 = generate_interhuman_pair(
        sampler,
        FakeDecodeVQ(),
        torch.zeros(2, 8),
        torch.zeros(2, 5, 8),
        torch.tensor([15, 299]),
        sampling_steps=50,
        temperature=0.5,
        cfg_scale=3.5,
    )
    assert sampler.call['m_length'].tolist() == [12, 296]
    assert sampler.call['max_steps'] == 50
    assert sampler.call['temperature'] == 0.5
    assert sampler.call['CFG'] == 3.5
    assert sampler.call['type'] == 'sample_intra_cas'
    assert motion1.shape == motion2.shape == (2, 300, 262)
    assert torch.count_nonzero(motion1[0, 12:]) == 0


def test_real_small_vq_and_transformer_use_generation_helper():
    args = SimpleNamespace(mu=0.99, interhuman_latent_dims='24,8')
    vq = VQVAE_SEP3_InterHuman(
        args, nb_code=16, code_dim=32, output_emb_width=32,
        down_t=2, stride_t=2, width=32, depth=1,
        dilation_growth_rate=2,
    ).eval()
    transformer = InterHumanText2MotionTransformer(
        vqvae=vq, num_vq=16, embed_dim=32, clip_dim=32,
        block_size=75, num_layers=1, num_local_layer=0, n_head=4,
        drop_out_rate=0.0, fc_rate=2, contact=False, shape=False,
        paper_architecture=True, conditioning_contract='paper_text_cfg',
    ).eval()
    motion1, motion2 = generate_interhuman_pair(
        transformer, vq, torch.randn(1, 32), torch.randn(1, 6, 32),
        torch.tensor([297]), sampling_steps=1, rand_pos=False,
    )
    assert motion1.shape == motion2.shape == (1, 300, 262)
    assert torch.isfinite(motion1).all() and torch.isfinite(motion2).all()
    assert torch.count_nonzero(motion1[:, 296:]) == 0
    assert torch.count_nonzero(motion2[:, 296:]) == 0


def test_in_memory_validation_uses_complete_official_batches_and_restores_rng():
    class FakeEvaluator:
        @staticmethod
        def _embeddings(names):
            indices = torch.tensor(
                [int(name.split('_')[-1]) for name in names], dtype=torch.float64
            )
            basis = torch.arange(16, dtype=torch.float64).view(1, -1)
            return torch.nn.functional.normalize(
                torch.cos(indices.view(-1, 1) * 0.13 + basis * 0.17), dim=-1
            )

        def get_co_embeddings(self, batch):
            embeddings = self._embeddings(batch[0])
            offset = torch.linspace(0.0, 1e-4, embeddings.shape[1], dtype=embeddings.dtype)
            prediction = torch.nn.functional.normalize(embeddings + offset, dim=-1)
            return embeddings, prediction

        def get_motion_embeddings(self, batch):
            return self._embeddings(batch[0])

    def loader():
        for batch_index in range(5):
            start = batch_index * 96
            names = [f'id_{index}' for index in range(start, start + 96)]
            motions = torch.zeros(96, 300, 262)
            yield names, ['text'] * 96, motions, motions.clone(), torch.full((96,), 296)

    generated_calls = []

    def generate(texts, lengths, sample_seeds):
        generated_calls.append((len(texts), lengths.clone(), list(sample_seeds)))
        return torch.zeros(96, 300, 262), torch.zeros(96, 300, 262)

    seed_map = {f'id_{index}': index + 7000 for index in range(480)}
    torch.manual_seed(987)
    state = torch.random.get_rng_state().clone()
    result = evaluate_interhuman_generator(
        generate, loader(), FakeEvaluator(), load_official_metrics_module(),
        random_seed=17, sample_seeds_by_name=seed_map,
    )
    assert result['samples'] == 480
    assert abs(result['fid']) < 1e-5
    assert len(generated_calls) == 5
    assert generated_calls[0][2] == list(range(7000, 7096))
    assert generated_calls[-1][2] == list(range(7384, 7480))
    assert torch.equal(torch.random.get_rng_state(), state)


def test_validation_multimodality_is_val_only_seeded_and_restores_rng():
    class FakeEvaluator:
        def get_motion_embeddings(self, batch):
            first, second = batch[2], batch[3]
            return torch.cat((first[:, 0, :8], second[:, 0, :8]), dim=-1)

    payload = {
        'repeats': 11, 'pairs': 10,
        'requests': [
            {
                'identifier': f'val_{index}', 'caption': f'text {index}',
                'processed_length': 100,
            }
            for index in range(100)
        ],
    }

    def generate(texts, lengths, repeat):
        assert len(texts) == 100 and lengths.shape == (100,) and 0 <= repeat < 11
        first = torch.zeros(100, 300, 262)
        second = torch.zeros_like(first)
        first[:, 0, :8] = torch.randn(100, 8)
        second[:, 0, :8] = torch.randn(100, 8)
        return first, second

    torch.manual_seed(321)
    state = torch.random.get_rng_state().clone()
    result = evaluate_interhuman_validation_multimodality(
        generate, payload, FakeEvaluator(), load_official_metrics_module(),
        random_seed=99,
    )
    assert result['multimodality'] > 0
    assert result['multimodality_prompts'] == 100
    assert result['multimodality_repeats'] == 11
    assert torch.equal(torch.random.get_rng_state(), state)


def test_per_sample_seeds_make_interhuman_sampling_batch_invariant():
    args = SimpleNamespace(mu=0.99, interhuman_latent_dims='24,8')
    vq = VQVAE_SEP3_InterHuman(
        args, nb_code=16, code_dim=32, output_emb_width=32,
        down_t=2, stride_t=2, width=32, depth=1,
        dilation_growth_rate=2,
    ).eval()
    transformer = InterHumanText2MotionTransformer(
        vqvae=vq, num_vq=16, embed_dim=32, clip_dim=32,
        block_size=75, num_layers=1, num_local_layer=0, n_head=4,
        drop_out_rate=0.0, fc_rate=2, contact=False, shape=False,
        paper_architecture=True, conditioning_contract='paper_text_cfg',
    ).eval()
    clip_features = torch.randn(2, 32)
    words = torch.randn(2, 6, 32)
    lengths = torch.tensor([100, 297])
    seeds = [2**63 + 7001, 2**64 - 2]
    together = generate_interhuman_pair(
        transformer, vq, clip_features, words, lengths,
        sampling_steps=2, sample_seeds=seeds,
    )
    separate_first, separate_second = [], []
    for index in range(2):
        first, second = generate_interhuman_pair(
            transformer, vq, clip_features[index:index + 1],
            words[index:index + 1], lengths[index:index + 1],
            sampling_steps=2, sample_seeds=seeds[index:index + 1],
        )
        separate_first.append(first)
        separate_second.append(second)
    assert torch.equal(together[0], torch.cat(separate_first))
    assert torch.equal(together[1], torch.cat(separate_second))


def test_seeded_generation_replays_after_global_rng_changes():
    args = SimpleNamespace(mu=0.99, interhuman_latent_dims='24,8')
    vq = VQVAE_SEP3_InterHuman(
        args, nb_code=16, code_dim=32, output_emb_width=32,
        down_t=2, stride_t=2, width=32, depth=1,
        dilation_growth_rate=2,
    ).eval()
    base = InterHumanText2MotionTransformer(
        vqvae=vq, num_vq=16, embed_dim=32, clip_dim=32,
        block_size=75, num_layers=1, num_local_layer=0, n_head=4,
        drop_out_rate=0.0, fc_rate=2, contact=False, shape=False,
        paper_architecture=True, conditioning_contract='paper_text_cfg',
    ).eval()
    extended = InterHumanText2MotionTransformer(
        vqvae=vq, num_vq=16, embed_dim=32, clip_dim=32,
        block_size=75, num_layers=1, num_local_layer=0, n_head=4,
        drop_out_rate=0.0, fc_rate=2, contact=False, shape=False,
        paper_architecture=True, conditioning_contract='paper_text_cfg',
    ).eval()
    extended.load_state_dict(base.state_dict())
    clip_features = torch.randn(1, 32)
    words = torch.randn(1, 6, 32)
    lengths = torch.tensor([297])
    seed = 7001
    torch.manual_seed(seed)
    expected = generate_interhuman_pair(
        base, vq, clip_features, words, lengths, sampling_steps=2, sample_seeds=[seed],
    )
    torch.manual_seed(seed + 99)
    actual = generate_interhuman_pair(
        extended, vq, clip_features, words, lengths, sampling_steps=2,
        sample_seeds=[seed],
    )
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
