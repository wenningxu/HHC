from types import SimpleNamespace

import torch

from models.model_tools import build_model
from models.interhuman_transformer import InterHumanText2MotionTransformer
from models.two_person_transformer import CrossCondTransBase
from models.two_person_transformer import Text2Motion_Transformer
from models.vqvae_sep3_interhuman import (
    VQVAE_SEP3_InterHuman,
    _resolve_latent_dims,
)


def _args(**overrides):
    values = dict(
        nb_code=16,
        code_dim=16,
        output_emb_width=16,
        down_t=1,
        stride_t=2,
        width=16,
        depth=1,
        dilation_growth_rate=2,
        vq_act='relu',
        vq_norm=None,
        mu=0.99,
        embed_dim_gpt=16,
        clip_dim=16,
        block_size=9,
        num_layers=2,
        num_local_layer=0,
        n_head_gpt=4,
        drop_out_rate=0.0,
        ff_rate=2,
        contact_conditioning=False,
        shape=False,
        paper_architecture=False,
        conditioning_contract='paper_text_cfg',
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_latent_partition_is_exact():
    assert _resolve_latent_dims(256) == (192, 64)
    assert _resolve_latent_dims(16) == (12, 4)


def test_invalid_latent_partition_fails_closed():
    try:
        _resolve_latent_dims(16, '12,3')
    except ValueError as exc:
        assert 'not code_dim=16' in str(exc)
    else:
        raise AssertionError('invalid latent partition was accepted')


def test_interhuman_vq_full_encode_decode_contract():
    torch.manual_seed(123)
    model = VQVAE_SEP3_InterHuman(_args(), nb_code=16, code_dim=16,
                                  down_t=1, stride_t=2, width=16, depth=1)
    motion = torch.randn(2, 16, 524)
    reconstructed, commitment, perplexity = model(motion, type='full')
    assert reconstructed.shape == motion.shape
    assert commitment.ndim == 0
    assert perplexity.shape == (2,)
    assert model.channel_masks.shape == (2, 524)
    assert torch.all(model.channel_masks.sum(dim=0) == 1)
    assert len(model.encoders) == len(model.quantizers) == 2
    loss = (reconstructed - motion).square().mean() + 0.02 * commitment
    loss.backward()
    for component in (*model.encoders, model.decoder):
        gradients = [p.grad for p in component.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)

    model.eval()
    tokens = model(motion, type='encode')
    assert tokens.shape == (2, 8, 2)
    decoded = model(tokens, beta=torch.randn(2, 20),
                    gender=torch.zeros(2, 2, dtype=torch.long), type='decode')
    assert decoded.shape == motion.shape
    assert torch.isfinite(decoded).all()


def test_model_builder_has_distinct_interhuman_type():
    model = build_model(_args(), 'interhuman_vq')
    assert isinstance(model, VQVAE_SEP3_InterHuman)
    assert model.feature_dim == 524


def test_two_part_vq_real_motion_geometric_loss_backward():
    from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
    from utils.interhuman_vq_loss import InterHumanPairVQLoss

    torch.manual_seed(124)
    first, second = OfficialInterHumanAdapter().process_pair('3292', normalize=True)
    motion = torch.cat((torch.from_numpy(first[:32]),
                        torch.from_numpy(second[:32])), dim=-1).unsqueeze(0)
    model = build_model(_args(), 'interhuman_vq')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    before = [next(encoder.parameters()).detach().clone() for encoder in model.encoders]
    prediction, commitment, _ = model(motion)
    terms = InterHumanPairVQLoss()(motion, prediction, commitment)
    assert all(torch.isfinite(value).all() for value in terms.values())
    terms['total'].backward()
    for component in (*model.encoders, model.decoder):
        gradients = [p.grad for p in component.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
    optimizer.step()
    for previous, encoder in zip(before, model.encoders):
        assert not torch.equal(previous, next(encoder.parameters()))
    model.eval()
    with torch.no_grad():
        tokens = model(motion, type='encode')
        assert tokens.shape == (1, 16, 2)
        decoded = model(tokens, type='decode')
        assert decoded.shape == motion.shape and torch.isfinite(decoded).all()


def test_transformer_builders_keep_interx_base_and_interhuman_seeded_subclass():
    class MinimalVQ(torch.nn.Module):
        code_dim = 16

    args = _args(interhuman_latent_dims='12,4')
    vq = MinimalVQ()
    interx = build_model(args, 'interx_transformer', vqvae=vq)
    interhuman = build_model(args, 'interhuman_transformer', vqvae=build_model(args, 'interhuman_vq'))
    assert type(interx) is Text2Motion_Transformer
    assert type(interhuman) is InterHumanText2MotionTransformer


def test_interhuman_transformer_binds_exact_vq_quantizers_and_widths():
    args = _args(interhuman_latent_dims='11,5')
    vq = build_model(args, 'interhuman_vq')
    transformer = build_model(args, 'interhuman_transformer', vqvae=vq)
    base = transformer.trans_base
    assert tuple(layer.in_features for layer in base.emb_proj) == (11, 5)
    assert base.part_emb.num_embeddings == 2
    assert base.vqvae.quantizer_upper is vq.quantizers[0]
    assert base.vqvae.quantizer_root is vq.quantizers[1]


def test_one_token_pair_encodes_and_predicts_one_paired_motion():
    args = _args(num_local_layer=1)
    vq = build_model(args, 'interhuman_vq')
    paired_motion = torch.randn(2, 16, 524)
    vq.train()
    vq(paired_motion, type='full')  # initialize EMA codebooks
    vq.eval()
    tokens = vq(paired_motion, type='encode')
    transformer = build_model(args, 'interhuman_transformer', vqvae=vq)
    logits = transformer(
        tokens,
        torch.randn(2, 16),
        src_mask=torch.ones(2, tokens.shape[1], dtype=torch.bool),
        word_emb=torch.randn(2, 5, 16),
    )
    assert tokens.shape == (2, 8, 2)
    assert logits.shape == (2, 8, 2, 16)
    decoded_pair = vq(tokens, type='decode')
    assert decoded_pair.shape == paired_motion.shape


def test_interhuman_transformer_rejects_shape_or_contact_conditioning():
    for forbidden in ({'shape': True}, {'contact_conditioning': True}):
        args = _args(**forbidden)
        vq = build_model(args, 'interhuman_vq')
        try:
            build_model(args, 'interhuman_transformer', vqvae=vq)
        except ValueError as exc:
            assert 'InterHuman HHC v1' in str(exc)
        else:
            raise AssertionError(f'forbidden InterHuman condition accepted: {forbidden}')


def test_legacy_interx_vq_width_fallback_is_unchanged():
    class LegacyVQ(torch.nn.Module):
        code_dim = 256

    base = CrossCondTransBase(
        LegacyVQ(), num_vq=4096, embed_dim=16, clip_dim=16,
        block_size=9, num_layers=2, num_local_layer=0, n_head=4,
        drop_out_rate=0.0, fc_rate=2, contact=False, shape=False,
        paper_architecture=False, conditioning_contract='paper_text_cfg',
    )
    assert (base.upper_dim, base.lower_dim, base.root_dim) == (96, 96, 64)


def test_two_part_mgt_sampling_backward_and_frozen_vq_integrity():
    for paper in (False, True):
        torch.manual_seed(81)
        args = _args(down_t=2, block_size=6, paper_architecture=paper)
        vq = build_model(args, 'interhuman_vq').eval()
        vq.requires_grad_(False)
        saved = {name: value.clone() for name, value in vq.state_dict().items()}
        transformer = build_model(args, 'interhuman_transformer', vqvae=vq)
        for name, value in vq.state_dict().items():
            assert torch.equal(value, saved[name]), name
        tokens = torch.full((2, 5, 2), 18, dtype=torch.long)
        features, words = torch.randn(2, 16), torch.randn(2, 4, 16)
        logits = transformer(tokens, features, word_emb=words,
                             src_mask=torch.ones(2, 5, dtype=torch.bool))
        assert logits.shape == (2, 5, 2, 16)
        torch.nn.functional.cross_entropy(logits.reshape(-1, 16),
                                         torch.zeros(20, dtype=torch.long)).backward()
        assert all(p.grad is None for p in vq.parameters())
        assert transformer.trans_base.part_emb.weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in transformer.parameters() if p.grad is not None)
        transformer.eval()
        with torch.no_grad():
            kwargs = dict(m_length=torch.tensor([16, 12]), max_steps=2,
                          sample_seeds=[17, 19], CFG=2.0)
            together = transformer.sample_intra_cas(features, words, **kwargs)
            assert together.shape == (2, 5, 2)
            for i, length in enumerate((4, 3)):
                single = transformer.sample_intra_cas(
                    features[i:i+1], words[i:i+1], m_length=kwargs['m_length'][i:i+1],
                    max_steps=2, sample_seeds=kwargs['sample_seeds'][i:i+1], CFG=2.0)
                assert torch.equal(single[0], together[i])
                assert torch.all((together[i, :length] >= 0) & (together[i, :length] < 16))
                if paper:
                    assert torch.all(together[i, length:] == 17)
                else:
                    assert torch.all(together[i, length] == 16)
                    assert torch.all(together[i, length+1:] == 17)
            unseeded = transformer.sample_intra_cas(
                features, words, m_length=kwargs['m_length'], max_steps=2)
            assert unseeded.shape == (2, 5, 2)


def test_two_part_ema_checkpoint_resumes_identical_next_forward():
    from utils.quantizer_state import quantizer_training_state, restore_quantizer_training_state

    torch.manual_seed(66)
    source = build_model(_args(), 'interhuman_vq').train()
    motion = torch.randn(2, 16, 524)
    source(motion)
    weights = {key: value.clone() for key, value in source.state_dict().items()}
    ema = quantizer_training_state(source)
    assert set(ema) == {'quantizer_upper', 'quantizer_root'}
    rng = torch.get_rng_state().clone()
    expected = source(motion)
    expected_ema = quantizer_training_state(source)
    restored = build_model(_args(), 'interhuman_vq').train()
    restored.load_state_dict(weights, strict=True)
    assert not restore_quantizer_training_state(restored, ema)
    torch.set_rng_state(rng)
    actual = restored(motion)
    for left, right in zip(expected, actual):
        assert torch.equal(left, right)
    for channel, values in quantizer_training_state(restored).items():
        for key, value in values.items():
            expected_value = expected_ema[channel][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, expected_value)
            else:
                assert value == expected_value
