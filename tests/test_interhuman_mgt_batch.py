import torch
from types import SimpleNamespace

from models.vqvae_sep3_interhuman import VQVAE_SEP3_InterHuman
from models.interhuman_transformer import InterHumanText2MotionTransformer
from utils.interhuman_mgt_batch import encode_interhuman_mgt_batch


class FakeVQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))
        self.seen_training_modes = []

    def forward(self, motion, type='full'):
        assert type == 'encode'
        assert motion.shape[-2:] == (300, 524)
        self.seen_training_modes.append(self.training)
        return torch.zeros(motion.shape[0], 75, 2, dtype=torch.long, device=motion.device)


def test_interhuman_mgt_batch_encodes_pair_online_with_floor_lengths():
    vq = FakeVQ().train()
    batch = (
        ['a', 'b'], ['first', 'second'],
        torch.randn(2, 300, 262), torch.randn(2, 300, 262),
        torch.tensor([299, 14]),
    )
    captions, tokens, lengths, betas, gender, contact = encode_interhuman_mgt_batch(
        vq, batch, 'cpu'
    )
    assert captions == ['first', 'second']
    assert tokens.shape == (2, 75, 2)
    assert lengths.tolist() == [74, 3]
    assert betas.shape == (2, 20) and torch.count_nonzero(betas) == 0
    assert gender.shape == (2, 2) and torch.count_nonzero(gender) == 0
    assert contact.shape == (2, 75) and torch.count_nonzero(contact) == 0
    assert vq.seen_training_modes == [False]
    assert vq.training is True


def test_interhuman_mgt_batch_rejects_ceil_or_out_of_range_contracts():
    vq = FakeVQ().eval()
    bad = (
        ['a'], ['first'], torch.zeros(1, 300, 262),
        torch.zeros(1, 300, 262), torch.tensor([300]),
    )
    try:
        encode_interhuman_mgt_batch(vq, bad, 'cpu')
    except ValueError as exc:
        assert 'processed lengths' in str(exc)
    else:
        raise AssertionError('300 processed frames were accepted despite 299 maximum')


def test_real_two_channel_vq_integrates_with_mgt_batch_bridge():
    args = SimpleNamespace(mu=0.99, interhuman_latent_dims='24,8')
    vq = VQVAE_SEP3_InterHuman(
        args, nb_code=16, code_dim=32, output_emb_width=32,
        down_t=2, stride_t=2, width=32, depth=1,
        dilation_growth_rate=2,
    ).train()
    batch = (
        ['a'], ['interaction'], torch.randn(1, 300, 262),
        torch.randn(1, 300, 262), torch.tensor([297]),
    )
    _, tokens, lengths, _, _, _ = encode_interhuman_mgt_batch(vq, batch, 'cpu')
    assert tokens.shape == (1, 75, 2)
    assert lengths.tolist() == [74]
    assert tokens.min() >= 0 and tokens.max() < 16
    assert vq.training is True
    # Online MGT encoding must not initialize/update EMA codebooks.
    assert all(not quantizer.init for quantizer in vq.quantizers)


def test_real_vq_tokens_run_through_interhuman_length_transformer():
    """The planned 300-frame/75-token contract must work end to end.

    This catches a class of silent Inter-X assumptions (short block sizes or
    fixed latent partitions) before the InterHuman GPU pilot is allowed to
    start.
    """
    args = SimpleNamespace(mu=0.99, interhuman_latent_dims='24,8')
    vq = VQVAE_SEP3_InterHuman(
        args, nb_code=16, code_dim=32, output_emb_width=32,
        down_t=2, stride_t=2, width=32, depth=1,
        dilation_growth_rate=2,
    ).eval()
    batch = (
        ['a'], ['interaction'], torch.randn(1, 300, 262),
        torch.randn(1, 300, 262), torch.tensor([297]),
    )
    _, tokens, lengths, _, _, _ = encode_interhuman_mgt_batch(vq, batch, 'cpu')

    transformer = InterHumanText2MotionTransformer(
        vqvae=vq,
        num_vq=16,
        embed_dim=32,
        clip_dim=32,
        block_size=75,
        num_layers=1,
        num_local_layer=0,
        n_head=4,
        drop_out_rate=0.0,
        fc_rate=2,
        contact=False,
        shape=False,
        paper_architecture=True,
        conditioning_contract='paper_text_cfg',
    ).eval()
    src_mask = torch.arange(75).unsqueeze(0) < lengths.unsqueeze(1)
    with torch.no_grad():
        logits = transformer(
            tokens,
            torch.randn(1, 32),
            src_mask=src_mask,
            word_emb=torch.randn(1, 6, 32),
        )
    assert logits.shape == (1, 75, 2, 16)
    assert torch.isfinite(logits).all()

    # The released InterMask protocol floors the processed frame length before
    # generation. Passing that token support back as an exact multiple of four
    # preserves HHC's existing ceil-based sampler without changing Inter-X.
    with torch.no_grad():
        sampled = transformer(
            torch.randn(1, 32),
            torch.randn(1, 6, 32),
            m_length=lengths * 4,
            max_steps=1,
            rand_pos=False,
            CFG=-1,
            type='sample_intra_cas',
        )
    assert sampled.shape == (1, 74, 2)
    assert sampled.min() >= 0 and sampled.max() < 16
