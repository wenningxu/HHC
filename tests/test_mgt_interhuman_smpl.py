"""CPU regression checks for native-SMPL InterHuman MGT."""

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.mgt_interhuman_smpl import SMPLMGT
from utils.interhuman_smpl_mgt import TokenData, load_vq


def _state_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(memoryview(tensor.numpy()).cast('B'))
    return digest.hexdigest()


def _token_batch(batch=2, time=7, vocab=12288):
    return torch.randint(vocab, (batch, time, 2))


def test_real_vq_freeze_forward_cfg_shape_sampler_and_token_data():
    torch.set_num_threads(4)
    vq, _, _ = load_vq('cpu')
    vq_digest = _state_sha256(vq)
    model = SMPLMGT(
        vq, embed_dim=32, layers=3, heads=4, dropout=0.0, block_size=8,
    )
    assert _state_sha256(vq) == vq_digest
    assert not vq.training
    assert all(not parameter.requires_grad for parameter in vq.parameters())

    batch, time, vocab = 2, 7, vq.num_code
    tokens = _token_batch(batch, time, vocab)
    clip = torch.randn(batch, 512)
    words = torch.randn(batch, 5, 512)
    lengths = torch.tensor([7, 5])
    mask = torch.arange(time)[None] < lengths[:, None]
    betas = torch.randn(batch, 20)
    gender = torch.ones(batch, 2, dtype=torch.long)

    model.train()
    vq.eval()  # The production trainer re-freezes the shared child after net.train().
    assert not vq.training
    logits = model.forward_function(
        tokens, clip, mask, word_emb=words, betas=betas, gender=gender,
    )
    assert logits.shape == (batch, time, 2, vocab)
    loss = F.cross_entropy(logits.reshape(-1, vocab), tokens.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for name, parameter in model.named_parameters()
        if not name.startswith('trans_base.vqvae.')
    )
    assert all(parameter.grad is None for parameter in vq.parameters())
    model.zero_grad(set_to_none=True)

    model.eval()
    model.conditioning_contract = 'paper_text_cfg'
    model.trans_base.conditioning_contract = 'paper_text_cfg'
    beta_pair = betas[:1].expand(2, -1).clone()
    beta_pair[1, 0] += 1.0
    with torch.no_grad():
        masked_a = model.forward_function(
            tokens[:1], clip[:1], mask[:1], word_emb=words[:1],
            betas=beta_pair[:1], gender=gender[:1], force_mask=True,
        )
        masked_text_changed = model.forward_function(
            tokens[:1], clip[:1] + 3, mask[:1], word_emb=words[:1] - 2,
            betas=beta_pair[:1], gender=gender[:1], force_mask=True,
        )
        changed_shape = model.forward_function(
            tokens[:1], clip[:1], mask[:1], word_emb=words[:1],
            betas=beta_pair[1:], gender=gender[:1], force_mask=True,
        )
    torch.testing.assert_close(masked_a, masked_text_changed, rtol=0, atol=0)
    assert not torch.equal(masked_a, changed_shape)

    sample_lengths = torch.tensor([4, 12])
    seeds = [1901, 1902]
    with torch.no_grad():
        first = model.sample_intra_cas(
            clip, words, betas, gender, sample_lengths, seeds,
            max_steps=3, temperature=0.5, CFG=3.5,
        )
        second = model.sample_intra_cas(
            clip, words, betas, gender, sample_lengths, seeds,
            max_steps=3, temperature=0.5, CFG=3.5,
        )
    assert first.shape == (2, 7, 2)
    assert torch.equal(first, second)
    for row, valid_length in zip(first, (1, 3)):
        assert torch.all((row[:valid_length] >= 0) & (row[:valid_length] < vocab))
        assert torch.all(row[valid_length] == vocab)  # END
        assert torch.all(row[valid_length + 1:] == vocab + 1)  # PAD

    source = torch.tensor([[i, i + 100] for i in range(12)])
    dataset = TokenData([{
        'name': 'fixture', 'tokens': source, 'texts': ['caption A', 'caption B'],
        'beta': torch.zeros(20),
    }])
    draw_a = dataset[(0, 7721)]
    draw_b = dataset[(0, 7721)]
    target, valid_length, caption, beta = draw_a
    assert torch.equal(target, draw_b[0])
    assert valid_length == draw_b[1] and caption == draw_b[2]
    assert torch.equal(beta, draw_b[3])
    assert target.shape == (76, 2)
    assert 1 <= valid_length <= 12
    assert torch.all((target[:valid_length] >= 0) & (target[:valid_length] < vocab))
    assert torch.all(target[valid_length] == vocab)  # END
    assert torch.all(target[valid_length + 1:] == vocab + 1)  # PAD
    assert _state_sha256(vq) == vq_digest


if __name__ == '__main__':
    test_real_vq_freeze_forward_cfg_shape_sampler_and_token_data()
    print('PASS: native-SMPL MGT freeze, forward, shape CFG, sampling, and TokenData')
