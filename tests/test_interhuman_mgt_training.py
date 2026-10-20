from types import SimpleNamespace

import torch

from models.interhuman_transformer import InterHumanText2MotionTransformer
from models.vqvae_sep3_interhuman import VQVAE_SEP3_InterHuman
from utils.interhuman_mgt_training import (
    freeze_interhuman_vq_for_mgt,
    interhuman_mgt_objective,
    prepare_interhuman_mgt_batch,
)


def _models(paper_architecture=True):
    args = SimpleNamespace(mu=0.99, interhuman_latent_dims='24,8')
    vq = freeze_interhuman_vq_for_mgt(VQVAE_SEP3_InterHuman(
        args, nb_code=16, code_dim=32, output_emb_width=32,
        down_t=2, stride_t=2, width=32, depth=1,
        dilation_growth_rate=2,
    ))
    transformer = InterHumanText2MotionTransformer(
        vqvae=vq, num_vq=16, embed_dim=32, clip_dim=32,
        block_size=75, num_layers=1,
        num_local_layer=0 if paper_architecture else 1, n_head=4,
        drop_out_rate=0.0, fc_rate=2, contact=False, shape=False,
        paper_architecture=paper_architecture,
        conditioning_contract='paper_text_cfg',
    )
    transformer.train()
    vq.eval()
    return vq, transformer


def test_interhuman_training_batch_uses_paper_ratio_order_and_floor_support():
    vq, _ = _models()
    batch = (
        ['a', 'b'], ['one', 'two'],
        torch.randn(2, 300, 262), torch.randn(2, 300, 262),
        torch.tensor([15, 299]),
    )
    prepared = prepare_interhuman_mgt_batch(
        vq, batch, 'cpu', num_codes=16, pkeep=0.5,
        paper_architecture=True,
        generator=torch.Generator().manual_seed(7),
    )
    assert prepared['target'].shape == (2, 75, 2)
    assert prepared['token_lengths'].tolist() == [3, 74]
    assert prepared['valid_mask'].sum(dim=1).tolist() == [[3, 3], [74, 74]]
    body, root = prepared['mask_ratios'].unbind(dim=-1)
    assert torch.all(root <= body)
    assert torch.all(prepared['masked_mask'] <= prepared['valid_mask'])


def test_real_interhuman_mgt_objective_backpropagates_only_transformer():
    vq, transformer = _models()
    batch = (
        ['a'], ['one'], torch.randn(1, 300, 262),
        torch.randn(1, 300, 262), torch.tensor([297]),
    )
    prepared = prepare_interhuman_mgt_batch(
        vq, batch, 'cpu', num_codes=16,
        paper_architecture=True,
        generator=torch.Generator().manual_seed(11),
    )
    logits = transformer(
        prepared['masked_input'], torch.randn(1, 32),
        src_mask=prepared['attention_mask'],
        word_emb=torch.randn(1, 6, 32),
    )
    metrics = interhuman_mgt_objective(logits, prepared)
    metrics['loss'].backward()
    assert torch.isfinite(metrics['loss'])
    assert 0 <= metrics['masked_fraction'] <= 1
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for name, parameter in transformer.named_parameters()
        if not name.startswith('trans_base.vqvae.') and parameter.requires_grad
    )
    assert all(parameter.grad is None for parameter in vq.parameters())
    assert all(not parameter.requires_grad for parameter in vq.parameters())


def test_historical_topology_gets_explicit_end_and_pad_not_zero_tail_codes():
    vq, transformer = _models(paper_architecture=False)
    batch = (
        ['a', 'b'], ['one', 'two'],
        torch.randn(2, 300, 262), torch.randn(2, 300, 262),
        torch.tensor([15, 299]),
    )
    prepared = prepare_interhuman_mgt_batch(
        vq, batch, 'cpu', num_codes=16, paper_architecture=False,
        generator=torch.Generator().manual_seed(13),
    )
    target = prepared['target']
    assert torch.all(target[0, 3] == 16)
    assert torch.all(target[0, 4:] == 17)
    assert torch.all(target[1, 74] == 16)
    assert prepared['attention_mask'].sum(dim=1).tolist() == [4, 75]
    logits = transformer(
        prepared['masked_input'], torch.randn(2, 32),
        src_mask=prepared['attention_mask'],
        word_emb=torch.randn(2, 6, 32),
    )
    metrics = interhuman_mgt_objective(logits, prepared)
    assert torch.isfinite(metrics['loss'])


def test_branch_accuracies_do_not_average_over_other_channels():
    target = torch.zeros(1, 3, 2, dtype=torch.long)
    valid = torch.ones_like(target, dtype=torch.bool)
    logits = torch.zeros(1, 3, 2, 4)
    logits[..., 0, 0] = 10  # body entirely correct
    logits[..., 1, 1] = 10  # root entirely wrong
    metrics = interhuman_mgt_objective(logits, {
        'target': target, 'valid_mask': valid, 'masked_mask': valid,
    })
    assert metrics['accuracy_body'] == 1
    assert metrics['accuracy_root'] == 0
    assert metrics['accuracy_valid'] == 0.5
    assert 'accuracy_hands' not in metrics
