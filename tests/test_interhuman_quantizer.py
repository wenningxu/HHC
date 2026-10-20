from types import SimpleNamespace

import torch

from models.interhuman_quantizer import InterHumanQuantizeEMAReset
from utils.quantizer_state import quantizer_training_state, restore_quantizer_training_state
from models.vqvae_sep3_interhuman import VQVAE_SEP2_InterHuman


def test_fp32_ema_and_gradients_under_autocast():
    q = InterHumanQuantizeEMAReset(8, 4, SimpleNamespace(mu=.99)).train()
    x = torch.randn(2, 4, 8, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y, commitment, _ = q(x.bfloat16())
    (y.square().mean() + commitment).backward()
    assert torch.isfinite(x.grad).all()
    assert all(v.dtype == torch.float32 for v in (q.codebook, q.code_sum, q.code_count))
    q.eval()
    flattened = x.detach().permute(0, 2, 1).reshape(-1, 4).bfloat16()
    reference = q.quantize(flattened)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert torch.equal(reference, q.quantize(flattened))


def test_expired_usage_resets_dead_codes_and_their_ema():
    q = InterHumanQuantizeEMAReset(4, 2, SimpleNamespace(mu=.99)).train()
    q.init_codebook(torch.ones(4, 2))
    q.usage.fill_(1)  # every code was used in an earlier window
    q.reset_count = 20
    q.update_codebook(torch.ones(4, 2), torch.zeros(4, dtype=torch.long))
    assert not q.usage.any()
    # Only code 0 receives assignments in the next window.
    for _ in range(21):
        q.update_codebook(torch.full((4, 2), 3.), torch.zeros(4, dtype=torch.long))
    assert not q.usage.any() and q.reset_count == 0
    assert torch.equal(q.code_count[1:], torch.ones(3))
    assert torch.equal(q.codebook[1:], torch.full((3, 2), 3.))
    assert torch.equal(q.code_sum[1:], q.codebook[1:])
    q.update_codebook(torch.full((4, 2), 3.), torch.zeros(4, dtype=torch.long))
    assert torch.allclose(q.codebook[1:], torch.full((3, 2), 3.))


def test_fixed_quantizer_resume_preserves_next_update():
    def make():
        model = torch.nn.Module()
        model.channel_names = ('body', 'root')
        model.quantizer_upper = InterHumanQuantizeEMAReset(8, 4, SimpleNamespace(mu=.99))
        model.quantizer_root = InterHumanQuantizeEMAReset(8, 4, SimpleNamespace(mu=.99))
        return model
    model, resumed = make(), make()
    x = torch.randn(2, 4, 8)
    for q in (model.quantizer_upper, model.quantizer_root):
        for _ in range(20):
            q(x)
    resumed.load_state_dict(model.state_dict())
    restore_quantizer_training_state(resumed, quantizer_training_state(model))
    rng = torch.get_rng_state()
    a = [q(x) for q in (model.quantizer_upper, model.quantizer_root)]
    torch.set_rng_state(rng)
    b = [q(x) for q in (resumed.quantizer_upper, resumed.quantizer_root)]
    for left, right in zip(a, b):
        assert all(torch.equal(u, v) for u, v in zip(left, right))
    for left, right in zip(model.state_dict().values(), resumed.state_dict().values()):
        assert torch.equal(left, right)


def test_vq_selects_explicit_fixed_quantizer_without_changing_legacy_default():
    def build(version):
        return VQVAE_SEP2_InterHuman(SimpleNamespace(mu=.99, interhuman_latent_dims='24,8', interhuman_quantizer=version), nb_code=16, code_dim=32, down_t=2, width=32, depth=1)
    fixed, legacy = build('fp32_window_reset_v1'), build('legacy')
    assert all(isinstance(q, InterHumanQuantizeEMAReset) for q in fixed.quantizers)
    assert all(not isinstance(q, InterHumanQuantizeEMAReset) for q in legacy.quantizers)
    assert fixed.state_dict().keys() == legacy.state_dict().keys()
    x = torch.randn(2, 64, 524)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        prediction, commitment, _ = fixed(x)
    (prediction.float().square().mean() + commitment).backward()
    assert all(q.code_sum.dtype == torch.float32 for q in fixed.quantizers)
    assert all(torch.isfinite(p.grad).all() for p in fixed.parameters() if p.grad is not None)
