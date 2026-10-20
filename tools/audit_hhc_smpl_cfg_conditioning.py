"""CPU structural audit of the HHC text-null training/CFG mismatch.

Uses a tiny randomly initialized instance of the real HHC network, not trained
weights. This verifies data routing and gradients; it cannot measure quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from models.mgt_interhuman_smpl import SMPLMGT
from models.mgt_interhuman_smpl_cfg import configure_conditioning


class Quantizer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer('codes', torch.randn(16, width))

    def dequantize(self, ids):
        return self.codes[ids]


class TinyFrozenVQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_code = 16
        self.quantizer_upper = Quantizer(96)
        self.quantizer_root = Quantizer(64)
        self.requires_grad_(False).eval()


def run():
    torch.set_num_threads(2)
    torch.manual_seed(20261016)
    model = SMPLMGT(TinyFrozenVQ(), embed_dim=32, layers=3, heads=4,
                    dropout=0., block_size=8)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    tokens = torch.randint(16, (4, 7, 2))
    global_text, words, beta = torch.randn(4, 512), torch.randn(4, 5, 512), torch.randn(4, 20)
    gender, mask = torch.ones(4, 2, dtype=torch.long), torch.ones(4, 7, dtype=torch.bool)
    captured = {}
    handles = []
    for name in ('word_emb', 'cond_emb', 'shape_projection'):
        def capture(module, inputs, key=name):
            captured[key] = inputs[0].detach().clone()
        handles.append(getattr(model.trans_base, name).register_forward_pre_hook(capture))

    def forward(g=global_text, w=words, b=beta, force=False):
        return model.forward_function(tokens, g, mask, word_emb=w, betas=b,
                                      gender=gender, force_mask=force)

    rows = []
    for arm in ('historical', 'matched'):
        model.train()
        configure_conditioning(model, arm)
        # Force the dropout event so the test is deterministic, then restore .2.
        model.trans_base.text_drop_prob = model.trans_base.shape_mask_prob = 1.
        out = forward()
        expected_word = words if arm == 'historical' else torch.zeros_like(words)
        assert torch.equal(captured['word_emb'], expected_word)
        assert torch.count_nonzero(captured['cond_emb']) == 0
        assert torch.count_nonzero(captured['shape_projection']) == 0
        torch.nn.functional.cross_entropy(out.reshape(-1, 16), tokens.reshape(-1)).backward()
        assert all(p.grad is None for p in model.trans_base.vqvae.parameters())
        assert any(p.grad is not None and p.grad.isfinite().all() for p in model.trans_head.parameters())
        rows.append({'arm': arm, 'forced_training_drop': True,
                     'word_text_removed': not bool(torch.count_nonzero(captured['word_emb'])),
                     'global_text_removed': True, 'independent_shape_drop_preserved': True})
        model.zero_grad(set_to_none=True)

    configure_conditioning(model, 'matched')
    torch.manual_seed(20261016)
    many_words, many_global = torch.randn(4096, 3, 512), torch.randn(4096, 512)
    dropped_word, dropped_global = model.trans_base.mask_text(many_words, many_global)
    word_drop = (dropped_word == 0).all(dim=(1, 2))
    global_drop = (dropped_global == 0).all(dim=1)
    assert torch.equal(word_drop, global_drop)
    assert .17 < float(word_drop.float().mean()) < .23

    model.eval()
    for arm in ('historical', 'matched'):
        configure_conditioning(model, arm, sampling=True)
        with torch.no_grad():
            null = forward(force=True)
            assert torch.equal(captured['shape_projection'], beta)
            changed_text = forward(global_text + 3, words - 2, force=True)
            assert torch.equal(null, changed_text)
            changed_shape = forward(b=beta + 1, force=True)
            assert not torch.equal(null, changed_shape)
        rows.append({'arm': arm, 'sampling_text_null_independent_of_text': True,
                     'sampling_shape_retained_and_effective': True})
    for handle in handles:
        handle.remove()
    assert all(torch.equal(value, original[key]) for key, value in model.state_dict().items())
    return {'status': 'passed', 'device': 'cpu', 'seed': 20261016,
            'evidence_level': 'random_tiny_network_routing_and_gradient_test_only',
            'performance_improvement_measured': False,
            'matched_joint_text_drop_rate_4096': float(word_drop.float().mean()),
            'state_dict_unchanged': True, 'rows': rows,
            'sources': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
                'models/mgt_interhuman_smpl.py', 'models/mgt_interhuman_smpl_cfg.py',
                'tools/train_hhc_smpl_cfg_ablation.py', 'tools/audit_hhc_smpl_cfg_conditioning.py')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
