#!/usr/bin/env python
"""Runtime-audit the paper HHC MGT topology before formal training."""

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.motion_transformer import AdaCrossSelfBlock  # noqa: E402
from models.two_person_transformer import Text2Motion_Transformer  # noqa: E402


class _Quantizer(nn.Module):
    def __init__(self, num_codes, width):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, width)

    def dequantize(self, indices):
        return self.embedding(indices)


class _Tokenizer(nn.Module):
    code_dim = 256

    def __init__(self, num_codes):
        super().__init__()
        self.quantizer_upper = _Quantizer(num_codes, 96)
        self.quantizer_lower = _Quantizer(num_codes, 96)
        self.quantizer_root = _Quantizer(num_codes, 64)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/hhc_model_structure_runtime_audit.json'),
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    num_codes = 17
    model = Text2Motion_Transformer(
        _Tokenizer(num_codes), num_vq=num_codes, embed_dim=512,
        clip_dim=512, block_size=51, num_layers=9, num_local_layer=9,
        n_head=8, drop_out_rate=0.1, fc_rate=4, contact=False,
        shape=True, paper_architecture=True,
        conditioning_contract='paper_text_cfg',
    ).eval()
    blocks = model.trans_base.paper_blocks
    checks = {
        'paper_architecture_enabled': model.paper_architecture is True,
        'exactly_nine_paper_blocks': len(blocks) == 9,
        'all_blocks_are_cross_self_adaln': all(
            isinstance(block, AdaCrossSelfBlock) for block in blocks
        ),
        'eight_attention_heads': (
            model.n_head == 8
            and all(block.cross_attn.n_head == 8 for block in blocks)
            and all(block.self_attn.n_head == 8 for block in blocks)
        ),
        'hidden_dimension_is_512': (
            model.trans_base.part_emb.embedding_dim == 512
            and model.trans_base.shared_special_emb.embedding_dim == 512
            and model.trans_head.ln_f.normalized_shape == (512,)
        ),
        'three_part_embeddings': model.trans_base.part_emb.num_embeddings == 3,
        'shared_special_tokens': (
            model.trans_base.shared_special_emb.num_embeddings == 3
        ),
        # The paper does not state whether the three codebooks have
        # parameter-independent classifiers.  The released/historical HHC
        # topology uses one vocabulary projection at all interleaved part
        # positions; record that source-derived ambiguity explicitly.
        'released_shared_vocab_projection': (
            isinstance(model.trans_head.head, nn.Linear)
            and model.trans_head.head.in_features == 512
            and model.trans_head.head.out_features == num_codes
        ),
        'legacy_base_blocks_absent': len(model.trans_base.blocks) == 0,
        'legacy_output_blocks_absent': len(model.trans_head.blocks) == 0,
        'shape_conditioning_enabled': model.shape is True,
        'paper_text_cfg_enabled': (
            model.conditioning_contract == 'paper_text_cfg'
        ),
    }

    execution_order = []
    hooks = []
    for index, block in enumerate(blocks):
        hooks.append(block.cross_attn.register_forward_hook(
            lambda _module, _inputs, _output, index=index:
            execution_order.append(f'{index}:cross')
        ))
        hooks.append(block.self_attn.register_forward_hook(
            lambda _module, _inputs, _output, index=index:
            execution_order.append(f'{index}:self')
        ))
    batch, tokens = 2, 2
    indices = torch.full(
        (batch, tokens, 3), num_codes + 2, dtype=torch.long,
    )
    clip_feature = torch.zeros(batch, 512)
    word_embedding = torch.zeros(batch, 77, 512)
    source_mask = torch.ones(batch, tokens, dtype=torch.bool)
    betas = torch.zeros(batch, 20)
    gender = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward_function(
            indices, clip_feature, src_mask=source_mask,
            word_emb=word_embedding, betas=betas, gender=gender,
        )
    for hook in hooks:
        hook.remove()
    expected_order = [
        value for index in range(9)
        for value in (f'{index}:cross', f'{index}:self')
    ]
    checks.update({
        'each_block_executes_cross_then_self': execution_order == expected_order,
        'forward_shape_is_batch_time_three_vocab': (
            tuple(logits.shape) == (batch, tokens, 3, num_codes)
        ),
        'forward_logits_are_finite': bool(torch.isfinite(logits).all()),
    })
    forward_source = inspect.getsource(model.trans_base.forward)
    checks['interleave_source_is_root_body_hand'] = all(
        marker in forward_source for marker in (
            'torch.stack([x_r, x_u, x_l], dim=2)',
            'x_stack.view(B, T * 3, D)',
        )
    )
    require(all(checks.values()), f'model-structure checks failed: {checks}')

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    payload = {
        'schema': 'hhc_model_structure_runtime_audit_v1',
        'status': 'pass',
        'contract': {
            'blocks': 9,
            'block_order': 'word cross-attention -> motion self-attention -> MLP',
            'attention_heads': 8,
            'hidden_dimension': 512,
            'motion_stream': 'Root -> Body -> Hand interleaved per time step',
            'legacy_output_blocks': 0,
            'output': '[batch, time, Body/Hand/Root, vocabulary]',
            'output_classifier': (
                'one released shared Linear(hidden,vocabulary) applied at '
                'Root/Body/Hand positions; paper does not specify whether '
                'classifier parameters are shared'
            ),
        },
        'runtime': {
            'seed': args.seed,
            'execution_order': execution_order,
            'output_shape': list(logits.shape),
            'tiny_vocab_trainable_parameters': trainable_parameters,
        },
        'checks': checks,
        'sources': {
            'two_person_transformer': {
                'path': str((ROOT / 'models/two_person_transformer.py').resolve()),
                'sha256': sha256_file(ROOT / 'models/two_person_transformer.py'),
            },
            'motion_transformer': {
                'path': str((ROOT / 'models/motion_transformer.py').resolve()),
                'sha256': sha256_file(ROOT / 'models/motion_transformer.py'),
            },
        },
        'implementation': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(__file__),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(
        args.output.suffix + f'.tmp.{os.getpid()}'
    )
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
