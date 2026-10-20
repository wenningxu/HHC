from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def test_zero_beta_diagnostic_preserves_input_and_cannot_resume_as_requested():
    import numpy as np
    from tools.export_hhc_official_interx import generation_betas, resolve_beta_policies, validate_resume_attributes

    betas = np.ones((2, 20), dtype=np.float32)
    assert generation_betas(betas, 'requested') is betas
    assert np.array_equal(generation_betas(betas, 'zero'), np.zeros_like(betas))
    assert np.all(betas == 1)
    assert resolve_beta_policies('requested') == ('requested', 'requested')
    assert resolve_beta_policies('requested', 'zero', None) == ('zero', 'requested')
    assert resolve_beta_policies('zero', None, 'requested') == ('zero', 'requested')
    assert resolve_beta_policies('requested', 'zero', 'zero') == ('zero', 'zero')
    source = SimpleNamespace(attrs={'beta_policy': 'zero'})
    try:
        validate_resume_attributes(source, {})
    except ValueError as error:
        assert 'beta policy' in str(error)
    else:
        raise AssertionError('zero-shape diagnostic accepted as requested shape')


def test_no_shape_flag_is_forwarded_to_hhc_exporter():
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    import run_fair_interx_method as runner

    args = SimpleNamespace(
        method="hhc",
        batch_size=32,
        checkpoint="candidate.ckpt",
        vq_checkpoint="vq.pth",
        hhc_stats_dir="dataset/interx",
        hhc_stats_prefix="Interx",
        hhc_vq_contract="precleanup_raw12288",
        hhc_mgt_contract="historical_mmm",
        hhc_mgt_shape=False,
        hhc_inference_conditioning_contract="paper_text_cfg",
        hhc_contact_map="contact.npy",
        hhc_contact_map_contract="paper_formula_train_only",
        gpu=1,
        sampling_steps=50,
        temperature=0.5,
        cfg_scale=3.5,
        contact_guidance_scale=1.0,
        hhc_contact_timing_source="none",
        hhc_contact_timing_file=None,
        contact_guidance=False,
    )
    command, _ = runner.exporter_command(
        args, "request.npz", "output.h5", 1234, 1
    )
    assert "--no-mgt-shape" in command
    assert "--mgt-shape" not in command


def test_historical_no_shape_model_args_only_remove_mgt_shape_branch():
    import sys
    import torch

    sys.path.insert(0, str(ROOT / "tools"))
    import export_hhc_official_interx as exporter

    expected = exporter.PRECLEANUP_RAW_12288_VQ_CONTRACT
    payload = {
        "args": dict(expected),
        "net": {
            "quantizer_upper.codebook": torch.empty(
                expected["nb_code"], expected["code_dim"] * 3 // 8
            ),
            "condition_mlp.2.weight": torch.empty(
                expected["shape_condition_dim"], 1
            ),
        },
    }
    args = exporter.paper_model_args(
        "cpu", "unused.pth", payload,
        contract="precleanup_raw12288",
        mgt_contract="historical_mmm",
        mgt_shape=False,
    )
    assert args.shape is False
    assert args.paper_architecture is False
    assert args.embed_dim_gpt == 1024
    assert args.n_head_gpt == 16
    assert args.num_local_layer == 2
    assert args.shape_condition_dim == 8


def test_mm_validator_requires_explicit_shape_and_accepts_split_beta_policies():
    from tools.score_hhc_validation_mm import validate_generation_metadata
    valid = {
        'sampling_steps': 50, 'temperature': 0.5, 'cfg_scale': 3.5,
        'sampler': 'sample_intra_cas', 'mgt_shape_conditioning': False,
        'contact_guidance': False, 'contact_timing_source': 'none',
        'vq_contract': 'precleanup_raw12288', 'mgt_contract': 'historical_mmm',
        'inference_conditioning_contract': 'paper_text_cfg', 'seed': 600123,
        'temperature_end': 'constant', 'temperature_schedule': 'constant',
    }
    validate_generation_metadata(valid)
    validate_generation_metadata(dict(valid, mgt_shape_conditioning=True),
                                 mgt_shape_conditioning=True)
    validate_generation_metadata(dict(valid, mgt_beta_policy='zero', vq_beta_policy='requested'),
                                 mgt_beta_policy='zero', vq_beta_policy='requested')
    for kwargs in ({'mgt_beta_policy': 'zero'}, {'vq_beta_policy': 'zero'}):
        try:
            validate_generation_metadata(valid, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('missing split beta policy accepted')
    try:
        validate_generation_metadata(dict(valid, mgt_shape_conditioning=True))
    except ValueError:
        pass
    else:
        raise AssertionError('shape-enabled metadata accepted without explicit opt-in')


def test_split_beta_routing_and_resume_mismatches():
    import ast
    import inspect
    import numpy as np
    import torch
    from tools.export_hhc_official_interx import (
        generation_betas, resolve_beta_policies, validate_resume_attributes,
    )
    source = inspect.getsource(__import__('tools.export_hhc_official_interx', fromlist=['main']))
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    transformer_call = next(node for node in calls if node.func.id == 'sample'
                            and any(k.arg == 'betas' for k in node.keywords))
    tokenizer_call = next(node for node in calls if node.func.id == 'tokenizer'
                          and len(node.args) >= 2)
    namespace = {
        'torch': torch, 'np': np,
        'mgt_betas': np.arange(8, dtype=np.float32).reshape(2, 4),
        'vq_betas': np.arange(8, dtype=np.float32).reshape(2, 4) + 10,
        'start': 0, 'end': 1, 'local_index': 0,
        'device': torch.device('cpu'),
    }
    mgt_value = eval(compile(ast.Expression(next(k.value for k in transformer_call.keywords
                                             if k.arg == 'betas')), '<ast>', 'eval'), namespace)
    vq_value = eval(compile(ast.Expression(tokenizer_call.args[1]), '<ast>', 'eval'), namespace)
    assert torch.equal(mgt_value, torch.tensor([[0., 1., 2., 3.]]))
    assert torch.equal(vq_value, torch.tensor([[10., 11., 12., 13.]]))
    original = np.arange(8, dtype=np.float32).reshape(2, 4)
    for base, mgt, vq in (('requested', None, None), ('zero', None, None),
                          ('requested', 'zero', 'requested'), ('requested', 'requested', 'zero')):
        mp, vp = resolve_beta_policies(base, mgt, vq)
        namespace.update(mgt_betas=generation_betas(original, mp),
                         vq_betas=generation_betas(original, vp))
        for expression, policy in ((next(k.value for k in transformer_call.keywords if k.arg == 'betas'), mp),
                                   (tokenizer_call.args[1], vp)):
            actual = eval(compile(ast.Expression(expression), '<ast>', 'eval'), namespace)
            expected = np.zeros_like(original[:1]) if policy == 'zero' else original[:1]
            np.testing.assert_array_equal(actual.numpy(), expected)
        assert np.array_equal(original, np.arange(8, dtype=np.float32).reshape(2, 4))
    for source_attrs, expected in (
        ({'beta_policy': 'requested', 'mgt_beta_policy': 'zero', 'vq_beta_policy': 'requested'},
         {'beta_policy': 'requested', 'mgt_beta_policy': 'requested', 'vq_beta_policy': 'requested'}),
        ({'beta_policy': 'requested', 'mgt_beta_policy': 'requested', 'vq_beta_policy': 'zero'},
         {'beta_policy': 'requested', 'mgt_beta_policy': 'requested', 'vq_beta_policy': 'requested'}),
        ({'beta_policy': 'zero'},
         {'beta_policy': 'requested', 'mgt_beta_policy': 'requested', 'vq_beta_policy': 'requested'}),
    ):
        for old, new in ((source_attrs, expected), (expected, source_attrs)):
            try:
                validate_resume_attributes(SimpleNamespace(attrs=old), new)
            except ValueError:
                pass
            else:
                raise AssertionError('mismatched beta policies accepted for resume')
