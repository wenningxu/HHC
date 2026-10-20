#!/usr/bin/env python3

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_wrapper_changes_only_in_memory_sampler_and_relabels_output():
    source = (ROOT / "tools/export_hhc_repredict_diagnostic.py").read_text()
    ast.parse(source)
    assert "Text2Motion_Transformer.sample_intra_cas_repredict" in source
    assert "Text2Motion_Transformer.sample_intra_cas = original" in source
    assert 'target.attrs["sampler"] = "sample_intra_cas_repredict"' in source
    assert 'target.attrs["diagnostic_changed_factor"] = "token_update_support_only"' in source
    assert 'if "--resume" in sys.argv' in source


def test_launcher_freezes_every_formal_setting():
    source = (ROOT / "tools/run_hhc_repredict_validation_diagnostic.sh").read_text()
    required = (
        "--sampling-steps 50", "--temperature 0.5", "--cfg-scale 3.5",
        "--contact-guidance-scale 1.0", "--contact-guidance",
        "--seed 500123", "--mgt-contract current",
        "--inference-conditioning-contract paper_text_cfg",
        "model-000060000-0.6364.ckpt",
    )
    for marker in required:
        assert marker in source
    assert "--resume" not in source


if __name__ == "__main__":
    test_wrapper_changes_only_in_memory_sampler_and_relabels_output()
    test_launcher_freezes_every_formal_setting()
