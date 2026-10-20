import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "joint_validation", ROOT / "tools/score_interact2ar_joint_validation.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SELECTOR_SPEC = importlib.util.spec_from_file_location(
    "no_shape_selector",
    ROOT / "tools/select_no_shape_checkpoint_on_official_interx_val.py",
)
SELECTOR = importlib.util.module_from_spec(SELECTOR_SPEC)
SELECTOR_SPEC.loader.exec_module(SELECTOR)


def test_fid_roundoff_is_normalized_but_real_negative_values_fail():
    assert MODULE.nonnegative_fid(-1e-9) == 0.0
    assert MODULE.nonnegative_fid(1e-9) == 0.0
    assert MODULE.nonnegative_fid(0.25) == 0.25
    for value in (-1e-3, float("nan"), float("inf")):
        try:
            MODULE.nonnegative_fid(value)
        except FloatingPointError:
            pass
        else:
            raise AssertionError(f"invalid FID {value} was accepted")


def test_selector_uses_joint_full_fid_and_not_legacy_672d_scorer():
    source = (
        ROOT / "tools/select_no_shape_checkpoint_on_official_interx_val.py"
    ).read_text()
    assert "score_interact2ar_joint_validation.py" in source
    assert 'joint_metrics["JointsFull"]["fid"]' in source
    assert "score_fair_interx_main.py" not in source


def test_joint_scorer_names_test_guided_selection_without_heldout_overclaim():
    source = (ROOT / "tools/score_interact2ar_joint_validation.py").read_text()
    assert '"test_guided_checkpoint_selection"' in source
    assert "user-directed three-replication official-test checkpoint selection" in source
    assert '"evaluation_role": args.evaluation_role' in source


def _write_selector_run(path, *, total_iter, diagnostic_stop, final_step):
    path.mkdir()
    resolved = {
        "shape": False,
        "contact": False,
        "contact_conditioning": False,
        "training_mask_contract": "historical_pipeline_paper_kinematic_ratio",
        "training_loss_contract": "historical_all_valid",
        "training_validation_contract": "paper_isolated_common",
        "sampler": "sample_intra_cas",
        "sampling_steps": 50,
        "temperature": 0.5,
        "cfg_scale": 3.5,
        "diagnostic_stop_iter": diagnostic_stop,
        "eval_iter": 10,
        "save_all_validation_checkpoints": True,
        "wandb_mode": "online",
        "total_iter": total_iter,
    }
    (path / "training_manifest.json").write_text(
        json.dumps({"resolved_arguments": resolved})
    )
    (path / f"latest-{final_step:09d}.ckpt").touch()
    for step in range(10, final_step + 1, 10):
        (path / f"model-{step:09d}-0.1234.ckpt").touch()
    return SimpleNamespace(run_dir=path, final_step=final_step, interval=10)


def test_selector_accepts_diagnostic_pause_and_normal_full_endpoint(tmp_path):
    diagnostic = _write_selector_run(
        tmp_path / "diagnostic", total_iter=30, diagnostic_stop=20, final_step=20
    )
    _, diagnostic_candidates = SELECTOR.validate_run(diagnostic)
    assert sorted(diagnostic_candidates) == [10, 20]

    full = _write_selector_run(
        tmp_path / "full", total_iter=30, diagnostic_stop=0, final_step=30
    )
    _, full_candidates = SELECTOR.validate_run(full)
    assert sorted(full_candidates) == [10, 20, 30]


def test_selector_rejects_diagnostic_stop_at_full_endpoint(tmp_path):
    invalid = _write_selector_run(
        tmp_path / "invalid", total_iter=30, diagnostic_stop=30, final_step=30
    )
    with pytest.raises(ValueError, match="training contract mismatch"):
        SELECTOR.validate_run(invalid)
