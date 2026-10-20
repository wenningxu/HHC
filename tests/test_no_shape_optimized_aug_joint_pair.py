from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paired_shape_augmentation_uses_identical_joint_val_selection():
    source = (
        ROOT / "tools/finalize_no_shape_optimized_aug_joint_pair.sh"
    ).read_text()
    assert source.count("select_no_shape_checkpoint_on_official_interx_val.py") == 2
    assert source.count('--raw-ground-truth-manifest "${raw_val_gt_manifest}"') == 2
    assert "replay_interx_validation_raw_ground_truth.py" in source
    assert "promote_interx_validation_request_contract.py" in source
    assert "wait \"${baseline_selection_pid}\"" in source
    assert "wait \"${augmented_selection_pid}\"" in source
    assert "compare_interact2ar_joint_augmentation_paired.py" in source
    assert "audit_lightning_checkpoint.py" in source
    assert "audit_wandb_online_metric_history.py" in source
    assert source.count("--expected-final-step 50000") == 1
