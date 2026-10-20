from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_finalizer_freezes_common50_selection_and_joint_comparison_contract():
    source = (ROOT / "tools/finalize_no_shape_control_common50.sh").read_text()
    assert "select_no_shape_checkpoint_on_official_interx_val.py" in source
    assert "run_hhc_no_shape_joint_evaluation.sh" in source
    assert "compare_interact2ar_joint_augmentation_paired.py" in source
    assert "output/interact2ar_joint_evaluator_suite/hhc" in source
    assert "--replications 20" in source
    assert "--comparison-kind shape_condition_ablation" in source
    assert "--claim-scope diagnostic_unmatched_training_budget" in source
    assert "shape_conditioned_vs_no_shape_paired.json" in source
    assert "replay_interx_validation_raw_ground_truth.py" in source
    assert "promote_interx_validation_request_contract.py" in source
    assert "--raw-ground-truth-manifest" in source
    assert "audit_lightning_checkpoint.py" in source
    assert "audit_wandb_online_metric_history.py" in source
    assert "build_hhc_no_shape_diagnostic_result_card.py" in source
    assert source.index("build_hhc_no_shape_diagnostic_result_card.py") > source.index(
        "compare_interact2ar_joint_augmentation_paired.py"
    )
    assert "--expected-final-step 50000" in source
    assert "--metric-profile masked_token_classification" in source

    selector = (
        ROOT / "tools/select_no_shape_checkpoint_on_official_interx_val.py"
    ).read_text()
    assert "output/interx_official_val_requests/manifest.json" in selector
    assert "score_interact2ar_joint_validation.py" in selector
    assert '"selection_metric": "JointsFull/FID"' in selector
    assert "score_fair_interx_main.py" not in selector


def test_no_shape_joint_runner_has_user_fixed_sampling_and_final_mm_contract():
    source = (ROOT / "tools/run_hhc_no_shape_joint_evaluation.sh").read_text()
    assert "--sampling-steps 50 --temperature 0.5 --cfg-scale 3.5" in source
    assert "--no-hhc-mgt-shape" in source
    assert "--start-replication 0 --end-replication 20" in source
    assert "--replications 20" in source
    assert 'replication_00_mm.h5' in source
