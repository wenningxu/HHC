from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_current_paired_finalizer_is_direct_and_uses_current_hhc():
    source = (
        ROOT / "tools/finalize_current_interx_paired_results.sh"
    ).read_text(encoding="utf-8")
    assert "while " not in source
    assert "sleep " not in source
    assert "systemctl" not in source
    assert "hhc_formal_raw12288_repaired_mgt" not in source
    assert "hhc_historical_plus_paper_mask_ratio_no_contact_common50" in source
    assert "hhc_historical_plus_paper_mask_ratio_contact_common50" in source
    for label in (
        "intermask_official_exact_common50",
        "interact2ar_official_fair_no_gt_first_frame",
        "coshmdm_interx_retrained_2daycap_seed123",
        "timotion_transformer_lpa_interx_retrained_fair",
    ):
        assert label in source
    assert ".replications == 20" in source
    assert ".samples_per_replication == 1696" in source
    assert "all(.values | length == 20)" in source
    assert "tools/rescore_final_interx_ordinary.py" in source
    assert "--device cpu --batch-size 32" in source
    assert "current_main_ordinary_only_metric_audit.json" in source
    assert "--ordinary-only-audit" in source
    assert "paired_fair_method_comparison_historical_paper_mask.json" in source
    assert "paired_oracle_timing_effect_historical_paper_mask.json" in source
    assert "tools/build_current_interx_main_table.py" in source
    assert "output/current_interx_main_table.json" in source
    assert "output/current_interx_main_table.md" in source


def test_cpu_rescore_refreshes_exhaustive_multimodality_binding():
    source = (
        ROOT / "tools/rescore_final_interx_ordinary.py"
    ).read_text(encoding="utf-8")
    metric_aggregate = "aggregate_fair_interx_metrics.py"
    mm_aggregate = "aggregate_fair_interx_multimodality_audits.py"
    assert metric_aggregate in source
    assert mm_aggregate in source
    assert "expected 20 exhaustive multimodality audits" in source
    assert source.index(metric_aggregate) < source.index(mm_aggregate)
    assert "multimodality_audit_summary.json" in source


def test_method_finalizers_trigger_one_nonwaiting_paired_coordinator():
    coordinator = (
        ROOT / "tools/finalize_current_interx_if_ready.sh"
    ).read_text(encoding="utf-8")
    assert "while " not in coordinator
    assert "sleep " not in coordinator
    assert "systemctl" not in coordinator
    assert "flock --nonblock" in coordinator
    assert "finalize_current_interx_paired_results.sh" in coordinator
    assert "run_current_interx_review_experiments_before_shape.sh" in coordinator
    assert "run_interx_shape_augmentation_nearest_last_stage.sh" not in coordinator
    for label in (
        "hhc_historical_plus_paper_mask_ratio_no_contact_common50",
        "hhc_historical_plus_paper_mask_ratio_contact_common50",
        "intermask_official_exact_common50",
        "interact2ar_official_fair_no_gt_first_frame",
        "coshmdm_interx_retrained_2daycap_seed123",
        "timotion_transformer_lpa_interx_retrained_fair",
    ):
        assert label in coordinator
    for finalizer, gpu in (
        ("tools/finalize_hhc_historical_plus_paper_mask_ratio.sh", "1"),
        ("tools/queue_timotion_post_training.sh", "0"),
    ):
        source = (ROOT / finalizer).read_text(encoding="utf-8")
        assert f"bash tools/finalize_current_interx_if_ready.sh {gpu}" in source


if __name__ == "__main__":
    test_current_paired_finalizer_is_direct_and_uses_current_hhc()
    test_cpu_rescore_refreshes_exhaustive_multimodality_binding()
    test_method_finalizers_trigger_one_nonwaiting_paired_coordinator()
    print("current Inter-X paired-finalizer contract passed")
