from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_budget_finalizer_uses_complete_endpoint_and_common50_protocol():
    source = (ROOT / "tools/finalize_no_shape_full300k_common50.sh").read_text()
    assert "latest-000300000.ckpt" in source
    assert "--expected-step 300000" in source
    assert "--expected-lr 0.00001" in source
    assert "--expected-final-step 300000" in source
    assert "--candidate-steps 80000 220000 260000 280000 300000" in source
    assert "select_no_shape_checkpoint_on_test3.py" in source
    assert "run_hhc_no_shape_joint_evaluation_two_gpu.sh" in source
    assert "compare_interact2ar_joint_augmentation_paired.py" in source
    assert "full_training_budget_test_guided_selection_and_validation_rng_differs" in source
    assert "build_hhc_no_shape_full300k_result_card.py" in source
    assert "amass" not in source.lower()


def test_full_budget_result_card_fails_closed_on_required_evidence():
    source = (ROOT / "tools/build_hhc_no_shape_full300k_result_card.py").read_text()
    assert 'args.get("total_iter") == 300000' in source
    assert 'args.get("diagnostic_stop_iter") == 0' in source
    assert 'len(selection.get("candidates", [])) == 5' in source
    assert 'expected_candidate_steps = [80000, 220000, 260000, 280000, 300000]' in source
    assert 'selection.get("selection_is_test_guided") is True' in source
    assert 'checkpoint.get("global_step") == 300000' in source
    assert 'wandb.get("remote_state") == "finished"' in source
    assert '"external_shape_dataset_used": False' in source
    assert 'len(replications) == 3' in source
    assert '[0, 1, 2]' in source
    assert 'selected step is not the minimum three-replication mean' in source
    assert '"legacy_672d_metrics": compact_legacy_metrics(legacy_summary)' in source
    assert '"fid_namespace_warning"' in source


def test_two_gpu_runner_uses_disjoint_complete_shards_and_merges_before_scoring():
    source = (ROOT / "tools/run_hhc_no_shape_joint_evaluation_two_gpu.sh").read_text()
    assert "--start-replication 0 --end-replication 10" in source
    assert "--start-replication 10 --end-replication 20" in source
    assert source.count("--shard-manifest") == 1
    assert "merge_fair_interx_shard_manifests.py" in source
    assert source.index("merge_fair_interx_shard_manifests.py") < source.index(
        "score_interact2ar_joint_evaluators.py"
    )
    assert "--sampling-steps 50 --temperature 0.5 --cfg-scale 3.5" in source
    assert "--replications 20" in source


def test_paired_comparator_accepts_full_budget_noncausal_disclosure():
    source = (ROOT / "tools/compare_interact2ar_joint_augmentation_paired.py").read_text()
    assert "full_training_budget_test_guided_selection_and_validation_rng_differs" in source
    assert 'args.claim_scope == "matched_training"' in source


def test_test3_selector_validates_generation_and_score_cache_provenance():
    source = (ROOT / "tools/select_no_shape_checkpoint_on_test3.py").read_text()
    # The exporter must always run in resume mode so an existing HDF5 is
    # checked against its checkpoint/request/sampler attributes before reuse.
    assert source.count('"--resume"') == 1
    assert source.index('run([') < source.index('if not score_is_current(')
    # Cached scores are reusable only with the current released evaluator
    # weights, SMPL-X model, scorer/wrapper/metrics code and repository commit.
    for contract in (
        'def expected_score_provenance()',
        '"repository_revision"',
        '"wrapper_source"',
        '"metrics_source"',
        '"scorer_source"',
        '"body_model"',
        '"checkpoints"',
        'expected_provenance.items()',
    ):
        assert contract in source
