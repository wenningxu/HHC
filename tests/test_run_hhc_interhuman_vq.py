from pathlib import Path


def test_interhuman_vq_runner_enforces_order_pause_resume_and_final_evidence():
    source = Path('tools/run_hhc_interhuman_vq.sh').read_text()
    assert 'shape_root="dataset/interx_shape_ssp62_physical_full6238_reviewfiltered_20260930"' in source
    assert 'manifest_sha256' in source and '.quality_selection.pass_fraction > 0.6' in source
    assert 'HHC_INTERHUMAN_OUTPUT_ROOT' in source
    assert '--output-root "$output_root/vq_interhuman"' in source
    assert 'optimized_comprehensively_beats_interact2ar_numerically' not in source
    assert '--total-steps 30000 --run-until-step "$run_until"' in source
    assert 'pilot_decision_ready' in source
    assert 'current_optimizer_step' in source
    assert '--resume "$resume_checkpoint"' in source
    assert 'resume-pilot' in source
    assert 'checkpoints/latest.pth' in source
    assert '--wandb-mode online' in source
    assert '--metric-profile interhuman_vq' in source
    assert '--expected-final-step 30000' in source
    assert 'evaluate_hhc_interhuman_vq_validation.py' in source
    assert 'export_official_intermask_interhuman_vq_reconstruction.py' in source
    assert 'score_interhuman_vq_validation.py' in source
    assert 'compare_interhuman_vq_validation.py' in source
    assert 'nohup' not in source and 'rm ' not in source
