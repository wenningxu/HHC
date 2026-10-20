from pathlib import Path


def test_interhuman_mgt_runner_enforces_order_resume_and_formal_protocol():
    source = Path('tools/run_hhc_interhuman_mgt.sh').read_text()
    assert 'manifest_sha256' in source and '.quality_selection.pass_fraction > 0.6' in source
    assert 'HHC_INTERHUMAN_OUTPUT_ROOT' in source
    assert '--output-root "$output_root/mgt_interhuman"' in source
    assert 'optimized_comprehensively_beats_interact2ar_numerically' not in source
    assert "final_optimizer_step' \"$vq_manifest\")\" = \"30000\"" in source
    assert 'vq_wandb_audit' in source
    assert 'vq_run="$output_root/vq_interhuman/hhc_interhuman_vq_bodyroot_ema_v3"' in source
    assert '--total-steps 300000 --run-until-step "$run_until"' in source
    assert '--micro-batch-size 64 --gradient-accumulation 1' in source
    assert 'pilot_decision_ready' in source
    assert '--resume "$resume_checkpoint"' in source
    assert 'resume-pilot' in source
    assert 'checkpoints/latest.pth' in source
    assert '--wandb-mode online' in source
    assert 'finalize_hhc_interhuman_common50.sh' in source
    assert 'hhc_interhuman_mgt_common50' in source
    assert 'nohup' not in source and 'rm ' not in source
