from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_selector_has_hash_verified_partial_resume():
    source = (
        ROOT / "tools/select_hhc_checkpoint_on_official_interx_val.py"
    ).read_text(encoding="utf-8")
    required = (
        "if output_path.is_file():",
        '"resume_verified_candidates"',
        "validate_artifact(",
        'record.get("checkpoint", {})',
        'record.get("generations", {})',
        'record.get("metrics", {})',
        "if step in completed_steps:",
        "existing validation selection belongs to another immutable ",
    )
    assert all(item in source for item in required)
    assert source.index("if output_path.is_file():") < source.index(
        "for step, checkpoint in candidates.items():"
    )


def test_selector_requires_method_neutral_official_validation_contract():
    source = (
        ROOT / "tools/select_hhc_checkpoint_on_official_interx_val.py"
    ).read_text(encoding="utf-8")
    assert 'REQUEST_SCHEMA = "hhc_interx_method_neutral_validation_request_v1"' in source
    assert "held-out official Inter-X checkpoint selection for all methods" in source
    assert 'output/interx_official_val_requests/manifest.json' in source
    assert 'request_manifest.get("test_set_used") is not False' in source
    assert 'request_manifest.get("request_is_byte_identical_to_source") is not True' in source
    assert 'output/coshmdm_interx_val_requests/manifest.json' not in source
    assert '"source_generation_manifest"' in source
    assert '"request_generation_implementation"' in source
    assert 'promote_interx_validation_request_contract.py' in source
    assert 'final_checkpoint_audit.get("status") != "pass"' in source
    assert 'wandb_history_audit.get("remote_state") != "finished"' in source
    assert '"final_checkpoint_audit": artifact(final_checkpoint_audit)' in source
    assert '"wandb_history_audit": artifact(wandb_history_audit)' in source
    assert '"sampling_runtime_audit": artifact(sampling_runtime_audit)' in source
    assert '"model_architecture_contract": "historical_1024d_with_end_token"' in source
    assert 'hhc_historical_model_checkpoint_structure_audit_v1' in source
    assert '"model_structure_audit": artifact(model_structure_audit)' in source


def test_selector_accepts_only_explicit_model_and_latest_checkpoint_names():
    source = (
        ROOT / "tools/select_hhc_checkpoint_on_official_interx_val.py"
    ).read_text(encoding="utf-8")
    assert 'MODEL_PATTERN = re.compile(r"model-(\\d{9})-[0-9.]+\\.ckpt$")' in source
    assert 'LATEST_PATTERN = re.compile(r"latest-(\\d{9})\\.ckpt$")' in source
    assert "match = LATEST_PATTERN.match(path.name)" in source


def test_finalizer_can_resume_at_selection_without_repeating_audits():
    source = (
        ROOT / "tools/finalize_hhc_historical_plus_paper_mask_ratio.sh"
    ).read_text(encoding="utf-8")
    assert 'resume_from="${HHC_FINALIZE_RESUME_FROM:-start}"' in source
    assert 'if [[ "${resume_from}" == start ]]; then' in source
    assert "prior final-state audits will be validated by the selector, not rerun" in source
    assert source.index('if [[ "${resume_from}" == start ]]') < source.index(
        "tools/audit_hhc_sampling_runtime.py"
    )
    assert source.index("tools/promote_interx_validation_request_contract.py") < source.index(
        "tools/select_hhc_checkpoint_on_official_interx_val.py"
    )


if __name__ == "__main__":
    test_selector_has_hash_verified_partial_resume()
    test_selector_requires_method_neutral_official_validation_contract()
    test_selector_accepts_only_explicit_model_and_latest_checkpoint_names()
    test_finalizer_can_resume_at_selection_without_repeating_audits()
    print("4 HHC official-val selector contract tests passed")
