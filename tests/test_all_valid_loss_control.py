import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_valid_control_is_explicit_single_factor_and_online():
    launcher = (ROOT / "tools/run_hhc_all_valid_loss_control.sh").read_text()
    required = (
        "--training-mask-contract paper_kinematic",
        "--training-loss-contract historical_all_valid",
        "--optimizer-contract paper_warmup_cosine",
        "--conditioning-contract paper_text_cfg",
        "--training-token-length-contract paper_50",
        "--training-validation-contract paper_isolated_common",
        "--sampler sample_intra_cas --temperature 0.5 --sampling-steps 50",
        "--cfg-scale 3.5 --contact-guidance-scale 1.0",
        "--wandb-mode online --seed 123",
        "--diagnostic-stop-iter 50000",
        "tools/audit_wandb_online_metric_history.py",
        "tools/audit_mgt_wandb_warmup.py",
        "tools/audit_mgt_historical_loss_contract.py",
        "historical_mgt_loss_reduction_audit.json",
        "tools/compare_mgt_wandb_validation.py",
        "--steps 9999 19999 29999 39999 49999",
        "--checkpoint-step-offset 1",
        "hhc_all_valid_vs_formal50_validation_10k_50k.json",
    )
    assert all(marker in launcher for marker in required)
    assert "--training-loss-contract paper_masked_only" not in launcher
    assert "--training-loss-contract paper_masked_global_token_mean" not in launcher


def test_all_valid_assessment_never_promotes_nonpaper_loss_to_formal():
    path = ROOT / "tools/assess_hhc_all_valid_loss_control.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    assert "exploratory single-factor diagnosis only" in source
    assert "paper masked-only" in source
    assert 'key != "training_loss_contract"' in source
    assert "substantial_gap_explained" in source


if __name__ == "__main__":
    test_all_valid_control_is_explicit_single_factor_and_online()
    test_all_valid_assessment_never_promotes_nonpaper_loss_to_formal()
