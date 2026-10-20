from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_20k_milestone_binds_validation_training_and_fixed_checkpoint_pair():
    source = (
        ROOT / "tools/audit_hhc_original38_20k_milestone.sh"
    ).read_text(encoding="utf-8")
    required = (
        "--steps 9999 19999 --checkpoint-step-offset 1",
        "--steps 999 4999 9999 14999 19999",
        "--expected-final-step 20000",
        "prevalidation-000020000.ckpt",
        "postvalidation-000020000.ckpt",
        "--expected-step 20000",
        "--require-fixed-pair-marker",
        "--expected-final-step 50000",
        "--acknowledge-failed-no-checkpoint-run-id ans4ij26",
        "--acknowledge-failed-no-checkpoint-run-id qgcs8cnb",
    )
    assert all(marker in source for marker in required)


if __name__ == "__main__":
    test_20k_milestone_binds_validation_training_and_fixed_checkpoint_pair()
