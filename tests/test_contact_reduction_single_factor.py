from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_contact_reduction_single_factor import build_checks


def test_contact_reduction_audit_allows_only_the_preregistered_factor():
    common = {
        "schema": "run", "plan_sha256": "p",
        "plan_records_sha256": "pr", "contact_manifest_sha256": "c",
        "source_motion_sha256": "s", "selection": {"seed": 1},
        "requested_samples": 25, "completed_samples": 25,
        "complete": True,
        "optimizer": {
            "iterations": 100, "learning_rate": 0.02,
            "foot_contact_mask_policy": "frozen_initial",
        },
    }
    reference = {**common, "optimizer": dict(common["optimizer"])}
    candidate = {**common, "optimizer": {
        **common["optimizer"],
        "contact_loss_reduction_policy": "active_contact_mean",
    }}
    preregistration = {
        "status": "conditionally_preregistered",
        "created_before_frozen_initial_pilot_completion": True,
        "single_factor_change": {
            "factor": "contact_loss_reduction_policy"
        },
    }
    activation = {"full_augmentation_authorized": False}
    checks = build_checks(
        reference, candidate, preregistration, activation
    )
    assert all(checks.values())

    changed = {**candidate, "optimizer": {
        **candidate["optimizer"], "learning_rate": 0.01,
    }}
    checks = build_checks(
        reference, changed, preregistration, activation
    )
    assert checks[
        "all_optimizer_settings_except_reduction_identical"
    ] is False


def test_contact_reduction_finalizer_preserves_causal_results_on_gate_failure():
    source = (
        ROOT / "tools/finalize_normalized_contact_reduction_pilot.sh"
    ).read_text(encoding="utf-8")
    causal = '--before "${reference_physical}" --after "${candidate_physical}"'
    assessment = "tools/assess_shape_augmentation_pilot.py"
    assert source.index(causal) < source.index(assessment)
    assert "|| assessment_status=$?" in source
    assert 'exit "${assessment_status}"' in source


if __name__ == "__main__":
    test_contact_reduction_audit_allows_only_the_preregistered_factor()
    test_contact_reduction_finalizer_preserves_causal_results_on_gate_failure()
