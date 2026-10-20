import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_metric_aligned_foot_single_factor import (
    FOOT_POLICY,
    build_checks,
    expected_candidate_sources,
)


def fixtures():
    identity = {
        "schema": "hhc_interx_shape_augmentation_run_v1",
        "plan_sha256": "a" * 64,
        "plan_records_sha256": "b" * 64,
        "contact_manifest_sha256": "c" * 64,
        "source_motion_sha256": "d" * 64,
        "selection": {"selected_names": [str(index) for index in range(25)]},
        "requested_samples": 25,
        "completed_samples": 25,
        "complete": True,
    }
    optimizer = {
        "mode": "optimized",
        "iterations": 100,
        "learning_rate": 0.02,
        "fine_stage_start": 50,
        "contact_loss_reduction_policy": "active_contact_mean",
        "foot_contact_mask_policy": "frozen_initial",
        "query_chunk": 10475,
    }
    reference = {**identity, "optimizer": optimizer}
    candidate = {
        **identity,
        "optimizer": {
            **optimizer,
            "foot_contact_mask_policy": FOOT_POLICY,
            "volumetric_smpl_frame_batch_size": 8,
        },
        "implementation": expected_candidate_sources(),
    }
    preregistration = json.loads((
        ROOT / "configs/interx_shape_augmentation/"
        "metric_aligned_foot_sliding_pilot_v1.json"
    ).read_text(encoding="utf-8"))
    batching = json.loads((
        ROOT / "output/volumetric_smpl_flat_hand_serial_sampling_"
        "frame_batching_assessment.json"
    ).read_text(encoding="utf-8"))
    activation = {
        "full_augmentation_authorized": False,
        "metric_decisions": {
            "mean_foot_sliding_m_per_frame": {
                "pass": False,
                "observed": {
                    "paired_delta_after_minus_before_mean": 0.001,
                },
            },
        },
    }
    return reference, candidate, preregistration, activation, batching


def test_exact_preregistered_candidate_passes():
    checks = build_checks(*fixtures())
    assert checks
    assert all(checks.values()), checks


def test_unregistered_objective_change_fails():
    reference, candidate, preregistration, activation, batching = fixtures()
    candidate["optimizer"]["learning_rate"] = 0.03
    checks = build_checks(
        reference, candidate, preregistration, activation, batching
    )
    assert not checks["all_objective_settings_except_foot_policy_identical"]


def test_non_foot_failure_does_not_activate_candidate():
    reference, candidate, preregistration, activation, batching = fixtures()
    activation["metric_decisions"]["mean_foot_sliding_m_per_frame"] = {
        "pass": True,
        "observed": {"paired_delta_after_minus_before_mean": -0.001},
    }
    checks = build_checks(
        reference, candidate, preregistration, activation, batching
    )
    assert not checks["foot_specific_activation_condition_was_met"]


if __name__ == "__main__":
    test_exact_preregistered_candidate_passes()
    test_unregistered_objective_change_fails()
    test_non_foot_failure_does_not_activate_candidate()
