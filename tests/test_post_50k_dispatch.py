import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dispatch_hhc_post_50k_training import decide
from tools.assess_hhc_global_masked_loss_control import load_selection


def assessment(decision, supported=None):
    payload = {"decision": decision}
    if supported is not None:
        payload["control"] = {"hypothesis_supported": supported}
    return payload


def shape(decision):
    return {"status": "pass", "decision": decision}


def test_retain_formal_goes_to_shape():
    result = decide(assessment(
        "retain_formal_contract_and_continue_shape_training"
    ), None)
    assert result["action"] == "continue_shape_augmentation"
    assert result["shape_training_loss_contract"] == "paper_masked_only"


def test_supported_global_control_continues_same_lineage_to_300k():
    result = decide(assessment(
        "promote_global_masked_loss_to_full_retrain_before_shape", True
    ), None)
    assert result["action"] == "continue_global_control_to_300k_then_shape"
    assert result["continue_global_control"] is True


def test_shape_shortcut_can_release_shape_training_after_negative_loss_control():
    result = decide(assessment(
        "global_loss_not_sufficient_investigate_mask_conditioning_next", False
    ), shape("train_only_shape_identity_shortcut_supported"))
    assert result["status"] == "ready"
    assert result["action"] == "continue_shape_augmentation"


def test_two_negative_diagnostics_preserve_unresolved_status_and_run_shape_arms():
    result = decide(assessment(
        "global_loss_not_sufficient_investigate_mask_conditioning_next", False
    ), shape("train_only_shape_identity_shortcut_not_established"))
    assert result["status"] == "ready"
    assert result["action"] == "continue_shape_augmentation"
    assert result["shape_training_loss_contract"] == "paper_masked_only"
    assert result["mask_conditioning_status"] == (
        "unresolved_after_two_negative_diagnostics"
    )
    assert result["diagnostic_followup_required"] is True


def test_unfinished_assessment_cannot_dispatch():
    try:
        decide(assessment("run_global_masked_token_mean_50k_control"), None)
    except ValueError as error:
        assert "not dispatchable" in str(error)
    else:
        raise AssertionError("unfinished 50k assessment was dispatched")


def valid_selection():
    return {
        "run_complete": True,
        "expected_final_step": 300000,
        "selected_step": 50000,
        "selected_validation_fid": 0.8,
        "test_metrics_used_for_selection": False,
        "selection_metric": "minimum fixed-protocol validation FID",
        "retained_validation_candidates": [
            {"step": 50000, "validation_fid": 0.8, "path": "fixture.ckpt"},
        ],
        "validation_inference_contract": {
            "sampler": "sample_intra_cas",
            "sampling_steps": 50,
            "temperature": 0.5,
            "cfg_scale": 3.5,
            "contact_guidance_scale": 1.0,
            "contact_timing_source": "validation_dataset_annotation_oracle",
            "contact_timing_is_test_data": False,
            "contact_as_transformer_training_input": False,
        },
    }


def test_selection_loader_rejects_test_selected_or_sampler_drift():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "selection.json"
        payload = valid_selection()
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert load_selection(path) == payload
        payload["test_metrics_used_for_selection"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_selection(path)
        except ValueError as error:
            assert "not validation-only" in str(error)
        else:
            raise AssertionError("test-selected checkpoint was accepted")
        payload = valid_selection()
        payload["validation_inference_contract"]["sampling_steps"] = 10
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_selection(path)
        except ValueError as error:
            assert "validation contract changed" in str(error)
        else:
            raise AssertionError("sampler-drifted checkpoint was accepted")


if __name__ == "__main__":
    test_retain_formal_goes_to_shape()
    test_supported_global_control_continues_same_lineage_to_300k()
    test_shape_shortcut_can_release_shape_training_after_negative_loss_control()
    test_two_negative_diagnostics_preserve_unresolved_status_and_run_shape_arms()
    test_unfinished_assessment_cannot_dispatch()
    test_selection_loader_rejects_test_selected_or_sampler_drift()
