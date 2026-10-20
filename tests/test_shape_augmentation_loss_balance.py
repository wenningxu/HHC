import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_shape_augmentation_loss_balance import audit


def test_loss_balance_audit_exposes_raw_sum_scale(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    contract = {
        "optimizer": {"foot_contact_mask_policy": "frozen_initial"}
    }
    manifest = {
        "complete": True, "completed_samples": 1, "requested_samples": 1
    }
    (run / "_run_contract.json").write_text(json.dumps(contract))
    (run / "_completion_manifest.json").write_text(json.dumps(manifest))
    history = []
    for iteration in range(4):
        history.append({
            "iteration": iteration,
            "total": 2.0 if iteration < 2 else 8.0,
            "attract": 4.0 if iteration < 2 else 20.0,
            "pene": 0.0, "reg": 0.0, "smooth": 0.0,
            "acc": 0.0, "skate": 0.01, "ground": 0.0,
        })
    sample = {
        "source_name": "G000", "timing_positive_frames_crop": 10,
        "structured_contact_rows": 2,
        "diagnostics": {
            "fine_stage_start": 2, "iterations": 4,
            "frozen_contact_region_pairs": 2,
            "frozen_contact_vertex_pairs": 8,
            "loss_weights": {
                "attract": 1.0, "pene": 5.0, "reg": 1.0,
                "smooth": 0.1, "skate": 100.0, "ground": 20.0,
            },
            "loss_history": history,
        },
    }
    (run / "G000.json").write_text(json.dumps(sample))
    result = audit(run, require_complete=True)
    row = result["records"][0]
    assert row["raw_sum_over_active_frame_pair_mean_scale"] == 20
    assert row["coarse_contact_to_skate_ratio"] == 4.0
    assert row["coarse_active_mean_contact_to_skate_ratio"] == 0.2
    assert row["fine_contact_to_skate_ratio"] == 20.0
    assert row["fine_stage_total_loss_jump_ratio"] == 4.0

    contract["optimizer"][
        "contact_loss_reduction_policy"
    ] = "active_contact_mean"
    (run / "_run_contract.json").write_text(json.dumps(contract))
    mean_result = audit(run, require_complete=True)
    mean_row = mean_result["records"][0]
    assert mean_row["coarse_contact_to_skate_ratio"] == 4.0
    assert mean_row["coarse_active_mean_contact_to_skate_ratio"] == 4.0


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        test_loss_balance_audit_exposes_raw_sum_scale(Path(directory))
