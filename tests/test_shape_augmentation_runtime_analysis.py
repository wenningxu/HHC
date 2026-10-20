import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_shape_augmentation_runtime import build_report


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_runtime_report_is_provisional_until_frozen_pilot_finishes(tmp_path):
    run = tmp_path / "run"
    plan = tmp_path / "plan.json"
    write_json(plan, {"eligible_samples": 100, "positive_timing_samples": 80})
    manifest = {
        "requested_samples": 2,
        "completed_samples": 1,
        "completed_names": ["one"],
        "selection": {"selected_names": ["one", "two"]},
        "optimizer": {"mode": "optimized", "query_chunk": 512, "iterations": 100},
    }
    write_json(run / "_completion_manifest.json", manifest)
    write_json(run / "one.json", {
        "augmented_name": "one",
        "crop_start": 0,
        "crop_stop": 20,
        "timing_positive_frames_crop": 5,
        "diagnostics": {"elapsed_seconds": 10.0},
    })
    report = build_report(run, plan, bootstrap_replicates=100, seed=1)
    assert report["status"] == "provisional"
    assert report["run"]["remaining_samples"] == 1
    assert report["runtime"]["naive_serial_projection"]["seconds"] == 1000.0
    assert "bootstrap_mean_projection_95" not in report["runtime"]


def test_runtime_report_bootstraps_complete_paired_selection(tmp_path):
    run = tmp_path / "run"
    plan = tmp_path / "plan.json"
    write_json(plan, {"eligible_samples": 10, "positive_timing_samples": 8})
    manifest = {
        "requested_samples": 2,
        "completed_samples": 2,
        "completed_names": ["one", "two"],
        "selection": {"selected_names": ["one", "two"]},
        "optimizer": {"mode": "optimized", "query_chunk": 512, "iterations": 100},
    }
    write_json(run / "_completion_manifest.json", manifest)
    for name, seconds, timing in (("one", 10.0, 5), ("two", 30.0, 10)):
        write_json(run / f"{name}.json", {
            "augmented_name": name,
            "crop_start": 0,
            "crop_stop": 20,
            "timing_positive_frames_crop": timing,
            "diagnostics": {"elapsed_seconds": seconds},
        })
    report = build_report(run, plan, bootstrap_replicates=1000, seed=2)
    assert report["status"] == "complete"
    assert report["runtime"]["elapsed_seconds"]["mean"] == 20.0
    assert report["runtime"]["naive_serial_projection"]["seconds"] == 200.0
    interval = report["runtime"]["bootstrap_mean_projection_95"]
    assert interval["gpu_days_lower"] <= 200.0 / 86400.0
    assert interval["gpu_days_upper"] >= 200.0 / 86400.0


if __name__ == "__main__":
    test_runtime_report_is_provisional_until_frozen_pilot_finishes(
        Path("/tmp/hhc_shape_runtime_test_one")
    )
    test_runtime_report_bootstraps_complete_paired_selection(
        Path("/tmp/hhc_shape_runtime_test_two")
    )
