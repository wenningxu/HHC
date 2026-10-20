import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_PATH))

from tools.assess_volumetric_smpl_frame_batching import ROOT, assess, sha256_file


def fake_record(path, identity, speedup=2.0, gradient_difference=1e-6):
    benchmark_path = ROOT / "tools/benchmark_volumetric_smpl_frame_batching.py"
    payload = {
        "schema": "hhc_volumetric_smpl_frame_batch_benchmark_v1",
        "status": "benchmark_complete_not_promoted",
        "device": "cuda:1",
        "frames": 12,
        "query_chunk": 10475,
        "inputs": {
            "p1": {"sha256": hashlib.sha256(f"{identity}/p1".encode()).hexdigest()},
            "p2": {"sha256": hashlib.sha256(f"{identity}/p2".encode()).hexdigest()},
        },
        "serial": {
            "loss": 3.0,
            "elapsed_seconds": 4.0,
            "peak_allocated_bytes": 4 * 1024 ** 3,
        },
        "experimental": [{
            "frame_batch_size": 4,
            "loss_relative_difference": 1e-7,
            "gradient_relative_to_serial_max": gradient_difference,
            "speedup_over_serial": speedup,
            "elapsed_seconds": 4.0 / speedup,
            "peak_allocated_bytes": 8 * 1024 ** 3,
        }],
        "implementation": {
            "production": {
                "sha256": sha256_file(
                    ROOT / "optimization/volumetric_smpl_backend.py"
                )
            },
            "experimental": {
                "sha256": sha256_file(
                    ROOT / "optimization/volumetric_smpl_batching.py"
                )
            },
            "benchmark": {"sha256": sha256_file(benchmark_path)},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def run_assessment(records):
    return assess(
        records,
        minimum_motions=2,
        maximum_loss_relative_difference=1e-5,
        maximum_gradient_relative_difference=1e-4,
        minimum_speedup=1.25,
        maximum_peak_allocated_gib=20.0,
        expected_query_chunk=10475,
    )


def test_two_distinct_passing_motions_authorize_one_common_size():
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        records = [
            fake_record(folder / "a.json", "a"),
            fake_record(folder / "b.json", "b", speedup=1.5),
        ]
        result, passed = run_assessment(records)
        assert passed
        assert result["production_edit_authorized"]
        assert result["selected_candidate"]["frame_batch_size"] == 4
        assert result["selected_candidate"]["minimum_speedup"] == 1.5


def test_one_failed_motion_denies_promotion():
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        records = [
            fake_record(folder / "a.json", "a"),
            fake_record(folder / "b.json", "b", gradient_difference=2e-4),
        ]
        result, passed = run_assessment(records)
        assert not passed
        assert not result["production_edit_authorized"]


def test_duplicate_motion_is_rejected():
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        records = [
            fake_record(folder / "a.json", "same"),
            fake_record(folder / "b.json", "same"),
        ]
        try:
            run_assessment(records)
        except ValueError as error:
            assert "duplicate real-motion" in str(error)
        else:
            raise AssertionError("duplicate inputs must fail closed")


def test_nonproduction_query_chunk_is_rejected():
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        records = [
            fake_record(folder / "a.json", "a"),
            fake_record(folder / "b.json", "b"),
        ]
        records[1][1]["query_chunk"] = 512
        records[1][0].write_text(json.dumps(records[1][1]), encoding="utf-8")
        try:
            run_assessment(records)
        except ValueError as error:
            assert "promoted serial optimizer contract 10475" in str(error)
        else:
            raise AssertionError("nonproduction query chunk must fail closed")


def test_one_oom_candidate_does_not_discard_a_safe_common_candidate():
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        records = [
            fake_record(folder / "a.json", "a"),
            fake_record(folder / "b.json", "b"),
        ]
        oom = {
            "mode": "experimental_frame_batched",
            "frame_batch_size": 8,
            "status": "cuda_out_of_memory",
            "speedup_over_serial": None,
            "loss_relative_difference": None,
            "gradient_relative_to_serial_max": None,
            "elapsed_seconds": None,
            "peak_allocated_bytes": 23 * 1024 ** 3,
        }
        for path, payload in records:
            payload["experimental"].append(dict(oom))
            path.write_text(json.dumps(payload), encoding="utf-8")
        result, passed = run_assessment(records)
        assert passed
        assert result["selected_candidate"]["frame_batch_size"] == 4
        assert result["motions"][0]["candidates"]["8"]["pass"] is False


if __name__ == "__main__":
    test_two_distinct_passing_motions_authorize_one_common_size()
    test_one_failed_motion_denies_promotion()
    test_duplicate_motion_is_rejected()
    test_nonproduction_query_chunk_is_rejected()
    test_one_oom_candidate_does_not_discard_a_safe_common_candidate()
