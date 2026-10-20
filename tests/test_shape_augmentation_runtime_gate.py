import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/require_shape_augmentation_runtime_gate.py"


def write_analysis(path: Path, *, backend: str, mean: float) -> None:
    target = 6238
    path.write_text(json.dumps({
        "schema": "hhc_interx_shape_augmentation_runtime_v1",
        "status": "complete",
        "run": {
            "mode": "optimized",
            "signed_distance_backend": backend,
            "completed_samples": 10,
        },
        "runtime": {
            "elapsed_seconds": {"mean": mean, "median": mean},
            "naive_serial_projection": {
                "target_samples": target,
                "seconds": mean * target,
            },
        },
    }), encoding="utf-8")


def run_gate(tmp_path: Path, backend: str, mean: float):
    analysis = tmp_path / f"{backend}_{mean}.json"
    output = tmp_path / f"{backend}_{mean}_gate.json"
    write_analysis(analysis, backend=backend, mean=mean)
    completed = subprocess.run([
        sys.executable, str(SCRIPT), "--analysis", str(analysis),
        "--output", str(output),
    ], cwd=ROOT, check=False, capture_output=True, text=True)
    return completed, json.loads(output.read_text(encoding="utf-8"))


def test_nearest_vertex_ten_seconds_passes(tmp_path):
    completed, result = run_gate(tmp_path, "nearest_vertex", 10.0)
    assert completed.returncode == 0
    assert result["status"] == "pass"
    assert result["full_augmentation_authorized"] is True
    assert result["observed"]["projected_hours"] < 24.0


def test_volumetric_backend_is_rejected_even_if_fast(tmp_path):
    completed, result = run_gate(tmp_path, "volumetric_smpl", 10.0)
    assert completed.returncode == 2
    assert result["checks"]["released_nearest_vertex_backend"] is False


def test_full_projection_over_one_day_is_rejected(tmp_path):
    completed, result = run_gate(tmp_path, "nearest_vertex", 14.0)
    assert completed.returncode == 2
    assert result["checks"]["full_serial_projection_within_limit"] is False


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        test_nearest_vertex_ten_seconds_passes(root)
        test_volumetric_backend_is_rejected_even_if_fast(root)
        test_full_projection_over_one_day_is_rejected(root)
    print("3 shape runtime-gate tests passed")
