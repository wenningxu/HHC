import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.select_timotion_checkpoint_on_interx_val import (
    artifact,
    main,
)


def build_fixture(root: Path):
    final_checkpoint = root / "last.ckpt"
    final_checkpoint.write_bytes(b"full-state")
    manifest_path = root / "training_manifest.json"
    manifest_path.write_text(json.dumps({
        "status": "complete",
        "final_global_step": 106500,
        "last_checkpoint": str(final_checkpoint),
        "last_checkpoint_sha256": artifact(final_checkpoint)["sha256"],
        "batch_emulation": {
            "expected_final_optimizer_updates": 106500,
            "optimizer_updates_per_epoch": 71,
        },
        "paper_protocol": {"epochs": 1500},
        "observability": {
            "semantic_validation_split": "official Inter-X val (544 samples)",
        },
        "official_split_counts": {"train": 9110, "val": 570, "test": 1708},
    }))
    records = []
    for epoch in range(100, 1501, 100):
        step = epoch * 71
        checkpoint = root / f"step_{step:09d}_weights.ckpt"
        generations = root / f"step_{step:09d}_main.h5"
        metrics = root / f"step_{step:09d}.json"
        checkpoint.write_bytes(f"checkpoint-{step}".encode())
        generations.write_bytes(f"generation-{step}".encode())
        fid = 1.0 + epoch / 1000.0
        if epoch == 300:
            fid = 0.25
        scored = {
            "status": "complete",
            "samples": 544,
            "generation_request_hash_verified": True,
            "generation_length_policy": "requested_lengths",
            "main_generations_sha256": artifact(generations)["sha256"],
            "fid": fid,
            "top1": 0.1,
            "top2": 0.2,
            "top3": 0.3,
            "matching": 4.0,
            "diversity": 8.0,
            "main_generation_metadata": {
                "formal_result_eligible": True,
                "rotation6d_contract": (
                    "a1=d6[...,0:3], a2=d6[...,3:6], Gram-Schmidt, row-stack"
                ),
                "exporter_sha256": artifact(
                    ROOT / "tools/export_timotion_official_interx_standard.py"
                )["sha256"],
                "base_exporter_sha256": artifact(
                    ROOT / "tools/export_timotion_official_interx.py"
                )["sha256"],
            },
        }
        metrics.write_text(json.dumps(scored))
        records.append({
            "epoch": epoch,
            "step": step,
            "samples": 544,
            "test_set_used": False,
            "checkpoint": artifact(checkpoint),
            "generations": artifact(generations),
            "metrics": artifact(metrics),
            "values": {
                "val/FID": fid,
                "val/R_precision_top1": 0.1,
                "val/R_precision_top2": 0.2,
                "val/R_precision_top3": 0.3,
                "val/Matching": 4.0,
                "val/Diversity": 8.0,
            },
            "validation_contract": "standard_row_concatenated_v1",
        })
    history_path = root / "official_validation_history.json"
    history_path.write_text(json.dumps({
        "schema": "timotion_official_interx_standard_validation_history_v1",
        "status": "complete",
        "final_global_step": 106500,
        "validation_contract": "standard_row_concatenated_v1",
        "completed_validation_epochs": list(range(100, 1501, 100)),
        "training_manifest": artifact(manifest_path),
        "implementation": artifact(
            ROOT / "tools/replay_timotion_standard_validation.py"
        ),
        "records": records,
    }))
    return manifest_path, history_path


def run_selector(manifest: Path, history: Path, output: Path):
    previous = sys.argv
    try:
        sys.argv = [
            "select_timotion_checkpoint_on_interx_val.py",
            "--manifest", str(manifest),
            "--history", str(history),
            "--output", str(output),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            main()
    finally:
        sys.argv = previous


def test_selects_exact_minimum_validation_fid_and_rejects_metric_drift():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest, history = build_fixture(root)
        output = root / "selection.json"
        run_selector(manifest, history, output)
        selected = json.loads(output.read_text())
        assert selected["selected_epoch"] == 300
        assert selected["selected_step"] == 21300
        assert selected["selected_validation_fid"] == 0.25
        assert selected["test_metrics_used_for_selection"] is False

        payload = json.loads(history.read_text())
        payload["records"][0]["values"]["val/FID"] += 1.0
        history.write_text(json.dumps(payload))
        try:
            run_selector(manifest, history, output)
        except ValueError as error:
            assert "differs from scored JSON" in str(error)
        else:
            raise AssertionError("stale W&B-style validation value was accepted")


def test_post_training_queue_uses_selection_before_common50():
    training_queue = (ROOT / "tools/queue_timotion_fair_after_cosh.sh").read_text()
    post_queue = (ROOT / "tools/queue_timotion_post_training.sh").read_text()
    final_queue = (ROOT / "tools/queue_final_report_after_v3.sh").read_text()
    assert "exec bash tools/queue_timotion_post_training.sh" in training_queue
    assert post_queue.index(
        "tools/select_timotion_checkpoint_on_interx_val.py"
    ) < post_queue.index("tools/run_fair_interx_method.py")
    assert "tools/audit_timotion_validation_checkpoint_pair.py" in post_queue
    assert post_queue.index(
        "tools/audit_timotion_validation_checkpoint_pair.py"
    ) < post_queue.index("tools/replay_timotion_standard_validation.py")
    assert "state_dicts are bitwise identical" in post_queue
    assert "--metric-profile diffusion_regression_20260831_launch" in post_queue
    assert post_queue.count(
        "--metric-profile diffusion_regression_20260831_launch"
    ) == 2
    replay = (ROOT / "tools/replay_timotion_standard_validation.py").read_text()
    assert "], environment, cwd=TIMOTION_ROOT)" in replay
    assert 'TIMOTION_FINALIZE_RESUME_FROM:-start' in post_queue
    assert 'resume_from}" == "replay"' in post_queue
    assert "skipping completed audits" in post_queue
    assert 'resume_from}" == "fair_test"' in post_queue
    assert "skipping validation replay" in post_queue
    driver = (ROOT / "tools/run_fair_interx_method.py").read_text()
    assert "'transformer' if args.method == 'hhc' else 'model'" in driver
    assert '--selection-manifest "${selection}"' in post_queue
    assert "output/timotion_interx_val_selection.json" in final_queue


if __name__ == "__main__":
    test_selects_exact_minimum_validation_fid_and_rejects_metric_drift()
    test_post_training_queue_uses_selection_before_common50()
