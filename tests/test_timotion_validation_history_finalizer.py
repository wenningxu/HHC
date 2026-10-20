#!/usr/bin/env python3
"""Regression tests for TiMotion validation-history finalization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_finalizer_binds_complete_history_before_selection() -> None:
    with tempfile.TemporaryDirectory() as temporary_name:
        temporary = Path(temporary_name)
        final_checkpoint = temporary / "last.ckpt"
        final_checkpoint.write_bytes(b"full state")
        manifest = temporary / "training_manifest.json"
        write_json(manifest, {
            "status": "complete",
            "final_global_step": 106500,
            "last_checkpoint": str(final_checkpoint),
            "last_checkpoint_sha256": sha256_file(final_checkpoint),
            "batch_emulation": {
                "expected_final_optimizer_updates": 106500,
                "optimizer_updates_per_epoch": 71,
            },
            "paper_protocol": {"epochs": 1500},
            "observability": {
                "semantic_validation_split": "official Inter-X val (544 samples)"
            },
            "official_split_counts": {"val": 570},
        })

        records = []
        for epoch in range(100, 1501, 100):
            step = epoch * 71
            checkpoint = temporary / f"step_{step:09d}_weights.ckpt"
            generations = temporary / f"step_{step:09d}_main.h5"
            metrics = temporary / f"step_{step:09d}.json"
            checkpoint.write_bytes(f"checkpoint {step}".encode())
            generations.write_bytes(f"generation {step}".encode())
            values = {
                "val/FID": float(epoch / 100),
                "val/R_precision_top1": 0.1,
                "val/R_precision_top2": 0.2,
                "val/R_precision_top3": 0.3,
                "val/Matching": 4.0,
                "val/Diversity": 8.0,
            }
            write_json(metrics, {
                "status": "complete",
                "samples": 544,
                "generation_request_hash_verified": True,
                "generation_length_policy": "requested_lengths",
                "main_generations_sha256": sha256_file(generations),
                "fid": values["val/FID"],
                "top1": values["val/R_precision_top1"],
                "top2": values["val/R_precision_top2"],
                "top3": values["val/R_precision_top3"],
                "matching": values["val/Matching"],
                "diversity": values["val/Diversity"],
            })
            records.append({
                "epoch": epoch,
                "step": step,
                "samples": 544,
                "test_set_used": False,
                "checkpoint": artifact(checkpoint),
                "generations": artifact(generations),
                "metrics": artifact(metrics),
                "values": values,
            })
        history = temporary / "official_validation_history.json"
        write_json(history, {
            "schema": "timotion_official_interx_validation_history_v1",
            "status": "running",
            "records": records,
        })

        premature_selection = temporary / "premature_selection.json"
        premature = subprocess.run([
            sys.executable,
            str(ROOT / "tools/select_timotion_checkpoint_on_interx_val.py"),
            "--manifest", str(manifest), "--history", str(history),
            "--output", str(premature_selection),
        ], cwd=ROOT, check=False, capture_output=True, text=True)
        assert premature.returncode != 0
        assert not premature_selection.exists()

        subprocess.run([
            sys.executable,
            str(ROOT / "tools/finalize_timotion_validation_history.py"),
            "--manifest", str(manifest), "--history", str(history),
        ], cwd=ROOT, check=True, capture_output=True, text=True)
        finalized = json.loads(history.read_text(encoding="utf-8"))
        assert finalized["status"] == "complete"
        assert finalized["final_global_step"] == 106500
        assert finalized["completed_validation_epochs"] == list(
            range(100, 1501, 100)
        )
        assert finalized["training_manifest"]["sha256"] == sha256_file(manifest)

        selection = temporary / "selection.json"
        subprocess.run([
            sys.executable,
            str(ROOT / "tools/select_timotion_checkpoint_on_interx_val.py"),
            "--manifest", str(manifest), "--history", str(history),
            "--output", str(selection),
        ], cwd=ROOT, check=True, capture_output=True, text=True)
        selected = json.loads(selection.read_text(encoding="utf-8"))
        assert selected["status"] == "complete"
        assert selected["selected_epoch"] == 100
        assert selected["selected_step"] == 7100


if __name__ == "__main__":
    test_finalizer_binds_complete_history_before_selection()
    print({
        "test": "test_finalizer_binds_complete_history_before_selection",
        "status": "pass",
    })
