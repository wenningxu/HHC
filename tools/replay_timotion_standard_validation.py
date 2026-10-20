#!/usr/bin/env python3
"""Regenerate every TiMotion validation candidate with the standard adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py


ROOT = Path(__file__).resolve().parents[1]
TIMOTION_ROOT = Path("/mnt/8T-HDD/wenning/TIMotion")
EXPORTER = ROOT / "tools/export_timotion_official_interx_standard.py"
HISTORICAL_EXPORTER = ROOT / "tools/export_timotion_official_interx.py"
SCORER = ROOT / "tools/score_fair_interx_main.py"
EQUIVALENCE = (
    ROOT / "output/timotion_cosh_adapter_equivalence_audit_standard_formal.json"
)
GENERATION_CONTRACT = ROOT / "output/timotion_generation_contract_audit_standard_formal.json"
SCHEMA = "timotion_official_interx_standard_validation_history_v1"
ROTATION_CONTRACT = "standard_row_concatenated_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_artifact(record: dict, label: str) -> Path:
    path = Path(record.get("path", "")).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise ValueError(f"{label}: artifact is absent or stale: {path}")
    return path


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(command, environment, *, cwd=ROOT) -> None:
    print(json.dumps({"command": command, "cwd": str(cwd)}), flush=True)
    subprocess.run(
        command, cwd=cwd, env=environment, check=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-history", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    historical = json.loads(args.history.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or int(manifest.get("final_global_step", -1)) != 106500
        or int(manifest.get("batch_emulation", {}).get(
            "optimizer_updates_per_epoch", -1
        )) != 71
    ):
        raise ValueError("TiMotion training is not complete at step 106500")
    if (
        historical.get("schema")
        != "timotion_official_interx_validation_history_v1"
        or historical.get("status") != "complete"
        or int(historical.get("final_global_step", -1)) != 106500
    ):
        raise ValueError("historical TiMotion validation inventory is incomplete")
    expected_epochs = list(range(100, 1501, 100))
    historical_records = {
        int(record["epoch"]): record for record in historical.get("records", [])
    }
    if sorted(historical_records) != expected_epochs:
        raise ValueError("historical validation checkpoint inventory changed")

    equivalence = json.loads(EQUIVALENCE.read_text(encoding="utf-8"))
    generation_contract = json.loads(
        GENERATION_CONTRACT.read_text(encoding="utf-8")
    )
    if (
        equivalence.get("status") != "pass"
        or not all(equivalence.get("checks", {}).values())
        or equivalence.get("sources", {}).get(
            "timotion_exporter", {}
        ).get("sha256") != sha256_file(EXPORTER)
        or generation_contract.get("status") != "pass"
        or generation_contract.get("representation", {}).get(
            "rotation6d_contract"
        ) != ROTATION_CONTRACT
        or generation_contract.get("sources", {}).get("exporter", {}).get(
            "sha256"
        ) != sha256_file(EXPORTER)
    ):
        raise ValueError("formal TiMotion standard-adapter gates are stale")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TIMOTION_STATS_PREFIX"] = "interx_global"
    records = []
    request_artifact = None
    for epoch in expected_epochs:
        old = historical_records[epoch]
        step = epoch * 71
        if int(old.get("step", -1)) != step:
            raise ValueError(f"epoch {epoch}: historical step changed")
        checkpoint = validate_artifact(
            old.get("checkpoint", {}), f"epoch {epoch}/checkpoint"
        )
        old_metrics_path = validate_artifact(
            old.get("metrics", {}), f"epoch {epoch}/historical metrics"
        )
        old_metrics = json.loads(old_metrics_path.read_text(encoding="utf-8"))
        request = Path(old_metrics.get("requests", "")).resolve()
        if (
            not request.is_file()
            or sha256_file(request) != old_metrics.get("requests_sha256")
        ):
            raise ValueError(f"epoch {epoch}: frozen validation request drifted")
        current_request_artifact = artifact(request)
        if request_artifact is None:
            request_artifact = current_request_artifact
        elif request_artifact != current_request_artifact:
            raise ValueError("validation epochs use different frozen requests")

        generation = output_root / f"step_{step:09d}_main.h5"
        metrics = output_root / f"step_{step:09d}.json"
        run([
            sys.executable, str(EXPORTER),
            "--checkpoint", str(checkpoint),
            "--requests", str(request),
            "--output", str(generation),
            "--gpu", str(args.gpu),
            "--batch-size", str(args.batch_size),
            "--repeats", "1", "--seed", "500123", "--resume",
        ], environment, cwd=TIMOTION_ROOT)
        run([
            sys.executable, str(SCORER),
            "--requests", str(request),
            "--main-generations", str(generation),
            "--output", str(metrics),
            "--device", f"cuda:{args.gpu}",
            "--batch-size", str(args.batch_size),
        ], environment)
        scored = json.loads(metrics.read_text(encoding="utf-8"))
        metadata = scored.get("main_generation_metadata", {})
        if (
            scored.get("status") != "complete"
            or int(scored.get("samples", -1)) != 544
            or metadata.get("rotation6d_contract") != (
                "a1=d6[...,0:3], a2=d6[...,3:6], Gram-Schmidt, row-stack"
            )
            or metadata.get("formal_result_eligible") is not True
            or metadata.get("exporter_sha256") != sha256_file(EXPORTER)
            or metadata.get("base_exporter_sha256")
            != sha256_file(HISTORICAL_EXPORTER)
        ):
            raise ValueError(f"epoch {epoch}: corrected score contract failed")
        values = {
            "val/FID": scored["fid"],
            "val/R_precision_top1": scored["top1"],
            "val/R_precision_top2": scored["top2"],
            "val/R_precision_top3": scored["top3"],
            "val/Matching": scored["matching"],
            "val/Diversity": scored["diversity"],
        }
        records.append({
            "epoch": epoch,
            "step": step,
            "samples": 544,
            "test_set_used": False,
            "validation_contract": ROTATION_CONTRACT,
            "checkpoint": artifact(checkpoint),
            "generations": artifact(generation),
            "metrics": artifact(metrics),
            "values": values,
            "historical_buggy_metrics": artifact(old_metrics_path),
        })
        atomic_json(args.output_history, {
            "schema": SCHEMA,
            "status": "running",
            "validation_contract": ROTATION_CONTRACT,
            "completed_validation_epochs": [row["epoch"] for row in records],
            "records": records,
            "training_manifest": artifact(args.manifest),
            "historical_validation_history": artifact(args.history),
            "validation_requests": request_artifact,
            "standard_exporter": artifact(EXPORTER),
            "historical_exporter": artifact(HISTORICAL_EXPORTER),
            "adapter_equivalence_audit": artifact(EQUIVALENCE),
            "generation_contract_audit": artifact(GENERATION_CONTRACT),
            "implementation": artifact(Path(__file__)),
        })

    payload = json.loads(args.output_history.read_text(encoding="utf-8"))
    payload.update({
        "status": "complete",
        "final_global_step": 106500,
        "completed_validation_epochs": expected_epochs,
    })
    atomic_json(args.output_history, payload)
    print(json.dumps({
        "status": "complete",
        "validations": len(records),
        "output_history": str(args.output_history.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
