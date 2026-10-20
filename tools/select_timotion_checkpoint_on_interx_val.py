#!/usr/bin/env python3
"""Select TiMotion by the frozen official Inter-X validation FID only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


SCHEMA = "timotion_official_interx_validation_history_v1"
STANDARD_SCHEMA = "timotion_official_interx_standard_validation_history_v1"
STANDARD_CONTRACT = "standard_row_concatenated_v1"
OUTPUT_SCHEMA = "hhc_timotion_interx_validation_selection_v1"
METRIC_KEYS = {
    "val/FID": "fid",
    "val/R_precision_top1": "top1",
    "val/R_precision_top2": "top2",
    "val/R_precision_top3": "top3",
    "val/Matching": "matching",
    "val/Diversity": "diversity",
}


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
    if not path.is_file():
        raise ValueError(f"{label}: missing artifact {path}")
    expected_bytes = int(record.get("bytes", -1))
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{label}: byte count changed for {path}")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label}: SHA-256 changed for {path}")
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


def validate_record(record: dict, expected_epoch: int, expected_step: int,
                    validation_contract: str = "historical_repository_decoder") -> dict:
    label = f"TiMotion validation epoch {expected_epoch}"
    if (
        int(record.get("epoch", -1)) != expected_epoch
        or int(record.get("step", -1)) != expected_step
        or int(record.get("samples", -1)) != 544
        or record.get("test_set_used") is not False
    ):
        raise ValueError(f"{label}: split/epoch/step contract changed")
    checkpoint = validate_artifact(record.get("checkpoint", {}), f"{label}/checkpoint")
    generations = validate_artifact(
        record.get("generations", {}), f"{label}/generations"
    )
    metrics_path = validate_artifact(record.get("metrics", {}), f"{label}/metrics")
    expected_checkpoint_name = f"step_{expected_step:09d}_weights.ckpt"
    expected_generation_name = f"step_{expected_step:09d}_main.h5"
    if checkpoint.name != expected_checkpoint_name:
        raise ValueError(f"{label}: checkpoint name does not bind its step")
    if generations.name != expected_generation_name:
        raise ValueError(f"{label}: generation name does not bind its step")

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        metrics_payload.get("status") != "complete"
        or int(metrics_payload.get("samples", -1)) != 544
        or metrics_payload.get("generation_request_hash_verified") is not True
        or metrics_payload.get("generation_length_policy") != "requested_lengths"
    ):
        raise ValueError(f"{label}: metrics are not a complete frozen request score")
    if metrics_payload.get("main_generations_sha256") != record["generations"]["sha256"]:
        raise ValueError(f"{label}: metrics do not bind the generated HDF5")
    if validation_contract == STANDARD_CONTRACT:
        metadata = metrics_payload.get("main_generation_metadata", {})
        expected_exporter = (
            Path(__file__).resolve().parent
            / "export_timotion_official_interx_standard.py"
        )
        expected_base = (
            Path(__file__).resolve().parent
            / "export_timotion_official_interx.py"
        )
        if (
            record.get("validation_contract") != STANDARD_CONTRACT
            or metadata.get("formal_result_eligible") is not True
            or metadata.get("rotation6d_contract")
            != "a1=d6[...,0:3], a2=d6[...,3:6], Gram-Schmidt, row-stack"
            or metadata.get("exporter_sha256") != sha256_file(expected_exporter)
            or metadata.get("base_exporter_sha256") != sha256_file(expected_base)
        ):
            raise ValueError(f"{label}: standard rotation6D contract is absent")

    values = record.get("values", {})
    if set(values) != set(METRIC_KEYS):
        raise ValueError(f"{label}: validation metric set changed")
    for logged_key, scored_key in METRIC_KEYS.items():
        logged = values[logged_key]
        scored = metrics_payload.get(scored_key)
        if (
            not isinstance(logged, (int, float))
            or not isinstance(scored, (int, float))
            or not math.isfinite(float(logged))
            or float(logged) != float(scored)
        ):
            raise ValueError(f"{label}: {logged_key} differs from scored JSON")
    if float(values["val/FID"]) < 0:
        raise ValueError(f"{label}: invalid negative FID")
    return {
        "epoch": expected_epoch,
        "step": expected_step,
        "samples": 544,
        "test_set_used": False,
        "checkpoint": record["checkpoint"],
        "generations": record["generations"],
        "metrics": record["metrics"],
        "values": values,
        "validation_contract": validation_contract,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Completed TiMotion fair-training manifest.",
    )
    parser.add_argument(
        "--history", type=Path, required=True,
        help="Official validation history written by the training callback.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    batch = manifest.get("batch_emulation", {})
    paper = manifest.get("paper_protocol", {})
    final_step = int(batch.get("expected_final_optimizer_updates", -1))
    updates_per_epoch = int(batch.get("optimizer_updates_per_epoch", -1))
    if (
        manifest.get("status") != "complete"
        or int(manifest.get("final_global_step", -1)) != 106500
        or final_step != 106500
        or updates_per_epoch != 71
        or int(paper.get("epochs", -1)) != 1500
        or manifest.get("observability", {}).get(
            "semantic_validation_split"
        ) != "official Inter-X val (544 samples)"
        or int(manifest.get("official_split_counts", {}).get("val", -1)) != 570
    ):
        raise ValueError("TiMotion training manifest is not the complete fair run")
    last_checkpoint = Path(manifest.get("last_checkpoint", "")).resolve()
    if (
        not last_checkpoint.is_file()
        or sha256_file(last_checkpoint) != manifest.get("last_checkpoint_sha256")
    ):
        raise ValueError("TiMotion final full-state checkpoint is absent or stale")

    history = json.loads(args.history.read_text(encoding="utf-8"))
    history_schema = history.get("schema")
    if (
        history_schema not in (SCHEMA, STANDARD_SCHEMA)
        or history.get("status") != "complete"
        or int(history.get("final_global_step", -1)) != 106500
    ):
        raise ValueError(f"unsupported TiMotion validation history: {history.get('schema')}")
    history_manifest = validate_artifact(
        history.get("training_manifest", {}),
        "TiMotion validation history/training manifest",
    )
    history_finalizer = validate_artifact(
        history.get("implementation", {}),
        "TiMotion validation history/finalizer",
    )
    validation_contract = (
        STANDARD_CONTRACT
        if history_schema == STANDARD_SCHEMA
        else "historical_repository_decoder"
    )
    expected_history_implementation = (
        Path(__file__).resolve().parent
        / (
            "replay_timotion_standard_validation.py"
            if validation_contract == STANDARD_CONTRACT
            else "finalize_timotion_validation_history.py"
        )
    )
    if (
        history_manifest != args.manifest.resolve()
        or history.get("training_manifest", {}).get("sha256")
        != sha256_file(args.manifest)
        or history_finalizer != expected_history_implementation
        or (
            validation_contract == STANDARD_CONTRACT
            and history.get("validation_contract") != STANDARD_CONTRACT
        )
    ):
        raise ValueError("TiMotion validation history is not bound to this completed run")
    records = history.get("records", [])
    by_epoch = {}
    for record in records:
        epoch = int(record.get("epoch", -1))
        if epoch in by_epoch:
            raise ValueError(f"duplicate TiMotion validation epoch: {epoch}")
        by_epoch[epoch] = record
    expected_epochs = list(range(100, 1501, 100))
    if history.get("completed_validation_epochs") != expected_epochs:
        raise ValueError("TiMotion finalized validation epoch inventory changed")
    if sorted(by_epoch) != expected_epochs:
        raise ValueError(
            "TiMotion validation history is incomplete: "
            f"expected={expected_epochs}, observed={sorted(by_epoch)}"
        )
    candidates = [
        validate_record(
            by_epoch[epoch], epoch, epoch * updates_per_epoch,
            validation_contract=validation_contract,
        )
        for epoch in expected_epochs
    ]
    selected = min(
        candidates,
        key=lambda row: (float(row["values"]["val/FID"]), int(row["step"])),
    )
    payload = {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "purpose": "held-out official Inter-X validation checkpoint selection only",
        "selection_metric": "minimum official validation FID",
        "test_metrics_used_for_selection": False,
        "validation_samples": 544,
        "validation_seed": 500123,
        "validation_contract": validation_contract,
        "candidate_epochs": expected_epochs,
        "candidate_steps": [row["step"] for row in candidates],
        "candidates": candidates,
        "selected_epoch": selected["epoch"],
        "selected_step": selected["step"],
        "selected_validation_fid": selected["values"]["val/FID"],
        "selected_checkpoint": selected["checkpoint"]["path"],
        "selected_checkpoint_sha256": selected["checkpoint"]["sha256"],
        "training_manifest": artifact(args.manifest),
        "validation_history": artifact(args.history),
        "final_full_state_checkpoint": artifact(last_checkpoint),
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
