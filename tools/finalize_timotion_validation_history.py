#!/usr/bin/env python3
"""Finalize TiMotion's official validation history after training completes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.select_timotion_checkpoint_on_interx_val import validate_record


SCHEMA = "timotion_official_interx_validation_history_v1"


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


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    batch = manifest.get("batch_emulation", {})
    if (
        manifest.get("status") != "complete"
        or int(manifest.get("final_global_step", -1)) != 106500
        or int(batch.get("expected_final_optimizer_updates", -1)) != 106500
        or int(batch.get("optimizer_updates_per_epoch", -1)) != 71
    ):
        raise ValueError("TiMotion training manifest is not complete at step 106500")

    history = json.loads(args.history.read_text(encoding="utf-8"))
    if history.get("schema") != SCHEMA:
        raise ValueError(f"unsupported validation history: {history.get('schema')}")
    records = history.get("records", [])
    by_epoch = {}
    for record in records:
        epoch = int(record.get("epoch", -1))
        if epoch in by_epoch:
            raise ValueError(f"duplicate TiMotion validation epoch: {epoch}")
        by_epoch[epoch] = record
    expected_epochs = list(range(100, 1501, 100))
    if sorted(by_epoch) != expected_epochs:
        raise ValueError(
            "TiMotion validation history is incomplete: "
            f"expected={expected_epochs}, observed={sorted(by_epoch)}"
        )
    for epoch in expected_epochs:
        validate_record(by_epoch[epoch], epoch, epoch * 71)

    history.update({
        "status": "complete",
        "final_global_step": 106500,
        "completed_validation_epochs": expected_epochs,
        "training_manifest": artifact(args.manifest),
        "implementation": artifact(Path(__file__)),
    })
    atomic_json(args.history, history)
    print(json.dumps({
        "history": str(args.history.resolve()),
        "status": "complete",
        "validations": len(expected_epochs),
        "final_global_step": 106500,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
