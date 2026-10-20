#!/usr/bin/env python3
"""Fail closed unless TiMotion post-training selection replays standard 6D."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "tools/queue_timotion_post_training.sh"
DRIVER = ROOT / "tools/run_fair_interx_method.py"
EXPORTER = ROOT / "tools/export_timotion_official_interx_standard.py"
REPLAY = ROOT / "tools/replay_timotion_standard_validation.py"
SELECTOR = ROOT / "tools/select_timotion_checkpoint_on_interx_val.py"
WANDB_AUDITOR = ROOT / "tools/audit_wandb_online_metric_history.py"
WANDB_POLICY = ROOT / "configs/online_wandb_training_policy.json"
OUTPUT = ROOT / "output/timotion_post_training_contract_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    queue = QUEUE.read_text(encoding="utf-8")
    driver = DRIVER.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    selector = SELECTOR.read_text(encoding="utf-8")
    required = (
        "tools/audit_timotion_cosh_adapter_equivalence.py",
        "--timotion-adapter tools/export_timotion_official_interx_standard.py",
        "tools/audit_timotion_generation_contract.py",
        "tools/replay_timotion_standard_validation.py",
        '--history "${standard_history}"',
        "tools/select_timotion_checkpoint_on_interx_val.py",
        "tools/run_fair_interx_method.py",
    )
    checks = {
        "all_required_post_steps_present": all(item in queue for item in required),
        "no_waiter_loop": all(
            marker not in queue for marker in ("while true", "sleep 30", "sleep 60")
        ),
        "replay_precedes_selection": queue.index(
            "tools/replay_timotion_standard_validation.py"
        ) < queue.index("tools/select_timotion_checkpoint_on_interx_val.py"),
        "selection_precedes_fair_test": queue.index(
            "tools/select_timotion_checkpoint_on_interx_val.py"
        ) < queue.index("tools/run_fair_interx_method.py"),
        "driver_uses_standard_exporter": (
            "tools/export_timotion_official_interx_standard.py" in driver
            and "'standard_row_concatenated_v1' if args.method == 'timotion'"
            in driver
        ),
        "replay_covers_all_15_epochs": (
            "list(range(100, 1501, 100))" in replay
            and '"test_set_used": False' in replay
            and '"--seed", "500123"' in replay
        ),
        "selector_requires_standard_contract": (
            'STANDARD_SCHEMA = "timotion_official_interx_standard_validation_history_v1"'
            in selector
            and 'STANDARD_CONTRACT = "standard_row_concatenated_v1"' in selector
            and 'metadata.get("formal_result_eligible") is not True' in selector
        ),
        "wandb_profile_is_bound_to_the_pre_expansion_launch": (
            "--metric-profile diffusion_regression_20260831_launch" in queue
            and '"diffusion_regression_20260831_launch"' in WANDB_POLICY.read_text(
                encoding="utf-8"
            )
            and '"run_id": "kdz5s3ra"' in WANDB_AUDITOR.read_text(
                encoding="utf-8"
            )
        ),
    }
    payload = {
        "schema": "hhc_timotion_post_training_contract_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "artifacts": {
            "queue": artifact(QUEUE),
            "driver": artifact(DRIVER),
            "formal_exporter": artifact(EXPORTER),
            "validation_replay": artifact(REPLAY),
            "selector": artifact(SELECTOR),
            "wandb_auditor": artifact(WANDB_AUDITOR),
            "wandb_policy": artifact(WANDB_POLICY),
            "implementation": artifact(Path(__file__)),
        },
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
