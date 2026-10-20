#!/usr/bin/env python3
"""Bind shape-augmentation arms to a promoted original-38 HHC baseline."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSESSMENT = ROOT / "output/hhc_original38_token_window_control_assessment.json"
SELECTION = ROOT / "output/hhc_original38_token_window_control_300k_selection.json"
TRAINING_AUDIT = ROOT / "output/hhc_original38_token_window_control_training_contract_audit.json"
WANDB_AUDIT = ROOT / "output/hhc_original38_token_window_control_300k_wandb_history_audit.json"
OLD_SELECTION = "output/hhc_formal_raw12288_repaired_mgt_selection.json"
NEW_SELECTION = "output/hhc_original38_token_window_control_300k_selection.json"
OLD_NO_CONTACT = "hhc_formal_raw12288_repaired_mgt_common50_no_contact"
NEW_NO_CONTACT = "hhc_original38_token_window_control_300k_common50_no_contact"
OLD_CONTACT = "hhc_formal_raw12288_repaired_mgt_common50_contact"
NEW_CONTACT = "hhc_original38_token_window_control_300k_common50_contact"
FILES = {
    "queue": ROOT / "tools/queue_interx_shape_augmentation_gpu1.sh",
    "audit": ROOT / "tools/audit_shape_augmentation_launch_contract.py",
    "results": ROOT / "tools/build_shape_augmentation_results.py",
    "tests": ROOT / "tests/test_paper_consistency.py",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def replace_exact(source: str, old: str, new: str, expected: int, label: str) -> str:
    count = source.count(old)
    if count != expected:
        raise ValueError(f"{label}: expected {expected}, found {count}")
    return source.replace(old, new)


def transform(sources: dict[str, str]) -> dict[str, str]:
    after = dict(sources)
    expected_counts = {
        "queue": (1, 1, 1, 1),
        "audit": (1, 1, 1, 1),
        "results": (0, 1, 1, 1),
        # The test module also contains paper_50 in independent token-window
        # and loss-control regressions.  Only its shape-launch CLI assertion
        # belongs to this promotion.
        "tests": (1, 1, 1, 0),
    }
    for key, counts in expected_counts.items():
        selection_count, no_count, contact_count, token_count = counts
        if selection_count:
            after[key] = replace_exact(
                after[key], OLD_SELECTION, NEW_SELECTION, selection_count,
                f"{key} selection",
            )
        after[key] = replace_exact(
            after[key], OLD_NO_CONTACT, NEW_NO_CONTACT, no_count,
            f"{key} no-contact baseline",
        )
        after[key] = replace_exact(
            after[key], OLD_CONTACT, NEW_CONTACT, contact_count,
            f"{key} contact baseline",
        )
        if token_count:
            after[key] = replace_exact(
                after[key], "paper_50", "original_mmm_38", token_count,
                f"{key} token contract",
            )
    after["tests"] = replace_exact(
        after["tests"],
        "'--training-token-length-contract paper_50',",
        "'--training-token-length-contract original_mmm_38',",
        1,
        "tests shape token contract assertion",
    )
    return after


def atomic_write(path: Path, source: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(source, encoding="utf-8")
    os.replace(temporary, path)


def artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/promoted_original38_shape_baseline_application.json",
    )
    args = parser.parse_args()

    assessment = json.loads(ASSESSMENT.read_text(encoding="utf-8"))
    if assessment.get("decision") != "promote_original_mmm_38_window":
        raise ValueError("original-38 window was not promoted")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    selected_fid = float(selection.get("selected_validation_fid", math.inf))
    if (
        not selection.get("run_complete")
        or selection.get("expected_final_step") != 300000
        or selection.get("test_metrics_used_for_selection") is not False
        or not math.isfinite(selected_fid)
        or selected_fid > 0.4
    ):
        raise ValueError("original-38 300k selection is incomplete")
    training_audit = json.loads(TRAINING_AUDIT.read_text(encoding="utf-8"))
    if training_audit.get("status") not in {
        "pass", "pass_with_disclosed_exceptions"
    }:
        raise ValueError("original-38 paper training-contract audit failed")
    wandb_audit = json.loads(WANDB_AUDIT.read_text(encoding="utf-8"))
    if (
        wandb_audit.get("schema")
        != "hhc_wandb_online_metric_history_audit_v1"
        or wandb_audit.get("status") != "pass"
        or wandb_audit.get("expected_final_step") != 300000
        or not all(wandb_audit.get("checks", {}).values())
    ):
        raise ValueError("original-38 online W&B history audit failed")
    for label in (NEW_NO_CONTACT, NEW_CONTACT):
        summary = ROOT / "output/fair_eval" / label / "summary.json"
        if not summary.is_file():
            raise FileNotFoundError(summary)

    before = {key: path.read_text(encoding="utf-8") for key, path in FILES.items()}
    already = NEW_SELECTION in before["queue"] and "original_mmm_38" in before["queue"]
    after = before if already else transform(before)
    for key in ("audit", "results", "tests"):
        ast.parse(after[key], filename=str(FILES[key]))
    if args.apply and not already:
        for key, path in FILES.items():
            atomic_write(path, after[key])

    payload = {
        "schema": "hhc_promoted_original38_shape_baseline_application_v1",
        "status": "already_applied" if already else "applied" if args.apply else "check_pass",
        "write_requested": bool(args.apply),
        "assessment": artifact(ASSESSMENT),
        "selection": artifact(SELECTION),
        "training_contract_audit": artifact(TRAINING_AUDIT),
        "wandb_history_audit": artifact(WANDB_AUDIT),
        "files": {
            key: {
                "path": str(path),
                "before_sha256": sha256_bytes(before[key].encode()),
                "after_sha256": sha256_bytes(after[key].encode()),
                "changed": before[key] != after[key],
            }
            for key, path in FILES.items()
        },
        "implementation": artifact(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
