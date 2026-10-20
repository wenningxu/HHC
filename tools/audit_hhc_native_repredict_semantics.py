#!/usr/bin/env python3
"""Bind HHC token re-prediction semantics to the pre-launch MMM source."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE = ROOT / (
    "output/historical_mmm_local_history/"
    "1770947689489_set-424054_change-424053_ContentChange.py"
)
DEFAULT_RECOVERY = ROOT / "output/historical_mmm_local_history/recovery_manifest.tsv"
DEFAULT_CURRENT = ROOT / "models/two_person_transformer.py"
LOW_FID_RUN_START_UTC = dt.datetime(
    2026, 2, 13, 1, 55, 15, tzinfo=dt.timezone.utc
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def function_source(source: str, name: str, required_arg: str) -> str:
    tree = ast.parse(source)
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = {argument.arg for argument in node.args.args}
            if node.name == name and required_arg in args:
                candidates.append(node)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one {name} with {required_arg}, got {len(candidates)}"
        )
    return ast.get_source_segment(source, candidates[0]) or ""


def recovery_record(path: Path, native: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"))) for line in lines[1:] if line]
    matches = [row for row in rows if row.get("output") == native.name]
    if len(matches) != 1:
        raise RuntimeError("native local-history file is not uniquely recorded")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--recovery-manifest", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--current-source", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/hhc_native_repredict_semantics_audit.json",
    )
    args = parser.parse_args()
    native_source = args.native_source.read_text(encoding="utf-8")
    current_source = args.current_source.read_text(encoding="utf-8")
    native_sampler = function_source(
        native_source, "sample_intra_cas", "clip_feature"
    )
    current_repredict = function_source(
        current_source, "sample_intra_cas_repredict", "clip_feature"
    )
    current_fixed = function_source(
        current_source, "sample_intra_cas", "clip_feature"
    )
    record = recovery_record(args.recovery_manifest, args.native_source)
    timestamp_ms = int(record["timestamp_ms"])
    native_time = dt.datetime.fromtimestamp(
        timestamp_ms / 1000, tz=dt.timezone.utc
    )
    seconds_before_launch = (
        LOW_FID_RUN_START_UTC - native_time
    ).total_seconds()
    native_hash = sha256_file(args.native_source)
    checks = {
        "recovery_hash_matches": record["sha256"] == native_hash,
        "recovery_byte_size_matches": int(record["bytes"]) == args.native_source.stat().st_size,
        "source_immediately_precedes_low_fid_launch": 0 < seconds_before_launch < 60,
        "native_default_is_ten_outer_steps": "max_steps=10" in native_sampler,
        "native_cascade_is_root_body_hand": (
            "cascade_order = [root_ch_idx, body_ch_idx, hand_ch_idx]"
            in native_sampler
        ),
        "native_repredicts_all_valid_positions": (
            "new_ids_ch = torch.where(is_mask_ch, mask_id, pred_ids_ch)"
            in native_sampler
        ),
        "native_writes_full_channel_prediction": (
            "ids[:, :, ch_idx] = new_ids_ch" in native_sampler
        ),
        "current_diagnostic_reproduces_update_support": (
            "ids[:, :, channel] = torch.where(" in current_repredict
            and "torch.full_like(predictions, mask_id)" in current_repredict
            and "predictions," in current_repredict
        ),
        "current_formal_sampler_presently_differs": (
            "newly_fixed = was_masked & ~keep_masked" in current_fixed
            and "updated[newly_fixed] = predictions[newly_fixed]" in current_fixed
        ),
    }
    passed = all(checks.values())
    payload = {
        "schema": "hhc_native_repredict_semantics_audit_v1",
        "status": "pass" if passed else "mismatch",
        "checks": checks,
        "native_source_time_utc": native_time.isoformat(),
        "known_low_fid_run_start_utc": LOW_FID_RUN_START_UTC.isoformat(),
        "seconds_before_known_low_fid_run_start": seconds_before_launch,
        "native_sampler_contract": {
            "dispatch": "sample_intra_cas",
            "default_outer_steps": 10,
            "cascade_order": "root,body,hands",
            "token_update_support": "all valid positions re-predicted each outer iteration",
        },
        "diagnostic_contract": {
            "outer_steps": 50,
            "temperature": 0.5,
            "cfg_scale": 3.5,
            "contact_guidance_scale": 1.0,
            "changed_factor_relative_to_current_formal_sampler": "token_update_support_only",
            "promotion_policy": "requires frozen 544-sample validation comparison",
        },
        "evidence": {
            "native_source": artifact(args.native_source),
            "recovery_manifest": artifact(args.recovery_manifest),
            "current_source": artifact(args.current_source),
            "implementation": artifact(Path(__file__)),
            "recovery_record": record,
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
