#!/usr/bin/env python3
"""Recover pointwise historical validation metrics hidden by best-so-far W&B logging."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path


BLOCK = re.compile(
    r"Eva\. Iter (?P<step>\d+)\s*:,.*?"
    r"FID\.\s*(?P<fid>[0-9.eE+-]+)\s*,.*?"
    r"Diversity Real\.\s*(?P<diversity_real>[0-9.eE+-]+)\s*,.*?"
    r"Diversity\.\s*(?P<diversity>[0-9.eE+-]+)\s*,.*?"
    r"R_precision_real\.\s*\[(?P<r_real>[^]]+)\]\s*,.*?"
    r"R_precision\.\s*\[(?P<r>[^]]+)\]\s*,.*?"
    r"matching_score_real\.\s*(?P<matching_real>[0-9.eE+-]+)\s*,.*?"
    r"matching_score_pred\.\s*(?P<matching>[0-9.eE+-]+)\s*,",
    re.DOTALL,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_vector(value: str) -> list[float]:
    return [float(item) for item in value.split()]


def parse_blocks(text: str) -> dict[int, dict]:
    rows = {}
    for match in BLOCK.finditer(text.replace("\\r", "\n").replace("\\n", "\n")):
        step = int(match.group("step"))
        r = parse_vector(match.group("r"))
        r_real = parse_vector(match.group("r_real"))
        if len(r) != 3 or len(r_real) != 3:
            raise ValueError(f"step {step}: malformed R-Precision vector")
        rows[step] = {
            "checkpoint_step": step,
            "FID": float(match.group("fid")),
            "Diversity": float(match.group("diversity")),
            "R_precision_top1": r[0],
            "R_precision_top2": r[1],
            "R_precision_top3": r[2],
            "Matching": float(match.group("matching")),
            "GT_Diversity": float(match.group("diversity_real")),
            "GT_R_precision_top1": r_real[0],
            "GT_R_precision_top2": r_real[1],
            "GT_R_precision_top3": r_real[2],
            "GT_Matching": float(match.group("matching_real")),
        }
    return rows


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-output-log", type=Path, required=True)
    parser.add_argument("--replay-validation-audit", type=Path, required=True)
    parser.add_argument(
        "--replay-output-log", type=Path,
        help=(
            "Preferred frozen v7 replay stdout log containing pointwise "
            "validation values that best-so-far W&B fields can hide."
        ),
    )
    parser.add_argument(
        "--codex-session", type=Path,
        help=(
            "Optional append-only Codex JSONL containing a captured replay "
            "stdout block. It is supporting evidence only when the preferred "
            "frozen replay stdout log is supplied."
        ),
    )
    parser.add_argument("--candidate-comparison", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=40000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    historical_rows = parse_blocks(
        args.historical_output_log.read_text(encoding="utf-8", errors="replace")
    )
    if args.checkpoint_step not in historical_rows:
        raise ValueError("requested checkpoint is absent from historical output.log")
    historical = historical_rows[args.checkpoint_step]

    replay = json.loads(args.replay_validation_audit.read_text(encoding="utf-8"))
    comparison = json.loads(args.candidate_comparison.read_text(encoding="utf-8"))
    wandb_step = args.checkpoint_step - 1
    comparison_row = next(
        row for row in comparison.get("comparisons", [])
        if row.get("trainer/global_step") == wandb_step
    )
    replay_row = next(
        row for row in replay.get("comparisons", [])
        if row.get("trainer/global_step") == wandb_step
    )

    # A v7 replay stdout observation may also exist in the append-only Codex
    # JSONL. Find the exact command/output pair and retain it as optional
    # corroboration rather than making conversational state a required input.
    replay_stdout_rows = []
    target_marker = f"Eva. Iter {args.checkpoint_step}"
    log_marker = "gpu1_mgt_known_good_historical_contract_v7_20260829.log"
    if args.codex_session is not None:
        with args.codex_session.open(
            "r", encoding="utf-8", errors="replace"
        ) as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                item = record.get("payload", {}).get("item", {})
                command = " ".join(item.get("command", []))
                stdout = item.get("stdout", "")
                is_direct_log_observation = (
                    log_marker in command
                    and "recover_historical_mgt_current_validation.py"
                    not in command
                )
                if is_direct_log_observation and target_marker in stdout:
                    parsed = parse_blocks(stdout)
                    if args.checkpoint_step in parsed:
                        replay_stdout_rows.append({
                            "session_line": line_number,
                            "session_line_sha256": hashlib.sha256(
                                line.encode()
                            ).hexdigest(),
                            "timestamp": record.get("timestamp"),
                            "command": command,
                            "stdout": stdout,
                            "stdout_sha256": hashlib.sha256(
                                stdout.encode()
                            ).hexdigest(),
                            "metrics": parsed[args.checkpoint_step],
                        })

    replay_log_current = None
    if args.replay_output_log is not None:
        replay_log_rows = parse_blocks(
            args.replay_output_log.read_text(
                encoding="utf-8", errors="replace"
            )
        )
        if args.checkpoint_step not in replay_log_rows:
            raise ValueError(
                "requested checkpoint is absent from frozen v7 replay log"
            )
        replay_log_current = replay_log_rows[args.checkpoint_step]

    names = (
        "FID", "Diversity", "R_precision_top1", "R_precision_top2",
        "R_precision_top3", "Matching",
    )
    historical_to_candidate = {
        "FID": "val/FID",
        "Diversity": "val/Diversity",
        "R_precision_top1": "val/TOP1",
        "R_precision_top2": "val/TOP2",
        "R_precision_top3": "val/TOP3",
        "Matching": "val/Matching",
    }
    if replay_log_current is not None:
        replay_current = replay_log_current
        replay_current_source = "frozen_v7_replay_output_log"
    elif replay_stdout_rows:
        replay_current = replay_stdout_rows[0]["metrics"]
        replay_current_source = "captured_codex_stdout"
    else:
        raise ValueError(
            "pointwise v7 replay stdout is unavailable; best-so-far W&B "
            "fields cannot recover current R-Precision"
        )
    historical_matches_replay_stdout = all(
        math.isclose(historical[name], replay_current[name], abs_tol=5e-8)
        for name in names
    )
    captured_stdout_agrees = all(
        all(
            math.isclose(
                row["metrics"][name], historical[name], abs_tol=5e-8
            )
            for name in names
        )
        for row in replay_stdout_rows
    )
    checks = {
        "replay_validation_audit_passes": replay.get("status") == "pass",
        "replay_wandb_exact_match": replay.get("maximum_absolute_difference") == 0.0,
        "candidate_comparison_complete": comparison.get("status") == "comparison_complete",
        "historical_output_matches_pointwise_v7_replay_stdout": (
            historical_matches_replay_stdout
        ),
        "captured_v7_stdout_agrees_when_available": captured_stdout_agrees,
        "historical_fid_rounding_matches_wandb": math.isclose(
            historical["FID"], float(comparison_row["reference"]["val/FID"]),
            abs_tol=5e-5,
        ),
    }
    differences = {}
    for historical_name, candidate_name in historical_to_candidate.items():
        candidate_value = float(comparison_row["candidate"][candidate_name])
        difference = candidate_value - historical[historical_name]
        differences[historical_name] = {
            "historical_current": historical[historical_name],
            "candidate_current": candidate_value,
            "signed_candidate_minus_historical": difference,
            "absolute": abs(difference),
        }
    status = "pass" if all(checks.values()) else "fail"
    payload = {
        "schema": "hhc_historical_current_validation_recovery_v1",
        "status": status,
        "checkpoint_step": args.checkpoint_step,
        "checks": checks,
        "historical_current_metrics": historical,
        "candidate_current_metrics": {
            name: float(comparison_row["candidate"][candidate_name])
            for name, candidate_name in historical_to_candidate.items()
        },
        "differences": differences,
        "role": (
            "training-time historical-sampler diagnostic only; formal model selection "
            "and claims still require frozen official validation/test regeneration"
        ),
        "replay_stdout_evidence": replay_stdout_rows,
        "replay_current_source": replay_current_source,
        "replay_evidence_mode": "+".join(filter(None, (
            "historical_output",
            "replay_wandb",
            "frozen_replay_output" if replay_log_current is not None else None,
            "optional_codex_stdout" if replay_stdout_rows else None,
        ))),
        "artifacts": {
            "historical_output_log": artifact(args.historical_output_log),
            "replay_validation_audit": artifact(args.replay_validation_audit),
            "candidate_comparison": artifact(args.candidate_comparison),
            "implementation": artifact(Path(__file__)),
        },
    }
    if args.codex_session is not None:
        payload["artifacts"]["codex_session_live_source"] = {
            "path": str(args.codex_session.resolve()),
            "bytes_at_scan": args.codex_session.stat().st_size,
            "mtime_ns_at_scan": args.codex_session.stat().st_mtime_ns,
            "captured_target_rows": len(replay_stdout_rows),
            "sha256_omitted_reason": (
                "append-only active session; any selected JSON lines and their "
                "SHA-256 values are materialized in replay_stdout_evidence"
            ),
        }
    if args.replay_output_log is not None:
        payload["artifacts"]["replay_output_log"] = artifact(
            args.replay_output_log
        )
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
