#!/usr/bin/env python3
"""Stop the formal CoShMDM run at a clean milestone or a hard wall-clock cap.

The monitor is intentionally external to the trainer: it does not modify the
frozen CoShMDM source tree or the numerical training contract.  It records the
user-imposed resource cap in a machine-readable manifest before signalling the
trainer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import time


def parse_time(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("deadline must include a UTC offset")
    return parsed


def command_line(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return None


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--expected-command", required=True)
    parser.add_argument("--queue-pid", type=int)
    parser.add_argument("--expected-queue-command")
    parser.add_argument("--deadline", type=parse_time, required=True)
    parser.add_argument("--preferred-checkpoint", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--checkpoint-stable-seconds", type=float, default=45.0)
    args = parser.parse_args()

    started = dt.datetime.now().astimezone()
    payload = {
        "schema": "hhc_coshmdm_user_training_cap_v1",
        "status": "monitoring",
        "trainer_pid": args.pid,
        "expected_command_substring": args.expected_command,
        "monitor_started_at": started.isoformat(),
        "hard_deadline": args.deadline.isoformat(),
        "maximum_additional_wallclock_hours": (
            args.deadline - started
        ).total_seconds() / 3600.0,
        "preferred_terminal_optimizer_step": 200000,
        "preferred_checkpoint": str(args.preferred_checkpoint.resolve()),
        "policy": "stop after stable 200k checkpoint, otherwise stop at hard deadline",
    }
    if args.queue_pid is not None:
        if not args.expected_queue_command:
            parser.error("--expected-queue-command is required with --queue-pid")
        payload.update({
            "superseded_queue_pid": args.queue_pid,
            "expected_superseded_queue_command_substring": (
                args.expected_queue_command
            ),
            "queue_policy": (
                "terminate the pre-cap waiter before trainer interrupt so it "
                "cannot auto-resume toward 568k"
            ),
        })
    args.status.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.status, payload)

    stable_since: float | None = None
    stable_signature: tuple[int, int] | None = None
    reason: str | None = None
    while reason is None:
        cmdline = command_line(args.pid)
        if cmdline is None:
            payload.update({
                "status": "trainer_exited_before_monitor_signal",
                "observed_at": dt.datetime.now().astimezone().isoformat(),
            })
            atomic_json(args.status, payload)
            return
        if args.expected_command not in cmdline:
            payload.update({
                "status": "failed_closed_pid_identity_mismatch",
                "observed_command": cmdline,
                "observed_at": dt.datetime.now().astimezone().isoformat(),
            })
            atomic_json(args.status, payload)
            raise SystemExit(2)

        try:
            stat = args.preferred_checkpoint.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if signature != stable_signature:
                stable_signature = signature
                stable_since = time.monotonic()
            elif (
                stable_since is not None
                and time.monotonic() - stable_since >= args.checkpoint_stable_seconds
            ):
                reason = "preferred_200k_checkpoint_complete"
        except FileNotFoundError:
            stable_since = None
            stable_signature = None

        now = dt.datetime.now().astimezone()
        if now >= args.deadline:
            reason = "hard_48h_deadline_reached"
        if reason is None:
            time.sleep(args.poll_seconds)

    payload.update({
        "status": "interrupt_sent",
        "stop_reason": reason,
        "interrupt_sent_at": dt.datetime.now().astimezone().isoformat(),
        "preferred_checkpoint_exists": args.preferred_checkpoint.is_file(),
        "preferred_checkpoint_size": (
            args.preferred_checkpoint.stat().st_size
            if args.preferred_checkpoint.is_file() else None
        ),
    })
    atomic_json(args.status, payload)

    if args.queue_pid is not None:
        queue_cmdline = command_line(args.queue_pid)
        if queue_cmdline is None:
            payload["superseded_queue_status"] = "already_exited"
        elif args.expected_queue_command not in queue_cmdline:
            payload.update({
                "status": "failed_closed_queue_pid_identity_mismatch",
                "observed_queue_command": queue_cmdline,
                "observed_at": dt.datetime.now().astimezone().isoformat(),
            })
            atomic_json(args.status, payload)
            raise SystemExit(2)
        else:
            os.kill(args.queue_pid, signal.SIGTERM)
            queue_deadline = time.monotonic() + 15.0
            while (
                command_line(args.queue_pid) is not None
                and time.monotonic() < queue_deadline
            ):
                time.sleep(0.25)
            if command_line(args.queue_pid) is not None:
                payload.update({
                    "status": "failed_closed_superseded_queue_did_not_exit",
                    "observed_at": dt.datetime.now().astimezone().isoformat(),
                })
                atomic_json(args.status, payload)
                raise SystemExit(2)
            payload.update({
                "superseded_queue_status": "terminated_before_trainer",
                "superseded_queue_exit_observed_at": (
                    dt.datetime.now().astimezone().isoformat()
                ),
            })
        atomic_json(args.status, payload)

    os.kill(args.pid, signal.SIGINT)

    graceful_deadline = time.monotonic() + 180.0
    while command_line(args.pid) is not None and time.monotonic() < graceful_deadline:
        time.sleep(2.0)
    if command_line(args.pid) is not None:
        os.kill(args.pid, signal.SIGTERM)
        payload["sigterm_sent_at"] = dt.datetime.now().astimezone().isoformat()
        payload["status"] = "sigterm_sent_after_180s_grace"
    else:
        payload["status"] = "stopped_gracefully"
        payload["trainer_exit_observed_at"] = dt.datetime.now().astimezone().isoformat()
    atomic_json(args.status, payload)


if __name__ == "__main__":
    main()
