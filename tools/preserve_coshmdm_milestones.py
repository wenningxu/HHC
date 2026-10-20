#!/usr/bin/env python3
"""Preserve stable CoShMDM 20k checkpoints as zero-copy hard links."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import time


SOURCE_PATTERN = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt$")


def command_line(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(
            b"\0", b" "
        ).decode()
    except FileNotFoundError:
        return None


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def eligible_sources(directory: Path, maximum_step: int) -> list[tuple[int, Path]]:
    records = []
    for path in directory.glob("epoch=*-step=*.ckpt"):
        match = SOURCE_PATTERN.fullmatch(path.name)
        if not match:
            continue
        step = int(match.group(2))
        if 0 < step <= maximum_step and step % 20_000 == 0:
            records.append((step, path.resolve()))
    return sorted(records)


def preserve(source: Path, target: Path) -> str:
    if target.exists():
        if not os.path.samefile(source, target):
            raise RuntimeError(
                f"milestone target exists but is not the source inode: {target}"
            )
        return "preexisting_same_inode"
    try:
        os.link(source, target)
    except FileExistsError:
        if not os.path.samefile(source, target):
            raise RuntimeError(
                f"milestone target raced with a different inode: {target}"
            )
        return "created_by_racing_peer"
    if not os.path.samefile(source, target):
        raise RuntimeError(f"hard-link preservation failed for {target}")
    return "created_hard_link"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--expected-command", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--maximum-step", type=int, default=200_000)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--stable-seconds", type=float, default=45.0)
    args = parser.parse_args()

    directory = args.checkpoint_dir.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    payload = {
        "schema": "hhc_coshmdm_milestone_preservation_v1",
        "status": "monitoring",
        "trainer_pid": args.pid,
        "expected_command_substring": args.expected_command,
        "checkpoint_directory": str(directory),
        "maximum_preserved_step": args.maximum_step,
        "interval_steps": 20_000,
        "method": "stable source inode hard-linked without copying",
        "stable_seconds": args.stable_seconds,
        "started_at": dt.datetime.now().astimezone().isoformat(),
        "preserved": [],
    }
    args.status.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.status, payload)

    signatures: dict[str, tuple[tuple[int, int], float]] = {}
    preserved_steps: set[int] = set()
    while True:
        cmdline = command_line(args.pid)
        if cmdline is None:
            payload.update({
                "status": "trainer_exited",
                "completed_at": dt.datetime.now().astimezone().isoformat(),
            })
            atomic_json(args.status, payload)
            return
        if args.expected_command not in cmdline:
            payload.update({
                "status": "failed_closed_pid_identity_mismatch",
                "observed_command": cmdline,
                "failed_at": dt.datetime.now().astimezone().isoformat(),
            })
            atomic_json(args.status, payload)
            raise SystemExit(2)

        now = time.monotonic()
        for step, source in eligible_sources(directory, args.maximum_step):
            if step in preserved_steps:
                continue
            stat = source.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            previous = signatures.get(str(source))
            if previous is None or previous[0] != signature:
                signatures[str(source)] = (signature, now)
                continue
            if now - previous[1] < args.stable_seconds:
                continue
            target = directory / f"milestone-{step:09d}.ckpt"
            result = preserve(source, target)
            target_stat = target.stat()
            payload["preserved"].append({
                "step": step,
                "source": str(source),
                "target": str(target),
                "result": result,
                "same_inode": os.path.samefile(source, target),
                "inode": target_stat.st_ino,
                "bytes": target_stat.st_size,
                "preserved_at": dt.datetime.now().astimezone().isoformat(),
            })
            preserved_steps.add(step)
            atomic_json(args.status, payload)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
