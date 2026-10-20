#!/usr/bin/env python3
"""Verify online W&B credentials and endpoint reachability without exposing keys."""

from __future__ import annotations

import argparse
import json
import netrc
import os
import socket
import urllib.parse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://api.wandb.ai")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    host = urllib.parse.urlparse(args.base_url).hostname
    if not host:
        raise ValueError("invalid W&B base URL")
    credential_source = None
    if os.environ.get("WANDB_API_KEY"):
        credential_source = "WANDB_API_KEY"
    else:
        try:
            auth = netrc.netrc().authenticators(host)
        except (FileNotFoundError, netrc.NetrcParseError, OSError):
            auth = None
        if auth and auth[2]:
            credential_source = "netrc"
    endpoint_reachable = False
    endpoint_error = None
    try:
        with socket.create_connection((host, 443), timeout=args.timeout):
            endpoint_reachable = True
    except OSError as error:
        endpoint_error = f"{type(error).__name__}: {error}"
    checks = {
        "credential_available": credential_source is not None,
        "endpoint_reachable": endpoint_reachable,
    }
    payload = {
        "schema": "hhc_wandb_online_runtime_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "credential_source": credential_source,
        "credential_value_disclosed": False,
        "base_url": args.base_url,
        "endpoint_host": host,
        "endpoint_error": endpoint_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
