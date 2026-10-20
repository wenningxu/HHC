#!/usr/bin/env python
"""Parse and validate the untouched Interact2AR author-evaluation stdout."""

import argparse
import hashlib
import json
import re
from pathlib import Path


SCALAR = re.compile(
    r"^---> \[(?P<label>.+)\] Mean: (?P<mean>-?[0-9.]+) "
    r"CInterval: (?P<ci>-?[0-9.]+)$"
)
TOPK = re.compile(
    r"\(top (?P<rank>[123])\) Mean: (?P<mean>-?[0-9.]+) "
    r"CInt: (?P<ci>-?[0-9.]+);"
)
WRAPPERS = ("JointsFull", "JointsBody", "JointsHands")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not str(audit.get("status", "")).startswith("pass"):
        raise ValueError("paper/repository/data audit did not pass")
    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    replication_ids = [int(match.group(1)) for line in lines if (match := re.match(r"=+ Replication (\d+) =+", line))]
    if replication_ids != list(range(5)):
        raise ValueError(f"expected released replications 0..4, got {replication_ids}")

    section = None
    metrics = {}
    for line in lines:
        match = re.match(r"^=+ (.+) Summary =+$", line)
        if match:
            section = match.group(1)
            continue
        if section is None or not line.startswith("---> ["):
            continue
        scalar = SCALAR.match(line)
        if scalar:
            metrics.setdefault(section, {})[scalar.group("label")] = {
                "mean": float(scalar.group("mean")), "ci95": float(scalar.group("ci"))
            }
            continue
        label_match = re.match(r"^---> \[(?P<label>.+?)\](?P<rest>.*)$", line)
        if label_match:
            ranks = TOPK.findall(label_match.group("rest"))
            if ranks:
                metrics.setdefault(section, {})[label_match.group("label")] = {
                    f"top{rank}": {"mean": float(mean), "ci95": float(ci)}
                    for rank, mean, ci in ranks
                }

    required_sections = {"Matching Score", "R_precision", "FID", "Diversity", "MultiModality"}
    if set(metrics) != required_sections:
        raise ValueError(f"summary sections differ: {sorted(metrics)}")
    for wrapper in WRAPPERS:
        for method in ("ground truth", "Interact2Ar"):
            label = f"{method} ({wrapper})"
            for section in required_sections - {"MultiModality"}:
                if label not in metrics[section]:
                    raise ValueError(f"missing {section}/{label}")
        if f"Interact2Ar ({wrapper})" not in metrics["MultiModality"]:
            raise ValueError(f"missing MultiModality/Interact2Ar ({wrapper})")

    full_fid = metrics["FID"]["Interact2Ar (JointsFull)"]["mean"]
    reported = float(audit["paper_contract"]["reported_interact2ar_joints_full_fid"])
    result = {
        "schema": "interact2ar_author_release_calibration_v1",
        "status": "complete",
        "scope": "untouched_public_README_protocol_with_GT_first_frame",
        "must_not_mix_with_unconditional_fair_table": True,
        "released_replications": 5,
        "paper_replications": 20,
        "samples_per_replication": 1696,
        "seed": 44,
        "cfg_weight": 2.5,
        "first_frame_conditioning": "random ground-truth frame at shared batch index 0..4",
        "metrics": metrics,
        "paper_joints_full_fid": reported,
        "released_five_rep_joints_full_fid": full_fid,
        "absolute_difference_from_paper_mean": abs(full_fid - reported),
        "diagnostic_tolerance_not_paper_claim": 0.05,
        "near_paper_mean": abs(full_fid - reported) <= 0.05,
        "audit": artifact(args.audit),
        "stdout": artifact(args.log),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), "joints_full_fid": full_fid, "near_paper_mean": result["near_paper_mean"]}, sort_keys=True))


if __name__ == "__main__":
    main()
