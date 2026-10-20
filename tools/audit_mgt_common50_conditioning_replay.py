#!/usr/bin/env python
"""Prove that separated inference CFG replays frozen pre-cleanup motions."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


REFERENCE_EXPORTER_SHA256 = (
    "312ee55ca12fe92f29108bf513e1da9fcde394508e119cc4b29bbbf4239dd76e"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    reference_exporter = manifest["provenance"]["implementation"]["exporter"]
    if reference_exporter.get("sha256") != REFERENCE_EXPORTER_SHA256:
        raise RuntimeError("reference exporter identity changed")

    with h5py.File(args.reference, "r") as reference, h5py.File(
        args.candidate, "r"
    ) as candidate:
        count = args.count
        if count <= 0 or len(candidate["motions"]) < count:
            raise ValueError("candidate does not contain the requested replay count")
        names_equal = np.array_equal(
            reference["names"][:count], candidate["names"][:count]
        )
        lengths_equal = np.array_equal(
            reference["lengths"][:count], candidate["lengths"][:count]
        )
        reference_motion = reference["motions"][:count, 0]
        candidate_motion = candidate["motions"][:count, 0]
        differences = np.abs(reference_motion - candidate_motion)
        exact = bool(np.array_equal(reference_motion, candidate_motion))
        contracts = {
            "training": candidate.attrs.get("training_conditioning_contract"),
            "inference": candidate.attrs.get("inference_conditioning_contract"),
        }
        status = (
            "pass"
            if names_equal
            and lengths_equal
            and exact
            and contracts
            == {
                "training": "historical_mmm_shape_dropout",
                "inference": "paper_text_cfg",
            }
            else "fail"
        )
        payload = {
            "schema": "hhc_mgt_common50_conditioning_replay_v1",
            "status": status,
            "samples": count,
            "names_equal": bool(names_equal),
            "lengths_equal": bool(lengths_equal),
            "motions_byte_exact": exact,
            "maximum_absolute_motion_difference": float(differences.max()),
            "nonzero_motion_element_count": int(np.count_nonzero(differences)),
            "reference_motion_bytes_sha256": hashlib.sha256(
                reference_motion.tobytes()
            ).hexdigest(),
            "candidate_motion_bytes_sha256": hashlib.sha256(
                candidate_motion.tobytes()
            ).hexdigest(),
            "conditioning_contracts": contracts,
            "reference_exporter": reference_exporter,
            "reference_generations": artifact(args.reference),
            "candidate_generations": artifact(args.candidate),
            "reference_manifest": artifact(args.reference_manifest),
            "implementation": artifact(Path(__file__)),
        }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
