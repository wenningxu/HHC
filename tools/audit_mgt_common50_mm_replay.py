#!/usr/bin/env python
"""Prove that every repaired common-50 MM repeat replays the frozen reference."""

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


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
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


def compare_mm_files(reference_path, candidate_path, expected_repeats=30):
    repeat_records = []
    combined_reference = hashlib.sha256()
    combined_candidate = hashlib.sha256()
    maximum_absolute_difference = 0.0
    nonzero_difference_count = 0
    with h5py.File(reference_path, "r") as reference, h5py.File(
        candidate_path, "r"
    ) as candidate:
        reference_shape = tuple(reference["motions"].shape)
        candidate_shape = tuple(candidate["motions"].shape)
        if reference_shape != candidate_shape:
            raise ValueError(
                f"MM motion shapes differ: {reference_shape} != {candidate_shape}"
            )
        if len(candidate_shape) != 4 or candidate_shape[1] != expected_repeats:
            raise ValueError(
                f"expected [sample,{expected_repeats},time,feature] MM data, "
                f"got {candidate_shape}"
            )
        completed = np.asarray(candidate["completed_repeats"][:], dtype=bool)
        if completed.shape != (expected_repeats,) or not completed.all():
            raise ValueError("candidate MM repeats are incomplete")
        names_equal = np.array_equal(reference["names"][:], candidate["names"][:])
        lengths_equal = np.array_equal(
            reference["lengths"][:], candidate["lengths"][:]
        )
        for repeat in range(expected_repeats):
            reference_motion = np.asarray(reference["motions"][:, repeat])
            candidate_motion = np.asarray(candidate["motions"][:, repeat])
            reference_bytes = reference_motion.tobytes()
            candidate_bytes = candidate_motion.tobytes()
            combined_reference.update(reference_bytes)
            combined_candidate.update(candidate_bytes)
            differences = np.abs(reference_motion - candidate_motion)
            repeat_maximum = float(differences.max())
            repeat_nonzero = int(np.count_nonzero(differences))
            maximum_absolute_difference = max(
                maximum_absolute_difference, repeat_maximum
            )
            nonzero_difference_count += repeat_nonzero
            repeat_records.append({
                "repeat": repeat,
                "motions_byte_exact": reference_bytes == candidate_bytes,
                "maximum_absolute_motion_difference": repeat_maximum,
                "nonzero_motion_element_count": repeat_nonzero,
                "reference_motion_bytes_sha256": hashlib.sha256(
                    reference_bytes
                ).hexdigest(),
                "candidate_motion_bytes_sha256": hashlib.sha256(
                    candidate_bytes
                ).hexdigest(),
            })
        contracts = {
            "training": candidate.attrs.get("training_conditioning_contract"),
            "inference": candidate.attrs.get("inference_conditioning_contract"),
        }
    exact = all(record["motions_byte_exact"] for record in repeat_records)
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
    return {
        "status": status,
        "samples": candidate_shape[0],
        "repeats": expected_repeats,
        "names_equal": bool(names_equal),
        "lengths_equal": bool(lengths_equal),
        "motions_byte_exact": exact,
        "maximum_absolute_motion_difference": maximum_absolute_difference,
        "nonzero_motion_element_count": nonzero_difference_count,
        "reference_motion_bytes_sha256": combined_reference.hexdigest(),
        "candidate_motion_bytes_sha256": combined_candidate.hexdigest(),
        "conditioning_contracts": contracts,
        "repeat_records": repeat_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--expected-repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    reference_exporter = manifest["provenance"]["implementation"]["exporter"]
    if reference_exporter.get("sha256") != REFERENCE_EXPORTER_SHA256:
        raise RuntimeError("reference exporter identity changed")
    result = compare_mm_files(
        args.reference, args.candidate, expected_repeats=args.expected_repeats
    )
    payload = {
        "schema": "hhc_mgt_common50_mm_replay_v1",
        **result,
        "reference_exporter": reference_exporter,
        "reference_generations": artifact(args.reference),
        "candidate_generations": artifact(args.candidate),
        "reference_manifest": artifact(args.reference_manifest),
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
